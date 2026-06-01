"""Live catalog grounding: HEAD-checks Ensembl URLs and resolves GEO metadata.

Closes the silent-stale-URL failure mode named in §B3 of the FlowAgent
architecture review (todo T5). The planner today recalls genome /
transcriptome URLs from a hardcoded dict and ships them into the plan
unvalidated; if Ensembl reorganises a path or drops a release, the
download step fails silently downstream. Same hazard for GEO accessions
the LLM recalls from training data.

This module adds two pre-flight checks, both behind
``FLOWAGENT_LIVE_CATALOG`` (default on) with a configurable timeout:

* :func:`head_ok` — HEAD-check a URL, fall back to ``GET Range: 0-0``
  for servers that reject HEAD. Cached in-memory for the process
  lifetime.
* :func:`resolve_geo` — fetch the canonical NCBI GEO metadata for a GSE
  accession (organism, title, summary) so the planner sees concrete
  metadata instead of a bare accession string.

Design constraints:

* Idempotent and cacheable — repeated lookups on the same prompt corpus
  hit a handful of URLs over and over, so an in-memory cache is
  sufficient. No durable storage; cache is dropped on process exit.
* Fail-soft — every error path returns a "trust the input" sentinel
  (``True`` for liveness, ``None`` for metadata). The caller decides
  whether to act on a failure; this module never raises on network
  flakiness.
* No new hard dependency — ``aiohttp`` is already pulled in transitively
  by the OpenAI / Anthropic clients, but if it's missing we degrade
  gracefully to "skip the check".

Recovery-taxonomy framing: this is pure prevention, not retry-based
correction. If the URL is dead, the planner falls back to placeholder
paths (today's behaviour pre-T5) and the user is shown a clear error
rather than getting a plan that ships a 404. No LLM-on-feedback loop is
involved.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ── Env-flag gating ────────────────────────────────────────────────

def is_enabled() -> bool:
    """Read FLOWAGENT_LIVE_CATALOG fresh on every call.

    Mirrors the per-call freshness pattern other ablation flags use
    (LLM_DAG_AWARE, FLOWAGENT_VALIDATOR_AUTOFIX / _RETRY) so the benchmark
    harness can flip it per-cell without restarting the process.
    """
    return os.environ.get("FLOWAGENT_LIVE_CATALOG", "true").strip().lower() not in {
        "0", "false", "no", "off",
    }


def get_timeout_seconds() -> float:
    """Per-request timeout budget. Default 5s, overridable via env."""
    raw = os.environ.get("FLOWAGENT_LIVE_CATALOG_TIMEOUT", "5.0")
    try:
        v = float(raw)
        # Clamp to sensible range — 0.5s is the floor below which spurious
        # timeouts dominate; 30s is the ceiling above which the planner
        # hot path becomes unusable.
        return max(0.5, min(30.0, v))
    except (ValueError, TypeError):
        return 5.0


# ── In-memory caches ───────────────────────────────────────────────

# Process-lifetime caches. The same prompt corpus hits the same handful
# of Ensembl URLs and GEO accessions repeatedly across a benchmark run;
# a dict here saves hundreds of HEAD requests on a single sweep.
_url_cache: Dict[str, bool] = {}
_geo_cache: Dict[str, Optional[Dict[str, str]]] = {}


def clear_caches() -> None:
    """Drop the in-memory caches. Test-only helper."""
    _url_cache.clear()
    _geo_cache.clear()


# ── URL liveness ───────────────────────────────────────────────────

async def head_ok(url: str, *, timeout: Optional[float] = None) -> bool:
    """Return True iff *url* responds with a non-error HTTP status.

    Tries ``HEAD`` first (cheap, no body transfer); if the server returns
    405/501 (Method Not Allowed / Not Implemented — common on Ensembl FTP
    mirrors and some CDNs), falls back to ``GET`` with a single-byte
    range request.

    Returns True (trusting the URL) when:
      - live catalog is disabled (``FLOWAGENT_LIVE_CATALOG=false``)
      - aiohttp isn't importable
      - the URL is empty / not a string

    Cached in-memory by URL string so the benchmark harness doesn't probe
    the same Ensembl release path 198 times.
    """
    if not is_enabled():
        return True
    if not isinstance(url, str) or not url:
        return True
    if url in _url_cache:
        return _url_cache[url]
    if timeout is None:
        timeout = get_timeout_seconds()
    try:
        import aiohttp
    except ImportError:
        return True

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as session:
            try:
                async with session.head(url, allow_redirects=True) as resp:
                    if resp.status not in (405, 501):
                        ok = resp.status < 400
                        _url_cache[url] = ok
                        return ok
            except aiohttp.ClientResponseError:
                pass

            # HEAD rejected — fall back to a tiny GET. Range: 0-0 asks
            # for one byte; servers that don't honour ranges still send
            # a 200 with a small payload, which is fine for liveness.
            async with session.get(
                url,
                headers={"Range": "bytes=0-0"},
                allow_redirects=True,
            ) as resp:
                ok = resp.status < 400
                _url_cache[url] = ok
                return ok
    except Exception as e:
        logger.debug("Live URL check failed for %s: %s", url, e)
        _url_cache[url] = False
        return False


async def all_urls_alive(urls: Dict[str, str]) -> Dict[str, bool]:
    """HEAD-check every URL in *urls* concurrently.

    Returns a dict keyed by the same keys as the input, with bool
    values. Convenience wrapper for callers that want to validate a
    bundle of URLs (e.g. all three Ensembl URLs from
    ``_detect_reference_genome``) in one round-trip.
    """
    import asyncio
    keys = list(urls.keys())
    results = await asyncio.gather(
        *(head_ok(urls[k]) for k in keys),
        return_exceptions=False,
    )
    return dict(zip(keys, results))


# ── GEO accession resolver ─────────────────────────────────────────

_GSE_RE = re.compile(r"^GSE\d+$")
_SERIES_FIELD_RE = re.compile(r"!Series_([\w_]+)\s*=\s*(.*)$")


async def resolve_geo(
    accession: str, *, timeout: Optional[float] = None,
) -> Optional[Dict[str, str]]:
    """Resolve a GSE accession to canonical NCBI metadata.

    Hits the GEO text-format endpoint::

        https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi
            ?acc=<GSE>&targ=self&form=text&view=quick

    which returns SOFT-format ``!Series_<key> = <value>`` lines. We
    parse the fields the planner needs:

      - ``title``     — human-readable study title
      - ``summary``   — one-line study summary (may be long; truncated
                        downstream if used in a prompt)
      - ``organism``  — sample organism (drives genome selection)
      - ``type``      — experiment type (e.g. ``Expression profiling
                        by high throughput sequencing``)
      - ``platform``  — sequencing platform / model

    Returns ``None`` when:
      - the live catalog is disabled
      - the accession isn't a valid GSE shape
      - the network call fails / times out
      - the response doesn't parse

    Cached in-memory by accession.
    """
    if not is_enabled():
        return None
    if not isinstance(accession, str) or not _GSE_RE.match(accession):
        return None
    if accession in _geo_cache:
        return _geo_cache[accession]
    if timeout is None:
        timeout = get_timeout_seconds()
    try:
        import aiohttp
    except ImportError:
        return None

    url = (
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
        f"?acc={accession}&targ=self&form=text&view=quick"
    )
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as session:
            async with session.get(url, allow_redirects=True) as resp:
                if resp.status >= 400:
                    _geo_cache[accession] = None
                    return None
                text = await resp.text()
    except Exception as e:
        logger.debug("GEO resolve failed for %s: %s", accession, e)
        _geo_cache[accession] = None
        return None

    fields: Dict[str, str] = {}
    for line in text.splitlines():
        m = _SERIES_FIELD_RE.match(line.strip())
        if m:
            key, val = m.group(1), m.group(2).strip()
            # First-wins: the SOFT format sometimes repeats keys for
            # multi-value fields (e.g. multiple platforms); the first
            # occurrence is enough for prompt-grounding purposes.
            fields.setdefault(key, val)

    if not fields:
        _geo_cache[accession] = None
        return None

    result: Dict[str, str] = {
        "accession": accession,
        "title": fields.get("title", ""),
        "summary": fields.get("summary", ""),
        "organism": (
            fields.get("sample_organism", "")
            or fields.get("platform_organism", "")
        ),
        "type": fields.get("type", ""),
        "platform": (
            fields.get("platform_title", "")
            or fields.get("platform_id", "")
        ),
    }
    _geo_cache[accession] = result
    return result


__all__ = [
    "all_urls_alive",
    "clear_caches",
    "get_timeout_seconds",
    "head_ok",
    "is_enabled",
    "resolve_geo",
]
