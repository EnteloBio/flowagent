"""Generate manuscript Supplementary Table 4 — pricing provenance.

Provider list prices change frequently, so any dollar figure in the
manuscript (Supplementary Table 2, ``planning_cost_summary``, the cost
figures) is only reproducible if the reader knows *which* unit prices
were used and *when* both the prices and the cost figures were captured.
This table records exactly that:

* the input / output USD-per-1,000-token rate used for every evaluated
  model (read straight from ``config/models.yaml``), and
* the pricing snapshot date (``defaults.pricing_snapshot`` in the YAML,
  overridable per model with ``pricing_as_of:``), and
* in the caption, when the cost figures themselves were computed — the
  date range and git revision(s) of the merged planning sweep that the
  costs were derived from.

A reader can reconstruct any reported cost as
``prompt_tokens * input_per_1k/1000 + completion_tokens * output_per_1k/1000``
using the rates below.

Outputs (under ``results/figures/``):

* ``supp_table4_pricing.md``   — manuscript-paste Markdown table.
* ``supp_table4_pricing.tsv``  — machine-readable version.

By default the model list is taken from the latest merged planning run
(so Table 4 lists the same models as Table 2); ``--run`` points at a
specific run, and ``--all-models`` lists every priced model in the YAML
regardless of whether it appears in a sweep.

Usage::

    python make_supp_table4.py
    python make_supp_table4.py --run results/planning/_merged/<TS>
    python make_supp_table4.py --all-models
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

_HERE = Path(__file__).parent


_PROVIDER_BY_PREFIX = [
    ("gpt", "openai"), ("o1", "openai"), ("o3", "openai"), ("o4", "openai"),
    ("claude", "anthropic"),
    ("gemini", "google"),
]


def _provider_for(model: str) -> str:
    for pref, prov in _PROVIDER_BY_PREFIX:
        if model.startswith(pref):
            return prov
    return ""


# ── Merged-run discovery (mirrors make_supp_tables / make_supp_table3) ──

def _latest_merged(base: Path) -> Optional[Path]:
    root = base / "planning" / "_merged"
    if not root.exists():
        return None
    runs = sorted([p for p in root.iterdir() if p.is_dir()],
                  key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def _load_manifest(run_dir: Path) -> Optional[Dict[str, Any]]:
    mf = run_dir / "manifest.json"
    if not mf.exists():
        return None
    try:
        return json.loads(mf.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _resolve(base: Path, run_dir: str) -> Path:
    p = Path(run_dir)
    if p.is_absolute():
        return p
    cand = _HERE / p
    if cand.exists():
        return cand
    return base.parent / p


def _models_in_run(run_dir: Path, tier: str = "transcription") -> List[str]:
    """Distinct model ids in the run's metrics.csv (explicit-tool tier).

    Filtered to ``tier`` so the model set matches Supplementary Table 2,
    which reports the explicit-tool (``transcription``) sweep. Rows with
    no ``tier`` column are kept (older runs predate the field).
    """
    metrics = run_dir / "metrics.csv"
    if not metrics.exists():
        return []
    seen: List[str] = []
    with metrics.open() as f:
        for r in csv.DictReader(f):
            if tier and "tier" in r and (r.get("tier") or "") != tier:
                continue
            m = (r.get("model") or "").strip()
            if m and m not in seen:
                seen.append(m)
    return seen


def _cost_provenance(run_dir: Path, base: Path) -> Dict[str, Any]:
    """When/where the cost figures were produced: dates + git SHAs."""
    top = _load_manifest(run_dir) or {}
    timestamps: List[str] = []
    shas: Dict[str, int] = {}
    sources = top.get("sources")
    if isinstance(sources, list):
        for s in sources:
            sd = _resolve(base, s.get("run_dir", ""))
            man = _load_manifest(sd) or {}
            ts = man.get("timestamp")
            if ts:
                timestamps.append(str(ts)[:10])
            sha = man.get("git_sha")
            if sha and str(sha).lower() not in ("", "unknown", "none"):
                shas[str(sha)] = shas.get(str(sha), 0) + 1
    # Plain (non-merged) run.
    if not timestamps and top.get("timestamp"):
        timestamps.append(str(top["timestamp"])[:10])
    top_sha = top.get("git_sha")
    if not shas and top_sha and str(top_sha).lower() not in ("", "unknown", "none"):
        shas[str(top_sha)] = 1
    merged_at = top.get("merged_at")
    return {
        "merged_at": (str(merged_at)[:10] if merged_at else None),
        "date_lo": (min(timestamps) if timestamps else None),
        "date_hi": (max(timestamps) if timestamps else None),
        "shas": sorted(shas, key=lambda s: -shas[s]),
    }


# ── Table build ───────────────────────────────────────────────────────

def _fmt_price(v: Any) -> str:
    try:
        return f"{float(v):.5f}"
    except (TypeError, ValueError):
        return "—"


def build_rows(models_yaml: Path,
               model_ids: Optional[List[str]]) -> Tuple[List[Dict[str, Any]], str]:
    doc = yaml.safe_load(models_yaml.read_text())
    default_snapshot = (doc.get("defaults") or {}).get("pricing_snapshot", "")
    reg = {m["id"]: m for m in doc.get("models", [])}

    ids = model_ids if model_ids else sorted(reg)
    rows: List[Dict[str, Any]] = []
    for mid in ids:
        meta = reg.get(mid, {})
        pricing = meta.get("pricing") or {}
        rows.append({
            "model": mid,
            "provider": meta.get("provider") or _provider_for(mid),
            "family": meta.get("family", ""),
            "tier": meta.get("tier", ""),
            "input_per_1k": pricing.get("input_per_1k", ""),
            "output_per_1k": pricing.get("output_per_1k", ""),
            "pricing_as_of": meta.get("pricing_as_of") or default_snapshot or "—",
        })
    order = {"openai": 0, "anthropic": 1, "google": 2}
    rows.sort(key=lambda r: (order.get(r["provider"], 9), r["model"]))
    return rows, default_snapshot


# ── Rendering ─────────────────────────────────────────────────────────

def _caption(rows: List[Dict[str, Any]], default_snapshot: str,
             prov: Dict[str, Any], source_name: Optional[str]) -> str:
    snaps = sorted({str(r["pricing_as_of"]) for r in rows if r["pricing_as_of"] not in ("", "—")})
    snap_str = ", ".join(snaps) if snaps else (default_snapshot or "unspecified")

    if prov.get("date_lo") and prov.get("date_hi"):
        if prov["date_lo"] == prov["date_hi"]:
            when = f"on {prov['date_lo']}"
        else:
            when = f"between {prov['date_lo']} and {prov['date_hi']}"
    elif prov.get("merged_at"):
        when = f"on {prov['merged_at']}"
    else:
        when = "on an unrecorded date"

    shas = prov.get("shas") or []
    if len(shas) == 1:
        sha_str = f" at git revision `{shas[0][:12]}`"
    elif len(shas) > 1:
        sha_str = (" at git revisions "
                   + ", ".join(f"`{s[:12]}`" for s in shas[:3])
                   + ("…" if len(shas) > 3 else ""))
    else:
        sha_str = ""

    src = f" (merged sweep `{source_name}`)" if source_name else ""

    return (
        f"**Supplementary Table 4. Provider pricing used for all cost "
        f"calculations.** Every monetary figure in this work "
        f"(Supplementary Table 2, the planning cost summary, and the cost "
        f"figures) was computed from the provider list prices below, in "
        f"USD per 1,000 tokens, captured as of {snap_str}. Cost = "
        f"`prompt_tokens × input_per_1k / 1000 + completion_tokens × "
        f"output_per_1k / 1000`. The cost figures themselves were generated "
        f"from the planning sweep run {when}{sha_str}{src}. Provider list "
        f"prices change frequently; reproducing the absolute dollar values "
        f"requires the rates in effect on the snapshot date, while relative "
        f"comparisons (cost-per-pass fold-changes) are robust to uniform "
        f"price drift.\n"
    )


def render_md(rows: List[Dict[str, Any]], default_snapshot: str,
              prov: Dict[str, Any], source_name: Optional[str]) -> str:
    cols = ["Model ID", "Provider", "Family", "Tier",
            "Input $/1k", "Output $/1k", "Pricing as of"]
    lines = [_caption(rows, default_snapshot, prov, source_name),
             "| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join([
            f"`{r['model']}`",
            r["provider"] or "—",
            r["family"] or "—",
            r["tier"] or "—",
            _fmt_price(r["input_per_1k"]),
            _fmt_price(r["output_per_1k"]),
            str(r["pricing_as_of"]),
        ]) + " |")
    return "\n".join(lines)


def _write_tsv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-base", default="results")
    ap.add_argument("--run", default=None,
                    help="Run dir whose models/provenance to use "
                         "(default: latest merged planning run)")
    ap.add_argument("--models-yaml", default="config/models.yaml")
    ap.add_argument("--out-dir", default="results/figures")
    ap.add_argument("--all-models", action="store_true",
                    help="List every priced model in the YAML, not just the "
                         "models present in the merged sweep")
    args = ap.parse_args()

    base = (_HERE / args.results_base) if not Path(args.results_base).is_absolute() \
        else Path(args.results_base)
    out_dir = (_HERE / args.out_dir) if not Path(args.out_dir).is_absolute() \
        else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir: Optional[Path] = Path(args.run) if args.run else _latest_merged(base)

    model_ids: Optional[List[str]] = None
    prov: Dict[str, Any] = {}
    source_name: Optional[str] = None
    if run_dir is not None and run_dir.exists() and not args.all_models:
        model_ids = _models_in_run(run_dir) or None
        prov = _cost_provenance(run_dir, base)
        source_name = run_dir.name
    elif run_dir is not None and run_dir.exists():
        # --all-models, but still record provenance from the run.
        prov = _cost_provenance(run_dir, base)
        source_name = run_dir.name

    rows, default_snapshot = build_rows(_HERE / args.models_yaml, model_ids)
    if not rows:
        raise SystemExit("No models to render; check config/models.yaml.")

    md = render_md(rows, default_snapshot, prov, source_name)
    md_path = out_dir / "supp_table4_pricing.md"
    md_path.write_text(md + "\n")
    tsv_path = out_dir / "supp_table4_pricing.tsv"
    _write_tsv(tsv_path, rows)

    print(f"[ok] Table 4 → {md_path} ({len(rows)} models; "
          f"pricing as of {default_snapshot or 'unspecified'})")
    print(f"[ok] pricing TSV → {tsv_path}")
    if prov.get("date_lo"):
        print(f"[info] cost figures computed "
              f"{prov['date_lo']}…{prov.get('date_hi')} "
              f"(source {source_name})")


if __name__ == "__main__":
    main()
