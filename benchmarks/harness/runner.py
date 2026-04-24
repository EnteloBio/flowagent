"""Shared sweep runner for the benchmark harnesses.

Handles:
  - switching LLM provider via environment variables
  - the (model × prompt × replicate) cartesian sweep
  - per-cell exception handling (one failure doesn't kill the run)
  - manifest emission (git SHA, versions, timing, seeds)
  - CSV + JSON result serialisation

Does NOT depend on any one benchmark's semantics — ``run_one`` is a
user-supplied async function receiving ``(model_cfg, input_entry, replicate)``
and returning a JSON-serialisable dict.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

import yaml


# ── .env loading ──────────────────────────────────────────────────

def _load_dotenv_once() -> Optional[Path]:
    """Walk up from this file looking for a ``.env`` and merge it into
    ``os.environ``. Shell-set values win (we don't override them).

    Runs once at import time so every benchmark script picks up API keys
    from the repo's ``.env`` file without the user having to ``source`` it.
    Uses ``python-dotenv`` if available, otherwise a minimal parser.
    """
    start = Path(__file__).resolve()
    for parent in [start.parent] + list(start.parents):
        candidate = parent / ".env"
        if candidate.is_file():
            try:
                from dotenv import load_dotenv  # type: ignore
                load_dotenv(candidate, override=False)
            except ImportError:
                # Tiny fallback parser — KEY=VALUE per line, no interpolation.
                for raw in candidate.read_text().splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    os.environ.setdefault(key, val)
            return candidate
    return None


_DOTENV_PATH = _load_dotenv_once()


# ── Provider switching ────────────────────────────────────────────

def set_provider(model_cfg: Dict[str, Any]) -> None:
    """Switch FlowAgent to a specific LLM provider+model for the current process.

    Mutates environment variables that ``flowagent.config.settings.Settings``
    reads. Calls to ``Settings()`` after this function return the new values
    because ``Settings`` is re-constructed on demand in FlowAgent.
    """
    os.environ["LLM_PROVIDER"] = model_cfg["provider"]
    os.environ["LLM_MODEL"] = model_cfg["id"]

    env_var = model_cfg.get("env_var")
    if env_var and env_var not in os.environ:
        raise RuntimeError(
            f"Environment variable {env_var!r} is required for model "
            f"{model_cfg['id']!r} but is not set."
        )

    base_url = model_cfg.get("base_url")
    if base_url:
        os.environ["LLM_BASE_URL"] = base_url


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Manifest ──────────────────────────────────────────────────────

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _package_versions() -> Dict[str, str]:
    """Best-effort snapshot of installed package versions."""
    try:
        from importlib.metadata import distributions
        return {d.metadata["Name"]: d.version for d in distributions()
                if d.metadata.get("Name")}
    except Exception:
        return {}


def write_manifest(out_dir: Path, *, benchmark: str,
                   models: Sequence[Dict[str, Any]],
                   extra: Optional[Dict[str, Any]] = None) -> None:
    """Write ``manifest.json`` with provenance for reviewer reproduction."""
    manifest = {
        "benchmark": benchmark,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "models": [{k: v for k, v in m.items() if k != "env_var"}
                   for m in models],
        # Redact sensitive env vars
        "env_snapshot": {
            k: (v if not any(s in k for s in ("KEY", "TOKEN", "SECRET"))
                else "<redacted>")
            for k, v in os.environ.items()
            if k.startswith(("LLM_", "OPENAI_", "ANTHROPIC_", "GOOGLE_", "OLLAMA_"))
        },
        "packages": _package_versions(),
    }
    if extra:
        manifest.update(extra)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


# ── Sweep driver ──────────────────────────────────────────────────

@dataclass
class SweepResult:
    out_dir: Path
    results: List[Dict[str, Any]]


async def sweep(
    run_one: Callable[[Dict[str, Any], Dict[str, Any], int], Awaitable[Dict[str, Any]]],
    *,
    models: Sequence[Dict[str, Any]],
    inputs: Sequence[Dict[str, Any]],
    replicates: int,
    out_dir: Path,
    benchmark_name: str,
    concurrency: int = 4,
) -> SweepResult:
    """Execute the (model × input × replicate) sweep.

    Incremental + resumable:
      * Appends each completed cell to ``results.jsonl`` as it finishes, so
        partial progress survives a ``Ctrl-C`` / timeout / OOM.
      * On startup, reads any existing ``results.jsonl`` in ``out_dir`` and
        skips cells already present (keyed by ``model, input_id, replicate``).
        To resume, point at the existing timestamped dir instead of creating
        a fresh one (``--resume`` in ``bench_planning.py``).
      * Logs one line per cell completion with index / total / model /
        prompt / status / cost / wall-time — so progress is visible in the
        terminal and trivially greppable (each line starts with ``[N/TOTAL]``).
      * At end, consolidates ``results.jsonl`` → ``results.json`` +
        ``metrics.csv`` with a schema that spans every field observed.

    ``run_one`` MUST be self-contained: a failure in one cell must not
    affect other cells. We catch all exceptions, record them as a
    ``{"error": ...}`` row, and continue.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()

    results_jsonl = out_dir / "results.jsonl"

    # ── Resume: load any existing rows and skip their (model, input, rep) cells.
    existing_rows: List[Dict[str, Any]] = []
    completed_keys: set = set()
    if results_jsonl.exists():
        with results_jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                existing_rows.append(row)
                completed_keys.add((
                    row.get("model"),
                    row.get("input_id"),
                    row.get("replicate"),
                ))
        if existing_rows:
            print(
                f"[resume] {results_jsonl.name}: {len(existing_rows)} cells "
                f"already complete, will be skipped",
                flush=True,
            )

    total_cells = len(models) * len(inputs) * replicates
    pending = [
        (model_cfg, entry, rep)
        for model_cfg in models
        for entry in inputs
        for rep in range(replicates)
        if (model_cfg["id"], entry.get("id"), rep) not in completed_keys
    ]
    print(
        f"[sweep] {total_cells} total cells "
        f"({len(existing_rows)} cached, {len(pending)} to run, "
        f"concurrency={concurrency})",
        flush=True,
    )

    async def _append_row(row: Dict[str, Any]) -> None:
        async with write_lock:
            with results_jsonl.open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")

    async def _guarded(model_cfg, entry, rep, idx):
        async with sem:
            t0 = time.perf_counter()
            try:
                result = await run_one(model_cfg, entry, rep)
            except Exception as exc:
                result = {
                    "model": model_cfg["id"],
                    "input_id": entry.get("id", "?"),
                    "replicate": rep,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            result.setdefault("wall_seconds", time.perf_counter() - t0)
            # Ensure key fields are present for the resume index.
            result.setdefault("model", model_cfg["id"])
            result.setdefault("input_id", entry.get("id", "?"))
            result.setdefault("replicate", rep)

            await _append_row(result)

            status = "err" if result.get("error") else "ok"
            cost = float(result.get("cost_usd") or 0.0)
            wall = float(result.get("wall_seconds") or 0.0)
            print(
                f"[{idx:>4}/{total_cells}] "
                f"{model_cfg['id']:<32s} "
                f"{str(entry.get('id', '?')):<28s} "
                f"rep={rep} {status} "
                f"${cost:6.4f} {wall:6.1f}s",
                flush=True,
            )
            return result

    # Enumerate pending with a 1-based global index that counts past the
    # cells already loaded from disk (so the [idx/total] line matches what
    # the user sees written to results.jsonl).
    offset = len(existing_rows)
    tasks = [
        _guarded(m, e, r, offset + i + 1)
        for i, (m, e, r) in enumerate(pending)
    ]
    new_results = list(await asyncio.gather(*tasks))
    results = existing_rows + new_results

    # Final consolidated writes — schema covers the union of all fields.
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    _write_csv(out_dir / "metrics.csv", results)
    write_manifest(out_dir, benchmark=benchmark_name, models=models,
                   extra={"num_inputs": len(inputs), "replicates": replicates,
                          "resumed_from": len(existing_rows)})
    return SweepResult(out_dir=out_dir, results=results)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Flat-CSV dump with every scalar field observed across rows."""
    import csv
    if not rows:
        path.write_text("")
        return
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k, v in r.items():
            if k in seen:
                continue
            # Skip nested objects; keep scalars and short lists
            if isinstance(v, (str, int, float, bool)) or v is None:
                keys.append(k); seen.add(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def timestamped_dir(base: Path, name: str) -> Path:
    """Return ``<base>/<name>/<YYYY-MM-DDTHH-MM-SS>``."""
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out = base / name / ts
    out.mkdir(parents=True, exist_ok=True)
    return out
