"""Driver for Benchmark F — runs FlowAgent end-to-end on every fidelity case.

This is the *runner* that ``bench_fidelity.py`` (the scorer) was designed
to feed off. It:

  1. Reads ``config/fidelity_cases.yaml``.
  2. For each (case × model × replicate), creates a workspace at
     ``<results>/fidelity_runs/<case>__<model>__rep<N>/``.
  3. Invokes ``flowagent prompt --workflow --model <m> --non-interactive
     "<case prompt>"`` inside that workspace.
  4. Writes per-cell stdout / stderr to ``run.log`` plus a top-level
     ``_driver.log`` for cross-cell progress.
  5. Checkpoints by output-file existence — a re-run skips any cell whose
     ``output_relpath`` already exists and is non-empty (use ``--force``
     to bypass).
  6. After all cells finish, automatically invokes the scorer
     (``bench_fidelity.py --bulk-dir``) so a single command takes you
     from "no outputs" → "metrics.csv with Spearman / Jaccard / F1".

Each cell costs real money + real wall time. The driver doesn't
parallelise across cells by default because pipelines are bandwidth- and
disk-heavy; pass ``--concurrency N`` to opt in.

Usage:

    # All cases × default model × 1 replicate, sequential
    python bench_fidelity_run.py --model gpt-4.1

    # Cross-model sweep across 3 models, 2 replicates each
    python bench_fidelity_run.py \\
        --models gpt-4.1,claude-sonnet-4-6,gemini-2.5-flash \\
        --replicates 2

    # Just one case, useful for re-running a flaky pipeline
    python bench_fidelity_run.py --model gpt-4.1 \\
        --case gse52778_dex_de

    # Two cells in flight at once (be careful — disk + network heavy)
    python bench_fidelity_run.py --models gpt-4.1,o3 --concurrency 2

    # Force a clean re-run even if outputs exist
    python bench_fidelity_run.py --model gpt-4.1 --force

    # Skip the scoring step (just produce candidate dirs)
    python bench_fidelity_run.py --model gpt-4.1 --no-score

    # Score-only run (existing dirs, no new flowagent invocations)
    python bench_fidelity_run.py --score-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

_HERE = Path(__file__).parent

# Load .env into our own process so the keys flow through to every
# flowagent subprocess via the inherited environment. flowagent's own
# dotenv loader (flowagent/config/settings.py) only checks ``./env`` and
# ``$USER_EXECUTION_DIR/.env``, neither of which resolve from a cell
# four directories deep — so the shim here is what makes
# ``flowagent prompt`` work in cell working directories.
def _load_dotenv_walking_up() -> Optional[Path]:
    sys.path.insert(0, str(_HERE))
    try:
        from harness.runner import _DOTENV_PATH  # type: ignore  # noqa: F401
        return _DOTENV_PATH
    except Exception:
        return None


_DOTENV_PATH = _load_dotenv_walking_up()


# ── Helpers ───────────────────────────────────────────────────────

def _slug(*parts: str) -> str:
    """Filesystem-safe slug for cell directory names."""
    return "__".join(p.replace("/", "-").replace(" ", "_") for p in parts)


def _candidate_complete(cell_dir: Path, output_relpath: str) -> bool:
    """A cell is considered done when its declared output exists + non-empty."""
    out = cell_dir / output_relpath
    return out.exists() and out.stat().st_size > 0


# Per-cell directories / paths that can safely be deleted after a successful
# pipeline run. These hold raw inputs and large intermediates that the
# scorer doesn't need — only the file at ``output_relpath`` is preserved.
_CLEANUP_TARGETS = (
    "raw_data",                                          # FASTQs + SRA cache + reference genome
    "results/rna_seq_kallisto/kallisto_index",           # ~3 GB index
    "results/rna_seq_kallisto/fastqc",                   # FastQC HTML reports
    "results/rna_seq/fastqc",                            # alt path for non-kallisto runs
    "results/macs2/bowtie2_alignments",                  # ChIP/ATAC alignments (if produced here)
    "results/gatk/aligned",                              # variant-calling intermediates
)


def _du_bytes(path: Path) -> int:
    """Recursive size of a directory (in bytes), 0 if missing."""
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def _cleanup_cell(cell_dir: Path, output_relpath: str,
                  logger: logging.Logger) -> int:
    """Delete heavy intermediates after a successful cell.

    Returns the number of bytes freed. Refuses to act unless the cell's
    declared ``output_relpath`` already exists and is non-empty — that's
    the marker that the pipeline finished and the intermediate files
    are safely no longer needed for scoring.
    """
    if not _candidate_complete(cell_dir, output_relpath):
        return 0  # never delete intermediates of a partial / failed run
    freed = 0
    for rel in _CLEANUP_TARGETS:
        target = cell_dir / rel
        if not target.exists():
            continue
        size = _du_bytes(target)
        try:
            shutil.rmtree(target)
        except OSError as exc:
            logger.warning("[cleanup] %s: failed to remove %s: %s",
                           cell_dir.name, rel, exc)
            continue
        freed += size
    if freed:
        logger.info("[cleanup] %s: freed %.1f GB",
                    cell_dir.name, freed / (1 << 30))
    return freed


def _resolve_flowagent() -> str:
    """Find the flowagent CLI for the active Python environment."""
    # Prefer the same Python that's running this script — keeps everyone on
    # the same dependency snapshot. Falls back to PATH if the conda layout
    # isn't standard (e.g. user-installed editable install).
    candidate = Path(sys.executable).parent / "flowagent"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("flowagent")
    if found is None:
        sys.exit("flowagent CLI not on PATH — install with `pip install -e .`")
    return found


# ── Cell dataclass ────────────────────────────────────────────────

@dataclass
class Cell:
    case_id:    str
    model:      str
    replicate:  int
    prompt:     str
    output_relpath: str
    workdir:    Path
    log_path:   Path = field(init=False)

    def __post_init__(self):
        self.log_path = self.workdir / "run.log"

    def is_complete(self) -> bool:
        return _candidate_complete(self.workdir, self.output_relpath)


# ── Driver ────────────────────────────────────────────────────────

def _build_cells(cases: List[Dict[str, Any]],
                 models: Sequence[str],
                 replicates: int,
                 base: Path) -> List[Cell]:
    cells: List[Cell] = []
    for case in cases:
        for model in models:
            for rep in range(replicates):
                workdir = base / _slug(case["id"], model, f"rep{rep}")
                cells.append(Cell(
                    case_id   = case["id"],
                    model     = model,
                    replicate = rep,
                    prompt    = case["prompt"].strip(),
                    output_relpath = case["output_relpath"],
                    workdir   = workdir,
                ))
    return cells


async def _run_cell(cell: Cell, *, flowagent_bin: str, force: bool,
                    timeout_seconds: int,
                    semaphore: asyncio.Semaphore,
                    cleanup: bool,
                    logger: logging.Logger) -> Dict[str, Any]:
    """Invoke ``flowagent prompt`` for one cell.

    Returns a small status dict so the top-level driver can summarise
    outcomes. ``flowagent``'s own logs are streamed line-by-line into
    ``run.log`` so they're inspectable post-hoc without holding the
    whole transcript in memory.
    """
    async with semaphore:
        cell.workdir.mkdir(parents=True, exist_ok=True)

        if cell.is_complete() and not force:
            logger.info("[skip] %s: output already exists",
                        cell.workdir.name)
            # Belt-and-suspenders: even on a skipped cell, run cleanup so
            # users who add ``--cleanup`` mid-sweep can recover space from
            # already-completed cells without re-running them.
            if cleanup:
                _cleanup_cell(cell.workdir, cell.output_relpath, logger)
            return {"cell": cell.workdir.name, "status": "skip",
                    "wall_seconds": 0.0}

        # Persist the prompt to disk so a future reviewer can replay it
        # exactly without re-reading the YAML.
        (cell.workdir / "prompt.txt").write_text(cell.prompt + "\n")

        cmd = [flowagent_bin, "prompt",
               "--workflow", "--non-interactive",
               "--model", cell.model,
               cell.prompt]
        logger.info("[run]  %s  (model=%s, timeout=%ds)",
                    cell.workdir.name, cell.model, timeout_seconds)

        t0 = time.perf_counter()
        # flowagent's settings loader looks at ``$USER_EXECUTION_DIR/.env``
        # as one of its fallback locations. We've already merged .env into
        # our own environment above, so the subprocess inherits the keys
        # directly; pointing USER_EXECUTION_DIR at the benchmarks dir is
        # belt-and-suspenders so a user who deletes the parent .env load
        # but leaves benchmarks/.env in place still gets a working sweep.
        env = os.environ.copy()
        if _DOTENV_PATH is not None:
            env.setdefault("USER_EXECUTION_DIR", str(_DOTENV_PATH.parent))
        with cell.log_path.open("w", buffering=1) as log_fh:
            log_fh.write(f"# command: {' '.join(cmd[:5])} <prompt elided>\n")
            log_fh.write(f"# started: {datetime.now(timezone.utc).isoformat()}\n\n")
            log_fh.flush()
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, cwd=str(cell.workdir), env=env,
                    stdout=log_fh, stderr=subprocess.STDOUT,
                )
                rc = await asyncio.wait_for(proc.wait(),
                                            timeout=timeout_seconds)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                rc = -9
                logger.warning("[timeout] %s after %ds",
                               cell.workdir.name, timeout_seconds)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("[error] %s: %s",
                             cell.workdir.name, exc)
                rc = -1

        wall = round(time.perf_counter() - t0, 1)
        produced = cell.is_complete()
        status = ("ok" if rc == 0 and produced
                  else "no_output" if rc == 0
                  else "timeout" if rc == -9
                  else "fail")
        logger.info("[done] %s  rc=%s  produced=%s  wall=%.1fs",
                    cell.workdir.name, rc, produced, wall)

        # Free disk on success only — never delete intermediates of a
        # failed cell, since the user may want to re-run it.
        freed_bytes = 0
        if cleanup and produced:
            freed_bytes = _cleanup_cell(cell.workdir, cell.output_relpath,
                                        logger)

        return {"cell": cell.workdir.name, "status": status,
                "returncode": rc, "wall_seconds": wall,
                "produced_output": produced,
                "bytes_freed": freed_bytes}


async def _run_all(cells: List[Cell], *, flowagent_bin: str, force: bool,
                   timeout_seconds: int, concurrency: int,
                   cleanup: bool,
                   logger: logging.Logger) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)
    tasks = [_run_cell(c, flowagent_bin=flowagent_bin, force=force,
                       timeout_seconds=timeout_seconds,
                       semaphore=sem, cleanup=cleanup, logger=logger)
             for c in cells]
    return await asyncio.gather(*tasks)


def _setup_logger(driver_log: Path) -> logging.Logger:
    logger = logging.getLogger("fidelity.run")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(driver_log)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _invoke_scorer(bulk_dir: Path, *, results_base: Path,
                   logger: logging.Logger) -> int:
    """Shell out to bench_fidelity.py --bulk-dir to score everything."""
    cmd = [sys.executable, str(_HERE / "bench_fidelity.py"),
           "--bulk-dir", str(bulk_dir),
           "--out", str(results_base)]
    logger.info("[score] %s", " ".join(cmd))
    return subprocess.call(cmd)


# ── CLI ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="config/fidelity_cases.yaml")
    ap.add_argument("--results-base", default="results")
    ap.add_argument("--bulk-dir-name", default="fidelity_runs",
                    help="Sub-directory under results/ where cell dirs live")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--model",   help="Single model ID to run")
    grp.add_argument("--models",  help="Comma-separated model IDs")
    ap.add_argument("--replicates", type=int, default=1,
                    help="Number of independent replicates per (case × model)")
    ap.add_argument("--case", default=None,
                    help="Run only this case ID (default: all in YAML)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="How many cells in flight at once (default: 1; "
                         "pipelines are I/O heavy so be conservative)")
    ap.add_argument("--timeout-hours", type=float, default=24.0,
                    help="Per-cell timeout (default: 24h). Killed if exceeded")
    ap.add_argument("--force", action="store_true",
                    help="Re-run cells even if their output already exists")
    ap.add_argument("--cleanup", action="store_true",
                    help="After each successful cell, delete heavy "
                         "intermediates (raw_data/ — FASTQs + SRA cache + "
                         "reference genome; kallisto index; FastQC HTML) "
                         "to recover disk space. Output file declared in "
                         "fidelity_cases.yaml is preserved. Re-running a "
                         "cleaned cell would re-download — only use after "
                         "you're confident the run is final.")
    ap.add_argument("--no-cleanup", action="store_true",
                    help="Disable cleanup even if --cleanup was set "
                         "(used by the Makefile to allow CLEANUP=0).")
    ap.add_argument("--no-score", action="store_true",
                    help="Skip the scoring step at the end")
    ap.add_argument("--score-only", action="store_true",
                    help="Don't invoke flowagent — just score whatever "
                         "cells are already on disk under --bulk-dir-name")
    args = ap.parse_args()

    cases_path = _HERE / args.cases if not Path(args.cases).is_absolute() \
        else Path(args.cases)
    cfg   = yaml.safe_load(cases_path.read_text())
    cases = cfg.get("cases", [])
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            sys.exit(f"case '{args.case}' not in {cases_path}")

    base = (_HERE / args.results_base if not Path(args.results_base).is_absolute()
            else Path(args.results_base)) / args.bulk_dir_name
    base.mkdir(parents=True, exist_ok=True)

    driver_log = base / "_driver.log"
    logger = _setup_logger(driver_log)
    logger.info("=" * 64)
    logger.info("Benchmark F driver starting")
    logger.info("  cases:        %s", [c["id"] for c in cases])
    logger.info("  bulk dir:     %s", base)
    logger.info("  driver log:   %s", driver_log)
    logger.info("  .env loaded:  %s",
                _DOTENV_PATH if _DOTENV_PATH else "NOT FOUND — flowagent will fail")
    if _DOTENV_PATH is None or not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY is not set in the driver's "
                       "environment — every cell will fail with "
                       "'Missing API key'. Either source your .env "
                       "before running or place .env at benchmarks/.env.")
    logger.info("=" * 64)

    if args.score_only:
        rc = _invoke_scorer(base, results_base=Path(args.results_base),
                            logger=logger)
        sys.exit(rc)

    # Resolve models
    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    elif args.model:
        models = [args.model]
    else:
        sys.exit("provide --model or --models (or use --score-only)")

    cells = _build_cells(cases, models, args.replicates, base)
    n_complete = sum(1 for c in cells if c.is_complete())
    logger.info("plan: %d cells (%d already complete; %d to run)",
                len(cells), n_complete, len(cells) - n_complete)

    flowagent_bin = _resolve_flowagent()
    timeout_s = int(args.timeout_hours * 3600)

    cleanup_enabled = bool(args.cleanup) and not args.no_cleanup
    if cleanup_enabled:
        logger.info("cleanup: ON — raw_data/ + kallisto_index/ + fastqc/ "
                    "deleted after each successful cell")

    t_total = time.perf_counter()
    results = asyncio.run(_run_all(
        cells, flowagent_bin=flowagent_bin, force=args.force,
        timeout_seconds=timeout_s, concurrency=args.concurrency,
        cleanup=cleanup_enabled, logger=logger,
    ))
    wall = time.perf_counter() - t_total

    by_status: Dict[str, int] = {}
    total_freed = 0
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        total_freed += r.get("bytes_freed") or 0
    logger.info("=" * 64)
    logger.info("driver finished in %.1f min", wall / 60.0)
    for status, n in sorted(by_status.items()):
        logger.info("  %-12s  %d", status, n)
    if total_freed:
        logger.info("  cleanup freed %.1f GB across all cells",
                    total_freed / (1 << 30))
    logger.info("=" * 64)

    # Persist the per-cell summary alongside the driver log
    (base / "_driver_summary.json").write_text(json.dumps({
        "ran_at":      datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(wall, 1),
        "results":     results,
    }, indent=2))

    if not args.no_score:
        rc = _invoke_scorer(base, results_base=Path(args.results_base),
                            logger=logger)
        sys.exit(rc)


if __name__ == "__main__":
    main()
