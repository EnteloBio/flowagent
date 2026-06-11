"""Drop selected model ids from planning run artifacts.

Use when a model is retired (e.g. Gemini 1.5) or a sweep should be
excluded from merge/report without deleting other models in the same run.

Touches, when present in each run dir (and ``rescored_*/`` subdirs):

* ``metrics.csv``
* ``results.json``
* ``results.jsonl``

Usage::

    # Remove Gemini 1.5 from every planning run under results/
    python prune_planning_models.py --models gemini-1.5-pro,gemini-1.5-flash --all

    # Then rebuild merged CSV + figures
    make merge REFRESH=1
    make report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Set

import pandas as pd

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))


def _artifact_dirs(run_dir: Path) -> List[Path]:
    dirs = [run_dir]
    dirs.extend(sorted(p for p in run_dir.glob("rescored_*") if p.is_dir()))
    return dirs


def _prune_json(path: Path, drop: Set[str]) -> int:
    if not path.exists():
        return 0
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        return 0
    kept = [r for r in rows if r.get("model") not in drop]
    removed = len(rows) - len(kept)
    if removed:
        path.write_text(json.dumps(kept, indent=2, default=str))
    return removed


def _prune_jsonl(path: Path, drop: Set[str]) -> int:
    if not path.exists():
        return 0
    kept_lines: List[str] = []
    removed = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("model") in drop:
            removed += 1
        else:
            kept_lines.append(line)
    if removed:
        path.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""))
    return removed


def _prune_csv(path: Path, drop: Set[str]) -> int:
    if not path.exists():
        return 0
    df = pd.read_csv(path)
    if "model" not in df.columns:
        return 0
    before = len(df)
    df = df[~df["model"].isin(drop)].copy()
    removed = before - len(df)
    if removed:
        df.to_csv(path, index=False)
    return removed


def prune_run(run_dir: Path, drop: Set[str]) -> int:
    total = 0
    for sub in _artifact_dirs(run_dir):
        total += _prune_csv(sub / "metrics.csv", drop)
        total += _prune_json(sub / "results.json", drop)
        total += _prune_jsonl(sub / "results.jsonl", drop)
    return total


def _planning_runs(base: Path, runs: Iterable[Path] | None) -> List[Path]:
    if runs:
        return [Path(r) for r in runs]
    pdir = base / "planning"
    if not pdir.exists():
        return []
    return sorted(
        (p for p in pdir.iterdir()
         if p.is_dir() and not p.name.startswith("_")),
        key=lambda p: p.stat().st_mtime,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True,
                    help="Comma-separated model ids to remove")
    ap.add_argument("--runs", nargs="+",
                    help="Specific run dirs (default: all under results/planning/)")
    ap.add_argument("--all", action="store_true",
                    help="Prune every run under results/planning/ (same as default)")
    ap.add_argument("--results-base", default="results")
    ap.add_argument("--delete-empty", action="store_true",
                    help="Delete a run dir when no metrics/results rows remain")
    args = ap.parse_args()

    drop = {m.strip() for m in args.models.split(",") if m.strip()}
    if not drop:
        raise SystemExit("No models given")

    base = Path(args.results_base)
    run_dirs = _planning_runs(base, args.runs)
    if not run_dirs:
        raise SystemExit(f"No planning runs under {base}/planning/")

    grand = 0
    for run in run_dirs:
        removed = prune_run(run, drop)
        if removed:
            print(f"[pruned] {run.name}: removed {removed} row(s) for {sorted(drop)}")
            grand += removed
        elif any((run / "metrics.csv").exists()
                 or (run / "results.json").exists()
                 or (run / "results.jsonl").exists()
                 for _ in [0]):
            print(f"[ok]     {run.name}: no matching rows")
        else:
            print(f"[skip]   {run.name}: no artifacts")

        if args.delete_empty:
            has_data = False
            for sub in _artifact_dirs(run):
                for name in ("metrics.csv", "results.json", "results.jsonl"):
                    p = sub / name
                    if p.exists() and p.stat().st_size > 2:
                        has_data = True
                        break
            if not has_data:
                import shutil
                shutil.rmtree(run)
                print(f"[delete] {run.name}: empty run dir removed")

    print(f"\n[done] removed {grand} row(s) across {len(run_dirs)} run dir(s)")
    if grand:
        print("Next: make merge REFRESH=1 && make report")


if __name__ == "__main__":
    main()
