"""Merge planning benchmark runs across multiple models / sessions.

Combines every ``results/planning/<TS>/metrics.csv`` (and optionally
``results.json``) into a single tabular file, keeping only the latest
result per (model × input_id × replicate). Useful when you've added
models incrementally and want a unified CSV for plotting.

By default, ``rescored_*`` subdirectories are preferred over the
original results because they reflect the current scoring code.

Writes:
    results/planning/_merged/<TS>/{metrics.csv, results.json, manifest.json}

Usage:
    # Merge all planning runs
    python merge_runs.py

    # Merge a specific subset
    python merge_runs.py --runs results/planning/2026-04-13T13-03-40 \
                                results/planning/2026-04-13T15-22-10

    # Re-run with --refresh to discard previous _merged outputs
    python merge_runs.py --refresh
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

_HERE_DIR = Path(__file__).parent
sys.path.insert(0, str(_HERE_DIR))
sys.path.insert(0, str(_HERE_DIR.parent))

from harness.runner import _write_csv  # noqa: E402


def _is_run_dir(p: Path) -> bool:
    """A planning run dir contains a metrics.csv (after rescore or fresh)."""
    if not p.is_dir():
        return False
    if p.name.startswith("_"):
        return False  # skip our own _merged outputs
    return (p / "metrics.csv").exists() or any(
        (sub / "metrics.csv").exists() for sub in p.iterdir() if sub.is_dir()
    )


def _resolve_metrics_csv(run_dir: Path, prefer_rescored: bool) -> Optional[Path]:
    """Pick the most relevant metrics.csv from a run directory.

    If prefer_rescored is True, return the latest rescored_* subdir if any;
    otherwise fall back to the run dir's own metrics.csv.
    """
    if prefer_rescored:
        rescored = sorted(
            (sub for sub in run_dir.iterdir()
             if sub.is_dir() and sub.name.startswith("rescored_")),
            key=lambda p: p.stat().st_mtime,
        )
        if rescored:
            csv = rescored[-1] / "metrics.csv"
            if csv.exists():
                return csv
    csv = run_dir / "metrics.csv"
    return csv if csv.exists() else None


def _resolve_results_json(metrics_csv: Path) -> Optional[Path]:
    j = metrics_csv.with_name("results.json")
    return j if j.exists() else None


def _discover_runs(base: Path) -> List[Path]:
    pdir = base / "planning"
    if not pdir.exists():
        return []
    return sorted([p for p in pdir.iterdir() if _is_run_dir(p)],
                  key=lambda p: p.stat().st_mtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    help="Specific run dirs to merge "
                         "(default: every dir under results/planning/)")
    ap.add_argument("--results-base", default="results")
    ap.add_argument("--no-rescored", action="store_true",
                    help="Use original metrics.csv even if rescored_* exists")
    ap.add_argument("--refresh", action="store_true",
                    help="Delete previous results/planning/_merged/ first")
    args = ap.parse_args()

    base = Path(args.results_base)
    if args.runs:
        run_dirs = [Path(r) for r in args.runs]
    else:
        run_dirs = _discover_runs(base)
    if not run_dirs:
        raise SystemExit(f"No planning runs found under {base/'planning'}")

    if args.refresh:
        merged_root = base / "planning" / "_merged"
        if merged_root.exists():
            shutil.rmtree(merged_root)

    # Collect metrics + results
    csv_frames: List[pd.DataFrame] = []
    all_json_rows: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []

    for run in run_dirs:
        csv = _resolve_metrics_csv(run, prefer_rescored=not args.no_rescored)
        if csv is None:
            print(f"  [skip] no metrics.csv: {run}")
            continue
        df = pd.read_csv(csv)
        df["_source_run"] = str(run.name)
        df["_source_csv_mtime"] = csv.stat().st_mtime
        csv_frames.append(df)

        rj = _resolve_results_json(csv)
        if rj is not None:
            try:
                rows = json.loads(rj.read_text())
                for row in rows:
                    row["_source_run"] = str(run.name)
                all_json_rows.extend(rows)
            except Exception as exc:
                print(f"  [warn] couldn't read {rj}: {exc}")

        sources.append({
            "run_dir": str(run),
            "metrics_csv": str(csv),
            "rows": len(df),
            "models": sorted(df["model"].dropna().unique().tolist())
                       if "model" in df.columns else [],
        })
        print(f"  [ok]  {run.name}  ({len(df)} rows; "
              f"models={sorted(df['model'].dropna().unique().tolist()) if 'model' in df else []})")

    if not csv_frames:
        raise SystemExit("Found run dirs but none had readable metrics.csv")

    merged = pd.concat(csv_frames, ignore_index=True)

    # Deduplicate: keep the most recent result per (model, input_id, replicate)
    if all(c in merged.columns for c in ("model", "input_id", "replicate")):
        before = len(merged)
        merged = (merged.sort_values("_source_csv_mtime")
                         .drop_duplicates(
                             subset=["model", "input_id", "replicate"],
                             keep="last")
                         .reset_index(drop=True))
        if len(merged) < before:
            print(f"  [info] dedup: {before} → {len(merged)} rows "
                  f"(kept latest per model/input/replicate)")

    # Drop the helper columns from the final CSV
    merged = merged.drop(columns=["_source_csv_mtime"], errors="ignore")

    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = base / "planning" / "_merged" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "metrics.csv", index=False)

    # Also dedup the per-row JSON if present
    if all_json_rows:
        seen: Dict[tuple, Dict[str, Any]] = {}
        for r in all_json_rows:
            key = (r.get("model"), r.get("input_id"), r.get("replicate"))
            seen[key] = r  # last write wins (matches CSV dedup direction)
        (out_dir / "results.json").write_text(
            json.dumps(list(seen.values()), indent=2, default=str)
        )

    manifest = {
        "merged_at": ts,
        "sources": sources,
        "total_rows": len(merged),
        "models": sorted(merged["model"].dropna().unique().tolist())
                  if "model" in merged.columns else [],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Per-model summary
    if "model" in merged.columns and "overall_pass" in merged.columns:
        summary = (merged.groupby("model")
                          .agg(rows=("overall_pass", "size"),
                               overall_pass_rate=("overall_pass", "mean"),
                               type_correct=("type_correct", "mean")
                                if "type_correct" in merged.columns else ("overall_pass", "mean"),
                               wall_seconds=("wall_seconds", "mean")
                                if "wall_seconds" in merged.columns else ("overall_pass", "mean"))
                          .round(3))
        print()
        print("Per-model summary:")
        print(summary.to_string())

    print()
    print(f"[ok] wrote merged CSV → {out_dir/'metrics.csv'}")
    print(f"     ({len(merged)} rows, {len(sources)} source runs)")


if __name__ == "__main__":
    main()
