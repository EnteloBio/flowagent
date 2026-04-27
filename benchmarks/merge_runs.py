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


def _discover_runs(base: Path, bench: str = "planning") -> List[Path]:
    pdir = base / bench
    if not pdir.exists():
        return []
    return sorted([p for p in pdir.iterdir() if _is_run_dir(p)],
                  key=lambda p: p.stat().st_mtime)


# Benchmark → list of columns that together identify a unique cell.
# Used for deduplication when the same cell appears in multiple runs.
_DEDUP_KEYS_BY_BENCH = {
    "planning":       ("model", "input_id", "replicate"),
    "competitors":    ("competitor", "input_id", "replicate"),
    "recovery":       ("model", "fault_id", "seed"),
    "interpretation": ("model", "dataset", "question_id"),
    "fidelity":       ("case_id", "model", "replicate"),
}


def _merge_benchmark(
    base: Path, bench: str, *, runs=None, refresh: bool = False,
    prefer_rescored: bool = True,
) -> Optional[Path]:
    """Merge every run under ``base/<bench>/`` into ``base/<bench>/_merged/<ts>/``.

    Returns the merged-output directory, or None if nothing was mergeable.
    Dedup key per cell is drawn from ``_DEDUP_KEYS_BY_BENCH[bench]``.
    """
    if runs:
        run_dirs = [Path(r) for r in runs]
    else:
        run_dirs = _discover_runs(base, bench)
    if not run_dirs:
        return None

    if refresh:
        merged_root = base / bench / "_merged"
        if merged_root.exists():
            shutil.rmtree(merged_root)

    csv_frames: List[pd.DataFrame] = []
    all_json_rows: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []

    for run in run_dirs:
        csv = _resolve_metrics_csv(run, prefer_rescored=prefer_rescored)
        if csv is None:
            print(f"  [skip] no metrics.csv: {run}")
            continue
        df = pd.read_csv(csv)

        # Drop rows from runs that errored out before the schema-stable
        # row layout was in place. For the interpretation benchmark a
        # row is meaningless without ``correct``; for fidelity it's
        # meaningless without at least one comparator metric. Filtering
        # at merge time keeps the per-model rollup honest — an erroring
        # run shouldn't be silently counted as ``correct=False`` for
        # every cell it failed on.
        if bench == "interpretation":
            if "correct" not in df.columns:
                # Whole CSV pre-dates the schema-stable fix and has no
                # signal we can rescue. Skip it entirely.
                print(f"  [skip] {run.name}: no 'correct' column "
                      f"(all-error run; pre-schema-fix)")
                continue
            n_before = len(df)
            df = df[df["correct"].notna()].copy()
            if len(df) < n_before:
                print(f"  [info] {run.name}: dropped "
                      f"{n_before - len(df)} schema-incomplete rows")
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

        # Use whichever grouping column the benchmark emits (model /
        # competitor / fault_id) so the summary printout is meaningful
        # for each.
        keys = _DEDUP_KEYS_BY_BENCH.get(bench, ("model", "input_id", "replicate"))
        key_col = keys[0] if keys[0] in df.columns else None
        seen_keys = (sorted(df[key_col].dropna().unique().tolist())
                     if key_col else [])
        sources.append({
            "run_dir": str(run),
            "metrics_csv": str(csv),
            "rows": len(df),
            key_col or "entries": seen_keys,
        })
        print(f"  [ok]  {bench}/{run.name}  ({len(df)} rows; "
              f"{key_col or 'entries'}={seen_keys})")

    if not csv_frames:
        print(f"  [skip] {bench}: run dirs exist but no readable metrics.csv")
        return None

    merged = pd.concat(csv_frames, ignore_index=True)

    # Deduplicate on the benchmark's identity columns.
    dedup_keys = _DEDUP_KEYS_BY_BENCH.get(
        bench, ("model", "input_id", "replicate"),
    )
    present_keys = [k for k in dedup_keys if k in merged.columns]
    if len(present_keys) == len(dedup_keys):
        before = len(merged)
        merged = (merged.sort_values("_source_csv_mtime")
                         .drop_duplicates(subset=list(dedup_keys), keep="last")
                         .reset_index(drop=True))
        if len(merged) < before:
            print(f"  [info] {bench} dedup: {before} → {len(merged)} rows "
                  f"(kept latest per {'/'.join(dedup_keys)})")

    merged = merged.drop(columns=["_source_csv_mtime"], errors="ignore")

    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = base / bench / "_merged" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "metrics.csv", index=False)

    if all_json_rows:
        seen: Dict[tuple, Dict[str, Any]] = {}
        for r in all_json_rows:
            key = tuple(r.get(k) for k in dedup_keys)
            seen[key] = r
        (out_dir / "results.json").write_text(
            json.dumps(list(seen.values()), indent=2, default=str)
        )

    group_col = present_keys[0] if present_keys else None
    manifest = {
        "merged_at": ts,
        "benchmark": bench,
        "sources": sources,
        "total_rows": len(merged),
        "group_column": group_col,
        "groups": (sorted(merged[group_col].dropna().unique().tolist())
                   if group_col and group_col in merged.columns else []),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Per-group summary
    if group_col and "overall_pass" in merged.columns:
        agg_cols = {"rows": ("overall_pass", "size"),
                    "overall_pass_rate": ("overall_pass", "mean")}
        if "type_correct" in merged.columns:
            agg_cols["type_correct"] = ("type_correct", "mean")
        if "wall_seconds" in merged.columns:
            agg_cols["wall_seconds"] = ("wall_seconds", "mean")
        summary = merged.groupby(group_col).agg(**agg_cols).round(3)
        print()
        print(f"{bench.title()} — per-{group_col} summary:")
        print(summary.to_string())

    print()
    print(f"[ok] wrote merged CSV → {out_dir/'metrics.csv'}")
    print(f"     ({len(merged)} rows, {len(sources)} source runs)")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    help="Specific run dirs to merge "
                         "(default: every dir under results/<bench>/)")
    ap.add_argument("--results-base", default="results")
    ap.add_argument("--no-rescored", action="store_true",
                    help="Use original metrics.csv even if rescored_* exists")
    ap.add_argument("--refresh", action="store_true",
                    help="Delete previous results/<bench>/_merged/ first")
    ap.add_argument("--benchmarks", default="planning,competitors,interpretation,fidelity",
                    help="Comma-separated list of benchmarks to merge "
                         "(default: planning,competitors,interpretation,fidelity)")
    args = ap.parse_args()

    base = Path(args.results_base)
    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]

    any_merged = False
    for bench in benchmarks:
        print(f"=== Merging {bench} ===")
        out = _merge_benchmark(
            base, bench,
            runs=args.runs if bench == benchmarks[0] else None,
            refresh=args.refresh,
            prefer_rescored=not args.no_rescored,
        )
        if out is not None:
            any_merged = True
        print()

    if not any_merged:
        raise SystemExit(
            f"No runs found under {base}/ for any of: {benchmarks}"
        )


if __name__ == "__main__":
    main()
