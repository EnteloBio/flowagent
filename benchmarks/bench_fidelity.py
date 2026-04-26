"""Benchmark F — output fidelity.

Score the *outputs* of a FlowAgent run against a published reference,
filling the BixBench-style gap where prior FlowAgent benchmarks only
checked plan-level or step-exit-level success.

This runner is deliberately a pure scoring layer: it does not invoke
``flowagent`` itself. End-to-end execution of a real bioinformatics
pipeline takes hours and consumes provider credits, so the ergonomic
default is "run FlowAgent once per (case × model), then re-score
deterministically as the metric code matures".

Usage:
    # Score a single case against an existing FlowAgent run.
    python bench_fidelity.py \\
        --cases config/fidelity_cases.yaml \\
        --case  gse52778_dex_de \\
        --candidate-dir results/realworld_GSE52778 \\
        --model gpt-4.1 \\
        --replicate 0

    # Bulk-score: scan a parent directory containing one subdirectory
    # per (case, model, replicate) named "<case>__<model>__rep<N>".
    python bench_fidelity.py --cases config/fidelity_cases.yaml \\
        --bulk-dir results/fidelity_runs

Outputs (mirrors the other Benchmark runners):
    results/fidelity/<TS>/{metrics.csv, results.json, manifest.json}
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from harness.fidelity_metrics import COMPARATORS  # noqa: E402
from harness.runner import (  # noqa: E402
    _write_csv, load_yaml, timestamped_dir, write_manifest,
)


def _resolve_case(cases: List[Dict[str, Any]], case_id: str) -> Dict[str, Any]:
    for c in cases:
        if c["id"] == case_id:
            return c
    raise SystemExit(f"case '{case_id}' not in fidelity_cases.yaml")


def _score_one(case: Dict[str, Any], candidate_root: Path,
               reference_root: Path) -> Dict[str, Any]:
    """Run the case's comparator on a single (case × candidate-dir) pair.

    Returns a flat dict suitable for writing as one row of metrics.csv.
    Errors are caught and reported as ``error`` rather than re-raised so
    one bad case doesn't abort a bulk sweep.
    """
    cmp_name = case["comparison"]
    fn = COMPARATORS.get(cmp_name)
    if fn is None:
        return {"error": f"unknown comparison '{cmp_name}'"}

    candidate = candidate_root / case["output_relpath"]
    reference = reference_root / case["reference"]
    if not candidate.exists():
        return {"error": f"candidate not found: {candidate}"}
    if not reference.exists():
        return {"error": f"reference not found: {reference}"}

    t0 = time.perf_counter()
    try:
        metrics = fn(candidate, reference, case.get("params", {}) or {})
        metrics["score_seconds"] = round(time.perf_counter() - t0, 3)
        return metrics
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": f"{type(exc).__name__}: {exc}"}


_SLUG_RE = re.compile(
    r"^(?P<case>[A-Za-z0-9-]+)__(?P<model>[A-Za-z0-9.\-_]+)__rep(?P<rep>\d+)$"
)


def _scan_bulk(bulk_dir: Path) -> List[Dict[str, str]]:
    """Discover (case, model, replicate) sub-directories.

    Convention: each subdirectory's name is ``<case>__<model>__rep<N>``;
    the runner only enumerates dirs that match this slug pattern, so
    you can mix bulk and ad-hoc layouts inside the same parent.
    """
    runs: List[Dict[str, str]] = []
    for p in sorted(bulk_dir.iterdir()):
        if not p.is_dir():
            continue
        m = _SLUG_RE.match(p.name)
        if not m:
            continue
        runs.append({
            "case_id":   m.group("case"),
            "model":     m.group("model"),
            "replicate": int(m.group("rep")),
            "candidate_dir": str(p),
        })
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="config/fidelity_cases.yaml")
    ap.add_argument("--reference-base", default=".",
                    help="Root for resolving each case's ``reference`` path")
    ap.add_argument("--out", default="results")
    sg = ap.add_argument_group("Single-case scoring")
    sg.add_argument("--case",          help="case ID from fidelity_cases.yaml")
    sg.add_argument("--candidate-dir", help="FlowAgent output dir to score")
    sg.add_argument("--model",         help="Model that produced the run")
    sg.add_argument("--replicate", type=int, default=0)
    bg = ap.add_argument_group("Bulk scoring")
    bg.add_argument("--bulk-dir", help="Parent dir of <case>__<model>__rep<N> subdirs")
    args = ap.parse_args()

    cases_path = _HERE / args.cases if not Path(args.cases).is_absolute() \
        else Path(args.cases)
    cfg = load_yaml(cases_path)
    cases = cfg.get("cases", [])
    if not cases:
        sys.exit(f"no cases defined in {cases_path}")

    work: List[Dict[str, Any]] = []
    if args.bulk_dir:
        for entry in _scan_bulk(Path(args.bulk_dir)):
            work.append(entry)
    elif args.case and args.candidate_dir and args.model:
        work.append({
            "case_id":   args.case,
            "model":     args.model,
            "replicate": args.replicate,
            "candidate_dir": args.candidate_dir,
        })
    else:
        sys.exit("provide either --bulk-dir, or all of --case --candidate-dir --model")

    rows: List[Dict[str, Any]] = []
    for w in work:
        try:
            case = _resolve_case(cases, w["case_id"])
        except SystemExit as exc:
            rows.append({**w, "error": str(exc)})
            print(f"[skip] {w['case_id']}: {exc}")
            continue
        metrics = _score_one(case, Path(w["candidate_dir"]),
                             Path(args.reference_base))
        row = {
            "case_id":       w["case_id"],
            "accession":     case.get("accession", ""),
            "comparison":    case["comparison"],
            "model":         w["model"],
            "replicate":     w["replicate"],
            "candidate_dir": w["candidate_dir"],
            **metrics,
        }
        rows.append(row)
        print(f"[ok]  {w['case_id']}  model={w['model']}  rep={w['replicate']}  "
              f"{_summarise(metrics)}")

    out_dir = timestamped_dir(Path(args.out), "fidelity")
    _write_csv(out_dir / "metrics.csv", rows)
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2, default=str))
    write_manifest(
        out_dir, benchmark="fidelity",
        # No LLM calls happen during fidelity scoring — this runner
        # is a pure comparator. Manifest still records the case set
        # and which scoring functions ran, for reviewer reproduction.
        models=[],
        extra={
            "cases":     [c["id"] for c in cases],
            "n_scored":  len(rows),
            "comparators_used": sorted(
                {r["comparison"] for r in rows if r.get("comparison")}),
        },
    )
    print(f"\n[ok] wrote {len(rows)} rows → {out_dir}/metrics.csv")


def _summarise(m: Dict[str, Any]) -> str:
    if "error" in m:
        return f"ERROR: {m['error']}"
    bits = []
    for k in ("spearman_lfc", "jaccard_top_n", "jaccard_peak", "f1"):
        v = m.get(k)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        bits.append(f"{k}={v:.3f}")
    return " ".join(bits) or "scored"


if __name__ == "__main__":
    main()
