"""Inter-judge calibration harness for Benchmark G.

The reviewer flagged the open-ended judge as uncalibrated: a single
LLM-as-judge with no anchored bands and no inter-rater check. This
harness re-scores a sample of N open-ended responses with a *second*
judge model and reports inter-judge agreement.

Inputs
------
A merged CSV (or directory containing one) emitted by
``bench_interpretation.py`` — i.e. rows with ``question_type ==
'open_ended'``, ``judge_score``, ``candidate_answer`` and the question
text recoverable via ``question_id`` against the YAML.

Outputs
-------
A CSV ``judge_calibration.csv`` with one row per re-scored response::

    question_id, dataset, model, judge_a, judge_a_score, judge_a_pass,
    judge_b, judge_b_score, judge_b_pass

Plus a small ``judge_calibration_summary.json`` reporting:

    n                      : sample size
    score_correlation      : Pearson r between the two judges
    pass_agreement         : fraction of responses where both judges
                             agree on pass / fail at the 60 mark
    cohens_kappa           : Cohen's κ on the binary pass / fail
                             decision
    mean_score_delta       : mean (judge_b - judge_a) score
    bias_flag              : true if |mean_score_delta| > 5

Usage
-----
    python benchmarks/judge_calibration.py \
        --metrics path/to/interpretation_metrics.csv \
        --questions config/interpretation_questions.yaml \
        --models-yaml config/models.yaml \
        --judge-a gpt-5.4 --judge-b claude-opus-4-7 \
        --n 30 --out results/judge_cal

The harness keeps ``judge_a_score`` from the merged CSV (no re-spend on
the original judge) and only spends API on ``judge_b``.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from harness.runner import load_yaml, set_provider, timestamped_dir  # noqa: E402

import bench_interpretation as bi  # noqa: E402


# ── Stats helpers ────────────────────────────────────────────────

def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    """Pearson r; returns None for degenerate inputs (n < 2 or zero variance)."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx2 = sum((x - mx) ** 2 for x in xs)
    dy2 = sum((y - my) ** 2 for y in ys)
    if dx2 == 0 or dy2 == 0:
        return None
    return num / math.sqrt(dx2 * dy2)


def _cohens_kappa(labels_a: List[bool], labels_b: List[bool]) -> Optional[float]:
    """Cohen's κ on a 2x2 contingency between two binary labellers."""
    n = len(labels_a)
    if n == 0 or len(labels_b) != n:
        return None
    obs = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    pa_pos = sum(1 for a in labels_a if a) / n
    pb_pos = sum(1 for b in labels_b if b) / n
    pe = pa_pos * pb_pos + (1 - pa_pos) * (1 - pb_pos)
    if pe == 1:
        return None
    return (obs - pe) / (1 - pe)


# ── Question lookup ──────────────────────────────────────────────

def _build_question_index(qcfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for ds in qcfg.get("datasets", []):
        for q in ds.get("questions", []):
            out[q["id"]] = q
    return out


# ── Re-score loop ────────────────────────────────────────────────

async def _re_score_one(question: Dict[str, Any], candidate: str,
                        judge_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run one judge call against a stored candidate answer."""
    prompt = bi._judge_prompt(question, candidate)
    reply = await bi._call_llm(prompt, model_cfg=judge_cfg)
    return bi._parse_judge_json(reply)


async def _run_calibration(rows: List[Dict[str, Any]],
                           judge_b_cfg: Dict[str, Any],
                           question_index: Dict[str, Dict[str, Any]],
                           ) -> List[Dict[str, Any]]:
    set_provider(judge_b_cfg)
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        qid = row.get("question_id")
        question = question_index.get(qid)
        candidate = row.get("candidate_answer") or ""
        if not question or not candidate.strip():
            print(f"  [{i:>3}/{len(rows)}] {qid} — skipped (no question/candidate)")
            continue
        t0 = time.perf_counter()
        try:
            parsed = await _re_score_one(question, candidate, judge_b_cfg)
        except Exception as exc:  # pragma: no cover - defensive
            parsed = {"score": None, "justification": f"{type(exc).__name__}: {exc}",
                      "hits": [], "misses": [], "fabrications": [], "grounding_quote": ""}
        wall = round(time.perf_counter() - t0, 2)
        try:
            judge_a_score = float(row.get("judge_score") or "")
        except (TypeError, ValueError):
            judge_a_score = None
        judge_b_score = parsed["score"]
        out.append({
            "question_id":    qid,
            "dataset":        row.get("dataset", ""),
            "model":          row.get("model", ""),
            "judge_a":        row.get("judge_model", ""),
            "judge_a_score":  judge_a_score,
            "judge_a_pass":   (judge_a_score is not None
                               and judge_a_score >= bi._PASS_MARK),
            "judge_b":        judge_b_cfg["id"],
            "judge_b_score":  judge_b_score,
            "judge_b_pass":   (judge_b_score is not None
                               and judge_b_score >= bi._PASS_MARK),
            "judge_b_justification": parsed["justification"],
            "wall_seconds":   wall,
        })
        print(f"  [{i:>3}/{len(rows)}] {qid:<40}  "
              f"a={judge_a_score}  b={judge_b_score}  ({wall:.1f}s)")
    return out


# ── Summary ──────────────────────────────────────────────────────

def _summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    paired = [(r["judge_a_score"], r["judge_b_score"])
              for r in rows
              if r["judge_a_score"] is not None and r["judge_b_score"] is not None]
    n = len(paired)
    if n == 0:
        return {"n": 0, "score_correlation": None, "pass_agreement": None,
                "cohens_kappa": None, "mean_score_delta": None,
                "bias_flag": False}
    xs = [a for a, _ in paired]
    ys = [b for _, b in paired]
    a_pass = [a >= bi._PASS_MARK for a in xs]
    b_pass = [b >= bi._PASS_MARK for b in ys]
    delta = sum(b - a for a, b in paired) / n
    pass_agree = sum(1 for a, b in zip(a_pass, b_pass) if a == b) / n
    return {
        "n":                  n,
        "score_correlation":  _pearson(xs, ys),
        "pass_agreement":     pass_agree,
        "cohens_kappa":       _cohens_kappa(a_pass, b_pass),
        "mean_score_delta":   delta,
        "bias_flag":          abs(delta) > 5.0,
    }


# ── CLI ──────────────────────────────────────────────────────────

def _read_metrics(path: Path) -> List[Dict[str, Any]]:
    if path.is_dir():
        path = path / "metrics.csv"
    if not path.exists():
        raise SystemExit(f"metrics file not found: {path}")
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics",     required=True,
                    help="Path to interpretation metrics.csv (or its dir)")
    ap.add_argument("--questions",   default="config/interpretation_questions.yaml")
    ap.add_argument("--models-yaml", default="config/models.yaml")
    ap.add_argument("--judge-a",     default="gpt-5.4",
                    help="Original judge model id (used as label only; "
                         "scores come from the metrics CSV)")
    ap.add_argument("--judge-b",     default="claude-opus-4-8",
                    help="Second judge to run for re-scoring")
    ap.add_argument("--n",           type=int, default=30,
                    help="Number of open-ended responses to sample")
    ap.add_argument("--seed",        type=int, default=20260506)
    ap.add_argument("--out",         default="results")
    ap.add_argument("--mock",        action="store_true",
                    help="Skip LLM calls; emit a deterministic mock pass")
    args = ap.parse_args()

    qpath = _HERE / args.questions if not Path(args.questions).is_absolute() \
        else Path(args.questions)
    mpath = _HERE / args.models_yaml if not Path(args.models_yaml).is_absolute() \
        else Path(args.models_yaml)
    qcfg = load_yaml(qpath)
    mcfg = load_yaml(mpath)
    qidx = _build_question_index(qcfg)

    metrics_path = Path(args.metrics)
    rows_all = _read_metrics(metrics_path)
    rows_open = [r for r in rows_all
                 if r.get("question_type") == "open_ended"
                 and (r.get("candidate_answer") or "").strip()
                 and (r.get("judge_score") or "").strip()]
    if not rows_open:
        raise SystemExit("no usable open-ended rows with judge_score in metrics CSV")

    rng = random.Random(args.seed)
    sample = rng.sample(rows_open, k=min(args.n, len(rows_open)))
    print(f"sampling {len(sample)} of {len(rows_open)} open-ended rows "
          f"(seed={args.seed})")

    out_dir = timestamped_dir(_HERE / args.out, "judge_calibration")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mock:
        cal_rows = [{
            "question_id":   r.get("question_id", ""),
            "dataset":       r.get("dataset", ""),
            "model":         r.get("model", ""),
            "judge_a":       args.judge_a,
            "judge_a_score": _try_float(r.get("judge_score")),
            "judge_a_pass":  _try_float(r.get("judge_score"), default=0.0)
                             >= bi._PASS_MARK,
            "judge_b":       args.judge_b,
            "judge_b_score": _try_float(r.get("judge_score")),
            "judge_b_pass":  _try_float(r.get("judge_score"), default=0.0)
                             >= bi._PASS_MARK,
            "judge_b_justification": "(mock)",
            "wall_seconds":  0.0,
        } for r in sample]
    else:
        judge_b_cfg = next((m for m in mcfg.get("models", [])
                            if m["id"] == args.judge_b), None)
        if judge_b_cfg is None:
            raise SystemExit(f"judge-b model '{args.judge_b}' not in models.yaml")
        cal_rows = asyncio.run(_run_calibration(sample, judge_b_cfg, qidx))

    if not cal_rows:
        raise SystemExit("no rows produced by calibration run")

    out_csv = out_dir / "judge_calibration.csv"
    fieldnames = list(cal_rows[0].keys())
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cal_rows)

    summary = _summarise(cal_rows)
    summary["judge_a"]   = args.judge_a
    summary["judge_b"]   = args.judge_b
    summary["seed"]      = args.seed
    summary["sample_n"]  = len(cal_rows)
    summary["pass_mark"] = bi._PASS_MARK
    (out_dir / "judge_calibration_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print("\n=== judge calibration summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_csv}")


def _try_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
