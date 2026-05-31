"""Export a blinded expert-audit pack for Benchmark G open-ended items.

Builds spreadsheet-ready TSVs that join interpretation ``metrics.csv`` rows
with question rubrics and the same input context bundle the benchmark fed
to models under test.

Outputs (under ``results/expert_audit/<timestamp>/``)::

    INSTRUCTIONS.md           — one-page protocol for annotators
    expert_audit_scoring.tsv  — blinded sheet (experts fill score columns)
    expert_audit_key.tsv        — audit_id → model + LLM judge (authors only)
    expert_audit_full.tsv       — all columns for internal reference
    overlap_ids.txt             — audit_ids flagged for dual annotation

Usage::

    python export_expert_audit_pack.py \\
        --metrics results/interpretation/2026-05-30T22-42-56/metrics.csv

    python export_expert_audit_pack.py \\
        --metrics results/interpretation/2026-05-30T22-42-56 \\
        --inputs-base . \\
        --overlap 12
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from harness.runner import load_yaml, timestamped_dir  # noqa: E402

import bench_interpretation as bi  # noqa: E402


def _read_metrics(path: Path) -> List[Dict[str, Any]]:
    if path.is_dir():
        path = path / "metrics.csv"
    if not path.exists():
        raise SystemExit(f"metrics file not found: {path}")
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _load_results_json(metrics_path: Path) -> Dict[Tuple[str, str], str]:
    """Map (question_id, model) → full candidate_answer from results.json."""
    run_dir = metrics_path.parent if metrics_path.is_file() else metrics_path
    rpath = run_dir / "results.json"
    if not rpath.exists():
        return {}
    data = json.loads(rpath.read_text())
    out: Dict[Tuple[str, str], str] = {}
    for row in data:
        if row.get("question_type") != "open_ended":
            continue
        key = (row.get("question_id", ""), row.get("model", ""))
        ans = row.get("candidate_answer") or ""
        if ans.strip():
            out[key] = ans
    return out


def _build_dataset_index(qcfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {ds["id"]: ds for ds in qcfg.get("datasets", [])}


def _build_question_index(qcfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for ds in qcfg.get("datasets", []):
        for q in ds.get("questions", []):
            out[q["id"]] = {**q, "_dataset_id": ds["id"]}
    return out


def _open_ended_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        r for r in rows
        if r.get("question_type") == "open_ended"
        and (r.get("candidate_answer") or "").strip()
    ]


def _try_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _select_overlap_ids(
    audit_rows: List[Dict[str, Any]],
    *,
    n_overlap: int,
    seed: int,
) -> Set[str]:
    """Pick overlap items, preferring borderline LLM scores (40–70)."""
    if n_overlap <= 0 or not audit_rows:
        return set()
    rng = random.Random(seed)

    def _borderline_key(row: Dict[str, Any]) -> Tuple[int, float, str]:
        score = _try_float(row.get("llm_judge_score"))
        if score is None:
            dist = 999.0
        else:
            dist = abs(score - 55.0) if 40 <= score <= 70 else score + 100
        return (0 if 40 <= (score or -1) <= 70 else 1, dist, row["audit_id"])

    by_question: Dict[str, List[Dict[str, Any]]] = {}
    for row in audit_rows:
        by_question.setdefault(row["question_id"], []).append(row)

    chosen: Set[str] = set()
    # Round-robin one borderline per question first.
    questions = sorted(by_question)
    per_q = max(1, n_overlap // max(1, len(questions)))
    for qid in questions:
        pool = sorted(by_question[qid], key=_borderline_key)
        for row in pool[:per_q]:
            if len(chosen) >= n_overlap:
                break
            chosen.add(row["audit_id"])

    if len(chosen) < n_overlap:
        remaining = [r for r in audit_rows if r["audit_id"] not in chosen]
        rng.shuffle(remaining)
        for row in remaining:
            if len(chosen) >= n_overlap:
                break
            chosen.add(row["audit_id"])
    return chosen


def _instructions_md(*, pass_mark: float, n_items: int, n_overlap: int) -> str:
    return f"""# Expert audit — Benchmark G open-ended responses

## Goal

Grade model-written summaries of bioinformatics outputs using the supplied
question, rubric, reference answer, and input excerpt. Score **only** from
the evidence shown — do not look up original papers or GEO records.

## Items

- **{n_items}** responses across **9** open-ended questions
- **{n_overlap}** items flagged `overlap_item=Y` — score these twice
  (two annotators) for inter-rater reliability

## Scoring

- **Pass mark:** {pass_mark:.0f} (same threshold as the LLM judge)
- Use integer scores **0–100**

| Band | Quality |
|------|---------|
| 0–20 | Fabricated, contradicted by evidence, or wrong question |
| 21–40 | Generic prose; at most one rubric item |
| 41–59 | Partially correct; misses most rubric items |
| 60–79 | Majority of rubric items; minor gaps (**passing**) |
| 80–100 | All rubric items; grounded; no fabrications |

Fill in ``expert_audit_scoring.tsv`` (tab-separated — import as TSV in
Google Sheets or Excel):

- ``expert_score`` — 0–100
- ``expert_pass`` — YES if score ≥ {pass_mark:.0f}, else NO
- ``expert_notes`` — optional, 1–3 sentences
- ``expert_id`` — your initials or ``A`` / ``B``

**Dual annotation:** Expert A scores all rows. Expert B scores only
``overlap_item=Y`` rows in a copy of the sheet (set ``expert_id=B``).
Submit both files when running ``score_expert_audit.py --scoring A.tsv
--scoring-extra B.tsv``.

**Do not** open ``expert_audit_key.tsv`` until scoring is complete.

## Tips

- Read ``rubric`` and ``reference_answer`` before ``response_text``
- Penalise gene names, counts, or directions not supported by ``input_excerpt``
- Reward calibrated uncertainty when evidence is insufficient
"""


def build_audit_rows(
    metrics_rows: List[Dict[str, Any]],
    *,
    qcfg: Dict[str, Any],
    inputs_base: Path,
    full_answers: Dict[Tuple[str, str], str],
    bench_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    if bench_root is None:
        bench_root = _HERE
    ds_index = _build_dataset_index(qcfg)
    q_index = _build_question_index(qcfg)
    context_cache: Dict[str, str] = {}

    rows = _open_ended_rows(metrics_rows)
    rows.sort(key=lambda r: (r.get("question_id", ""), r.get("model", "")))

    out: List[Dict[str, Any]] = []
    for i, mrow in enumerate(rows, start=1):
        qid = mrow.get("question_id", "")
        question = q_index.get(qid)
        if not question:
            continue
        ds_id = question.get("_dataset_id") or mrow.get("dataset", "")
        ds = ds_index.get(ds_id, {})
        if ds_id not in context_cache:
            input_paths = {
                k: inputs_base / v
                for k, v in (ds.get("inputs") or {}).items()
            }
            context_cache[ds_id] = _relativize_excerpt(
                bi._bundle_inputs(
                    input_paths,
                    dataset_context=ds.get("analysis_context", ""),
                ),
                bench_root,
            )
        model = mrow.get("model", "")
        key = (qid, model)
        response = full_answers.get(key) or mrow.get("candidate_answer") or ""
        llm_score = _try_float(mrow.get("judge_score"))

        out.append({
            "audit_id":           f"AUDIT_{i:03d}",
            "question_id":        qid,
            "dataset":            ds_id,
            "accession":          mrow.get("accession", ds.get("accession", "")),
            "evidence_class":     mrow.get("evidence_class", question.get("evidence_class", "")),
            "question_text":      question.get("question", "").strip(),
            "rubric":             (question.get("rubric") or "").strip(),
            "reference_answer":   (question.get("reference_answer") or "").strip(),
            "dataset_context":    (ds.get("analysis_context") or "").strip(),
            "input_excerpt":      context_cache.get(ds_id, ""),
            "response_text":      response.strip(),
            "model":              model,
            "provider":           mrow.get("provider", ""),
            "judge_model":        mrow.get("judge_model", ""),
            "llm_judge_score":    "" if llm_score is None else llm_score,
            "llm_judge_pass":     "" if llm_score is None else ("YES" if llm_score >= bi._PASS_MARK else "NO"),
            "llm_judge_justification": mrow.get("judge_justification", ""),
            "llm_judge_hits":     mrow.get("judge_hits", ""),
            "llm_judge_misses":   mrow.get("judge_misses", ""),
            "llm_judge_fabrications": mrow.get("judge_fabrications", ""),
            "overlap_item":       "N",
            "expert_score":       "",
            "expert_pass":        "",
            "expert_notes":       "",
            "expert_id":          "",
        })
    return out


def _relativize_excerpt(text: str, bench_root: Path) -> str:
    """Strip absolute benchmark paths from input excerpts for portability."""
    root = bench_root.resolve()

    def _repl(match: re.Match[str]) -> str:
        raw = match.group(1)
        try:
            rel = Path(raw).expanduser().resolve().relative_to(root)
            return f"({rel.as_posix()})"
        except (ValueError, OSError):
            return match.group(0)

    text = re.sub(r"\(([^)]+)\)", _repl, text)
    root_str = str(root)
    return text.replace(root_str + "/", "").replace(root_str, "")


def _write_tsv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=fieldnames, extrasaction="ignore",
            delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--metrics", required=True,
                    help="Path to interpretation metrics.csv (or run directory)")
    ap.add_argument("--questions", default="config/interpretation_questions.yaml")
    ap.add_argument("--inputs-base", default=".",
                    help="Base dir for dataset input paths (default: benchmarks/)")
    ap.add_argument("--overlap", type=int, default=12,
                    help="Number of items flagged for dual expert annotation")
    ap.add_argument("--seed", type=int, default=20260531)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.is_absolute():
        metrics_path = _HERE / metrics_path
    qpath = _HERE / args.questions if not Path(args.questions).is_absolute() \
        else Path(args.questions)
    inputs_base = Path(args.inputs_base)
    if not inputs_base.is_absolute():
        inputs_base = _HERE / inputs_base

    qcfg = load_yaml(qpath)
    metrics_rows = _read_metrics(metrics_path)
    full_answers = _load_results_json(metrics_path)

    audit_rows = build_audit_rows(
        metrics_rows,
        qcfg=qcfg,
        inputs_base=inputs_base,
        full_answers=full_answers,
    )
    if not audit_rows:
        raise SystemExit("no open-ended rows with candidate_answer in metrics CSV")

    overlap_ids = _select_overlap_ids(
        audit_rows, n_overlap=args.overlap, seed=args.seed,
    )
    for row in audit_rows:
        if row["audit_id"] in overlap_ids:
            row["overlap_item"] = "Y"

    out_dir = timestamped_dir(_HERE / args.out, "expert_audit")
    out_dir.mkdir(parents=True, exist_ok=True)

    scoring_fields = [
        "audit_id", "overlap_item", "question_id", "dataset", "accession",
        "evidence_class", "question_text", "rubric", "reference_answer",
        "dataset_context", "input_excerpt", "response_text",
        "expert_score", "expert_pass", "expert_notes", "expert_id",
    ]
    key_fields = [
        "audit_id", "question_id", "model", "provider", "judge_model",
        "llm_judge_score", "llm_judge_pass", "llm_judge_justification",
        "llm_judge_hits", "llm_judge_misses", "llm_judge_fabrications",
    ]
    full_fields = list(audit_rows[0].keys())

    _write_tsv(out_dir / "expert_audit_scoring.tsv", audit_rows, scoring_fields)
    _write_tsv(out_dir / "expert_audit_key.tsv", audit_rows, key_fields)
    _write_tsv(out_dir / "expert_audit_full.tsv", audit_rows, full_fields)

    (out_dir / "overlap_ids.txt").write_text(
        "\n".join(sorted(overlap_ids)) + ("\n" if overlap_ids else "")
    )
    (out_dir / "INSTRUCTIONS.md").write_text(
        _instructions_md(
            pass_mark=bi._PASS_MARK,
            n_items=len(audit_rows),
            n_overlap=len(overlap_ids),
        )
    )

    manifest = {
        "source_metrics": str(metrics_path),
        "n_items": len(audit_rows),
        "n_overlap": len(overlap_ids),
        "overlap_seed": args.seed,
        "pass_mark": bi._PASS_MARK,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"wrote {len(audit_rows)} items to {out_dir}")
    print(f"  expert_audit_scoring.tsv  (send to annotators)")
    print(f"  expert_audit_key.tsv        (authors only — unblind after scoring)")
    print(f"  overlap_ids.txt             ({len(overlap_ids)} dual-score items)")


if __name__ == "__main__":
    main()
