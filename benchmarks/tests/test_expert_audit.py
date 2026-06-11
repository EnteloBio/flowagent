"""Tests for expert audit export + scoring helpers."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE.parent
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(BENCH_DIR.parent))

from export_expert_audit_pack import (  # noqa: E402
    _select_overlap_ids,
    build_audit_rows,
)
from harness.expert_audit_stats import cohens_kappa, pearson, summarise_paired_scores  # noqa: E402
from score_expert_audit import (  # noqa: E402
    _expert_entries,
    _expert_vs_llm_summary,
    _human_human_summary,
    _merge_key,
    _parse_pass,
)


def test_pearson_and_kappa():
    assert pearson([1, 2, 3], [1, 2, 3]) == 1.0
    assert cohens_kappa([True, True, False], [True, False, False]) is not None


def test_parse_pass():
    assert _parse_pass("YES", score=55.0, pass_mark=60.0) is True
    assert _parse_pass("NO", score=55.0, pass_mark=60.0) is False
    assert _parse_pass("", score=65.0, pass_mark=60.0) is True


def test_build_audit_rows_minimal():
    qcfg = {
        "datasets": [{
            "id": "ds1",
            "accession": "GSE1",
            "analysis_context": "test context",
            "inputs": {},
            "questions": [{
                "id": "ds1_q01",
                "type": "open_ended",
                "evidence_class": "data_required",
                "question": "Summarise the finding.",
                "rubric": "Award credit for direction.",
                "reference_answer": "Upregulated genes.",
            }],
        }],
    }
    metrics = [{
        "question_type": "open_ended",
        "question_id": "ds1_q01",
        "dataset": "ds1",
        "accession": "GSE1",
        "model": "gpt-test",
        "provider": "openai",
        "judge_model": "gpt-judge",
        "judge_score": "72",
        "candidate_answer": "Many genes are up.",
        "judge_justification": "ok",
        "evidence_class": "data_required",
    }]
    rows = build_audit_rows(
        metrics, qcfg=qcfg, inputs_base=BENCH_DIR, full_answers={},
    )
    assert len(rows) == 1
    assert rows[0]["audit_id"] == "AUDIT_001"
    assert rows[0]["response_text"] == "Many genes are up."
    assert rows[0]["llm_judge_score"] == 72.0


def test_overlap_selection_prefers_borderline():
    rows = [
        {"audit_id": f"A{i:03d}", "question_id": f"q{i // 2}", "llm_judge_score": score}
        for i, score in enumerate([90, 45, 30, 55, 62, 20])
    ]
    overlap = _select_overlap_ids(rows, n_overlap=3, seed=1)
    assert len(overlap) == 3
    # Borderline scores 45, 55, 62 should beat extremes 90/30/20.
    assert "A001" in overlap or "A003" in overlap or "A004" in overlap


def test_expert_vs_llm_and_human_human():
    entries = [
        {"audit_id": "A1", "question_id": "q1", "expert_id": "A",
         "expert_score": 70.0, "expert_pass": True, "llm_judge_score": 65.0,
         "overlap_item": True, "model": "m1"},
        {"audit_id": "A1", "question_id": "q1", "expert_id": "B",
         "expert_score": 68.0, "expert_pass": True, "llm_judge_score": 65.0,
         "overlap_item": True, "model": "m1"},
        {"audit_id": "A2", "question_id": "q2", "expert_id": "A",
         "expert_score": 40.0, "expert_pass": False, "llm_judge_score": 80.0,
         "overlap_item": False, "model": "m2"},
    ]
    llm_summary = _expert_vs_llm_summary(entries)
    assert llm_summary["n"] == 2
    human = _human_human_summary(entries)
    assert human is not None
    assert human["n_overlap_pairs"] == 1


def test_merge_key(tmp_path: Path):
    scoring = tmp_path / "scoring.tsv"
    key = tmp_path / "key.tsv"
    with scoring.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["audit_id", "expert_score"], delimiter="\t")
        w.writeheader()
        w.writerow({"audit_id": "A1", "expert_score": "70"})
    with key.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["audit_id", "llm_judge_score", "model"], delimiter="\t")
        w.writeheader()
        w.writerow({"audit_id": "A1", "llm_judge_score": "65", "model": "gpt-4"})
    merged = _merge_key(_read_csv(scoring), _read_csv(key))
    assert merged[0]["llm_judge_score"] == "65"
    assert merged[0]["model"] == "gpt-4"


def _read_csv(path: Path):
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def test_write_tsv_roundtrip(tmp_path: Path):
    from export_expert_audit_pack import _write_tsv
    import csv
    rows = [{"audit_id": "A1", "notes": "line1\nline2", "score": ""}]
    out = tmp_path / "test.tsv"
    _write_tsv(out, rows, ["audit_id", "notes", "score"])
    with out.open(newline="") as fh:
        back = list(csv.DictReader(fh, delimiter="\t"))
    assert len(back) == 1
    assert back[0]["notes"] == "line1\nline2"


def test_relativize_excerpt():
    from export_expert_audit_pack import _relativize_excerpt
    root = Path("/tmp/benchmarks")
    text = "=== peaks (/tmp/benchmarks/references/foo.bed) ===\n"
    assert "/tmp/benchmarks/" not in _relativize_excerpt(text, root)
    assert "references/foo.bed" in _relativize_excerpt(text, root)


def test_summarise_paired_scores():
    s = summarise_paired_scores([60, 70], [62, 68], pass_mark=60.0)
    assert s["n"] == 2
    assert s["score_correlation"] is not None
