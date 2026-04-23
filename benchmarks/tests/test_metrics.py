"""Unit tests for :mod:`harness.metrics` and mock plan synthesis.

Keeps Benchmark A scoring rules honest (overall_pass, tools, forbidden, DAG).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE.parent
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(BENCH_DIR.parent))

from harness.metrics import plan_schema_valid, score_plan  # noqa: E402
from harness.mock_plans import mock_plan_from_prompt  # noqa: E402


class TestScorePlan:

    def test_rnaseq_stub_passes_when_expectations_met(self):
        plan = {
            "workflow_type": "rna_seq_kallisto",
            "steps": [
                {"name": "a", "command": "fastqc reads.fq.gz",
                 "dependencies": [], "outputs": [], "description": ""},
                {"name": "b", "command": "kallisto quant -i idx -o out reads.fq.gz",
                 "dependencies": ["a"], "outputs": [], "description": ""},
                {"name": "c", "command": "multiqc out",
                 "dependencies": ["b"], "outputs": [], "description": ""},
                {"name": "d", "command": "samtools flagstat out/abundance.tsv",
                 "dependencies": ["c"], "outputs": [], "description": ""},
            ],
        }
        expected = {
            "expected_workflow_type": "rna_seq_kallisto",
            "expected_tools": ["fastqc", "kallisto", "multiqc"],
            "expected_min_steps": 4,
            "forbidden_tools": ["wget"],
        }
        m = score_plan(plan, expected)
        assert m["overall_pass"] is True
        assert m["type_correct"] is True
        assert m["tools_present_fraction"] == 1.0
        assert m["dag_valid"] is True

    def test_forbidden_tool_fails_even_if_elsewhere_correct(self):
        plan = {
            "workflow_type": "rna_seq_kallisto",
            "steps": [
                {"name": "q", "command": "wget https://example.com/ref.fa",
                 "dependencies": [], "outputs": [], "description": ""},
                {"name": "a", "command": "kallisto quant -i idx reads.fq.gz",
                 "dependencies": ["q"], "outputs": [], "description": ""},
            ],
        }
        expected = {
            "expected_workflow_type": "rna_seq_kallisto",
            "expected_tools": ["kallisto"],
            "expected_min_steps": 2,
            "forbidden_tools": ["wget"],
        }
        m = score_plan(plan, expected)
        assert m["no_forbidden_tools"] is False
        assert m["overall_pass"] is False

    def test_empty_plan_not_schema_valid(self):
        assert plan_schema_valid({"workflow_type": "custom", "steps": []}) is False


class TestMockPlanFromPrompt:

    def test_pads_to_expected_min_steps(self):
        entry = {
            "expected_workflow_type": "rna_seq_kallisto",
            "expected_tools": ["fastqc", "kallisto", "multiqc"],
            "expected_min_steps": 6,
        }
        plan = mock_plan_from_prompt(entry, step_name=lambda i: f"t{i}")
        assert len(plan["steps"]) >= 6
        m = score_plan(plan, entry)
        assert m["step_count_ok"] is True

    def test_list_workflow_type_normalized(self):
        entry = {
            "expected_workflow_type": ["rna_seq_star", "custom"],
            "expected_tools": ["star"],
            "expected_min_steps": 3,
            "forbidden_tools": [],
        }
        plan = mock_plan_from_prompt(entry, step_name=lambda i: f"t{i}")
        assert isinstance(plan["workflow_type"], str)
