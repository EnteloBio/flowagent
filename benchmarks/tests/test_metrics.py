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

from harness.metrics import (  # noqa: E402
    completeness_metrics,
    dag_shape,
    plan_schema_valid,
    score_plan,
)
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

    def test_narrative_commands_credit_tools_by_prose(self):
        """CLI tokens mid-sentence (no shell lead token) still cover rubric tools."""
        plan = {
            "workflow_type": "rna_seq_kallisto",
            "steps": [
                {
                    "name": "qc",
                    "command": "Run FastQC on raw reads for quality control.",
                    "dependencies": [],
                    "outputs": [],
                    "description": "",
                },
                {
                    "name": "quant",
                    "command": "Use Kallisto for transcript abundance estimation.",
                    "dependencies": ["qc"],
                    "outputs": [],
                    "description": "",
                },
                {
                    "name": "report",
                    "command": "Aggregate QC with MultiQC.",
                    "dependencies": ["quant"],
                    "outputs": [],
                    "description": "",
                },
                {
                    "name": "extra",
                    "command": "samtools flagstat out/abundance.tsv",
                    "dependencies": ["report"],
                    "outputs": [],
                    "description": "",
                },
            ],
        }
        expected = {
            "expected_workflow_type": "rna_seq_kallisto",
            "expected_tools": ["fastqc", "kallisto", "multiqc"],
            "expected_min_steps": 4,
            "forbidden_tools": [],
        }
        m = score_plan(plan, expected)
        assert m["tools_present_fraction"] == 1.0
        assert m["overall_pass"] is True

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


class TestDagShape:
    """Structural shape metrics: edge density, parallel width, stage efficiency."""

    def _plan(self, edges):
        # Helper: build a plan from a list of (name, [deps]) tuples.
        return {
            "workflow_type": "x",
            "steps": [
                {"name": n, "command": "fastqc r.fq", "dependencies": list(deps)}
                for n, deps in edges
            ],
        }

    def test_linear_chain_stage_efficiency_is_one(self):
        # Linear A -> B -> C -> D: 4 steps, 4 layers (one each), efficiency 1.0
        plan = self._plan([
            ("a", []), ("b", ["a"]), ("c", ["b"]), ("d", ["c"]),
        ])
        m = dag_shape(plan)
        assert m["num_dag_layers"] == 4
        assert m["stage_efficiency"] == pytest.approx(1.0)
        assert m["stage_efficiency_raw"] == pytest.approx(1.0)
        assert m["parallel_width"] == 1

    def test_fully_parallel_branches_increase_stage_efficiency(self):
        # a is the source; b, c, d all run in parallel after a.
        # 4 steps, 2 layers -> efficiency 2.0.
        plan = self._plan([
            ("a", []),
            ("b", ["a"]),
            ("c", ["a"]),
            ("d", ["a"]),
        ])
        m = dag_shape(plan)
        assert m["num_dag_layers"] == 2
        assert m["stage_efficiency"] == pytest.approx(4 / 2)
        assert m["parallel_width"] == 3

    def test_dag_blind_zero_edges_normalised_to_one(self):
        # The DAG-blind ablation arm produces a flat plan with no edges.
        # We normalise stage_efficiency to 1.0 (so the headline is "no
        # exposed parallelism") while keeping the raw ratio available.
        plan = self._plan([
            ("a", []), ("b", []), ("c", []), ("d", []),
        ])
        m = dag_shape(plan)
        assert m["num_dag_edges"] == 0
        assert m["stage_efficiency"] == pytest.approx(1.0)
        assert m["stage_efficiency_raw"] == pytest.approx(4.0)
        assert m["parallel_width"] == 1

    def test_empty_plan_returns_zero_metrics(self):
        m = dag_shape({"workflow_type": "x", "steps": []})
        assert m["stage_efficiency"] == 0.0
        assert m["num_dag_layers"] == 0


class TestCompletenessMetrics:
    """``completeness_metrics`` reads from envelope or runs validator live."""

    def test_envelope_short_circuits_validator(self):
        plan = {
            "workflow_type": "x",
            "steps": [{"name": "a", "command": "fastqc r.fq", "dependencies": []}],
            "_completeness": {
                "pass": False,
                "failures": ["foo failed", "bar failed"],
                "attempts": 3,
            },
        }
        m = completeness_metrics(plan)
        assert m["completeness_pass"] is False
        assert m["num_completeness_failures"] == 2
        assert "foo failed" in m["completeness_failures"]
        assert "bar failed" in m["completeness_failures"]
        assert m["completeness_attempts"] == 3

    def test_runs_validator_when_no_envelope(self):
        plan = {
            "workflow_type": "rna_seq_kallisto",
            "steps": [
                {"name": "mk", "command": "mkdir -p out", "dependencies": []},
                {"name": "ix", "command": "kallisto index -i x x.fa",
                 "dependencies": ["mk"]},
                {"name": "qt", "command": "kallisto quant -i x -o o r.fq",
                 "dependencies": ["ix"]},
                {"name": "rep", "command": "multiqc -f -n multiqc_report .",
                 "dependencies": ["qt"]},
            ],
        }
        m = completeness_metrics(plan)
        assert m["completeness_pass"] is True
        assert m["num_completeness_failures"] == 0

    def test_validator_flags_align_without_index(self):
        plan = {
            "workflow_type": "x",
            "steps": [
                {"name": "mk", "command": "mkdir -p out", "dependencies": []},
                {"name": "al", "command": "bwa mem ref.fa r.fq",
                 "dependencies": ["mk"]},
                {"name": "rep", "command": "multiqc .", "dependencies": ["al"]},
            ],
        }
        m = completeness_metrics(plan)
        assert m["completeness_pass"] is False
        assert m["num_completeness_failures"] >= 1


class TestScorePlanIncludesNewMetrics:
    """``score_plan`` exposes stage_efficiency and completeness_pass per row."""

    def test_well_formed_plan_carries_completeness_and_stage(self):
        plan = {
            "workflow_type": "rna_seq_kallisto",
            "steps": [
                {"name": "mk", "command": "mkdir -p out", "dependencies": []},
                {"name": "qc", "command": "fastqc r.fq", "dependencies": ["mk"]},
                {"name": "ix", "command": "kallisto index -i x x.fa",
                 "dependencies": ["mk"]},
                {"name": "qt", "command": "kallisto quant -i x -o o r.fq",
                 "dependencies": ["ix"]},
                {"name": "rep", "command": "multiqc -f -n multiqc_report .",
                 "dependencies": ["qc", "qt"]},
            ],
        }
        expected = {
            "expected_workflow_type": "rna_seq_kallisto",
            "expected_tools": ["fastqc", "kallisto", "multiqc"],
            "expected_min_steps": 4,
            "forbidden_tools": [],
        }
        m = score_plan(plan, expected)
        # New metrics are present.
        assert "stage_efficiency" in m
        assert "stage_efficiency_raw" in m
        assert "completeness_pass" in m
        assert "num_completeness_failures" in m
        # Plan is structurally complete.
        assert m["completeness_pass"] is True
        # Stage efficiency: 5 steps, 4 layers (mk; qc+ix; qt; rep) -> 1.25.
        assert m["stage_efficiency"] == pytest.approx(5 / 4)
        # Overall pass is unchanged by new (non-gating) metrics.
        assert m["overall_pass"] is True


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
