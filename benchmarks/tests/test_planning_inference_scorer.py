"""Unit tests for the inference-tier planning scorer.

The reviewer flagged Benchmark A as saturated because every prompt
named the expected tools. The fix introduced a second prompt tier
(``inference``) that only declares ``acceptable_tool_sets`` — the
plan must cover at least one of those sets *strictly*, with the prose
fallback turned off, hallucinated tools optionally blocking
``overall_pass``, and command-semantic validation gating
``commands_well_formed_fraction == 1.0``.

These tests pin those guarantees so future refactors cannot silently
re-saturate the benchmark.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE.parent
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(BENCH_DIR.parent))

from harness.metrics import (  # noqa: E402
    score_plan, score_plan_inference, tool_covered,
)


def _wrap(steps):
    return {"workflow_type": "rna_seq_kallisto", "steps": steps}


def _step(name, command, deps=None):
    return {
        "name": name, "command": command,
        "dependencies": deps or [],
        "outputs": [], "description": "",
    }


class TestProseFallbackFlag:

    def test_default_prose_fallback_credits_tool_in_narrative(self):
        plan = _wrap([
            _step("qc", "Run FastQC on raw reads to inspect quality."),
        ])
        # ``plan_tools`` is empty — no command-leading FastQC token —
        # so credit must come from the prose fallback path.
        assert tool_covered("fastqc", set(), plan=plan) is True

    def test_disabled_prose_fallback_requires_invocation(self):
        plan = _wrap([
            _step("qc", "Run FastQC on raw reads to inspect quality."),
        ])
        assert tool_covered(
            "fastqc", set(), plan=plan, prose_fallback=False,
        ) is False

    def test_disabled_prose_fallback_still_accepts_invocation(self):
        # Tool present in plan_tools (the extraction layer) -> credited
        # regardless of fallback flag.
        assert tool_covered("fastqc", {"fastqc"}, prose_fallback=False) is True


class TestInferenceScorer:

    def _good_kallisto_plan(self):
        return _wrap([
            _step("qc", "fastqc reads_1.fq.gz reads_2.fq.gz -o qc/"),
            _step("idx", "kallisto index -i tx.idx tx.fa", deps=["qc"]),
            _step(
                "quant",
                "kallisto quant -i tx.idx -o quant/ "
                "reads_1.fq.gz reads_2.fq.gz",
                deps=["idx"],
            ),
            _step("mqc", "multiqc -o mqc quant/", deps=["quant"]),
        ])

    def test_passes_when_one_acceptable_set_strictly_covered(self):
        expected = {
            "tier": "inference",
            "acceptable_tool_sets": [
                ["salmon", "tximport", "multiqc"],
                ["kallisto", "multiqc"],
                ["star", "featurecounts", "multiqc"],
            ],
            "expected_min_steps": 4,
            "forbidden_tools": ["wget"],
        }
        m = score_plan_inference(self._good_kallisto_plan(), expected)
        assert m["overall_pass"] is True
        assert m["any_tool_set_matched"] is True
        assert m["matched_tool_set"] == "kallisto,multiqc"
        assert m["commands_well_formed_fraction"] == 1.0

    def test_prose_only_mention_does_not_satisfy_inference(self):
        plan = _wrap([
            _step("qc", "fastqc reads_1.fq.gz reads_2.fq.gz -o qc/"),
            _step("idx", "kallisto index -i tx.idx tx.fa", deps=["qc"]),
            _step(
                "quant",
                "kallisto quant -i tx.idx -o quant/ "
                "reads_1.fq.gz reads_2.fq.gz",
                deps=["idx"],
            ),
            _step(
                "mqc",
                "echo 'Run MultiQC over the kallisto outputs to aggregate.'",
                deps=["quant"],
            ),
        ])
        expected = {
            "tier": "inference",
            "acceptable_tool_sets": [["kallisto", "multiqc"]],
            "expected_min_steps": 4,
            "forbidden_tools": [],
        }
        m = score_plan_inference(plan, expected)
        assert m["any_tool_set_matched"] is False, (
            "prose-only mention of multiqc must NOT satisfy the "
            "inference scorer"
        )
        assert m["overall_pass"] is False

    def test_forbidden_tool_blocks_pass(self):
        plan = _wrap([
            _step("dl", "wget -O reads.fq.gz http://example.com/r.fq.gz"),
            _step("idx", "kallisto index -i tx.idx tx.fa", deps=["dl"]),
            _step(
                "quant",
                "kallisto quant -i tx.idx -o quant/ reads.fq.gz",
                deps=["idx"],
            ),
            _step("mqc", "multiqc -o mqc quant/", deps=["quant"]),
        ])
        expected = {
            "tier": "inference",
            "acceptable_tool_sets": [["kallisto", "multiqc"]],
            "expected_min_steps": 4,
            "forbidden_tools": ["wget"],
        }
        m = score_plan_inference(plan, expected)
        assert m["no_forbidden_tools"] is False
        assert m["overall_pass"] is False

    def test_malformed_command_blocks_pass(self):
        plan = _wrap([
            _step("qc", "fastqc reads_1.fq.gz reads_2.fq.gz -o qc/"),
            _step("idx", "kallisto index", deps=["qc"]),
            _step("quant", "kallisto quant", deps=["idx"]),
            _step("mqc", "multiqc -o mqc quant/", deps=["quant"]),
        ])
        expected = {
            "tier": "inference",
            "acceptable_tool_sets": [["kallisto", "multiqc"]],
            "expected_min_steps": 4,
            "forbidden_tools": [],
        }
        m = score_plan_inference(plan, expected)
        assert m["commands_well_formed_fraction"] < 1.0
        assert m["overall_pass"] is False

    def test_dispatcher_routes_inference_when_tier_set(self):
        plan = self._good_kallisto_plan()
        expected = {
            "tier": "inference",
            "acceptable_tool_sets": [["kallisto", "multiqc"]],
            "expected_min_steps": 4,
            "forbidden_tools": [],
        }
        m_dispatch = score_plan(plan, expected)
        m_direct = score_plan_inference(plan, expected)
        assert m_dispatch["overall_pass"] == m_direct["overall_pass"]
        assert m_dispatch.get("any_tool_set_matched") == \
            m_direct.get("any_tool_set_matched")


class TestStrictHallucinationsFlag:

    def test_strict_flag_demotes_pass_for_hallucinations(self):
        plan = _wrap([
            _step("qc", "fastqc reads.fq.gz"),
            _step(
                "weird",
                "totally_made_up_tool --magic ref.fa",
                deps=["qc"],
            ),
            _step(
                "idx",
                "kallisto index -i x.idx ref.fa",
                deps=["weird"],
            ),
            _step(
                "quant",
                "kallisto quant -i x.idx -o out reads.fq.gz",
                deps=["idx"],
            ),
            _step("mqc", "multiqc out", deps=["quant"]),
        ])
        expected = {
            "expected_workflow_type": "rna_seq_kallisto",
            "expected_tools": ["fastqc", "kallisto", "multiqc"],
            "expected_min_steps": 4,
            "forbidden_tools": [],
        }
        m_loose = score_plan(plan, expected)
        m_strict = score_plan(plan, expected, strict_hallucinations=True)
        # The plan covers all expected tools, so loose mode passes
        # despite the hallucinated step.
        assert m_loose["hallucination_rate"] > 0.0
        # Strict mode demotes the pass.
        assert m_strict["overall_pass"] is False
