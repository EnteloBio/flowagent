"""Tests for the raw-LLM baseline's DAG-awareness toggle.

The raw-LLM lane is the zero-shot competitor for FlowAgent: one
provider call, no scaffolding. To make the head-to-head against
FlowAgent isolate FlowAgent's full stack (scaffolding + DAG-aware
planner + retry loop), the raw-LLM system prompt must default to
DAG-blind. Any researcher who wants to study a different question --
"what does FlowAgent's scaffolding add *over and above* the same DAG
instruction the planner already uses?" -- can opt-in via
``RawLLMCompetitor(with_dag=True)``.

Symmetric with ``test_claude_code_shim.py`` and
``test_edison_shim_dag_toggle.py``: pin the same fairness invariants
(default DAG-blind, opt-in distinct slug, two prompts diverge only in
DAG-related sentences) so a future change can't quietly re-introduce
a DAG instruction in the head-to-head baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE.parent
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(BENCH_DIR.parent))

from harness import competitors                                 # noqa: E402
from harness.competitors import RawLLMCompetitor                # noqa: E402


class TestSelectRawLLMSystemPrompt:
    def test_dag_aware_prompt_has_dependencies(self):
        p = competitors._select_raw_llm_system_prompt(dag_aware=True)
        assert '"dependencies"' in p
        assert "topological order" in p.lower()
        assert "``dependencies`` must reference names of prior steps" in p

    def test_dag_blind_prompt_strips_dependencies(self):
        p = competitors._select_raw_llm_system_prompt(dag_aware=False)
        assert '"dependencies"' not in p
        assert "topological" not in p.lower()
        assert "must reference names of prior steps" not in p

    def test_non_dag_rules_unchanged_between_arms(self):
        """The two raw-LLM prompts must diverge only in DAG-related
        sentences. If a future edit drifts the tool list, schema
        framing, or "Return ONLY the JSON object" rule, this test
        catches it before the experimental control breaks."""
        aware = competitors._select_raw_llm_system_prompt(dag_aware=True)
        blind = competitors._select_raw_llm_system_prompt(dag_aware=False)
        for shared_marker in [
            "You are a bioinformatics pipeline planner.",
            "Commands should be runnable shell pipelines",
            "fastqc, kallisto, salmon, STAR, bwa, samtools",
            "Include every step needed to go from raw input to the requested output.",
            "Return ONLY the JSON object. No markdown fences. No commentary.",
        ]:
            assert shared_marker in aware, f"missing in DAG-aware: {shared_marker!r}"
            assert shared_marker in blind, f"missing in DAG-blind: {shared_marker!r}"

    def test_legacy_alias_points_to_blind(self):
        """``_RAW_LLM_SYSTEM_PROMPT`` (the historical name) must now
        alias the DAG-blind variant. Any module that imported it
        directly inherits the fair head-to-head prompt for free."""
        assert (
            competitors._RAW_LLM_SYSTEM_PROMPT
            is competitors._RAW_LLM_SYSTEM_PROMPT_NO_DAG
        )


class TestRawLLMCompetitorDefaults:
    def test_default_with_dag_false(self):
        comp = RawLLMCompetitor(model_id="gpt-5.5-medium")
        assert comp.with_dag is False
        # Slug is the historical ``raw_<model>`` shape so existing
        # registries / results CSVs still resolve cleanly.
        assert comp.id == "raw_gpt-5.5-medium"

    def test_explicit_with_dag_true_renames_slug(self):
        comp = RawLLMCompetitor(model_id="gpt-5.5-medium", with_dag=True)
        assert comp.with_dag is True
        assert comp.id == "raw_gpt-5.5-medium_dag_aware"
        assert "dag-aware" in comp.name.lower()
