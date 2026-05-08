"""Tests for the Edison shim's DAG-awareness toggle.

Symmetric with ``test_claude_code_shim.py``: pin the same fairness
invariants (default DAG-blind, ``--with-dag-instruction`` opts-in,
the two system-prompt guidelines diverge only in DAG-related
sentences) so a future change to Edison's prompt scaffolding can't
quietly re-introduce a DAG instruction in the head-to-head baseline.

The ``edison-client`` SDK is not assumed available -- these tests
cover the pure-Python prompt selector + envelope plumbing only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE.parent
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(BENCH_DIR.parent))

from harness import edison_shim                                  # noqa: E402
from harness.competitors import EdisonCompetitor                 # noqa: E402


class TestSelectGuidelines:
    """The two prompts must diverge ONLY in DAG-related sentences.

    Same invariant as the Claude Code shim's ``TestSelectTemplate``:
    if a future edit accidentally drifts an unrelated rule (tool-first
    command, no markdown fences, etc.) between the two arms, the
    ablation no longer measures DAG awareness in isolation.
    """

    def test_dag_aware_guidelines_have_dependencies(self):
        g = edison_shim._select_guidelines(dag_aware=True)
        assert '"dependencies"' in g
        assert "topological order" in g.lower()
        assert "must reference prior step names" in g

    def test_dag_blind_guidelines_strip_dependencies(self):
        g = edison_shim._select_guidelines(dag_aware=False)
        assert '"dependencies"' not in g
        assert "topological" not in g.lower()
        assert "must reference prior step names" not in g

    def test_non_dag_rules_unchanged_between_arms(self):
        """Every rule that *isn't* about DAGs must be byte-identical
        across the two arms. Any drift here is a bug."""
        aware = edison_shim._select_guidelines(dag_aware=True)
        blind = edison_shim._select_guidelines(dag_aware=False)

        for shared_marker in [
            "STRICT OUTPUT FORMAT.",
            "Do not execute any tool",
            "The first token of each ``command`` MUST be the bioinformatics tool",
            "Cover every step from raw input to the requested final output.",
            "Return ONLY the JSON object. No markdown fences, no commentary.",
        ]:
            assert shared_marker in aware, f"missing marker in DAG-aware: {shared_marker!r}"
            assert shared_marker in blind, f"missing marker in DAG-blind: {shared_marker!r}"

    def test_default_module_alias_points_to_blind(self):
        """The legacy ``_PLAN_GUIDELINES`` symbol now aliases the
        DAG-blind variant -- any old code that imported the symbol
        directly should pick up the fair head-to-head prompt for
        free, not the historical DAG-aware one."""
        assert edison_shim._PLAN_GUIDELINES is edison_shim._PLAN_GUIDELINES_NO_DAG


class TestEnvelopeDagAwareKey:
    """The envelope must carry the arm label so downstream sweeps can
    sanity-check the active prompt template."""

    def test_default_envelope_dag_aware_false(self):
        env = edison_shim._envelope({"workflow_type": "custom", "steps": []})
        # The shim's runtime default is DAG-blind; mirror that here so
        # callers that build an envelope without specifying the arm
        # see the same default the CLI uses.
        assert env["dag_aware"] is False

    def test_explicit_dag_aware_envelope(self):
        env = edison_shim._envelope(
            {"workflow_type": "custom", "steps": []},
            dag_aware=True,
        )
        assert env["dag_aware"] is True


class TestCompetitorDefaults:
    """``EdisonCompetitor()`` must default to DAG-blind, mirroring the
    Claude Code competitor. The opt-in DAG-aware arm must use a
    distinct slug so paired Benchmark J runs don't collide."""

    def test_default_with_dag_false(self):
        comp = EdisonCompetitor()
        assert comp.with_dag is False
        assert comp.id == "edison"

    def test_explicit_with_dag_true_renames_slug(self):
        comp = EdisonCompetitor(with_dag=True)
        assert comp.with_dag is True
        assert comp.id == "edison_dag_aware"
        assert "dag" in comp.name.lower()


class TestShimMainParsesFlag:
    """Smoke-test ``edison_shim.main()``: with ``EDISON_API_KEY``
    unset the CLI returns its soft-skip envelope, but the
    ``dag_aware`` field still reflects the parsed flag, so the
    ablation harness sees the correct arm even when Edison isn't
    available locally."""

    def test_main_default_is_dag_blind(self, capsys, monkeypatch):
        monkeypatch.delenv("EDISON_API_KEY", raising=False)
        rc = edison_shim.main([
            "--prompt", "rna-seq",
            "--files", "[]",
        ])
        assert rc == 0
        import json as _json
        env = _json.loads(capsys.readouterr().out.strip())
        assert env["dag_aware"] is False
        assert env["error"]  # soft-skipped, EDISON_API_KEY missing

    def test_main_with_dag_instruction_opt_in(self, capsys, monkeypatch):
        monkeypatch.delenv("EDISON_API_KEY", raising=False)
        rc = edison_shim.main([
            "--prompt", "rna-seq",
            "--files", "[]",
            "--with-dag-instruction",
        ])
        assert rc == 0
        import json as _json
        env = _json.loads(capsys.readouterr().out.strip())
        assert env["dag_aware"] is True
