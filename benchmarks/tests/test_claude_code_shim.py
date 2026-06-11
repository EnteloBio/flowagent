"""Tests for the Claude Code shim's DAG-awareness ablation (Benchmark J).

Two arms must be provable from the prompt-template alone, without
invoking the ``claude`` CLI: that's the point of the ablation
(prompt-level intervention is the *only* knob we have on a competitor
framework's planning behaviour). These tests pin:

  * the **default** template is DAG-blind -- no ``dependencies``
    field, no topological-order rule. This is the fair head-to-head
    baseline for Benchmark E.
  * passing ``--with-dag-instruction`` switches to the DAG-aware
    template (Benchmark J's opt-in arm).
  * ``ClaudeCodeCompetitor()`` defaults to ``with_dag=False`` and
    propagates the flag to the shim's argv only when explicitly
    opted-in.

End-to-end shim execution is not covered here -- it requires the
``claude`` CLI, which is mocked in the broader competitor-harness
tests via the ``StubCompetitor``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE.parent
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(BENCH_DIR.parent))

from harness import claude_code_shim                        # noqa: E402
from harness.competitors import ClaudeCodeCompetitor        # noqa: E402


class TestSelectTemplate:
    """The two templates must diverge ONLY in DAG-related sentences.

    If a future edit accidentally drifts a non-DAG rule between the two
    arms, the ablation no longer measures DAG awareness in isolation --
    it measures DAG awareness plus the drift. Pinning the divergence
    here protects the experimental control.
    """

    def test_dag_aware_template_has_dependencies_field(self):
        tpl = claude_code_shim._select_template(dag_aware=True)
        assert '"dependencies"' in tpl
        assert "topological order" in tpl.lower()
        assert "must reference names of prior steps" in tpl

    def test_dag_blind_template_has_no_dependencies_field(self):
        tpl = claude_code_shim._select_template(dag_aware=False)
        assert '"dependencies"' not in tpl
        assert "topological order" not in tpl.lower()
        assert "prior step" not in tpl

    def test_non_dag_rules_unchanged_between_arms(self):
        """The bioinformatics-specific rules (tool-first command,
        no-side-effects, no-fences) must be identical -- the only
        experimental variable is the DAG instruction."""
        aware = claude_code_shim._select_template(dag_aware=True)
        blind = claude_code_shim._select_template(dag_aware=False)
        # Shared rules that must survive in BOTH arms.
        for shared in [
            "first token of each ``command`` MUST be the bioinformatics tool",
            "Cover every step from raw input to the requested final output",
            "Do NOT actually create files",
            "Return ONLY the JSON object, no markdown fences",
        ]:
            assert shared in aware, f"DAG-aware template missing: {shared!r}"
            assert shared in blind, f"DAG-blind template missing: {shared!r}"

    def test_template_formattable_without_keyerror(self):
        """Both templates must accept the same {prompt} / {files} fields,
        otherwise switching arms at runtime crashes the shim."""
        for dag_aware in (True, False):
            tpl = claude_code_shim._select_template(dag_aware=dag_aware)
            tpl.format(prompt="run kallisto quant on r.fq", files="r.fq")


class TestEnvelopeDagAwareKey:
    """The envelope must carry the arm label so downstream sweeps can
    sanity-check that each cell ran under the requested template.

    The shim's ``_envelope`` helper still defaults ``dag_aware=True``
    for backwards compatibility on internal call-sites (most early
    return paths historically passed nothing); the actual *runtime*
    default is set by ``main()`` based on ``--with-dag-instruction``,
    and that default is OFF (DAG-blind). See
    ``TestShimMainParsesFlag`` for the runtime check.
    """

    def test_explicit_dag_aware_envelope(self):
        env = claude_code_shim._envelope(
            {"workflow_type": "custom", "steps": []},
            dag_aware=True,
        )
        assert env["dag_aware"] is True

    def test_explicit_dag_blind_envelope(self):
        env = claude_code_shim._envelope(
            {"workflow_type": "custom", "steps": []},
            dag_aware=False,
        )
        assert env["dag_aware"] is False


class TestCliFlagPropagation:
    """``ClaudeCodeCompetitor`` defaults to DAG-blind, and only passes
    ``--with-dag-instruction`` to the shim when ``with_dag=True``.

    This is the fairness invariant: the default ``claude_code`` slug
    used by Benchmark E must never silently inject a DAG instruction
    into Claude Code's prompt -- that would confound the head-to-head
    against FlowAgent (whose differentiator is its DAG-aware planner).
    """

    def test_default_with_dag_false(self):
        """Default ``ClaudeCodeCompetitor()`` is DAG-blind. The slug
        ``claude_code`` therefore refers to the DAG-blind variant in
        Benchmark E."""
        comp = ClaudeCodeCompetitor()
        assert comp.with_dag is False
        assert comp.id == "claude_code"

    def test_explicit_with_dag_true_renames_slug(self):
        """The DAG-aware opt-in arm must use a distinct slug + display
        name so Benchmark J's two arms don't collide in the registry /
        CSV."""
        comp = ClaudeCodeCompetitor(with_dag=True)
        assert comp.with_dag is True
        assert comp.id == "claude_code_dag_aware"
        assert "dag" in comp.name.lower()

    def test_invoke_shim_argv_includes_flag_only_when_opt_in(self):
        """Capture the argv built by ``_invoke_shim`` without actually
        spawning the subprocess; assert ``--with-dag-instruction`` only
        lands when ``with_dag=True``."""
        captured: dict = {}

        async def _fake_create_subprocess_exec(*argv, **kwargs):
            captured["argv"] = list(argv)
            # Return a minimal stub that satisfies the .communicate() call.
            class _StubProc:
                returncode = 0
                async def communicate(self):
                    return (
                        b'{"plan": {"workflow_type": "custom", "steps": []}, '
                        b'"prompt_tokens": 0, "completion_tokens": 0, '
                        b'"llm_calls": 0, "cost_usd": 0.0, '
                        b'"wall_seconds": 0.0, "dag_aware": false, '
                        b'"error": null}',
                        b"",
                    )
            return _StubProc()

        for with_dag, want_flag in [(True, True), (False, False)]:
            captured.clear()
            comp = ClaudeCodeCompetitor(with_dag=with_dag)
            with patch(
                "harness.competitors.asyncio.create_subprocess_exec",
                side_effect=_fake_create_subprocess_exec,
            ):
                import asyncio
                asyncio.run(comp._invoke_shim("rna-seq with kallisto", None))
            argv = captured["argv"]
            has_flag = "--with-dag-instruction" in argv
            # Negative invariant: legacy ``--no-dag-instruction`` must
            # never appear -- the flag was renamed in the fairness flip
            # and a stale reference would silently break the ablation.
            assert "--no-dag-instruction" not in argv, (
                f"stale --no-dag-instruction flag in argv: {argv}")
            assert has_flag is want_flag, (
                f"with_dag={with_dag}: expected --with-dag-instruction "
                f"present={want_flag}, got argv={argv}")


class TestResolveClaudeBin:
    def test_invalid_env_falls_back_to_path(self, monkeypatch, tmp_path):
        fake = tmp_path / "claude"
        fake.write_text("#!/bin/sh\n")
        fake.chmod(0o755)
        monkeypatch.setenv("CLAUDE_CODE_BIN", "/no/such/binary/exists")
        monkeypatch.setattr(
            claude_code_shim.shutil, "which",
            lambda name: str(fake) if name == "claude" else None,
        )
        assert claude_code_shim._resolve_claude_bin() == str(fake)


class TestInvokeClaudeEnv:
    def test_strips_anthropic_api_key_by_default(self, monkeypatch, tmp_path):
        captured: dict = {}

        class _Proc:
            returncode = 0
            stdout = b'{"result": "{\\"workflow_type\\": \\"custom\\", \\"steps\\": []}", "usage": {}}'
            stderr = b""

        def _fake_run(*_a, **kwargs):
            captured["env"] = kwargs.get("env")
            return _Proc()

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-should-be-stripped")
        monkeypatch.setattr(claude_code_shim.subprocess, "run", _fake_run)
        claude_code_shim._invoke_claude(
            "plan something",
            binary=str(tmp_path / "claude"),
            model=None,
            timeout=30.0,
            cwd=tmp_path,
        )
        assert "ANTHROPIC_API_KEY" not in captured["env"]


class TestShimMainParsesFlag:
    """Smoke-test ``main()`` directly: assert the runtime default is
    DAG-blind, and that ``--with-dag-instruction`` flips to DAG-aware.
    The CLI is short-circuited by clearing ``CLAUDE_CODE_BIN`` so the
    shim returns its "binary not found" error envelope without
    spawning anything -- enough to confirm the flag flows through
    arg-parsing and is reflected in the envelope on the soft-skip
    path."""

    def test_main_default_is_dag_blind(self, capsys, monkeypatch):
        """No flag passed: envelope must show ``dag_aware=false``.
        This is THE invariant for fair head-to-head comparisons."""
        monkeypatch.setenv("CLAUDE_CODE_BIN", "/no/such/binary/exists")
        monkeypatch.setattr(claude_code_shim.shutil, "which", lambda _name: None)
        rc = claude_code_shim.main([
            "--prompt", "rna-seq",
            "--files", "[]",
        ])
        assert rc == 0
        import json as _json
        env = _json.loads(capsys.readouterr().out.strip())
        assert env["dag_aware"] is False
        assert env["error"]  # soft-skipped, error message present

    def test_main_with_dag_instruction_opt_in(self, capsys, monkeypatch):
        """``--with-dag-instruction`` switches to the DAG-aware template
        (Benchmark J's opt-in arm)."""
        monkeypatch.setenv("CLAUDE_CODE_BIN", "/no/such/binary/exists")
        monkeypatch.setattr(claude_code_shim.shutil, "which", lambda _name: None)
        rc = claude_code_shim.main([
            "--prompt", "rna-seq",
            "--files", "[]",
            "--with-dag-instruction",
        ])
        assert rc == 0
        import json as _json
        env = _json.loads(capsys.readouterr().out.strip())
        assert env["dag_aware"] is True

    def test_main_rejects_legacy_no_dag_instruction_flag(self, monkeypatch):
        """The flag was renamed during the fairness flip. Reject the
        old form loudly so a stale shell-script using the legacy flag
        doesn't silently run with the new defaults (which would mean
        "DAG-blind" instead of the user's intended "DAG-blind")."""
        monkeypatch.setenv("CLAUDE_CODE_BIN", "/no/such/binary/exists")
        with pytest.raises(SystemExit):
            claude_code_shim.main([
                "--prompt", "rna-seq",
                "--files", "[]",
                "--no-dag-instruction",
            ])
