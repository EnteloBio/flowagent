"""Tests for Benchmark E — head-to-head competitor harness.

These tests cover:
  * the abstract Competitor interface
  * the plan-normaliser's tolerance to variant output shapes
  * FlowAgent and BioMaster adapters' availability checks and fallbacks
  * the per-cell runner's error handling (timeout / not-available / exception)
  * end-to-end smoke of ``bench_competitors.py --mock`` writing a results CSV

All tests run offline — no real LLM calls. A ``StubCompetitor`` returns a
canned plan so we exercise the scoring pipeline deterministically.
"""

from __future__ import annotations

import asyncio
import csv
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE.parent
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(BENCH_DIR.parent))

from harness.competitors import (                       # noqa: E402
    BioMasterCompetitor, Competitor, CompetitorResult,
    FlowAgentCompetitor, _empty_plan, _normalise_plan, _normalise_step,
    build_registry,
)
from harness.metrics import score_plan                  # noqa: E402


# ── Plan normaliser ──────────────────────────────────────────────

class TestNormalisePlan:

    def test_empty_plan_shape(self):
        p = _empty_plan()
        assert p["workflow_type"] == "custom"
        assert p["steps"] == []

    def test_canonical_shape_passes_through(self):
        original = {
            "workflow_type": "rna_seq_kallisto",
            "steps": [
                {"name": "s1", "command": "fastqc x.fq.gz",
                 "dependencies": [], "outputs": ["fastqc/x"],
                 "description": "QC"}
            ],
        }
        out = _normalise_plan(original)
        assert out["workflow_type"] == "rna_seq_kallisto"
        assert len(out["steps"]) == 1
        assert out["steps"][0]["command"] == "fastqc x.fq.gz"

    @pytest.mark.parametrize("alt_key,alt_val,expected", [
        ("pipeline_type", "chip_seq",      "chip_seq"),
        ("workflow",      "variant_call",  "variant_call"),
    ])
    def test_alternative_workflow_keys(self, alt_key, alt_val, expected):
        out = _normalise_plan({alt_key: alt_val, "steps": []})
        assert out["workflow_type"] == expected

    def test_list_workflow_type_takes_first(self):
        out = _normalise_plan({"workflow_type": ["methylation", "custom"], "steps": []})
        assert out["workflow_type"] == "methylation"

    def test_step_key_tolerance(self):
        # BioMaster / AutoBA-style variants: ``tasks`` with ``id``/``cmd``/``deps``
        out = _normalise_plan({
            "workflow": "rna_seq_hisat2",
            "tasks": [
                {"id": "align", "cmd": "hisat2 -x idx -U r.fq -S r.sam",
                 "deps": []},
                {"id": "sort", "cmd": "samtools sort r.sam -o r.bam",
                 "deps": ["align"]},
            ],
        })
        assert out["workflow_type"] == "rna_seq_hisat2"
        assert len(out["steps"]) == 2
        names = [s["name"] for s in out["steps"]]
        assert names == ["align", "sort"]
        assert out["steps"][1]["dependencies"] == ["align"]

    def test_missing_fields_filled(self):
        step = _normalise_step({"name": "x", "command": "y"})
        assert step["dependencies"] == []
        assert step["outputs"] == []
        assert step["description"] == ""


# ── Adapter availability ─────────────────────────────────────────

class TestFlowAgentCompetitor:

    def test_id_and_name(self):
        c = FlowAgentCompetitor()
        assert c.id == "flowagent"
        assert c.name == "FlowAgent"

    def test_available_when_flowagent_installed(self):
        c = FlowAgentCompetitor()
        ok, why = c.available()
        # Either flowagent is installed (pass) or gracefully reports why
        assert isinstance(ok, bool)
        assert isinstance(why, str)
        if not ok:
            assert "flowagent" in why.lower()


class TestBioMasterCompetitor:

    def test_id_and_url(self):
        c = BioMasterCompetitor()
        assert c.id == "biomaster"
        assert "biorxiv" in c.url.lower()

    def test_unavailable_explains_install(self, monkeypatch):
        # With no env var + no package installed, should produce a clean hint
        monkeypatch.delenv("BIOMASTER_CLI", raising=False)
        # Force the import-probe loop to fail for all candidates
        real_import = importlib.import_module

        def _fake_import(name, *a, **kw):
            if "biomaster" in name:
                raise ImportError(f"no module {name}")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(importlib, "import_module", _fake_import)
        c = BioMasterCompetitor()
        ok, why = c.available()
        assert ok is False
        assert "BIOMASTER_CLI" in why or "not installed" in why.lower()

    def test_cli_env_var_takes_precedence_only_if_executable(self, tmp_path, monkeypatch):
        # Non-existent path → still unavailable
        monkeypatch.setenv("BIOMASTER_CLI", str(tmp_path / "does-not-exist"))
        real_import = importlib.import_module

        def _fake_import(name, *a, **kw):
            if "biomaster" in name:
                raise ImportError()
            return real_import(name, *a, **kw)

        monkeypatch.setattr(importlib, "import_module", _fake_import)
        c = BioMasterCompetitor()
        ok, _ = c.available()
        assert ok is False

    def test_plan_returns_error_when_unavailable(self, monkeypatch):
        monkeypatch.delenv("BIOMASTER_CLI", raising=False)
        real_import = importlib.import_module

        def _fake_import(name, *a, **kw):
            if "biomaster" in name:
                raise ImportError()
            return real_import(name, *a, **kw)

        monkeypatch.setattr(importlib, "import_module", _fake_import)

        c = BioMasterCompetitor()
        result = asyncio.run(c.plan("Run RNA-seq with kallisto"))
        assert isinstance(result, CompetitorResult)
        assert result.error is not None
        assert "not-available" in result.error
        assert result.plan["steps"] == []


# ── Registry ─────────────────────────────────────────────────────

class TestRegistry:

    def test_default_registry_has_both(self):
        reg = build_registry()
        assert set(reg) == {"flowagent", "biomaster"}
        assert all(isinstance(v, Competitor) for v in reg.values())

    def test_unique_slugs_and_colours(self):
        reg = build_registry()
        from harness.plot import _COMPETITOR_COLOURS
        for slug in reg:
            assert slug in _COMPETITOR_COLOURS, (
                f"Competitor {slug!r} missing from _COMPETITOR_COLOURS — "
                f"figure would fall back to grey")


# ── Stub competitor — exercises scoring end-to-end ───────────────

class StubCompetitor(Competitor):
    """A competitor that returns a fixed plan without touching any LLM.
    Used to verify the runner + scoring pipeline deterministically."""

    id = "stub"
    name = "Stub"

    def __init__(self, plan_to_return: Dict[str, Any], fail: bool = False):
        self._plan = plan_to_return
        self._fail = fail

    def available(self):
        return True, ""

    async def plan(self, prompt, *, context=None):
        await asyncio.sleep(0)
        if self._fail:
            raise RuntimeError("stub failure for test")
        return CompetitorResult(
            plan=self._plan, wall_seconds=0.01,
            prompt_tokens=100, completion_tokens=50, llm_calls=1,
        )


class TestRunCell:

    def _entry(self):
        return {
            "id": "rnaseq_basic",
            "prompt": "RNA-seq with kallisto and multiqc",
            "expected_workflow_type": "rna_seq_kallisto",
            "expected_tools": ["kallisto", "multiqc"],
            "expected_min_steps": 2,
            "forbidden_tools": ["wget"],
        }

    def _good_plan(self):
        return {
            "workflow_type": "rna_seq_kallisto",
            "steps": [
                {"name": "kallisto", "command": "kallisto quant -i idx -o out r.fq",
                 "dependencies": [], "outputs": [], "description": ""},
                {"name": "multiqc",  "command": "multiqc -f out",
                 "dependencies": ["kallisto"], "outputs": [], "description": ""},
            ],
        }

    def test_stub_good_plan_passes_scoring(self):
        from bench_competitors import _run_cell
        comp = StubCompetitor(self._good_plan())
        row = asyncio.run(_run_cell(
            comp, self._entry(), replicate=0,
            mock=False, timeout=5.0, model_cfg={"id": "test-model"}))
        assert row["overall_pass"] is True
        assert row["competitor"] == "stub"
        assert row["prompt_tokens"] == 100
        assert row["completion_tokens"] == 50

    def test_exception_is_caught_and_scored(self):
        from bench_competitors import _run_cell
        comp = StubCompetitor(self._good_plan(), fail=True)
        row = asyncio.run(_run_cell(
            comp, self._entry(), replicate=0,
            mock=False, timeout=5.0, model_cfg={"id": "test-model"}))
        assert row["overall_pass"] is False
        assert row["error"] is not None
        assert "RuntimeError" in row["error"]

    def test_timeout_is_recorded(self):
        from bench_competitors import _run_cell

        class SlowComp(Competitor):
            id = "slow"; name = "Slow"
            def available(self): return True, ""
            async def plan(self, prompt, *, context=None):
                await asyncio.sleep(5)
                return CompetitorResult(plan=_empty_plan(), wall_seconds=5)

        row = asyncio.run(_run_cell(
            SlowComp(), self._entry(), replicate=0,
            mock=False, timeout=0.2, model_cfg={"id": "test-model"}))
        assert row["error"] is not None
        assert "timeout" in row["error"].lower()

    def test_mock_mode_produces_deterministic_row(self):
        from bench_competitors import _run_cell
        comp = StubCompetitor({"workflow_type": "x", "steps": []})
        # In mock mode the competitor isn't invoked at all
        row = asyncio.run(_run_cell(
            comp, self._entry(), replicate=0,
            mock=True, timeout=5.0, model_cfg={"id": "test-model"}))
        assert row["error"] is None
        assert "plan" in row


# ── End-to-end smoke ─────────────────────────────────────────────

class TestEndToEndMock:

    def test_cli_mock_mode_writes_csv(self, tmp_path, monkeypatch):
        import subprocess
        py = sys.executable
        env = dict(os.environ)
        env.pop("BIOMASTER_CLI", None)

        result = subprocess.run(
            [py, str(BENCH_DIR / "bench_competitors.py"),
             "--mock",
             "--competitors=flowagent,biomaster",
             "--prompts=rnaseq_kallisto_basic,chipseq_macs2",
             "--replicates=1",
             "--out", str(tmp_path)],
            capture_output=True, text=True, env=env, timeout=60,
        )
        assert result.returncode == 0, (
            f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
        )
        # Find the timestamped output dir
        comp_dirs = list((tmp_path / "competitors").iterdir())
        assert comp_dirs, "bench_competitors did not create any output dir"
        metrics = comp_dirs[0] / "metrics.csv"
        assert metrics.exists()

        with open(metrics) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 4, f"expected 4 rows, got {len(rows)}"
        competitors = {r["competitor"] for r in rows}
        assert competitors == {"flowagent", "biomaster"}
        # Every row in mock mode should have a plan + metrics
        for r in rows:
            assert r["plan_valid"].lower() in ("true", "false")
