"""Pin the universal no-DAG convention for competitor shims.

After the fairness flip, **no** competitor shim may synthesise a
``dependencies`` chain from execution / enumeration order. The shims
that historically wrote ``[step_N-1]`` linear chains -- Biomni,
BioMaster, AutoBA -- now leave ``dependencies`` empty so the metric
pipeline sees a flat-list plan that matches the DAG-blind defaults of
the other three competitors (Claude Code, Edison, raw-LLM).

These tests run the shims' pure-Python parsers against synthetic input
(no real agent invocation needed) and assert ``dependencies == []`` on
every emitted step. They're cheap and deterministic, so they're the
fastest way to catch a regression where a future edit re-introduces
synthetic linear deps.

Symmetry with the prompt-level toggles for Claude Code / Edison /
raw-LLM is enforced separately by ``test_competitors.py``'s
``test_default_competitors_are_dag_blind``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE.parent
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(BENCH_DIR.parent))

from harness import biomni_shim, biomaster_shim, autoba_shim   # noqa: E402


# ── Biomni ─────────────────────────────────────────────────────────

class TestBiomniShimNoDagSynthesis:

    def _stub_message(self, tool_calls: List[Dict[str, Any]]) -> SimpleNamespace:
        """LangChain message stub: only ``tool_calls`` is read by the parser."""
        return SimpleNamespace(tool_calls=tool_calls)

    def test_messages_to_steps_emits_empty_deps(self):
        msgs = [
            self._stub_message([{"name": "fastqc", "args": {"input": "x.fq"}}]),
            self._stub_message([{"name": "kallisto", "args": {"index": "k.idx"}}]),
            self._stub_message([{"name": "multiqc", "args": {"d": "."}}]),
        ]
        steps = biomni_shim._messages_to_steps(msgs)
        assert len(steps) == 3
        for s in steps:
            assert s["dependencies"] == [], (
                f"Biomni shim must not synthesise deps; got {s['dependencies']!r}"
            )
        # Step ordering / naming still works (this is execution order
        # captured in the step list itself, NOT in dependencies).
        assert [s["name"] for s in steps] == ["fastqc_1", "kallisto_2", "multiqc_3"]

    def test_messages_to_steps_handles_object_tool_calls(self):
        # Real LangChain often produces tool-call objects rather than dicts;
        # the parser falls back to attribute access. The empty-deps invariant
        # must hold on that path too.
        tc = SimpleNamespace(name="bwa", args={"ref": "hg38.fa"})
        msgs = [self._stub_message([tc]), self._stub_message([tc])]
        steps = biomni_shim._messages_to_steps(msgs)
        assert len(steps) == 2
        for s in steps:
            assert s["dependencies"] == []

    def test_fallback_steps_from_text_emits_empty_deps(self):
        text = (
            "1. Run fastqc on the input fastq.\n"
            "2. Index the reference with kallisto.\n"
            "3. Quantify with kallisto quant.\n"
        )
        steps = biomni_shim._fallback_steps_from_text(text)
        assert len(steps) == 3
        for s in steps:
            assert s["dependencies"] == []
        assert [s["name"] for s in steps] == [
            "narrative_1", "narrative_2", "narrative_3",
        ]

    def test_empty_input_returns_empty_list(self):
        assert biomni_shim._messages_to_steps([]) == []
        assert biomni_shim._fallback_steps_from_text("") == []


# ── BioMaster ──────────────────────────────────────────────────────

class TestBiomasterShimNoDagSynthesis:

    def test_map_step_emits_empty_deps(self):
        # Three sequential PLAN.json entries. Pre-flip we'd have written
        # dependencies = [], ["step_1"], ["step_2"]; now all three are [].
        for idx, raw in enumerate([
            {"step_number": 1, "tools": "fastqc",
             "description": "QC", "output_filename": ["qc.html"]},
            {"step_number": 2, "tools": "kallisto",
             "description": "Quant", "output_filename": ["abundance.h5"]},
            {"step_number": 3, "tools": "deseq2",
             "description": "DE", "output_filename": []},
        ]):
            step = biomaster_shim._map_step(raw, idx)
            assert step["dependencies"] == [], (
                f"BioMaster shim must not synthesise deps; "
                f"got {step['dependencies']!r} on step_number="
                f"{raw.get('step_number')}"
            )

    def test_map_step_preserves_step_naming(self):
        """Empty deps doesn't break step naming -- ``step_N`` is still
        assigned from the raw ``step_number`` so the metrics pipeline
        sees stable, unique names."""
        step = biomaster_shim._map_step(
            {"step_number": 4, "tools": "samtools sort"}, idx=99,
        )
        assert step["name"] == "step_4"

    def test_map_step_handles_missing_step_number(self):
        # Falls back to idx + 1 when step_number is absent. Empty deps must
        # still be the result -- the fallback path is no excuse to invent a chain.
        step = biomaster_shim._map_step({"tools": "kallisto"}, idx=2)
        assert step["dependencies"] == []
        assert step["name"] == "step_3"


# ── AutoBA ─────────────────────────────────────────────────────────

class TestAutobaShimNoDagSynthesis:

    def test_map_plan_emits_empty_deps(self):
        tasks = [
            "Run FastQC on the input FASTQ files",
            "Build a kallisto index from the transcriptome FASTA",
            "Quantify expression using kallisto quant",
        ]
        steps = autoba_shim._map_plan(tasks, shells={})
        assert len(steps) == 3
        for s in steps:
            assert s["dependencies"] == [], (
                f"AutoBA shim must not synthesise deps; got {s['dependencies']!r}"
            )

    def test_map_plan_preserves_command_resolution(self):
        """Stripping deps must not break the shell-body / task-sentence
        resolution -- ``command`` should still come from the per-task
        ``<N>.sh`` if available, otherwise from ``_command_from_task``."""
        steps = autoba_shim._map_plan(
            ["Run STAR alignment"],
            shells={1: "STAR --runMode alignReads --runThreadN 8\n"},
        )
        assert steps[0]["dependencies"] == []
        assert "STAR" in steps[0]["command"]
        assert steps[0]["name"] == "step_1"


# ── Cross-shim invariant ───────────────────────────────────────────

class TestUniversalEmptyDepsInvariant:
    """One condensed sweep that catches a future regression in any of the
    three shims at once. If a maintainer reaches for ``[step_N-1]`` again,
    this test goes red regardless of which shim they touched."""

    def test_no_competitor_shim_synthesises_dependencies(self):
        # Biomni: 2 tool calls.
        biomni_steps = biomni_shim._messages_to_steps([
            SimpleNamespace(tool_calls=[{"name": "fastqc", "args": {}}]),
            SimpleNamespace(tool_calls=[{"name": "kallisto", "args": {}}]),
        ])
        # BioMaster: 2 PLAN entries.
        biomaster_steps = [
            biomaster_shim._map_step({"step_number": i, "tools": "tool"}, i - 1)
            for i in (1, 2)
        ]
        # AutoBA: 2 tasks, no shells.
        autoba_steps = autoba_shim._map_plan(["task A", "task B"], shells={})

        all_steps = biomni_steps + biomaster_steps + autoba_steps
        assert len(all_steps) == 6  # sanity: parsers actually ran

        for shim_name, steps in [
            ("biomni",    biomni_steps),
            ("biomaster", biomaster_steps),
            ("autoba",    autoba_steps),
        ]:
            for i, s in enumerate(steps):
                assert s["dependencies"] == [], (
                    f"{shim_name} shim emitted non-empty deps "
                    f"{s['dependencies']!r} on step {i}; this breaks the "
                    f"universal no-DAG-synthesis fairness invariant for "
                    f"Benchmark E."
                )
