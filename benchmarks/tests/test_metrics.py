"""Unit tests for :mod:`harness.metrics` and mock plan synthesis.

Keeps Benchmark A scoring rules honest (overall_pass, tools, forbidden, DAG).
"""

from __future__ import annotations

import sys
import unittest.mock
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE.parent
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(BENCH_DIR.parent))

from harness.metrics import (  # noqa: E402
    _classify_unknown_token,
    _load_known_tools,
    completeness_metrics,
    dag_shape,
    hallucinated_tool_names,
    hallucinated_tools,
    plan_schema_valid,
    score_plan,
    score_plan_inference,
    tool_covered,
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

    def test_macs3_satisfies_macs2_rubric(self):
        """MACS3 is a drop-in successor — covers ``expected: macs2``.

        Regression: ``chipseq_macs2`` used to fail when frontier models
        emitted ``macs3 callpeak`` even though the call is identical to
        ``macs2 callpeak`` for narrow-peak ChIP-seq. See
        ``_TOOL_FAMILY_ALIASES`` in :mod:`harness.metrics`.
        """
        plan = {
            "workflow_type": "chip_seq",
            "steps": [
                {"name": "align", "command": "bowtie2 -x idx -U r.fq -S a.sam",
                 "dependencies": [], "outputs": [], "description": ""},
                {"name": "peaks", "command": "macs3 callpeak -t a.bam -f BAM -g hs -n s",
                 "dependencies": ["align"], "outputs": [], "description": ""},
                {"name": "qc", "command": "multiqc results",
                 "dependencies": ["peaks"], "outputs": [], "description": ""},
                {"name": "extra", "command": "samtools sort a.bam -o s.bam",
                 "dependencies": ["align"], "outputs": [], "description": ""},
            ],
        }
        expected = {
            "expected_workflow_type": "chip_seq",
            "expected_tools": ["bowtie2", "macs2", "multiqc"],
            "expected_min_steps": 4,
            "forbidden_tools": ["wget"],
        }
        m = score_plan(plan, expected)
        assert m["tools_present_fraction"] == 1.0
        assert m["overall_pass"] is True

    def test_macs2_still_satisfies_macs2_rubric(self):
        """Sanity: the alias map doesn't break the exact-match path."""
        assert tool_covered("macs2", {"macs2"}) is True
        assert tool_covered("macs3", {"macs3"}) is True

    def test_aliases_are_bidirectional(self):
        assert tool_covered("macs2", {"macs3"}) is True
        assert tool_covered("macs3", {"macs2"}) is True

    def test_alias_does_not_credit_unrelated_tool(self):
        """``macs2`` must NOT match ``bowtie2`` or other non-family tools."""
        assert tool_covered("macs2", {"bowtie2"}) is False
        assert tool_covered("macs2", {"samtools"}) is False
        # Bowtie / Bowtie2 are deliberately NOT in the alias map.
        assert tool_covered("bowtie2", {"bowtie"}) is False

    def test_macs3_in_prose_satisfies_macs2_rubric(self):
        """Narrative-style plan (Biomni-shape) must also benefit from alias."""
        plan = {
            "workflow_type": "chip_seq",
            "steps": [
                {"name": "narrative",
                 "command": "Run macs3 callpeak on the sorted BAM",
                 "dependencies": [], "outputs": [], "description": ""},
            ],
        }
        # No CLI lead token (the leading word is "Run") -> exercises the
        # prose-fallback branch in tool_covered.
        assert tool_covered("macs2", set(), plan=plan) is True


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


# ── Hallucination-detector tests (v2 classifier) ────────────────────────────

class TestClassifyUnknownToken:
    """Unit tests for _classify_unknown_token()."""

    # A small but sufficient known-tool set used to avoid loading the full
    # snapshot on every call.
    _KNOWN: frozenset = frozenset({"kallisto", "fastqc", "samtools", "bowtie2"})

    def test_typo_detected_with_correction(self):
        """'kalsito' is close enough to 'kallisto' to be flagged as a typo."""
        cat, correction = _classify_unknown_token("kalsito", set(self._KNOWN))
        assert cat == "typo"
        assert correction == "kallisto"

    def test_filename_classified_as_filename(self):
        """Tokens with biodata extensions are classified as filenames."""
        cat, correction = _classify_unknown_token("reads.fastq.gz", set(self._KNOWN))
        assert cat == "filename"
        assert correction is None

    def test_runtime_glue_not_hallucination(self):
        """Common infrastructure commands should never be labelled hallucinations."""
        cat, correction = _classify_unknown_token("parallel", set(self._KNOWN))
        assert cat == "runtime_glue"
        assert correction is None

    def test_unknown_implausible_stays_unknown(self):
        """An invented name with no near match is classified as unknown."""
        cat, correction = _classify_unknown_token(
            "super_aligner_pro", set(self._KNOWN)
        )
        assert cat == "unknown"
        assert correction is None

    def test_short_token_no_typo_correction(self):
        """Tokens shorter than 5 characters are not eligible for fuzzy matching."""
        # 'kb' is a real tool but also very short; it must not match 'kallisto'
        # (Levenshtein distance 6) or any other entry via the typo path.
        cat, _corr = _classify_unknown_token("kb", set(self._KNOWN))
        # Short token: either runtime_glue or unknown, but never typo.
        assert cat != "typo"


class TestHallucinatedTools:
    """Tests for the rewritten hallucinated_tools() and back-compat shim."""

    def _make_known(self) -> set:
        return {"kallisto", "fastqc", "samtools", "bowtie2", "multiqc"}

    def test_strict_hallucinations_gates_pass(self):
        """strict_hallucinations=True → overall_pass=False if any hallucination."""
        plan = {
            "workflow_type": "rna_seq_kallisto",
            "steps": [
                {"name": "qc",   "command": "fastqc reads.fq.gz",
                 "dependencies": [], "outputs": [], "description": ""},
                {"name": "quant","command": "kalsito quant -i idx -o out reads.fq.gz",
                 "dependencies": ["qc"], "outputs": [], "description": ""},
                {"name": "rep",  "command": "multiqc out",
                 "dependencies": ["quant"], "outputs": [], "description": ""},
            ],
        }
        expected = {
            "expected_workflow_type": "rna_seq_kallisto",
            "expected_tools": ["fastqc", "multiqc"],
            "expected_min_steps": 3,
            "forbidden_tools": [],
        }
        m = score_plan(plan, expected, strict_hallucinations=True)
        # 'kalsito' is not in the known-tools snapshot → hallucination detected.
        assert m["num_hallucinated_tools"] >= 1
        assert m["overall_pass"] is False

    def test_known_tools_yaml_fallback(self):
        """If known_tools.yaml is absent, _load_known_tools() falls back
        to _BIOINFO_TOOLS without raising."""
        from flowagent import tool_catalog as tc
        # Point every snapshot candidate at a non-existent path so the
        # loader has to fall back to the bundled _BIOINFO_TOOLS set.
        missing = Path("/tmp/__nonexistent_known_tools__.yaml")
        tc._load_known_tools.cache_clear()
        with unittest.mock.patch.dict(
            "os.environ", {"FLOWAGENT_KNOWN_TOOLS_PATH": str(missing)}
        ), unittest.mock.patch.object(tc, "_SNAPSHOT_PATH", missing), \
             unittest.mock.patch.object(tc, "_PKG_ROOT", missing.parent):
            known = tc._load_known_tools()
        # Fallback must contain core bioinfo tools.
        assert "kallisto" in known
        assert "fastqc" in known
        # Restore cache for subsequent tests.
        tc._load_known_tools.cache_clear()

    def test_score_plan_emits_new_columns(self):
        """score_plan must include num_hallucinated_typos and hallucinated_typos."""
        plan = {
            "workflow_type": "rna_seq_kallisto",
            "steps": [
                {"name": "qc",   "command": "fastqc reads.fq.gz",
                 "dependencies": [], "outputs": [], "description": ""},
                {"name": "quant","command": "kallisto quant -i idx -o out reads.fq.gz",
                 "dependencies": ["qc"], "outputs": [], "description": ""},
                {"name": "rep",  "command": "multiqc out",
                 "dependencies": ["quant"], "outputs": [], "description": ""},
            ],
        }
        expected = {
            "expected_workflow_type": "rna_seq_kallisto",
            "expected_tools": ["fastqc", "kallisto", "multiqc"],
            "expected_min_steps": 3,
            "forbidden_tools": [],
        }
        m = score_plan(plan, expected)
        assert "num_hallucinated_typos" in m
        assert "hallucinated_typos" in m
        assert isinstance(m["num_hallucinated_typos"], int)
        # A clean plan has no typos.
        assert m["num_hallucinated_typos"] == 0
        assert m["hallucinated_typos"] == ""

    def test_score_plan_inference_emits_new_columns(self):
        """score_plan_inference must also include the new hallucination columns."""
        plan = {
            "workflow_type": "chip_seq",
            "steps": [
                {"name": "align", "command": "bowtie2 -x idx -U reads.fq -S out.sam",
                 "dependencies": [], "outputs": [], "description": ""},
                {"name": "sort",  "command": "samtools sort out.sam -o out.bam",
                 "dependencies": ["align"], "outputs": [], "description": ""},
                {"name": "peaks", "command": "macs2 callpeak -t out.bam -n sample",
                 "dependencies": ["sort"], "outputs": [], "description": ""},
            ],
        }
        expected = {
            "acceptable_tool_sets": [["bowtie2", "samtools", "macs2"]],
            "expected_min_steps": 3,
            "forbidden_tools": [],
        }
        m = score_plan_inference(plan, expected)
        assert "num_hallucinated_typos" in m
        assert "hallucinated_typos" in m


class TestHallucinationParserFixes:
    """Quote-aware parsing, CLI aliases, and r_code classification."""

    def test_rscript_inline_r_not_split_on_semicolons(self):
        from harness.metrics import extract_tools_from_plan

        plan = {
            "workflow_type": "rna_seq",
            "steps": [{
                "name": "de",
                "command": (
                    "Rscript -e 'library(DESeq2); dds <- DESeqDataSetFromMatrix("
                    "countData=cts, colData=coldata, design=~condition); "
                    "write.csv(as.data.frame(results(dds)), \"de.csv\")'"
                ),
                "dependencies": [], "outputs": [], "description": "",
            }],
        }
        tools = extract_tools_from_plan(plan)
        assert tools == set()
        assert "dds" not in tools
        assert "write.csv(as.data.frame(results(dds))" not in tools

    def test_featurecounts_recognised_via_cli_alias(self):
        from harness.metrics import extract_tools_from_plan, score_plan

        plan = {
            "workflow_type": "rna_seq",
            "steps": [
                {"name": "count", "command": "featureCounts -a genes.gtf -o c.txt a.bam",
                 "dependencies": [], "outputs": [], "description": ""},
            ],
        }
        assert "featurecounts" in extract_tools_from_plan(plan)
        m = score_plan(plan, {
            "expected_workflow_type": "rna_seq",
            "expected_tools": ["featurecounts"],
            "expected_min_steps": 1,
            "forbidden_tools": [],
        })
        assert m["num_hallucinated_tools"] == 0

    def test_deeptools_subcommand_not_unknown(self):
        from flowagent.tool_catalog import hallucinated_tools

        flagged = hallucinated_tools({"hicfindtads", "hicplotmatrix"})
        assert flagged == []

    def test_bare_r_variables_classified_as_r_code(self):
        from flowagent.tool_catalog import _classify_unknown_token

        cat, corr = _classify_unknown_token("dds", set())
        assert cat == "r_code"
        assert corr is None

    def test_true_unknown_still_unknown(self):
        from flowagent.tool_catalog import _classify_unknown_token

        cat, corr = _classify_unknown_token("super_aligner_pro", set())
        assert cat == "unknown"
        assert corr is None

    def test_hallucination_rate_ignores_r_code(self):
        from harness.metrics import _hallucination_metrics

        stats = _hallucination_metrics({"dds", "kalsito"})
        assert stats["num_hallucinated_tools"] == 1
        assert stats["num_r_code_tokens"] == 1
        assert "kalsito:typo" in stats["hallucinated_tools"] or \
               "kalsito:unknown" in stats["hallucinated_tools"]

    def test_piped_commands_still_split(self):
        from harness.metrics import extract_tools_from_plan

        plan = {
            "workflow_type": "chip_seq",
            "steps": [{
                "name": "sort",
                "command": "bowtie2 -x idx -U reads.fq | samtools sort -o out.bam",
                "dependencies": [], "outputs": [], "description": "",
            }],
        }
        tools = extract_tools_from_plan(plan)
        assert tools == {"bowtie2", "samtools"}

    def test_nan_hallucinated_tools_not_parsed_as_unknown(self):
        import math
        import pandas as pd
        from harness.plot import _parse_hallucinated_tools_column

        cat = _parse_hallucinated_tools_column(pd.Series([math.nan, "", "nan"]))
        assert cat["unknown"].sum() == 0
        assert cat["r_code"].sum() == 0

    def test_py_script_classified_as_filename_not_unknown(self):
        from flowagent.tool_catalog import _classify_unknown_token

        cat, corr = _classify_unknown_token("collapse_reads.py", set())
        assert cat == "filename"
        assert corr is None
