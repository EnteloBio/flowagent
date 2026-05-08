"""Unit tests for ``flowagent.core.completeness``.

Covers:
  - Heuristic kind inference from name + command (``infer_step_kind``).
  - ``fill_missing_kinds`` populates absent / invalid kinds in-place.
  - ``validate_workflow_completeness`` enforces the four structural rules:
      * align needs an index/download ancestor
      * download must be consumed downstream
      * quantify/call/de must feed a report or be a sink
      * graph is weakly connected, has at least one terminal sink
  - ``render_completeness_feedback`` produces a non-empty reflection
    prompt for the LLM.
"""

from __future__ import annotations

import pytest

from flowagent.core.completeness import (
    fill_missing_kinds,
    infer_step_kind,
    normalize_plan_steps,
    render_completeness_feedback,
    validate_workflow_completeness,
)
from flowagent.core.schemas import StepKind


class TestInferStepKind:
    """Heuristic kind inference from ``name`` / ``command``."""

    @pytest.mark.parametrize(
        "command,expected",
        [
            ("fastqc reads.fq.gz -o qc/", StepKind.QC),
            ("multiqc -f -n multiqc_report .", StepKind.REPORT),
            ("kallisto index -i ref.idx ref.fa", StepKind.INDEX),
            ("kallisto quant -i ref.idx -o out reads.fq.gz", StepKind.ALIGN),
            ("salmon quant -i idx -l A -r r.fq -o out", StepKind.ALIGN),
            ("bwa mem ref.fa r1.fq r2.fq", StepKind.ALIGN),
            ("hisat2 -x idx -U r.fq -S out.sam", StepKind.ALIGN),
            ("hisat2-build ref.fa idx", StepKind.INDEX),
            ("samtools sort -o sorted.bam in.bam", StepKind.SORT),
            ("samtools flagstat in.bam", StepKind.QC),
            ("picard MarkDuplicates I=in.bam O=dedup.bam M=m.txt", StepKind.DEDUP),
            ("gatk HaplotypeCaller -R ref.fa -I bam -O vcf", StepKind.CALL),
            ("macs2 callpeak -t bam -c ctrl -n peaks", StepKind.CALL),
            ("featureCounts -a gtf -o counts.txt bam", StepKind.QUANTIFY),
            ("Rscript de_analysis.R counts.txt", StepKind.DE),
            ("trim_galore --paired r1.fq r2.fq", StepKind.TRIM),
            ("fastp -i r1.fq -I r2.fq -o o1.fq -O o2.fq", StepKind.TRIM),
            ("curl -fSL -o ref.fa.gz https://example/ref.fa.gz", StepKind.DOWNLOAD),
            ("wget https://example/ref.fa.gz -O ref.fa.gz", StepKind.DOWNLOAD),
            ("prefetch SRR123456", StepKind.DOWNLOAD),
            ("mkdir -p results/qc results/align", StepKind.OTHER),
            ("echo done", StepKind.OTHER),
            ("", StepKind.OTHER),
        ],
    )
    def test_inference_by_command(self, command: str, expected: StepKind) -> None:
        assert infer_step_kind({"name": "step", "command": command}) is expected

    def test_inference_falls_back_to_other_for_unknown(self) -> None:
        kind = infer_step_kind({"name": "x", "command": "some_unknown_tool --foo bar"})
        assert kind is StepKind.OTHER


class TestFillMissingKinds:
    """``fill_missing_kinds`` mutates a plan in-place."""

    def test_fills_missing_field(self) -> None:
        plan = {
            "workflow_type": "x",
            "steps": [
                {"name": "qc", "command": "fastqc r.fq"},
                {"name": "ix", "command": "bwa index ref.fa"},
                {"name": "al", "command": "bwa mem ref.fa r.fq"},
            ],
        }
        fill_missing_kinds(plan)
        kinds = [s["kind"] for s in plan["steps"]]
        assert kinds == [
            StepKind.QC.value,
            StepKind.INDEX.value,
            StepKind.ALIGN.value,
        ]

    def test_preserves_valid_kind(self) -> None:
        plan = {
            "workflow_type": "x",
            "steps": [{"name": "x", "command": "fastqc r.fq", "kind": "report"}],
        }
        fill_missing_kinds(plan)
        # Valid pre-set value is preserved (even if it disagrees with the
        # heuristic — the LLM-emitted value wins).
        assert plan["steps"][0]["kind"] == "report"

    def test_replaces_invalid_kind(self) -> None:
        plan = {
            "workflow_type": "x",
            "steps": [{"name": "x", "command": "fastqc r.fq", "kind": "bogus"}],
        }
        fill_missing_kinds(plan)
        assert plan["steps"][0]["kind"] == StepKind.QC.value

    def test_coerces_enum_to_string(self) -> None:
        plan = {
            "workflow_type": "x",
            "steps": [
                {"name": "x", "command": "fastqc r.fq", "kind": StepKind.QC},
            ],
        }
        fill_missing_kinds(plan)
        assert plan["steps"][0]["kind"] == "qc"
        assert isinstance(plan["steps"][0]["kind"], str)


class TestValidateWorkflowCompleteness:
    """Each domain rule is exercised by a positive and a negative case."""

    def test_well_formed_rnaseq_passes(self) -> None:
        plan = {
            "workflow_type": "rna_seq_kallisto",
            "steps": [
                {"name": "mk", "command": "mkdir -p results/", "dependencies": []},
                {"name": "qc", "command": "fastqc r.fq.gz", "dependencies": ["mk"]},
                {"name": "ix", "command": "kallisto index -i x.idx x.fa",
                 "dependencies": ["mk"]},
                {"name": "qt", "command": "kallisto quant -i x.idx -o o r.fq.gz",
                 "dependencies": ["ix"]},
                {"name": "rep", "command": "multiqc -f -n multiqc_report .",
                 "dependencies": ["qc", "qt"]},
            ],
        }
        ok, fails = validate_workflow_completeness(plan)
        assert ok, fails
        assert fails == []

    def test_align_without_index_ancestor_fails(self) -> None:
        plan = {
            "workflow_type": "rna_seq",
            "steps": [
                {"name": "mk", "command": "mkdir -p results/", "dependencies": []},
                {"name": "al", "command": "star --runMode alignReads --readFilesIn r.fq",
                 "dependencies": ["mk"]},
                {"name": "rep", "command": "multiqc .", "dependencies": ["al"]},
            ],
        }
        ok, fails = validate_workflow_completeness(plan)
        assert not ok
        assert any("index" in f.lower() and "al" in f for f in fails)

    def test_align_with_download_ancestor_passes(self) -> None:
        # DAG-Plan analogue: a pre-built index downloaded from a server
        # satisfies the "index ancestor" requirement.
        plan = {
            "workflow_type": "rna_seq",
            "steps": [
                {"name": "mk", "command": "mkdir -p ref/", "dependencies": []},
                {"name": "fetch", "command": "curl -fSL -o idx.tar.gz http://x/idx.tar.gz",
                 "dependencies": ["mk"]},
                {"name": "al", "command": "kallisto quant -i idx -o o r.fq",
                 "dependencies": ["fetch"]},
                {"name": "rep", "command": "multiqc .", "dependencies": ["al"]},
            ],
        }
        ok, fails = validate_workflow_completeness(plan)
        assert ok, fails

    def test_dangling_download_fails(self) -> None:
        plan = {
            "workflow_type": "rna_seq",
            "steps": [
                {"name": "mk", "command": "mkdir -p ref/", "dependencies": []},
                {"name": "fetch", "command": "curl -fSL -o ref.fa.gz http://x/ref.fa.gz",
                 "dependencies": ["mk"]},
                {"name": "qc", "command": "fastqc r.fq", "dependencies": ["mk"]},
                {"name": "rep", "command": "multiqc .", "dependencies": ["qc"]},
            ],
        }
        ok, fails = validate_workflow_completeness(plan)
        assert not ok
        assert any("download" in f.lower() and "fetch" in f for f in fails)

    def test_quantify_without_report_fails(self) -> None:
        # featureCounts is "quantify" in the taxonomy; if it feeds a
        # downstream non-report step instead of a report or being a
        # sink, the rule fires.
        plan = {
            "workflow_type": "rna_seq",
            "steps": [
                {"name": "mk", "command": "mkdir -p out/", "dependencies": []},
                {"name": "ix", "command": "bwa index ref.fa", "dependencies": ["mk"]},
                {"name": "al", "command": "bwa mem ref.fa r.fq",
                 "dependencies": ["ix"]},
                {"name": "qt", "command": "featureCounts -a gtf -o counts.txt al.bam",
                 "dependencies": ["al"]},
                # qt feeds a glue step that is not a report.
                {"name": "post", "command": "mkdir -p analysis/",
                 "dependencies": ["qt"]},
            ],
        }
        ok, fails = validate_workflow_completeness(plan)
        assert not ok
        assert any(
            "report" in f.lower() and "qt" in f for f in fails
        ), f"expected 'qt needs report descendant' failure, got: {fails}"

    def test_de_step_as_sink_passes(self) -> None:
        # A DE analysis that IS the final artefact (a sink) is allowed
        # without an explicit report.
        plan = {
            "workflow_type": "rna_seq",
            "steps": [
                {"name": "mk", "command": "mkdir -p out/", "dependencies": []},
                {"name": "ix", "command": "bwa index ref.fa", "dependencies": ["mk"]},
                {"name": "al", "command": "bwa mem ref.fa r.fq",
                 "dependencies": ["ix"]},
                {"name": "qt", "command": "featureCounts -a gtf -o counts.txt al.bam",
                 "dependencies": ["al"]},
                {"name": "de", "command": "Rscript de_analysis.R counts.txt",
                 "dependencies": ["qt"]},
            ],
        }
        ok, fails = validate_workflow_completeness(plan)
        assert ok, fails

    def test_quantify_as_sink_passes(self) -> None:
        # If the analysis step IS the final artefact (no descendants),
        # the rule allows it without an explicit report.
        plan = {
            "workflow_type": "wgs",
            "steps": [
                {"name": "mk", "command": "mkdir -p out/", "dependencies": []},
                {"name": "ix", "command": "bwa index ref.fa", "dependencies": ["mk"]},
                {"name": "al", "command": "bwa mem ref.fa r.fq", "dependencies": ["ix"]},
                {"name": "call", "command": "gatk HaplotypeCaller -R ref.fa -I al.bam -O out.vcf",
                 "dependencies": ["al"]},
            ],
        }
        ok, fails = validate_workflow_completeness(plan)
        assert ok, fails

    def test_disconnected_components_fail(self) -> None:
        plan = {
            "workflow_type": "mixed",
            "steps": [
                {"name": "a1", "command": "fastqc r.fq", "dependencies": []},
                {"name": "a2", "command": "multiqc .", "dependencies": ["a1"]},
                {"name": "b1", "command": "kallisto index -i x x.fa",
                 "dependencies": []},
                {"name": "b2", "command": "kallisto quant -i x -o o r.fq",
                 "dependencies": ["b1"]},
            ],
        }
        ok, fails = validate_workflow_completeness(plan)
        assert not ok
        assert any("disconnected" in f.lower() for f in fails)

    def test_cycle_fails(self) -> None:
        plan = {
            "workflow_type": "x",
            "steps": [
                {"name": "a", "command": "fastqc r.fq", "dependencies": ["b"]},
                {"name": "b", "command": "multiqc .", "dependencies": ["a"]},
            ],
        }
        ok, fails = validate_workflow_completeness(plan)
        assert not ok
        assert any("cycle" in f.lower() for f in fails)

    def test_empty_plan_fails(self) -> None:
        ok, fails = validate_workflow_completeness({"workflow_type": "x", "steps": []})
        assert not ok
        assert "zero steps" in fails[0]


class TestRenderCompletenessFeedback:
    """The reflection prompt sent back to the LLM."""

    def test_empty_failures_returns_empty_string(self) -> None:
        assert render_completeness_feedback([]) == ""

    def test_failures_render_with_numbered_list(self) -> None:
        out = render_completeness_feedback([
            "alignment step bad has no index ancestor",
            "download dl1 has no consumer",
        ])
        assert "1." in out and "2." in out
        assert "alignment step bad" in out
        assert "download dl1" in out
        # Must contain explicit guidance to retry / regenerate.
        assert "regenerate" in out.lower()


class TestNormalizePlanSteps:
    """Defensive guard for malformed plans where ``steps`` mixes
    strings and dicts.

    The bug this guards against: Opus / Gemini occasionally emit plans
    via the regex-repair fallback path where the first step is a prose
    sentence instead of a step dict. Without normalisation, every
    downstream ``step.get(...)`` call raises ``AttributeError`` and
    discards the entire plan -- even when 19 of 20 steps were valid.
    """

    def test_all_dicts_unchanged(self) -> None:
        plan = {"steps": [
            {"name": "a", "command": "echo a"},
            {"name": "b", "command": "echo b"},
        ]}
        out = normalize_plan_steps(plan)
        assert out is plan  # mutates in-place
        assert len(out["steps"]) == 2

    def test_drops_string_steps(self) -> None:
        plan = {"steps": [
            "Download FASTQ via fasterq-dump",
            {"name": "trim", "command": "fastp r1.fq -o t.fq"},
            {"name": "align", "command": "bwa mem ref t.fq"},
        ]}
        out = normalize_plan_steps(plan)
        assert len(out["steps"]) == 2
        assert {s["name"] for s in out["steps"]} == {"trim", "align"}

    def test_handles_missing_steps_key(self) -> None:
        plan = {"workflow_type": "x"}
        out = normalize_plan_steps(plan)
        assert out["steps"] == []

    def test_handles_non_list_steps(self) -> None:
        plan = {"steps": "not a list"}
        out = normalize_plan_steps(plan)
        assert out["steps"] == []

    def test_drops_other_non_dict_types(self) -> None:
        plan = {"steps": [
            42,
            None,
            ["nested", "list"],
            {"name": "valid", "command": "echo ok"},
        ]}
        out = normalize_plan_steps(plan)
        assert len(out["steps"]) == 1
        assert out["steps"][0]["name"] == "valid"

    def test_fill_missing_kinds_survives_string_steps(self) -> None:
        """``fill_missing_kinds`` calls ``normalize_plan_steps`` defensively
        so a malformed plan no longer raises AttributeError."""
        plan = {"steps": [
            "narrative description",
            {"name": "trim", "command": "fastp r.fq -o t.fq"},
        ]}
        # Before the fix this raised: AttributeError: 'str' has no .get
        out = fill_missing_kinds(plan)
        assert len(out["steps"]) == 1
        assert out["steps"][0]["kind"] == StepKind.TRIM.value

    def test_validate_completeness_survives_string_steps(self) -> None:
        """The completeness validator also normalises defensively, so a
        partially malformed plan can still be scored."""
        plan = {
            "workflow_type": "rna_seq",
            "steps": [
                "Download reads",
                {"name": "idx", "command": "bwa index ref.fa", "kind": "index", "dependencies": []},
                {"name": "aln", "command": "bwa mem ref.fa r.fq", "kind": "align", "dependencies": ["idx"]},
            ],
        }
        ok, _ = validate_workflow_completeness(plan)
        # The string is dropped; the bwa index→align chain is valid.
        assert ok
