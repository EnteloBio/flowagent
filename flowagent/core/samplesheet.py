"""Samplesheet parsing for FlowAgent.

Accepts nf-core-style CSV samplesheets:

    sample,fastq_1,fastq_2,condition
    SRR123,data/SRR123_R1.fastq.gz,data/SRR123_R2.fastq.gz,treated
    SRR456,data/SRR456_R1.fastq.gz,data/SRR456_R2.fastq.gz,untreated

Also handles single-end (no fastq_2 column or empty values) and
tab-delimited files.

The ``expand_preset_for_samples`` function rewrites preset step commands
that contain shell globs (``data/*.fastq.gz``) into per-sample loops so
each sample is processed individually with correct output naming.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


from dataclasses import dataclass, field


@dataclass
class Sample:
    """One row in a samplesheet."""
    name: str
    fastq_1: str = ""
    fastq_2: str = ""
    condition: str = ""
    extra: Dict[str, str] = field(default_factory=dict)

    @property
    def paired_end(self) -> bool:
        return bool(self.fastq_2)


@dataclass
class Samplesheet:
    """Parsed samplesheet."""
    samples: List[Sample] = field(default_factory=list)
    source: str = ""

    @property
    def conditions(self) -> List[str]:
        seen: List[str] = []
        for s in self.samples:
            if s.condition and s.condition not in seen:
                seen.append(s.condition)
        return seen

    @property
    def paired_end(self) -> bool:
        return any(s.paired_end for s in self.samples)

    def summary(self) -> str:
        """One-line summary for inclusion in planner prompts."""
        n = len(self.samples)
        pe = "paired-end" if self.paired_end else "single-end"
        conds = self.conditions
        cond_str = f", conditions: {', '.join(conds)}" if conds else ""
        return f"{n} samples ({pe}){cond_str}"

    def to_planner_text(self) -> str:
        """Multi-line description of samples, suitable for injection into LLM prompts."""
        lines = [f"Samplesheet: {self.summary()}"]
        for s in self.samples:
            fq = s.fastq_1
            if s.fastq_2:
                fq = f"{s.fastq_1}, {s.fastq_2}"
            cond = f" [{s.condition}]" if s.condition else ""
            lines.append(f"  - {s.name}{cond}: {fq}")
        return "\n".join(lines)


# ── Parsing ───────────────────────────────────────────────────

def load_samplesheet(path: str | Path) -> Samplesheet:
    """Parse a samplesheet CSV (or TSV) and return a :class:`Samplesheet`.

    Accepts any of:
    - nf-core style: ``sample,fastq_1,fastq_2,condition``
    - Minimal:       ``sample,fastq_1``
    - Extra columns beyond these four are stored in ``Sample.extra``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Samplesheet not found: {path}")

    text = path.read_text(errors="replace")
    # Detect delimiter
    delimiter = "\t" if "\t" in text.split("\n")[0] else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)

    # Normalise header keys (lowercase, strip)
    raw_rows = list(reader)
    if not raw_rows:
        return Samplesheet(source=str(path))

    samples: List[Sample] = []
    for row in raw_rows:
        # Normalise keys
        norm: Dict[str, str] = {k.strip().lower(): (v or "").strip() for k, v in row.items()}
        name = norm.pop("sample", "") or norm.pop("sample_name", "") or norm.pop("id", "")
        fq1 = norm.pop("fastq_1", "") or norm.pop("fastq1", "") or norm.pop("read1", "")
        fq2 = norm.pop("fastq_2", "") or norm.pop("fastq2", "") or norm.pop("read2", "")
        condition = norm.pop("condition", "") or norm.pop("group", "") or norm.pop("treatment", "")
        if not name:
            continue
        samples.append(Sample(
            name=name,
            fastq_1=fq1,
            fastq_2=fq2,
            condition=condition,
            extra=norm,
        ))

    return Samplesheet(samples=samples, source=str(path))


# ── Preset expansion ──────────────────────────────────────────

# Patterns that indicate a command uses a glob over data/
_GLOB_PATTERNS = [
    r"data/\*",
    r"data/\$sample",
    r'"\$sample"',
    r"for sample in data",
    r"for f in data",
    r"for f in results",
]


def expand_preset_for_samples(
    preset: Dict[str, Any],
    sheet: Samplesheet,
) -> Dict[str, Any]:
    """Rewrite a preset so each step runs per-sample rather than over globs.

    Steps that already contain an explicit ``for`` loop are left unchanged.
    Steps that reference ``data/*.fastq.gz`` (or similar globs) are
    rewritten to loop over the specific samples in the samplesheet.

    For single-tool steps (fastqc, kallisto quant, etc.) the command is
    wrapped in a ``for``-loop that iterates over the samplesheet samples.

    The ``kallisto_quant`` step gets a tailored paired/single-end expansion.
    """
    import copy
    plan = copy.deepcopy(preset)

    if not sheet.samples:
        return plan

    for step in plan.get("steps", []):
        cmd = step.get("command", "")
        # Rewrite even existing loops when they use glob patterns
        new_cmd = _rewrite_command(step["name"], cmd, sheet)
        if new_cmd != cmd:
            step["command"] = new_cmd

    # Annotate plan with samplesheet info
    plan.setdefault("_meta", {})
    plan["_meta"]["samplesheet"] = sheet.summary()
    plan["_meta"]["samples"] = [s.name for s in sheet.samples]
    plan["_meta"]["conditions"] = sheet.conditions

    return plan


def _is_already_loop(cmd: str) -> bool:
    return "for " in cmd and " do" in cmd


def _rewrite_command(step_name: str, cmd: str, sheet: Samplesheet) -> str:
    """Attempt to convert a glob-based command into a per-sample loop."""
    step_lower = step_name.lower()
    pe = sheet.paired_end

    # fastqc: fastqc -t 4 data/*.fastq.gz -o results/fastqc
    if step_lower == "fastqc":
        files = " ".join(
            f"{s.fastq_1} {s.fastq_2}".strip() if pe else s.fastq_1
            for s in sheet.samples
        )
        # Replace glob with explicit list if it fits, else keep as-is
        new_cmd = re.sub(r"data/\*\.fastq\.gz", files, cmd)
        return new_cmd if new_cmd != cmd else cmd

    # kallisto quant: per-sample loop
    if "kallisto" in step_lower and "quant" in step_lower:
        if pe:
            loop_lines = [
                f"kallisto quant -i results/kallisto_index/transcripts.idx "
                f"-o results/kallisto_quant/{s.name} -t 4 {s.fastq_1} {s.fastq_2}"
                for s in sheet.samples
            ]
        else:
            loop_lines = [
                f"kallisto quant -i results/kallisto_index/transcripts.idx "
                f"-o results/kallisto_quant/{s.name} --single -l 200 -s 20 {s.fastq_1}"
                for s in sheet.samples
            ]
        return " && ".join(loop_lines)

    # STAR align: per-sample loop
    if "star_align" in step_lower or ("star" in step_lower and "align" in step_lower):
        if pe:
            loop_lines = [
                f"STAR --genomeDir results/star_index "
                f"--readFilesIn {s.fastq_1} {s.fastq_2} "
                f"--readFilesCommand zcat --outSAMtype BAM SortedByCoordinate "
                f"--runThreadN 8 --outFileNamePrefix results/star_align/{s.name}_"
                for s in sheet.samples
            ]
        else:
            loop_lines = [
                f"STAR --genomeDir results/star_index "
                f"--readFilesIn {s.fastq_1} "
                f"--readFilesCommand zcat --outSAMtype BAM SortedByCoordinate "
                f"--runThreadN 8 --outFileNamePrefix results/star_align/{s.name}_"
                for s in sheet.samples
            ]
        return " && ".join(loop_lines)

    # bowtie2 align: per-sample loop
    if "bowtie2_align" in step_lower or ("bowtie2" in step_lower and "align" in step_lower):
        if pe:
            loop_lines = [
                f"bowtie2 -x reference/genome -1 {s.fastq_1} -2 {s.fastq_2} "
                f"--very-sensitive -X 2000 -p 8 | "
                f"samtools sort -@ 4 -o results/aligned/{s.name}.bam && "
                f"samtools index results/aligned/{s.name}.bam"
                for s in sheet.samples
            ]
        else:
            loop_lines = [
                f"bowtie2 -x reference/genome -U {s.fastq_1} "
                f"-S results/aligned/{s.name}.sam -p 8 && "
                f"samtools sort -@ 4 results/aligned/{s.name}.sam "
                f"-o results/aligned/{s.name}.bam && "
                f"samtools index results/aligned/{s.name}.bam"
                for s in sheet.samples
            ]
        return " && ".join(loop_lines)

    # trim_galore: per-sample
    if "trim_galore" in step_lower or "trim" in step_lower:
        if pe:
            loop_lines = [
                f"trim_galore --cores 4 --paired {s.fastq_1} {s.fastq_2} -o results/trimmed"
                for s in sheet.samples
            ]
        else:
            loop_lines = [
                f"trim_galore --cores 4 -o results/trimmed {s.fastq_1}"
                for s in sheet.samples
            ]
        return " && ".join(loop_lines)

    return cmd
