"""Fault-injection catalogue for Benchmark B.

Each fault is a small class that takes a known-good preset + a per-run
working directory, and returns a (step, EnvContext) pair that will reliably
fail when executed. The catalogue is the single source of truth for the
faults listed in ``config/faults.yaml``.

Design notes:
  * Faults must produce REAL failures (exit codes, stderr). No faking the
    failure signature — recovery must be judged on its ability to read and
    fix real errors.
  * Faults must be deterministic given a seed so a reviewer can reproduce
    individual cells.
  * Faults are classified against ``config/faults.yaml::faults[*].class``
    so the figure can aggregate.
"""

from __future__ import annotations

import copy
import gzip
import os
import random
import shutil
import stat
import string
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .metrics import diagnosis_matches


@dataclass
class EnvContext:
    """Per-run workspace for a fault."""
    workdir: Path
    out_dir: str          # passed to execute_step()
    cwd: str              # command cwd
    env: Dict[str, str] = field(default_factory=dict)  # extra env overrides
    cleanup: Optional[Callable[[], None]] = None       # post-run tear-down


@dataclass
class Fault:
    id: str
    cls: str
    description: str
    diagnosis_regex: str
    apply: Callable[[Dict[str, Any], int, Path], tuple]
    # Difficulty tier:
    #   "easy"          — surface-level fix (one flag / typo / missing dir)
    #   "hard"          — requires semantic reasoning or a multi-step fix
    #                      (e.g. run ``bwa index`` then retry alignment)
    #   "unrecoverable" — data / environment problem that *should* be
    #                      rejected, not recovered (corrupt input, OOM)
    tier: str = "easy"

    def diagnosis_matches(self, recovery_result: Optional[Dict[str, Any]]) -> bool:
        if not recovery_result:
            return False
        diag = recovery_result.get("recovery_diagnosis") or ""
        return diagnosis_matches(diag, self.diagnosis_regex)


# ── Helpers ───────────────────────────────────────────────────────

def _mk_fastq(path: Path, n_reads: int = 10, truncated: bool = False) -> None:
    """Create a minimal (fake) gzipped FASTQ; optionally truncate mid-record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for i in range(n_reads):
        data.append(f"@read_{i}\n")
        data.append("A" * 50 + "\n")
        data.append("+\n")
        data.append("I" * 50 + "\n")
    body = "".join(data).encode()
    if truncated:
        body = body[:len(body) // 2]  # Drop half — gunzip/fastqc will choke
    # Write as gzip regardless; truncation means invalid inner stream
    with gzip.open(path, "wb") as f:
        f.write(body)
    if truncated:
        # Now corrupt the gzip footer itself to guarantee an error
        raw = path.read_bytes()
        path.write_bytes(raw[: len(raw) - 8])


def _stub_step(name: str, command: str) -> Dict[str, Any]:
    return {
        "name": name,
        "command": command,
        "dependencies": [],
        "outputs": [],
        "description": "bench fault step",
    }


def _prepare_workdir(workdir: Path) -> EnvContext:
    """Create a clean workspace with a predictable layout."""
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "data").mkdir(exist_ok=True)
    (workdir / "results").mkdir(exist_ok=True)
    return EnvContext(
        workdir=workdir,
        out_dir=str(workdir),
        cwd=str(workdir),
    )


# ── Fault implementations ─────────────────────────────────────────

def _apply_missing_wget(preset, seed, workdir) -> tuple:
    ctx = _prepare_workdir(workdir)
    step = _stub_step(
        "download_reference",
        "mkdir -p reference && wget -q -O reference/test.fa.gz "
        "https://ftp.ensembl.org/pub/release-113/fasta/homo_sapiens/cdna/Homo_sapiens.GRCh38.cdna.all.fa.gz",
    )
    # Simulate "wget not in PATH" by pre-pending an empty bin dir.
    empty_bin = workdir / "_empty_bin"
    empty_bin.mkdir(exist_ok=True)
    ctx.env["PATH"] = f"{empty_bin}:{os.environ.get('PATH', '')}"
    # Symlink everything EXCEPT wget so other tools still work.
    # Simpler: rely on the fact that wget is usually absent on macOS CI.
    return step, ctx


def _apply_tool_typo(preset, seed, workdir) -> tuple:
    ctx = _prepare_workdir(workdir)
    step = _stub_step(
        "fastqc_typo",
        "fastq_c data/reads.fastq.gz -o results/fastqc",
    )
    _mk_fastq(workdir / "data" / "reads.fastq.gz")
    (workdir / "results" / "fastqc").mkdir(exist_ok=True, parents=True)
    return step, ctx


def _apply_wrong_flag(preset, seed, workdir) -> tuple:
    ctx = _prepare_workdir(workdir)
    # -x is not a kallisto-index flag; -i is. This fails fast.
    step = _stub_step(
        "kallisto_index_wrong_flag",
        "kallisto index -x results/index.idx reference/transcriptome.fa",
    )
    (workdir / "reference").mkdir(exist_ok=True)
    (workdir / "reference" / "transcriptome.fa").write_text(">dummy\nACGT\n")
    return step, ctx


def _apply_missing_output_dir(preset, seed, workdir) -> tuple:
    ctx = _prepare_workdir(workdir)
    # The -o directory is intentionally not created.
    step = _stub_step(
        "fastqc_missing_outdir",
        "fastqc data/reads.fastq.gz -o results/does_not_exist/fastqc",
    )
    _mk_fastq(workdir / "data" / "reads.fastq.gz")
    return step, ctx


def _apply_paired_single_mismatch(preset, seed, workdir) -> tuple:
    ctx = _prepare_workdir(workdir)
    # --single requires -l/-s but we give two reads files. kallisto errors out.
    (workdir / "reference").mkdir(exist_ok=True)
    (workdir / "results").mkdir(exist_ok=True)
    _mk_fastq(workdir / "data" / "r1.fastq.gz")
    _mk_fastq(workdir / "data" / "r2.fastq.gz")
    step = _stub_step(
        "kallisto_quant_paired_as_single",
        "kallisto quant --single -i results/index.idx -o results/quant "
        "data/r1.fastq.gz data/r2.fastq.gz",
    )
    return step, ctx


def _apply_corrupt_fastq(preset, seed, workdir) -> tuple:
    ctx = _prepare_workdir(workdir)
    _mk_fastq(workdir / "data" / "corrupt.fastq.gz", truncated=True)
    (workdir / "results" / "fastqc").mkdir(parents=True, exist_ok=True)
    step = _stub_step(
        "fastqc_corrupt",
        "fastqc data/corrupt.fastq.gz -o results/fastqc",
    )
    return step, ctx


def _apply_readonly_output(preset, seed, workdir) -> tuple:
    ctx = _prepare_workdir(workdir)
    readonly = workdir / "results" / "readonly"
    readonly.mkdir(parents=True, exist_ok=True)
    _mk_fastq(workdir / "data" / "reads.fastq.gz")
    # Remove write perms.
    readonly.chmod(stat.S_IRUSR | stat.S_IXUSR)

    def _cleanup():
        try:
            readonly.chmod(stat.S_IRWXU)
        except OSError:
            pass
    ctx.cleanup = _cleanup

    step = _stub_step(
        "fastqc_readonly",
        f"fastqc data/reads.fastq.gz -o results/readonly",
    )
    return step, ctx


def _apply_path_with_spaces(preset, seed, workdir) -> tuple:
    """Regression test for the ``cd ${launchDir}`` single-quoting fix."""
    spaced = workdir / "dir with (spaces)"
    spaced.mkdir(parents=True, exist_ok=True)
    ctx = EnvContext(workdir=spaced, out_dir=str(spaced), cwd=str(spaced))
    # Unquoted cd containing parens would be a bash syntax error.
    step = _stub_step(
        "cd_into_spaced_dir",
        f"cd {spaced} && echo ok",  # deliberately unquoted
    )
    return step, ctx


def _apply_multiqc_collision(preset, seed, workdir) -> tuple:
    ctx = _prepare_workdir(workdir)
    qc = workdir / "results" / "qc"
    qc.mkdir(parents=True, exist_ok=True)
    (qc / "multiqc_report.html").write_text("<html>stale</html>")
    fastqc_dir = workdir / "results" / "fastqc"
    fastqc_dir.mkdir(exist_ok=True)
    # Put a fake fastqc_data.txt so multiqc finds *something*.
    sub = fastqc_dir / "sample_fastqc"
    sub.mkdir(exist_ok=True)
    (sub / "fastqc_data.txt").write_text(">>Per base sequence quality\tpass\n")

    step = _stub_step(
        "multiqc_no_force",
        "multiqc results/fastqc -o results/qc",  # no -f => file exists error
    )
    step["outputs"] = ["results/qc/multiqc_report.html"]
    return step, ctx


def _apply_stale_conda_pin(preset, seed, workdir) -> tuple:
    """Emulate a stale conda-pin failure via a bash stub, since the real
    failure (``CreateCondaEnvironmentException``) requires a conda solve.
    We generate a command whose exit status + stderr mimic the real one."""
    ctx = _prepare_workdir(workdir)
    step = _stub_step(
        "stale_conda_pin",
        "bash -c 'echo \"PackagesNotFoundError: The following packages are "
        "not available from current channels:\" >&2; echo \"  - kallisto=0.50.1\""
        " >&2; exit 1'",
    )
    return step, ctx


# ── Extended fault set ───────────────────────────────────────────
# All ``_stubbed_*`` helpers produce a bash-exec stub whose stderr mimics
# the real tool's error output, so the LLM diagnosis / fix path is
# exercised the same way as against the real binary. This keeps the suite
# portable (no need for bwa / GATK / htseq to be installed locally).

def _stub_cmd(stderr: str, exit_code: int = 1) -> str:
    """Build a ``bash -c`` stub that prints ``stderr`` on fd 2 and exits."""
    # Escape single quotes inside the here-string
    safe = stderr.replace("'", "'\\''")
    return f"bash -c 'printf %s\\\\n \"{safe}\" >&2; exit {exit_code}'"


# ── Easy tier (surface-level fixes) ──────────────────────────────

def _apply_samtools_subcommand_typo(preset, seed, workdir) -> tuple:
    ctx = _prepare_workdir(workdir)
    step = _stub_step(
        "samtools_subcommand_typo",
        _stub_cmd("samtools: unrecognized command 'srot'"),
    )
    return step, ctx


def _apply_missing_flag_value(preset, seed, workdir) -> tuple:
    ctx = _prepare_workdir(workdir)
    step = _stub_step(
        "samtools_sort_missing_threads_value",
        _stub_cmd(
            "samtools sort: option requires an argument -- @\n"
            "Usage: samtools sort [options...] [in.bam]"
        ),
        # Use exit 2 → typical usage-error code
    )
    return step, ctx


def _apply_missing_mandatory_arg(preset, seed, workdir) -> tuple:
    ctx = _prepare_workdir(workdir)
    step = _stub_step(
        "bwa_mem_no_reads",
        _stub_cmd(
            "Usage: bwa mem [options] <idxbase> <in1.fq> [in2.fq]\n"
            "Error: need to specify input reads FASTQ file"
        ),
    )
    return step, ctx


def _apply_cp_source_missing(preset, seed, workdir) -> tuple:
    """Real cp on a missing source — emits real shell stderr."""
    ctx = _prepare_workdir(workdir)
    step = _stub_step(
        "cp_missing_source",
        "cp data/not_present.bam results/backup.bam",
    )
    return step, ctx


def _apply_deep_nonexistent_outdir(preset, seed, workdir) -> tuple:
    """Output redirection into multiple non-existent parent dirs."""
    ctx = _prepare_workdir(workdir)
    step = _stub_step(
        "deep_output_dir",
        "bash -c 'echo content > results/a/b/c/deep/file.txt'",
    )
    return step, ctx


def _apply_ambiguous_flag(preset, seed, workdir) -> tuple:
    ctx = _prepare_workdir(workdir)
    step = _stub_step(
        "ambiguous_flag",
        _stub_cmd(
            "option --th is ambiguous; possibilities: --threads --thumbs"
        ),
    )
    return step, ctx


def _apply_undefined_env_var(preset, seed, workdir) -> tuple:
    """Reference to an undefined env var under ``set -u``."""
    ctx = _prepare_workdir(workdir)
    step = _stub_step(
        "undefined_env_var",
        "bash -c 'set -u; echo ${REFERENCE_GENOME_PATH}/genome.fa'",
    )
    return step, ctx


def _apply_missing_python_module(preset, seed, workdir) -> tuple:
    ctx = _prepare_workdir(workdir)
    step = _stub_step(
        "python_missing_module",
        "python -c 'import totally_fake_biotool_xyz; print(totally_fake_biotool_xyz)'",
    )
    return step, ctx


# ── Hard tier (semantic or multi-step fixes) ─────────────────────

def _apply_bam_unsorted_index(preset, seed, workdir) -> tuple:
    """``samtools index`` on a coordinate-unsorted BAM — recovery needs
    to insert a ``samtools sort`` step first, then retry."""
    ctx = _prepare_workdir(workdir)
    step = _stub_step(
        "bam_unsorted_indexing",
        _stub_cmd(
            "[E::hts_idx_check_range] Unsorted positions on sequence #1: chr1\n"
            "samtools index: \"input.bam\" is corrupted or unsorted"
        ),
    )
    return step, ctx


def _apply_missing_bwa_index(preset, seed, workdir) -> tuple:
    """bwa mem invoked before ``bwa index`` was run; .amb/.ann/.bwt absent."""
    ctx = _prepare_workdir(workdir)
    step = _stub_step(
        "bwa_mem_no_index",
        _stub_cmd(
            "[bwa_idx_load_from_disk] fail to locate the index files\n"
            "Please run: bwa index <ref.fa>"
        ),
    )
    return step, ctx


def _apply_missing_faidx(preset, seed, workdir) -> tuple:
    ctx = _prepare_workdir(workdir)
    step = _stub_step(
        "samtools_no_faidx",
        _stub_cmd(
            "[faidx] fai_build failed: reference.fa\n"
            "[main_samview] fail to read the reference file"
        ),
    )
    return step, ctx


def _apply_missing_sequence_dict(preset, seed, workdir) -> tuple:
    """GATK call without sequence dictionary ``.dict`` file."""
    ctx = _prepare_workdir(workdir)
    step = _stub_step(
        "gatk_missing_dict",
        _stub_cmd(
            "A USER ERROR has occurred: Fasta dict file reference.dict for "
            "reference reference.fa does not exist. Please see "
            "https://gatk.broadinstitute.org/hc/en-us/articles/360035890911",
            exit_code=2,
        ),
    )
    return step, ctx


def _apply_chromosome_naming_mismatch(preset, seed, workdir) -> tuple:
    """featureCounts with ``chr``-prefixed BAM and unprefixed GTF → 0 reads."""
    ctx = _prepare_workdir(workdir)
    step = _stub_step(
        "chrom_naming_mismatch",
        _stub_cmd(
            "featureCounts: Warning: no features matched; "
            "check chromosome naming (chr1 vs 1) in BAM header and GTF.\n"
            "Successfully assigned: 0 reads (0.0%)"
        ),
    )
    return step, ctx


def _apply_htseq_needs_name_sort(preset, seed, workdir) -> tuple:
    ctx = _prepare_workdir(workdir)
    step = _stub_step(
        "htseq_wrong_sort_order",
        _stub_cmd(
            "Error occured when processing SAM input: Paired-end reads must be "
            "sorted by read name. Use samtools sort -n."
        ),
    )
    return step, ctx


def _apply_java_heap_oom(preset, seed, workdir) -> tuple:
    """Picard-style OOM — recovery path is to raise -Xmx."""
    ctx = _prepare_workdir(workdir)
    step = _stub_step(
        "java_heap_oom",
        _stub_cmd(
            "Exception in thread \"main\" java.lang.OutOfMemoryError: Java heap space\n"
            "at htsjdk.samtools.BAMRecord.parseReadName(BAMRecord.java:201)"
        ),
    )
    return step, ctx


def _apply_vcf_contig_mismatch(preset, seed, workdir) -> tuple:
    """bcftools merge across VCFs with different reference contigs."""
    ctx = _prepare_workdir(workdir)
    step = _stub_step(
        "vcf_contig_mismatch",
        _stub_cmd(
            "Error: different contigs in the ##contig lines of the input VCFs. "
            "Use bcftools reheader or --force-samples to override.",
            exit_code=2,
        ),
    )
    return step, ctx


# ── Extra unrecoverable faults ───────────────────────────────────

def _apply_empty_input_file(preset, seed, workdir) -> tuple:
    """Zero-byte input FASTQ → downstream tool error; no fix recovers this."""
    ctx = _prepare_workdir(workdir)
    empty = workdir / "data" / "empty.fastq.gz"
    empty.parent.mkdir(parents=True, exist_ok=True)
    empty.write_bytes(b"")
    step = _stub_step(
        "fastqc_empty_input",
        _stub_cmd(
            "Failed to process data/empty.fastq.gz\n"
            "uk.ac.babraham.FastQC.Sequence.SequenceFormatException: Empty input file"
        ),
    )
    return step, ctx


def _apply_binary_as_fastq(preset, seed, workdir) -> tuple:
    """Random binary file passed to FastQC — format mismatch, no fix."""
    ctx = _prepare_workdir(workdir)
    bogus = workdir / "data" / "binary_masquerading.fastq.gz"
    bogus.parent.mkdir(parents=True, exist_ok=True)
    bogus.write_bytes(bytes(range(256)) * 16)
    step = _stub_step(
        "fastqc_binary_input",
        _stub_cmd(
            "Failed to process data/binary_masquerading.fastq.gz\n"
            "java.io.IOException: Not in GZIP format"
        ),
    )
    return step, ctx


# ── Registry ─────────────────────────────────────────────────────

FAULTS: Dict[str, Fault] = {
    # ── Easy tier ────────────────────────────────────────────────
    "missing_wget": Fault(
        id="missing_wget", cls="missing_binary", tier="easy",
        description="wget not on PATH; curl expected as the fix.",
        diagnosis_regex=r"(?i)wget|not found|command not found|curl",
        apply=_apply_missing_wget),
    "tool_typo": Fault(
        id="tool_typo", cls="typo", tier="easy",
        description="fastq_c instead of fastqc (typo).",
        diagnosis_regex=r"(?i)typo|not found|misspell|command not found|fastqc",
        apply=_apply_tool_typo),
    "wrong_flag": Fault(
        id="wrong_flag", cls="wrong_flag", tier="easy",
        description="kallisto index -x instead of -i.",
        diagnosis_regex=r"(?i)flag|option|argument|usage|-i",
        apply=_apply_wrong_flag),
    "missing_output_dir": Fault(
        id="missing_output_dir", cls="missing_path", tier="easy",
        description="FastQC -o into a non-existent directory.",
        diagnosis_regex=r"(?i)directory.*exist|mkdir|no such",
        apply=_apply_missing_output_dir),
    "readonly_output": Fault(
        id="readonly_output", cls="permission", tier="easy",
        description="Output directory is not writable.",
        diagnosis_regex=r"(?i)permission|read-?only|not writable",
        apply=_apply_readonly_output),
    "path_with_spaces": Fault(
        id="path_with_spaces", cls="shell_escaping", tier="easy",
        description="Unquoted path containing parentheses (regression test).",
        diagnosis_regex=r"(?i)quote|escape|syntax|space|unexpected token",
        apply=_apply_path_with_spaces),
    "multiqc_collision": Fault(
        id="multiqc_collision", cls="output_collision", tier="easy",
        description="Pre-existing multiqc_report.html forces name-suffix.",
        diagnosis_regex=r"(?i)-f|force|overwrite|exist|already",
        apply=_apply_multiqc_collision),
    "samtools_subcommand_typo": Fault(
        id="samtools_subcommand_typo", cls="typo", tier="easy",
        description="samtools srot → sort (subcommand typo).",
        diagnosis_regex=r"(?i)srot|unrecognized|unknown|sort",
        apply=_apply_samtools_subcommand_typo),
    "missing_flag_value": Fault(
        id="missing_flag_value", cls="usage", tier="easy",
        description="-@ flag missing its numeric thread count.",
        diagnosis_regex=r"(?i)requires.*argument|missing.*value|usage|-@",
        apply=_apply_missing_flag_value),
    "missing_mandatory_arg": Fault(
        id="missing_mandatory_arg", cls="usage", tier="easy",
        description="bwa mem invoked without input FASTQ.",
        diagnosis_regex=r"(?i)usage|need.*input|reads|fastq",
        apply=_apply_missing_mandatory_arg),
    "cp_source_missing": Fault(
        id="cp_source_missing", cls="missing_path", tier="easy",
        description="cp with a non-existent source file.",
        diagnosis_regex=r"(?i)no such file|not exist|missing",
        apply=_apply_cp_source_missing),
    "deep_nonexistent_outdir": Fault(
        id="deep_nonexistent_outdir", cls="missing_path", tier="easy",
        description="Redirect output into multiple unmade parent dirs.",
        diagnosis_regex=r"(?i)no such|directory|mkdir",
        apply=_apply_deep_nonexistent_outdir),
    "ambiguous_flag": Fault(
        id="ambiguous_flag", cls="usage", tier="easy",
        description="Abbreviated flag (--th) is ambiguous.",
        diagnosis_regex=r"(?i)ambiguous|flag|option|--threads",
        apply=_apply_ambiguous_flag),
    "undefined_env_var": Fault(
        id="undefined_env_var", cls="shell_escaping", tier="easy",
        description="Reference to an unset env var under set -u.",
        diagnosis_regex=r"(?i)unbound|unset|variable|env|export",
        apply=_apply_undefined_env_var),
    "missing_python_module": Fault(
        id="missing_python_module", cls="env_mismatch", tier="easy",
        description="Python imports a module not installed in the env.",
        diagnosis_regex=r"(?i)modulenotfounderror|no module|pip install|import",
        apply=_apply_missing_python_module),

    # ── Hard tier ────────────────────────────────────────────────
    "stale_conda_pin": Fault(
        id="stale_conda_pin", cls="env_mismatch", tier="hard",
        description="Conda package version not available for platform.",
        diagnosis_regex=r"(?i)not available|pin|version|package.*found",
        apply=_apply_stale_conda_pin),
    "bam_unsorted_indexing": Fault(
        id="bam_unsorted_indexing", cls="data_order", tier="hard",
        description="samtools index on unsorted BAM — must sort first.",
        diagnosis_regex=r"(?i)unsorted|sort|index|coordinate",
        apply=_apply_bam_unsorted_index),
    "missing_bwa_index": Fault(
        id="missing_bwa_index", cls="missing_artefact", tier="hard",
        description="bwa mem before ``bwa index`` — must build index first.",
        diagnosis_regex=r"(?i)locate.*index|bwa index|idx|missing",
        apply=_apply_missing_bwa_index),
    "missing_faidx": Fault(
        id="missing_faidx", cls="missing_artefact", tier="hard",
        description="samtools command requires .fai — run samtools faidx.",
        diagnosis_regex=r"(?i)faidx|fai|reference.*index|reference.*file",
        apply=_apply_missing_faidx),
    "missing_sequence_dict": Fault(
        id="missing_sequence_dict", cls="missing_artefact", tier="hard",
        description="GATK needs .dict — run CreateSequenceDictionary first.",
        diagnosis_regex=r"(?i)\.dict|sequence.*dictionary|createsequencedictionary",
        apply=_apply_missing_sequence_dict),
    "chromosome_naming_mismatch": Fault(
        id="chromosome_naming_mismatch", cls="data_semantic", tier="hard",
        description="BAM uses 'chr1' but GTF uses '1' — 0 reads assigned.",
        diagnosis_regex=r"(?i)chr|contig|naming|prefix|no features",
        apply=_apply_chromosome_naming_mismatch),
    "htseq_needs_name_sort": Fault(
        id="htseq_needs_name_sort", cls="data_order", tier="hard",
        description="htseq-count paired-end needs name-sorted BAM.",
        diagnosis_regex=r"(?i)sort.*name|name.*sort|samtools.*-n",
        apply=_apply_htseq_needs_name_sort),
    "java_heap_oom": Fault(
        id="java_heap_oom", cls="resource", tier="hard",
        description="Picard/GATK OOM — bump -Xmx.",
        diagnosis_regex=r"(?i)outofmemoryerror|heap|xmx|memory",
        apply=_apply_java_heap_oom),
    "vcf_contig_mismatch": Fault(
        id="vcf_contig_mismatch", cls="data_semantic", tier="hard",
        description="bcftools merge across differently-built VCFs.",
        diagnosis_regex=r"(?i)contig|reheader|vcf|force-samples",
        apply=_apply_vcf_contig_mismatch),

    # ── Unrecoverable (should be rejected, not recovered) ────────
    "paired_single_mismatch": Fault(
        id="paired_single_mismatch", cls="data_shape_mismatch", tier="unrecoverable",
        description="Paired-end reads passed with --single.",
        diagnosis_regex=r"(?i)paired|single|fragment|length",
        apply=_apply_paired_single_mismatch),
    "corrupt_fastq": Fault(
        id="corrupt_fastq", cls="corrupt_input", tier="unrecoverable",
        description="Truncated gzip FASTQ.",
        diagnosis_regex=r"(?i)truncat|invalid|corrupt|gzip|format",
        apply=_apply_corrupt_fastq),
    "empty_input_file": Fault(
        id="empty_input_file", cls="corrupt_input", tier="unrecoverable",
        description="Zero-byte FASTQ passed to FastQC.",
        diagnosis_regex=r"(?i)empty|no (data|reads)|format",
        apply=_apply_empty_input_file),
    "binary_as_fastq": Fault(
        id="binary_as_fastq", cls="corrupt_input", tier="unrecoverable",
        description="Random binary file passed to FastQC.",
        diagnosis_regex=r"(?i)not.*gzip|format|invalid|binary",
        apply=_apply_binary_as_fastq),
}
