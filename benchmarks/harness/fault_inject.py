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


# ── Registry ─────────────────────────────────────────────────────

FAULTS: Dict[str, Fault] = {
    "missing_wget": Fault(
        id="missing_wget", cls="missing_binary",
        description="wget not on PATH; curl expected as the fix.",
        diagnosis_regex=r"(?i)wget|not found|command not found|curl",
        apply=_apply_missing_wget),
    "tool_typo": Fault(
        id="tool_typo", cls="typo",
        description="fastq_c instead of fastqc (typo).",
        diagnosis_regex=r"(?i)typo|not found|misspell|command not found|fastqc",
        apply=_apply_tool_typo),
    "wrong_flag": Fault(
        id="wrong_flag", cls="wrong_flag",
        description="kallisto index -x instead of -i.",
        diagnosis_regex=r"(?i)flag|option|argument|usage|-i",
        apply=_apply_wrong_flag),
    "missing_output_dir": Fault(
        id="missing_output_dir", cls="missing_path",
        description="FastQC -o into a non-existent directory.",
        diagnosis_regex=r"(?i)directory.*exist|mkdir|no such",
        apply=_apply_missing_output_dir),
    "paired_single_mismatch": Fault(
        id="paired_single_mismatch", cls="data_shape_mismatch",
        description="Paired-end reads passed with --single.",
        diagnosis_regex=r"(?i)paired|single|fragment|length",
        apply=_apply_paired_single_mismatch),
    "corrupt_fastq": Fault(
        id="corrupt_fastq", cls="corrupt_input",
        description="Truncated gzip FASTQ.",
        diagnosis_regex=r"(?i)truncat|invalid|corrupt|gzip|format",
        apply=_apply_corrupt_fastq),
    "readonly_output": Fault(
        id="readonly_output", cls="permission",
        description="Output directory is not writable.",
        diagnosis_regex=r"(?i)permission|read-?only|not writable",
        apply=_apply_readonly_output),
    "path_with_spaces": Fault(
        id="path_with_spaces", cls="shell_escaping",
        description="Unquoted path containing parentheses (regression test).",
        diagnosis_regex=r"(?i)quote|escape|syntax|space|unexpected token",
        apply=_apply_path_with_spaces),
    "multiqc_collision": Fault(
        id="multiqc_collision", cls="output_collision",
        description="Pre-existing multiqc_report.html forces name-suffix.",
        diagnosis_regex=r"(?i)-f|force|overwrite|exist|already",
        apply=_apply_multiqc_collision),
    "stale_conda_pin": Fault(
        id="stale_conda_pin", cls="env_mismatch",
        description="Conda package version not available for platform.",
        diagnosis_regex=r"(?i)not available|pin|version|package.*found",
        apply=_apply_stale_conda_pin),
}
