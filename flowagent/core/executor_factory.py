"""Unified executor factory that wires up all execution backends.

Replaces the disconnect where ``WorkflowManager`` created the basic
``Executor`` class (local/slurm only) while ``CGATExecutor``,
``HPCExecutor``, and ``KubernetesExecutor`` sat unused in ``executors.py``.
"""

import asyncio
import logging
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from ..config.settings import Settings
from ..utils.logging import get_logger
from .executors import BaseExecutor, LocalExecutor, CGATExecutor, HPCExecutor, KubernetesExecutor

logger = get_logger(__name__)
settings = Settings()


# ── Nextflow executor ─────────────────────────────────────────

class NextflowExecutor(BaseExecutor):
    """Execute a Nextflow pipeline file (``main.nf``)."""

    def __init__(self, profile: str = "local"):
        self.profile = profile
        logger.info("Initialized NextflowExecutor (profile=%s)", profile)

    async def execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """For Nextflow, 'step' is expected to carry a ``pipeline_file`` key.

        If not present, fall back to running ``step['command']`` directly.

        Stdout/stderr are connected directly to the terminal so the user
        sees Nextflow's live ANSI progress display.  On failure the output
        is recovered from ``.nextflow.log`` for the error-recovery loop.
        """
        pipeline_file = step.get("pipeline_file", step.get("command", "main.nf"))
        cmd = f"nextflow run {shlex.quote(str(pipeline_file))} -profile {shlex.quote(self.profile)} -resume"

        work_dir = step.get("cwd", ".")
        logger.info("Running Nextflow: %s (cwd=%s)", cmd, work_dir)

        # Connect directly to the terminal for live progress display.
        # Nextflow uses ANSI escape codes (\r, cursor movement) that only
        # render correctly when attached to a real TTY / unbuffered stdout.
        proc = await asyncio.create_subprocess_exec(
            "bash", "-c", cmd,
            stdout=None,   # inherit parent stdout (live display)
            stderr=None,   # inherit parent stderr
            cwd=work_dir,
        )
        await proc.wait()

        # Recover output for the result dict (needed by the recovery loop).
        # Since we didn't capture stdout, read .nextflow.log instead.
        output = ""
        nf_log = Path(work_dir) / ".nextflow.log"
        if nf_log.exists():
            try:
                output = nf_log.read_text()[-5000:]
            except OSError:
                pass

        return {
            "step_id": step.get("name", "nextflow_run"),
            "status": "completed" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode,
            "stdout": output,
            "stderr": output,
        }

    async def wait_for_completion(self, jobs: Dict[str, Any]) -> Dict[str, Any]:
        return jobs


# ── Snakemake executor ────────────────────────────────────────

class SnakemakeExecutor(BaseExecutor):
    """Execute a Snakemake pipeline (``Snakefile``)."""

    def __init__(self, cores: int = 4, use_conda: bool = True):
        self.cores = cores
        self.use_conda = use_conda
        logger.info("Initialized SnakemakeExecutor (cores=%d, conda=%s)", cores, use_conda)

    async def execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        snakefile = Path(step.get("pipeline_file", step.get("command", "Snakefile")))
        parts = ["snakemake", f"--cores {self.cores}", f"-s {shlex.quote(str(snakefile))}"]
        if self.use_conda:
            parts.append("--use-conda")
        # Point to the config.yaml next to the Snakefile
        config_yaml = snakefile.parent / "config.yaml"
        if config_yaml.exists():
            parts.append(f"--configfile {shlex.quote(str(config_yaml))}")
        cmd = " ".join(parts)

        work_dir = step.get("cwd", ".")
        logger.info("Running Snakemake: %s (cwd=%s)", cmd, work_dir)

        # Write Snakemake's log to a file while streaming to the terminal
        log_file = snakefile.parent / "snakemake.log"

        proc = await asyncio.create_subprocess_exec(
            "bash", "-c", f"set -o pipefail; {cmd} 2>&1 | tee {shlex.quote(str(log_file))}",
            stdout=None,   # inherit parent stdout (live display)
            stderr=None,   # inherit parent stderr
            cwd=work_dir,
        )
        await proc.wait()

        # Read captured log for the result dict (needed by the recovery loop)
        output = ""
        if log_file.exists():
            try:
                output = log_file.read_text()[-5000:]
            except OSError:
                pass

        return {
            "step_id": step.get("name", "snakemake_run"),
            "status": "completed" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode,
            "stdout": output,
            "stderr": output,
        }

    async def wait_for_completion(self, jobs: Dict[str, Any]) -> Dict[str, Any]:
        return jobs


# ── Factory ───────────────────────────────────────────────────

class ExecutorFactory:
    """Create the right executor based on a type string.

    Supported types: ``local``, ``cgat``, ``hpc``, ``slurm``,
    ``kubernetes``, ``nextflow``, ``snakemake``.
    """

    @staticmethod
    def create(executor_type: str, **kwargs) -> BaseExecutor:
        executor_type = executor_type.lower().strip()

        if executor_type == "local":
            return LocalExecutor()

        if executor_type == "cgat":
            return CGATExecutor()

        if executor_type in ("hpc", "slurm", "sge", "torque"):
            return HPCExecutor()

        if executor_type == "kubernetes":
            return KubernetesExecutor()

        if executor_type == "nextflow":
            profile = kwargs.get("profile", settings.PIPELINE_PROFILE)
            return NextflowExecutor(profile=profile)

        if executor_type == "snakemake":
            cores = kwargs.get("cores", settings.DEFAULT_WORKFLOW_PARAMS.get("threads", 4))
            return SnakemakeExecutor(cores=cores)

        logger.warning("Unknown executor type '%s', falling back to local", executor_type)
        return LocalExecutor()
