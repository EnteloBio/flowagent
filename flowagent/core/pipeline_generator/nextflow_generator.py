"""Generate Nextflow DSL2 pipelines from a WorkflowPlan."""

import logging
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import PipelineGenerator

logger = logging.getLogger(__name__)


class NextflowGenerator(PipelineGenerator):
    """Converts a FlowAgent workflow plan to Nextflow DSL2.

    Because LLM-generated commands are self-contained (file paths baked in),
    processes use val-based signalling for dependency ordering rather than
    Nextflow channels for data flow.  Each process emits ``val(true)`` on
    completion; downstream processes ``.collect()`` their dependency signals
    before starting.
    """

    def default_filename(self) -> str:
        return "main.nf"

    # ── public API ────────────────────────────────────────────

    def generate(self, workflow_plan: Dict[str, Any], *, output_dir: Optional[Path] = None) -> str:
        steps = workflow_plan.get("steps", [])
        if not steps:
            return "// Empty workflow plan"

        step_map: Dict[str, Dict[str, Any]] = {}
        for i, step in enumerate(steps):
            name = self._safe_name(step.get("name", f"step_{i}"))
            step_map[name] = step

        process_blocks: List[str] = []
        workflow_lines: List[str] = []

        for i, step in enumerate(steps):
            name = self._safe_name(step.get("name", f"step_{i}"))
            command = step.get("command", "echo 'no command'")
            resources = step.get("resources", {})
            deps = [self._safe_name(d) for d in step.get("dependencies", [])]

            container = self._resolve_container(command)

            process_blocks.append(self._render_process(
                name=name,
                command=command,
                container=container,
                cpus=resources.get("cpus"),
                memory=resources.get("memory"),
            ))

            upper = name.upper()
            if not deps:
                workflow_lines.append(f"{upper}(Channel.value(true))")
            elif len(deps) == 1:
                workflow_lines.append(f"{upper}({deps[0].upper()}.out.done)")
            else:
                signals = [f"{d.upper()}.out.done" for d in deps]
                merged = f"{signals[0]}.mix({', '.join(signals[1:])}).collect()"
                workflow_lines.append(f"{upper}({merged})")

        code = self._assemble(process_blocks, workflow_lines, steps)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / self.default_filename()).write_text(code)
            self._write_nextflow_config(output_dir)
            logger.info("Wrote Nextflow pipeline to %s", output_dir / self.default_filename())

        return code

    def validate(self, code: str, *, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Advisory validation -- warnings never block execution."""
        import shutil
        import subprocess
        import tempfile as _tmp

        if not shutil.which("nextflow"):
            return {"valid": True, "errors": [], "warnings": ["nextflow not in PATH, skipping validation"]}

        tmp_cleanup = None
        if not output_dir:
            tmp_cleanup = _tmp.TemporaryDirectory()
            output_dir = Path(tmp_cleanup.name)

        warnings: list[str] = []
        try:
            nf_file = output_dir / self.default_filename()
            nf_file.write_text(code)
            result = subprocess.run(
                ["nextflow", "run", str(nf_file), "-preview"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                msg = (result.stderr or result.stdout or "").strip()
                if msg:
                    warnings.append(msg)
        except Exception as e:
            warnings.append(f"Could not run nextflow -preview: {e}")
        finally:
            if tmp_cleanup:
                tmp_cleanup.cleanup()

        return {"valid": True, "errors": [], "warnings": warnings}

    # ── private helpers ───────────────────────────────────────

    @staticmethod
    def _safe_name(name) -> str:
        return str(name or "unnamed").lower().replace(" ", "_").replace("-", "_")

    @staticmethod
    def _resolve_container(command: str) -> str:
        containers = {
            "fastqc": "biocontainers/fastqc:0.12.1--hdfd78af_0",
            "multiqc": "ewels/multiqc:latest",
            "kallisto": "biocontainers/kallisto:0.50.1--h6de1650_2",
            "samtools": "biocontainers/samtools:1.19.2--h50ea8bc_1",
            "hisat2": "biocontainers/hisat2:2.2.1--hdbdd923_6",
            "trim_galore": "biocontainers/trim-galore:0.6.10--hdfd78af_0",
            "star": "biocontainers/star:2.7.11b--h43eeafb_0",
            "salmon": "biocontainers/salmon:1.10.3--h6dccd9a_1",
            "bowtie2": "biocontainers/bowtie2:2.5.3--he20e202_0",
            "bwa": "biocontainers/bwa:0.7.18--he4a0461_0",
            "picard": "broadinstitute/picard:latest",
            "gatk": "broadinstitute/gatk:latest",
            "bedtools": "biocontainers/bedtools:2.31.1--hf5e1c6e_1",
            "featurecounts": "biocontainers/subread:2.0.6--h9a82719_0",
            "stringtie": "biocontainers/stringtie:2.2.3--h43eeafb_0",
        }
        # Find the first bioinformatics tool in the command, skipping
        # shell prefixes like ``mkdir -p foo && fastqc ...``
        import re
        shell_words = {"mkdir", "cd", "rm", "mv", "cp", "ln", "touch", "test",
                       "set", "export", "echo", "source", "bash", "sh"}
        tokens = re.split(r'[\s;|&()<>]+', (command or "").strip())
        for tok in tokens:
            if not tok or "=" in tok:
                continue
            name = tok.split("/")[-1]
            if name in containers:
                return containers[name]
            # Don't treat shell builtins or flag-looking tokens as "the tool"
            if name in shell_words or name.startswith("-"):
                continue
        return ""

    @staticmethod
    def _infer_reads_pattern(steps: List[Dict[str, Any]]) -> str:
        for step in steps:
            cmd = step.get("command", "")
            if ".fastq" in cmd or ".fq" in cmd:
                return "*_{R1,R2}.fastq.gz"
        return "*.fastq.gz"

    def _render_process(
        self, *, name: str, command: str, container: str,
        cpus: Optional[int] = None, memory: Optional[str] = None,
    ) -> str:
        lines = [f"process {name.upper()} {{"]

        if container:
            lines.append(f"    container '{container}'")
        if cpus:
            lines.append(f"    cpus {cpus}")
        if memory:
            lines.append(f"    memory '{memory}'")
        lines.append(f"    publishDir \"${{params.outdir}}/{name}\", mode: 'copy'")
        lines.append("")
        lines.append("    input:")
        lines.append("    val(ready)")
        lines.append("")
        lines.append("    output:")
        lines.append("    val(true), emit: done")
        lines.append("")
        # Nextflow isolates each process in its own work directory.
        # Commands generated by the LLM assume a shared filesystem with
        # relative paths from the user's launch directory, so we cd back
        # to ${launchDir} before running each command.
        lines.append("    script:")
        lines.append('    """')
        lines.append("    cd '${launchDir}'")
        lines.append(f"    {command}")
        lines.append('    """')
        lines.append("}")
        return "\n".join(lines)

    def _assemble(
        self, process_blocks: List[str], workflow_lines: List[str],
        steps: List[Dict[str, Any]],
    ) -> str:
        reads_pattern = self._infer_reads_pattern(steps)
        header = textwrap.dedent(f"""\
            #!/usr/bin/env nextflow
            nextflow.enable.dsl = 2

            // ── Parameters ──────────────────────────────────────────────
            params.outdir = 'results'
            params.reads  = '{reads_pattern}'
        """)

        process_section = "\n// ── Processes ───────────────────────────────────────────────\n\n"
        process_section += "\n\n".join(process_blocks)

        wf_body = "\n    ".join(workflow_lines)
        workflow_section = textwrap.dedent(f"""

            // ── Workflow ────────────────────────────────────────────────
            workflow {{
                {wf_body}
            }}
        """)

        return header + process_section + workflow_section

    @staticmethod
    def _write_nextflow_config(output_dir: Path):
        config = textwrap.dedent("""\
        // Nextflow configuration generated by FlowAgent
        profiles {
            local {
                process.executor = 'local'
            }
            docker {
                docker.enabled = true
            }
            singularity {
                singularity.enabled = true
            }
            slurm {
                process.executor = 'slurm'
                process.queue = 'all.q'
            }
        }

        // Default resources
        process {
            cpus = 1
            memory = '4 GB'
            time = '1h'
        }
        """)
        (output_dir / "nextflow.config").write_text(config)
