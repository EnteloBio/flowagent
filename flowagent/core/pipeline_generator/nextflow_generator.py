"""Generate Nextflow DSL2 pipelines from a WorkflowPlan."""

import logging
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, BaseLoader

from .base import PipelineGenerator

logger = logging.getLogger(__name__)

NEXTFLOW_PROCESS_TEMPLATE = """\
process {{ step_name | upper }} {
    {% if container %}container '{{ container }}'{% endif %}
    {% if cpus %}cpus {{ cpus }}{% endif %}
    {% if memory %}memory '{{ memory }}'{% endif %}
    {% if publish_dir %}publishDir '{{ publish_dir }}', mode: 'copy'{% endif %}

    input:
    {% for inp in inputs %}
    {{ inp }}
    {% endfor %}

    output:
    {% for out in outputs %}
    {{ out }}
    {% endfor %}

    script:
    \"\"\"
    {{ command }}
    \"\"\"
}
"""

NEXTFLOW_WORKFLOW_TEMPLATE = """\
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

// ── Parameters ──────────────────────────────────────────────
params.outdir   = '{{ outdir }}'
params.reads    = '{{ reads_pattern }}'

// ── Processes ───────────────────────────────────────────────
{{ processes }}

// ── Workflow ────────────────────────────────────────────────
workflow {
    {{ workflow_body }}
}
"""


class NextflowGenerator(PipelineGenerator):
    """Converts a FlowAgent workflow plan to Nextflow DSL2."""

    def __init__(self):
        self.env = Environment(loader=BaseLoader())
        self.process_tmpl = self.env.from_string(NEXTFLOW_PROCESS_TEMPLATE)
        self.workflow_tmpl = self.env.from_string(NEXTFLOW_WORKFLOW_TEMPLATE)

    def default_filename(self) -> str:
        return "main.nf"

    def generate(self, workflow_plan: Dict[str, Any], *, output_dir: Optional[Path] = None) -> str:
        steps = workflow_plan.get("steps", [])
        if not steps:
            return "// Empty workflow plan"

        processes: List[str] = []
        workflow_lines: List[str] = []
        prev_output_var: Optional[str] = None

        for i, step in enumerate(steps):
            name = self._safe_name(step.get("name", f"step_{i}"))
            command = step.get("command", "echo 'no command'")
            resources = step.get("resources", {})
            outputs = step.get("outputs", [])

            # Determine container from tool name if possible
            container = self._resolve_container(command)

            # Build input/output declarations
            input_decls = []
            if prev_output_var:
                input_decls.append(f"path(input_files)")
            else:
                input_decls.append("val(ready)")

            output_decls = []
            out_var = f"{name}_out"
            if outputs:
                for out in outputs:
                    output_decls.append(f"path('{out}'), emit: {out_var}")
            else:
                output_decls.append(f"path('*'), emit: {out_var}")

            process_code = self.process_tmpl.render(
                step_name=name,
                container=container,
                cpus=resources.get("cpus"),
                memory=resources.get("memory"),
                publish_dir=f"${{params.outdir}}/{name}",
                inputs=input_decls,
                outputs=output_decls,
                command=command,
            )
            processes.append(process_code)

            # Build workflow wiring
            upper_name = name.upper()
            if prev_output_var:
                workflow_lines.append(f"{upper_name}({prev_output_var})")
            else:
                workflow_lines.append(f"{upper_name}(Channel.value('ready'))")

            prev_output_var = f"{upper_name}.out.{out_var}"

        # Assemble
        reads_pattern = self._infer_reads_pattern(steps)
        code = self.workflow_tmpl.render(
            outdir="results",
            reads_pattern=reads_pattern,
            processes="\n".join(processes),
            workflow_body="\n    ".join(workflow_lines),
        )

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / self.default_filename()).write_text(code)
            self._write_nextflow_config(output_dir)
            logger.info("Wrote Nextflow pipeline to %s", output_dir / self.default_filename())

        return code

    def validate(self, code: str, *, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Validate by running ``nextflow -preview`` if available."""
        import shutil
        import subprocess
        import tempfile as _tmp

        if not shutil.which("nextflow"):
            return {"valid": True, "errors": [], "warnings": ["nextflow not in PATH, skipping validation"]}

        tmp_cleanup = None
        if not output_dir:
            tmp_cleanup = _tmp.TemporaryDirectory()
            output_dir = Path(tmp_cleanup.name)

        try:
            nf_file = output_dir / self.default_filename()
            nf_file.write_text(code)

            result = subprocess.run(
                ["nextflow", "run", str(nf_file), "-preview"],
                capture_output=True, text=True, timeout=120,
            )
            errors = []
            warnings = []
            if result.returncode != 0:
                errors.append(result.stderr)
            return {"valid": result.returncode == 0, "errors": errors, "warnings": warnings}
        except Exception as e:
            return {"valid": False, "errors": [str(e)], "warnings": []}
        finally:
            if tmp_cleanup:
                tmp_cleanup.cleanup()

    # ── helpers ────────────────────────────────────────────────

    @staticmethod
    def _safe_name(name) -> str:
        return str(name or "unnamed").lower().replace(" ", "_").replace("-", "_")

    @staticmethod
    def _resolve_container(command: str) -> str:
        """Best-effort BioContainers lookup from the first word in the command."""
        tool = command.strip().split()[0].split("/")[-1] if command.strip() else ""
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
        }
        return containers.get(tool, "")

    @staticmethod
    def _infer_reads_pattern(steps: List[Dict[str, Any]]) -> str:
        for step in steps:
            cmd = step.get("command", "")
            if ".fastq" in cmd or ".fq" in cmd:
                return "data/*_{R1,R2}.fastq.gz"
        return "data/*.fastq.gz"

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
