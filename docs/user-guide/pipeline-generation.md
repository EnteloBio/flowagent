# Pipeline Generation

FlowAgent can emit your workflow as a portable **Nextflow DSL2** or
**Snakemake** file, ready to commit, distribute, or submit to any cluster
that supports the runtime. This is the recommended path when you need
reproducibility beyond a single FlowAgent invocation.

## Quick start

```bash
# Generate + run
flowagent prompt "RNA-seq with Kallisto" --pipeline-format nextflow

# Generate only (don't execute)
flowagent prompt "RNA-seq with Kallisto" \
    --pipeline-format snakemake --no-execute
```

Output lands in `flowagent_pipeline_output/`.

## What gets generated

### Nextflow

```
flowagent_pipeline_output/
├── main.nf          # DSL2 pipeline with one process per plan step
└── nextflow.config  # profiles: local, docker, singularity, slurm
```

Each process:

- Inherits a container image from a built-in registry (FastQC, Kallisto,
  STAR, MultiQC, ...).
- Specifies `cpus` and `memory` from the plan's `resources` block.
- Uses **`val(true)`** signalling for dependencies — every process emits
  a "done" signal that downstream processes consume via
  `.collect()`. This avoids hard-coding channel topology.
- Begins with `cd '${launchDir}'` so commands resolve relative paths
  against the user's launch directory (Nextflow processes otherwise run
  in isolated `work/xx/yyyy/` directories).

### Snakemake

```
flowagent_pipeline_output/
├── Snakefile        # one rule per plan step, plus 'rule all'
├── config.yaml      # samples, threads, outdir
└── envs/            # conda env YAMLs (one per tool)
    ├── base.yaml
    ├── fastqc.yaml
    ├── kallisto.yaml
    └── ...
```

Each rule:

- Declares `input:` from upstream rules' outputs (or a `.done` marker
  for directory-setup steps).
- Declares `output:` from the plan's `outputs` field; falls back to a
  `.done` marker if outputs look like directories.
- Routes to the correct conda env via the new `_primary_tool()` helper
  (so `mkdir -p foo && fastqc ...` correctly maps to the `fastqc` env,
  not `base`).
- Runs the command with `2> logs/<rule>.log` for per-rule logging.

## Validation

Both generators have a `.validate()` method called automatically:

| Generator | Validation command | Behaviour on failure |
|---|---|---|
| `NextflowGenerator` | `nextflow run main.nf -preview` | Logs warnings; never blocks |
| `SnakemakeGenerator` | `snakemake --lint` + `snakemake -n` | Logs warnings; never blocks |

If the runtime isn't installed, validation is silently skipped with a
warning. You can run validation manually:

```bash
nextflow run flowagent_pipeline_output/main.nf -preview
snakemake --lint -s flowagent_pipeline_output/Snakefile
```

## Live progress display

Both `NextflowExecutor` and `SnakemakeExecutor` connect their stdout
directly to your terminal so you see live ANSI progress (Nextflow's
process bars, Snakemake's percent-complete) in real time. On failure,
output is recovered from `.nextflow.log` (Nextflow) or `snakemake.log`
(Snakemake) for the [error recovery loop](error-recovery.md).

## Profiles

Nextflow profiles in the generated `nextflow.config`:

| Profile | Use |
|---|---|
| `local` | Single machine (default) |
| `docker` | Pull biocontainers via Docker |
| `singularity` | Pull biocontainers via Singularity (HPC-friendly) |
| `slurm` | Submit each process as a SLURM job |

Pick one with `--profile`:

```bash
flowagent prompt "..." --pipeline-format nextflow --profile slurm
```

Snakemake config knobs are in `flowagent_pipeline_output/config.yaml` —
edit before running.

## Container resolution

The Nextflow generator maps tools to biocontainers:

| Tool | Container |
|---|---|
| `fastqc` | `biocontainers/fastqc:0.12.1--hdfd78af_0` |
| `kallisto` | `biocontainers/kallisto:0.50.1--h6de1650_2` |
| `multiqc` | `ewels/multiqc:latest` |
| `star`, `bwa`, `bowtie2`, `samtools`, ... | biocontainers |
| `gatk`, `picard` | `broadinstitute/...` |

Tools not in the registry fall back to running on the host.

## Conda environments (Snakemake)

The Snakemake generator writes `envs/<tool>.yaml` for each tool. Version
pins are intentionally **relaxed** so conda can solve for the best build
on any platform (especially `osx-arm64` where many bioconda packages have
different versions or builds than `linux-64`).

If you need pinned versions, edit the YAMLs after generation.

## Caveats

- **One generator per format.** FlowAgent generates Nextflow and
  Snakemake; CGAT-core / WDL / CWL are not supported as generators
  (CGAT-core is supported as an executor — see [Executors](executors.md)).
- **Path with spaces** is now handled (the `cd '${launchDir}'` is
  single-quoted). If you upgrade from an older version and have a
  cached `main.nf`, regenerate it.
- **MultiQC overwrites.** The LLM is prompted to use `multiqc -f -n
  multiqc_report` to avoid `multiqc_report_1.html` collisions when
  re-running. Older runs may need a manual `rm` of the prior report.

## Programmatic generation

```python
from pathlib import Path
from flowagent.core.pipeline_generator.nextflow_generator import NextflowGenerator
from flowagent.presets.catalog import get_preset

plan = get_preset("rnaseq-kallisto")
gen  = NextflowGenerator()
code = gen.generate(plan, output_dir=Path("out/"))
print(gen.validate(code, output_dir=Path("out/")))
```

## See also

- [Executors](executors.md) for choosing where to run the generated pipeline
- [Error Recovery](error-recovery.md) for what happens when a generated step fails
- [Benchmarking](../benchmarking.md) — Benchmark C measures generation fidelity
