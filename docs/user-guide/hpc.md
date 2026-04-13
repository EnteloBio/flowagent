# HPC Configuration

FlowAgent supports HPC clusters via three different paths. Pick the one
that matches your environment.

## At a glance

| Path | Backend | Best for |
|---|---|---|
| **`--executor hpc`** | `HPCExecutor` (cgatcore + DRMAA) | Native SLURM/SGE/TORQUE submission per step |
| **`--executor cgat`** | `CGATExecutor` | CGAT-core managed clusters |
| **`--executor nextflow`** + `--profile slurm` | Nextflow runtime | Distributable Nextflow pipelines that ship to any cluster |

For end-to-end portability we recommend generating a Nextflow pipeline
(`--pipeline-format nextflow`) and submitting it through Nextflow's own
SLURM profile. For tighter control over per-step submission, use
`--executor hpc` directly.

## Optional dependencies

```bash
# cgatcore + DRMAA (cgat & hpc executors)
pip install -e ".[hpc]"

# Plus your scheduler's DRMAA library:
#   SLURM:  https://github.com/natefoo/slurm-drmaa
#   SGE:    libdrmaa.so from SGE installation
#   TORQUE: pbs-drmaa
```

## Path 1 — Native HPC executor (`--executor hpc`)

This submits every step as its own scheduler job using cgatcore's
submission layer over DRMAA.

### Environment

```bash
EXECUTOR_TYPE=hpc
HPC_SYSTEM=slurm                 # slurm | sge | torque
HPC_QUEUE=all.q
HPC_DEFAULT_MEMORY=4G
HPC_DEFAULT_CPUS=1
HPC_DEFAULT_TIME=60              # minutes
```

### CLI

```bash
flowagent prompt "RNA-seq with Kallisto" \
    --executor hpc --hpc-system slurm
```

### How resources are mapped

The LLM-generated plan includes per-step resources
(`{"cpus": 4, "memory": "8G", "time_min": 60}`), which `HPCExecutor`
translates into scheduler options:

| Plan field | SLURM | SGE | TORQUE |
|---|---|---|---|
| `cpus` | `--cpus-per-task` | `-pe smp N` | `-l nodes=1:ppn=N` |
| `memory` | `--mem` | `-l h_vmem` | `-l mem` |
| `time_min` | `--time` | `-l h_rt` | `-l walltime` |

If the LLM doesn't specify resources, the `HPC_DEFAULT_*` values are used.

## Path 2 — CGAT-core executor (`--executor cgat`)

If your site already runs CGAT-core, use the cgat executor — it inherits
your existing `.cgat.yml` (cluster queue, memory defaults, parallel
environment).

```bash
flowagent prompt "ChIP-seq workflow" --executor cgat
```

The `.cgat.yml` at the repo root provides defaults for cluster queue,
parallel environment, and per-job overrides. See the cgatcore
documentation for the full schema.

## Path 3 — Nextflow with SLURM profile (recommended for portability)

Generate a Nextflow pipeline and let Nextflow handle submission:

```bash
flowagent prompt "RNA-seq with Kallisto" \
    --pipeline-format nextflow --profile slurm
```

This writes `flowagent_pipeline_output/main.nf` plus a `nextflow.config`
that already contains a `slurm` profile:

```groovy
profiles {
    slurm {
        process.executor = 'slurm'
        process.queue    = 'all.q'
    }
}
```

Edit `nextflow.config` to set queue, account, container engine, etc.
Then re-run with `nextflow run main.nf -profile slurm` directly, or let
FlowAgent invoke it.

## Combining with error recovery

The error-recovery loop works identically for all three paths. If a
SLURM job exits non-zero, the LLM sees the actual job stderr (recovered
from the scheduler's output file), proposes a fix, and the corrected
step is re-submitted. See [Error Recovery](error-recovery.md).

## Live progress

Both Nextflow and Snakemake executors stream their output directly to
your terminal so you can watch SLURM job IDs and per-task completion
in real time. The HPC executor logs each submission ID and polls until
all jobs complete.

## Inspecting submitted jobs

```bash
# Per-step logs (any executor)
ls flowagent_pipeline_output/logs/

# SLURM job details
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode

# Nextflow trace
nextflow log
```

## Resource overrides on the CLI

You can pin global resource limits regardless of what the LLM produces:

```bash
flowagent prompt "..." --executor hpc \
    --memory 32G --threads 16
```

These are applied as floor values: a step requesting more than the floor
keeps its larger request.

## When NOT to use HPC

For small datasets (< 1 GB total FASTQ, < 1 hour wall time), the cluster
submission overhead often costs more than running locally. Stick with
`--executor local` (the default) for those cases.

## See also

- [Execution Backends](executors.md) — full matrix including Kubernetes
- [Pipeline Generation](pipeline-generation.md) — Nextflow / Snakemake details
- [Error Recovery](error-recovery.md) — how recovery works on the cluster
