# Execution Backends

FlowAgent's `ExecutorFactory` provides six execution backends. They all
implement the same `BaseExecutor.execute_step(step) -> dict` interface,
so the rest of the system is backend-agnostic.

## At a glance

| `--executor` | Backend | Optional install | Live progress | Use case |
|---|---|---|---|---|
| `local` | `asyncio.subprocess` | none | yes | dev, small datasets |
| `cgat` | cgatcore submission layer | `pip install '.[hpc]'` | yes | CGAT-core managed clusters |
| `hpc` | DRMAA via cgatcore | `pip install '.[hpc]'` + DRMAA lib | yes | direct SLURM/SGE/TORQUE |
| `kubernetes` | `kubernetes.client.BatchV1Api` | `pip install '.[kubernetes]'` | poll | K8s clusters |
| `nextflow` | `nextflow run` subprocess | `nextflow` binary | yes (live ANSI) | distributable Nextflow |
| `snakemake` | `snakemake` subprocess | `snakemake` binary | yes (tee'd to file) | distributable Snakemake |

## Important asymmetry

FlowAgent **generates** two pipeline formats (Nextflow, Snakemake) but
**executes** through six backends. The relationship:

| Format | Generator | Executor |
|---|---|---|
| Nextflow | ✅ | ✅ |
| Snakemake | ✅ | ✅ |
| CGAT-core | ❌ (not implemented) | ✅ (consumes plan dict directly) |
| HPC (SLURM/SGE/TORQUE) | ❌ | ✅ (consumes plan dict) |
| Kubernetes | ❌ | ✅ (consumes plan dict) |
| Local shell | ❌ | ✅ (consumes plan dict) |

The four "executor-only" backends invoke each step's `command` directly
without intermediate codegen. This is faster but less portable than the
generated pipelines.

## Picking the right executor

```
Need to run on... ?
├── My laptop                          → local
├── A SLURM/SGE/TORQUE cluster
│   ├── Already use CGAT-core          → cgat
│   ├── Want native scheduler control  → hpc
│   └── Want a portable artefact       → nextflow + --profile slurm
├── A Kubernetes cluster               → kubernetes
└── Multi-site / regulated environment → nextflow or snakemake
                                          (commit the generated file)
```

## `local`

The default. Each step runs as a `bash -c` subprocess. Stdout/stderr are
captured to `<output>/logs/<step_name>.log`.

```bash
flowagent prompt "..." --executor local   # or omit; this is default
```

No external dependencies. Honours `cwd` from the step dict.

## `cgat`

Submits each step through `cgatcore.pipeline.submit()`. Inherits cluster
config from `.cgat.yml` at the repo root.

```bash
pip install -e ".[hpc]"
flowagent prompt "..." --executor cgat
```

Requires a working CGAT-core install + cluster credentials.

## `hpc`

Direct DRMAA submission. Maps the plan's `resources` block to scheduler
options (see [HPC Configuration](hpc.md) for the per-scheduler mapping).

```bash
pip install -e ".[hpc]"
# Plus DRMAA library matching your scheduler:
#   slurm-drmaa, libdrmaa.so (SGE), pbs-drmaa (TORQUE)

flowagent prompt "..." --executor hpc --hpc-system slurm
```

Each step becomes one job. Job IDs are logged; failures fall through to
the [error recovery loop](error-recovery.md) which sees the actual job
stderr.

## `kubernetes`

Each step becomes a Kubernetes `Job` via `BatchV1Api.create_namespaced_job()`.
Container images come from the same registry as the Nextflow generator;
falls back to `bash:latest` for shell steps.

```bash
pip install -e ".[kubernetes]"
export KUBERNETES_ENABLED=true
export KUBERNETES_NAMESPACE=flowagent
flowagent prompt "..." --executor kubernetes
```

Honours `KUBERNETES_JOB_TTL` for cleanup. Logs are retrieved via
`CoreV1Api.read_namespaced_pod_log()`.

## `nextflow`

Generates a `main.nf` (via `NextflowGenerator`) and runs it through
`nextflow run main.nf -profile <profile> -resume`. This is the
recommended path for HPC + container environments.

```bash
flowagent prompt "..." --pipeline-format nextflow --profile slurm
```

Live ANSI progress is shown in your terminal; on failure,
`.nextflow.log` is read for error recovery.

## `snakemake`

Generates a `Snakefile` (via `SnakemakeGenerator`) and runs
`snakemake --cores N -s Snakefile --use-conda`. Per-rule conda envs are
auto-created.

```bash
flowagent prompt "..." --pipeline-format snakemake
```

Output is `tee`-ed to `snakemake.log` for both live display and post-mortem.
`set -o pipefail` is enabled so the actual snakemake exit code propagates
through the `tee` pipe (this matters for failure detection).

## Programmatic instantiation

```python
from flowagent.core.executor_factory import ExecutorFactory

# Local
exe = ExecutorFactory.create("local")

# Nextflow with a custom profile
exe = ExecutorFactory.create("nextflow", profile="docker")

# Snakemake with N cores
exe = ExecutorFactory.create("snakemake", cores=8)

result = await exe.execute_step({
    "name": "fastqc",
    "command": "fastqc reads.fastq.gz -o results/",
    "dependencies": [],
    "outputs": ["results/reads_fastqc.html"],
    "resources": {"cpus": 1, "memory": "1G", "time_min": 5},
})
```

## How execution backends are validated

The [Benchmarking](../benchmarking.md) suite (Benchmark D) tests every
backend at three levels:

1. **Interface compliance** — class instantiates, `execute_step` is async,
   returns a dict with required keys.
2. **Mock submission** — job-spec construction passes with external APIs
   patched.
3. **Live execution** — trivial `echo "ok"` runs end-to-end (only when
   the infrastructure is present).

This guarantees the executor matrix is exercised on every CI run, even
without SLURM or K8s available.

## See also

- [HPC Configuration](hpc.md) — DRMAA setup, scheduler-specific options
- [Pipeline Generation](pipeline-generation.md) — Nextflow / Snakemake details
- [Error Recovery](error-recovery.md) — recovers regardless of backend
