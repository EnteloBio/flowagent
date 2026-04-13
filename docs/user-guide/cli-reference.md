# CLI Reference

The full surface area of the `flowagent` command. There are two
sub-commands: **`prompt`** (the one you'll use 99% of the time) and
**`serve`** (Chainlit web UI).

## `flowagent prompt`

```
flowagent prompt "<text>" [options]
```

### Mode selection (mutually exclusive)

| Flag | Effect |
|---|---|
| `--workflow` | Force workflow mode (plan → execute pipeline) |
| `--agent` | Force agent mode (interactive tool-calling loop) |

If neither is supplied, FlowAgent inspects the prompt + other flags and
routes automatically. See [Smart routing](#smart-routing) below.

### Workflow planning

| Flag | Type | Default | Effect |
|---|---|---|---|
| `--preset NAME` | string | none | Skip LLM planning; load a preset from the catalogue. Valid: `rnaseq-kallisto`, `rnaseq-star`, `chipseq`, `atacseq`. |
| `--non-interactive` | bool | false | Don't prompt for organism/reference; use defaults (human / GRCh38 / Ensembl). |

### Pipeline generation

| Flag | Type | Default | Effect |
|---|---|---|---|
| `--pipeline-format FMT` | `nextflow`/`snakemake` | none | Generate a portable pipeline file. See [Pipeline Generation](pipeline-generation.md). |
| `--profile NAME` | string | `local` | Nextflow profile (`local`, `docker`, `singularity`, `slurm`). |
| `--no-execute` | bool | false | Generate the pipeline file but don't run it. |

### Execution backend

| Flag | Type | Default | Effect |
|---|---|---|---|
| `--executor TYPE` | enum | `local` | One of `local`, `cgat`, `hpc`, `kubernetes`, `nextflow`, `snakemake`. See [Executors](executors.md). |
| `--hpc-system NAME` | enum | none | One of `slurm`, `sge`, `torque` — only with `--executor hpc`. |

### Resume / checkpointing

| Flag | Type | Default | Effect |
|---|---|---|---|
| `--checkpoint-dir DIR` | path | `<output>/.checkpoint` | Where to write `checkpoint.json` after each step. |
| `--resume` | bool | false | Resume from `--checkpoint-dir`; smart-resume skips steps whose outputs already exist. |
| `--force-resume` | bool | false | Like `--resume` but re-runs every step regardless of completion state. |

### Analysis

| Flag | Type | Default | Effect |
|---|---|---|---|
| `--analysis-dir DIR` | path | none | Run an LLM analysis report over an existing results directory. Forces agent mode. |

---

## `flowagent serve`

Launch the Chainlit-based web interface.

```
flowagent serve [--host HOST] [--port PORT]
```

| Flag | Default |
|---|---|
| `--host` | `0.0.0.0` |
| `--port` | `8000` |

See [Web Interface](web-interface.md).

---

## Smart routing

When neither `--workflow` nor `--agent` is given, the CLI picks a mode
based on the prompt:

- **Workflow mode** is forced by any of: `--preset`, `--pipeline-format`,
  `--resume`, `--checkpoint-dir`, or strong workflow keywords ("run",
  "execute", "kallisto", "chip-seq", "alignment", ...).
- **Agent mode** is used for exploratory prompts ("check which tools are
  installed", "what files are here?") or whenever `--analysis-dir` is set.
- Ambiguous prompts default to agent mode (safe — no side effects).

To always be explicit, pass the flag yourself.

---

## Environment variables

These supplement / override CLI flags. The CLI flag wins if both are set.

### LLM provider

```bash
LLM_PROVIDER=openai          # openai | anthropic | google | ollama
LLM_MODEL=gpt-4.1            # see config/models.yaml or provider docs
LLM_FALLBACK_MODEL=gpt-4o    # used if primary model fails
LLM_BASE_URL=...             # custom endpoint (Ollama, vLLM, OpenAI-compatible)

OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

See [LLM Providers](llm-providers.md).

### Execution

```bash
EXECUTOR_TYPE=local          # default executor when --executor not given
HPC_SYSTEM=slurm             # default for --executor hpc
HPC_QUEUE=all.q
HPC_DEFAULT_MEMORY=4G
HPC_DEFAULT_CPUS=1
HPC_DEFAULT_TIME=60
PIPELINE_PROFILE=local       # default for --profile (Nextflow)
KUBERNETES_ENABLED=false     # safety switch
KUBERNETES_NAMESPACE=default
KUBERNETES_JOB_TTL=3600
```

### Other

```bash
SECRET_KEY=...                # JWT signing for the web UI
FLOWAGENT_DATA_DIR=./data     # default search path for input files
LOG_LEVEL=INFO                # DEBUG | INFO | WARNING | ERROR
```

See [`.env.example`](https://github.com/cribbslab/flowagent/blob/main/.env.example)
for the authoritative list.

---

## Examples

```bash
# 1. Smart routing (workflow mode picked automatically)
flowagent prompt "Run RNA-seq with Kallisto on data/*.fastq.gz"

# 2. Skip LLM with a preset
flowagent prompt "go" --preset rnaseq-kallisto

# 3. Generate Nextflow pipeline, don't run
flowagent prompt "ChIP-seq with Bowtie2 + MACS2" \
    --pipeline-format nextflow --no-execute

# 4. SLURM submission via the HPC executor
flowagent prompt "RNA-seq with Kallisto" \
    --executor hpc --hpc-system slurm

# 5. Resume after a failure (smart-resume skips completed steps)
flowagent prompt "Continue RNA-seq" \
    --checkpoint-dir wf_state --resume

# 6. Generate a Snakemake pipeline using Anthropic
LLM_PROVIDER=anthropic LLM_MODEL=claude-sonnet-4-20250514 \
flowagent prompt "ATAC-seq paired-end" --pipeline-format snakemake

# 7. Analyze an existing results directory
flowagent prompt "summarise QC" --analysis-dir results/

# 8. Web interface
flowagent serve --port 8001
```
