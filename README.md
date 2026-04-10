# FlowAgent

FlowAgent is a multi-agent framework for automating bioinformatics workflows. It uses large language models (LLMs) to plan pipelines from natural language, execute shell steps locally or on HPC (including **cgat-core** / SLURM), optionally generate **Nextflow** or **Snakemake** pipelines, and produce QC-focused analysis reports.

**Version:** 0.2.0 (see `pyproject.toml`).

---

## Table of contents

1. [Features](#features)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [LLM providers](#llm-providers)
6. [Command-line interface](#command-line-interface)
7. [Pipeline formats: shell, Nextflow, Snakemake](#pipeline-formats-shell-nextflow-snakemake)
8. [Execution backends](#execution-backends)
9. [Web interface (Chainlit)](#web-interface-chainlit)
10. [Workflow presets](#workflow-presets)
11. [Analysis reports](#analysis-reports)
12. [Checkpoints and resume](#checkpoints-and-resume)
13. [Custom scripts](#custom-scripts)
14. [HPC and cluster notes](#hpc-and-cluster-notes)
15. [Architecture](#architecture)
16. [Development](#development)
17. [Documentation (MkDocs)](#documentation-mkdocs)
18. [Contributing and license](#contributing-and-license)

---

## Features

- **Natural-language workflows** — Describe RNA-seq, ChIP-seq, ATAC-seq, Hi-C, single-cell, and other analyses; the LLM proposes structured steps (commands, dependencies, resources).
- **Multiple LLM backends** — OpenAI, Anthropic Claude, Google Gemini, and local models via **Ollama** (OpenAI-compatible API), selected with `LLM_PROVIDER` and related env vars.
- **Tool-calling agent loop** — Optional interactive mode (Chainlit `/Agent`) where the model can list files, check binaries, run commands, and read/write files before finalizing a plan.
- **Portable pipelines** — Generate **Nextflow** DSL2 (`main.nf` + `nextflow.config`) or **Snakemake** (`Snakefile` + `config.yaml`) from the same workflow plan, with optional validation and execution.
- **Flexible execution** — Local subprocess execution, **SLURM** via raw `sbatch`, **cgat-core** cluster submission, **DRMAA** (SGE/TORQUE), **Kubernetes** jobs, or delegated **Nextflow** / **Snakemake** runs.
- **DAG-aware scheduling** — Workflow steps are organized as a DAG; independent steps can run in parallel where the executor supports it.
- **Smart resume** — Skip steps that already completed based on outputs and logs.
- **Reports** — JSON/HTML execution reports plus LLM-assisted narrative analysis (FastQC, MultiQC, Kallisto, logs).
- **Presets** — Curated shell-style workflow templates (e.g. Kallisto RNA-seq, STAR, ChIP-seq, ATAC-seq) under `flowagent/presets/` for reproducible plans without an LLM call.
- **Custom scripts** — Drop R/Python/Bash tools into `flowagent/custom_scripts/` with `metadata.json` for discovery.

---

## Requirements

- **Python** 3.10+ (CI tests 3.10–3.11; conda env pins 3.11).
- An **API key** (or local stack) for your chosen LLM provider.
- Bioinformatics tools are **not** bundled; install what your workflows need (e.g. Kallisto, FastQC, MultiQC) or use conda/docker profiles for generated pipelines.

---

## Installation

### From the repository (recommended for development)

```bash
git clone https://github.com/cribbslab/flowagent.git
cd flowagent

# Optional: full conda env with many bio tools (see conda/environment/environment.yml)
conda env create -f conda/environment/environment.yml
conda activate flowagent

# Editable install with core dependencies
pip install -e .

# Optional dependency groups
pip install -e ".[hpc]"        # cgatcore, drmaa (cluster)
pip install -e ".[kubernetes]" # Kubernetes executor
pip install -e ".[web]"       # Chainlit UI
pip install -e ".[dev]"       # pytest, linters, etc.

flowagent --help
```

Core dependencies are listed in [`pyproject.toml`](pyproject.toml). Legacy `setup.py` remains for compatibility; **`pip install -e .`** resolves dependencies from `pyproject.toml`.

### Verify bio tools (optional)

```bash
kallisto version
fastqc --version
multiqc --version
```

---

## Configuration

Copy the template and edit values:

```bash
cp .env.example .env
```

The following is a **conceptual map** of the main variables. For the exact list and comments, see [`.env.example`](.env.example).

| Area | Variables (examples) | Purpose |
|------|----------------------|---------|
| **LLM** | `LLM_PROVIDER`, `LLM_MODEL`, `LLM_FALLBACK_MODEL`, `LLM_BASE_URL` | Provider and model; use `LLM_BASE_URL` for Ollama (e.g. `http://localhost:11434/v1`). |
| **API keys** | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` | Set the key that matches `LLM_PROVIDER`. |
| **Legacy OpenAI** | `OPENAI_MODEL`, `OPENAI_FALLBACK_MODEL`, `OPENAI_BASE_URL` | Still read for backward compatibility; prefer `LLM_*` for new setups. |
| **Pipeline** | `PIPELINE_FORMAT`, `PIPELINE_PROFILE`, `CONTAINER_ENGINE`, `AUTO_EXECUTE_PIPELINE` | Default shell vs Nextflow vs Snakemake; Nextflow profile; container strategy. |
| **Executor** | `EXECUTOR_TYPE`, `HPC_SYSTEM`, `HPC_QUEUE`, `HPC_DEFAULT_*` | Where shell steps run: `local`, `cgat`, `hpc`, `kubernetes`, `nextflow`, `snakemake`. |
| **Kubernetes** | `KUBERNETES_ENABLED`, `KUBERNETES_NAMESPACE`, `KUBERNETES_IMAGE`, … | Enable and tune K8s jobs when using the kubernetes executor. |
| **Agents / HTTP** | `MAX_RETRIES`, `TIMEOUT`, `AGENT_*`, `WORKFLOW_TIMEOUT` | Retries and timeouts for LLM and workflow runs. |
| **App** | `SECRET_KEY`, `ENVIRONMENT`, `DEBUG`, `LOG_LEVEL` | Security and logging. |

**Shell workflows** read `EXECUTOR_TYPE` from the environment when you use `WorkflowManager` programmatically. The **CLI** can override with `--executor` and `--hpc-system` (see below).

---

## LLM providers

Set `LLM_PROVIDER` to one of: `openai`, `anthropic`, `google`, `ollama`.

| Provider | Typical `LLM_MODEL` | API key / notes |
|----------|---------------------|-----------------|
| **openai** | `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `o3`, `o4-mini` | `OPENAI_API_KEY`; optional `OPENAI_BASE_URL` for proxies. |
| **anthropic** | `claude-sonnet-4-20250514`, `claude-opus-4-20250514` | `ANTHROPIC_API_KEY`; install `anthropic`. |
| **google** | `gemini-2.5-flash`, `gemini-2.5-pro` | `GOOGLE_API_KEY`; install `google-genai`. |
| **ollama** | `llama4`, `mistral`, `qwen`, `deepseek` | Usually `LLM_BASE_URL=http://localhost:11434/v1`; key can be a placeholder. |

Install optional provider packages if needed:

```bash
pip install anthropic google-genai
```

Workflow planning and analysis still expect **JSON-shaped** answers from the model; stronger models (e.g. GPT-4 class, Claude Sonnet, Gemini Pro/Flash) generally give more reliable pipeline JSON than small local models.

---

## Command-line interface

The entry point is **`flowagent`**. Most work goes through the **`prompt`** subcommand.

```bash
flowagent prompt "<natural language>" [options]
flowagent serve [--host HOST] [--port PORT]
```

### `prompt` options

| Option | Description |
|--------|-------------|
| `--checkpoint-dir DIR` | Store / load checkpoints for resume. |
| `--resume` | Resume from checkpoint in `DIR`. |
| `--force-resume` | Run all steps even if some appear complete. |
| `--analysis-dir DIR` | Analyze existing results instead of running a new workflow. |
| `--pipeline-format {shell,nextflow,snakemake}` | Generate shell steps (default path), or emit Nextflow/Snakemake under `flowagent_pipeline_output/`. |
| `--profile NAME` | Nextflow profile (e.g. `local`, `docker`, `singularity`, `slurm`). |
| `--no-execute` | With Nextflow/Snakemake: write files only, do not run the engine. |
| `--executor {local,cgat,hpc,kubernetes,nextflow,snakemake}` | Override `EXECUTOR_TYPE` for this process (shell path uses internal executor wiring). |
| `--hpc-system {slurm,sge,torque}` | Intended for HPC configuration (see settings). |

### Examples

```bash
# Shell workflow (default): plan and run via existing workflow stack
flowagent prompt "Analyze RNA-seq with Kallisto; FASTQs in . ; reference transcriptome at ref.fa" \
  --checkpoint-dir workflow_state

# Resume
flowagent prompt "same analysis" --checkpoint-dir workflow_state --resume

# Analyze outputs on disk
flowagent prompt "summarize QC and quantification" --analysis-dir results/

# Nextflow: generate + validate + run (requires `nextflow` on PATH)
flowagent prompt "RNA-seq QC and quantification" --pipeline-format nextflow --profile docker

# Snakemake: generate only
flowagent prompt "variant calling outline" --pipeline-format snakemake --no-execute
```

**Note:** The subcommand is required: use `flowagent prompt "..."`, not `flowagent "..."` alone.

---

## Pipeline formats: shell, Nextflow, Snakemake

- **`shell` (default)** — The LLM returns a JSON plan of shell commands; FlowAgent runs them stepwise (local, SLURM, cgat-core, etc., depending on configuration).
- **`nextflow`** — After planning, a **Nextflow DSL2** scaffold is written (`main.nf`, `nextflow.config`). Container images are chosen heuristically (e.g. BioContainers-style tags) when the first token of a step command matches a known tool. Validation runs `nextflow run … -preview` when `nextflow` is available.
- **`snakemake`** — Writes a **Snakefile** and `config.yaml`; can run `snakemake --lint` and dry-run when `snakemake` is installed.

`PIPELINE_FORMAT` in `.env` sets the default; CLI `--pipeline-format` overrides for a single invocation.

---

## Execution backends

`ExecutorFactory` maps `EXECUTOR_TYPE` (or CLI `--executor`) to:

| Type | Role |
|------|------|
| **local** | `bash -c` per step, logs under the run directory. |
| **slurm** | Legacy `Executor` path: per-step `sbatch` scripts (see `flowagent/core/executor.py`). |
| **cgat** | **cgat-core** `pipeline.submit(..., to_cluster=True)` for cluster jobs (requires cgatcore). |
| **hpc** | SLURM via cgat-core, or SGE/TORQUE via DRMAA where configured. |
| **kubernetes** | Kubernetes `Job` objects (`KUBERNETES_ENABLED` must be true in settings). |
| **nextflow** / **snakemake** | Run the whole generated pipeline as one command (see pipeline section). |

Shell step-by-step execution in `WorkflowManager` still uses the **legacy `Executor`** interface for compatibility (`local` / `slurm`); the factory is used for cluster-capable backends and pipeline drivers. Tune **HPC** defaults with `HPC_*` variables and cluster config files such as `.cgat.yml` where applicable.

---

## Web interface (Chainlit)

```bash
export USER_EXECUTION_DIR="$(pwd)"   # required: working directory for the chat session
flowagent serve --host 0.0.0.0 --port 8000
```

Open the URL shown in the terminal (typically `http://127.0.0.1:8000` or `http://0.0.0.0:8000` — use the **same port** you passed to `--port`).

Commands exposed in the UI:

- **`/Run`** — Parse intent, then run `run_workflow` (checkpoint/resume aware).
- **`/Analyse`** — Point at a results directory; runs `analyze_workflow`.
- **`/Agent`** — Interactive loop with **tool calling** (list files, check tools, run commands, read/write files) before answering.

---

## Workflow presets

Validated template plans live in **`flowagent/presets/`** (see `catalog.py`). Use them from Python when you want a fixed graph without calling the LLM:

```python
from flowagent.presets import get_preset, list_presets

print(list_presets())
plan = get_preset("rnaseq-kallisto")  # dict with "steps", "workflow_type", ...
```

You can pass `plan` into your own runner or feed it to the Nextflow/Snakemake generators in `flowagent.core.pipeline_generator`.

---

## Analysis reports

Point the CLI at an output directory:

```bash
flowagent prompt "analyze workflow results" --analysis-dir /path/to/results
```

The tool searches for FastQC, MultiQC, Kallisto outputs, logs, and similar artifacts, then combines rule-based extraction with an LLM narrative. In the **Chainlit** flow, whether the report is saved to disk is controlled via the parsed prompt (`save_report`), not a separate CLI flag.

---

## Checkpoints and resume

```bash
flowagent prompt "Your experiment" --checkpoint-dir my_run_state
flowagent prompt "Your experiment" --checkpoint-dir my_run_state --resume
```

The checkpoint directory should contain `checkpoint.json` and align with how `WorkflowManager.resume_workflow` expects workflow metadata. **Smart resume** can skip steps that already produced expected outputs (see `flowagent/core/smart_resume.py`).

---

## Custom scripts

Place scripts under `flowagent/custom_scripts/<workflow_type>/.../` with a **`metadata.json`** next to each script. The script manager discovers them for template-driven workflows.

**Layout:**

```
flowagent/custom_scripts/
├── rna_seq/normalization/
│   ├── custom_normalize.R
│   └── metadata.json
├── chip_seq/peak_analysis/
│   ├── custom_peaks.py
│   └── metadata.json
├── common/utils/
│   ├── data_cleanup.sh
│   └── metadata.json
└── templates/metadata_template.json
```

**`metadata.json` (minimal shape):**

```json
{
  "name": "script_name",
  "description": "What the script does",
  "script_file": "script_name.ext",
  "language": "python",
  "input_requirements": [
    {"name": "counts_matrix", "type": "csv", "description": "Gene count matrix"}
  ],
  "output_types": [
    {"name": "normalized_counts", "type": "csv", "description": "Normalized matrix"}
  ],
  "workflow_types": ["rna_seq"],
  "execution_order": {"before": [], "after": ["alignment"]},
  "requirements": {
    "r_packages": [],
    "python_packages": ["pandas"],
    "system_dependencies": []
  }
}
```

**Script contract:** accept CLI arguments (e.g. `--input_name path`); on success print **JSON** to stdout mapping logical output names to file paths; exit non-zero on error. See `docs/custom_scripts/` for longer examples.

---

## HPC and cluster notes

1. Install optional HPC extras: `pip install -e ".[hpc]"`.
2. Set `EXECUTOR_TYPE=cgat` or `hpc` and configure `HPC_SYSTEM`, `HPC_QUEUE`, and memory/CPU/time defaults in `.env`.
3. For **Nextflow** on a cluster, prefer `--pipeline-format nextflow --profile slurm` (or your site-specific profile) and maintain `nextflow.config` profiles.
4. Ensure **DRMAA** and scheduler libraries match your site if you use `hpc` with SGE/TORQUE.

---

## Architecture

At a high level:

- **`flowagent/core/llm.py`** — Domain prompts, workflow-type heuristics, resource hints; calls the pluggable **`flowagent/core/providers/`** layer for chat completions.
- **`flowagent/core/workflow_manager.py`** — Plans dependencies, runs steps, reports, DAG images.
- **`flowagent/core/executor.py`** — Per-step local/SLURM subprocess execution used by the main shell path.
- **`flowagent/core/executors.py`** — `LocalExecutor`, `CGATExecutor`, `HPCExecutor`, `KubernetesExecutor` for DAG/cluster-style submission.
- **`flowagent/core/executor_factory.py`** — Selects the backend from `EXECUTOR_TYPE` / CLI.
- **`flowagent/core/pipeline_generator/`** — Nextflow and Snakemake codegen.
- **`flowagent/core/agent_loop.py`** + **`tool_definitions.py`** — Tool-calling agent for `/Agent`.
- **`flowagent/workflow.py`** — CLI-oriented `run_workflow` / `analyze_workflow` helpers.
- **`flowagent/web.py`** — Chainlit app.

The shipped package is focused on **CLI + Chainlit**. Any broader “API layer” in docs may refer to future or external integrations; check the codebase for FastAPI/GraphQL if you need HTTP APIs.

---

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Optional: `mypy`, `ruff`, `black`, `isort` as in your team conventions.

---

## Documentation (MkDocs)

```bash
pip install mkdocs mkdocs-material mkdocstrings[python]
mkdocs build
mkdocs serve   # http://127.0.0.1:8000 — use another port if Chainlit already uses 8000
```

---

## Contributing

1. Fork the repository  
2. Create a feature branch  
3. Run tests and linters  
4. Open a pull request  

---

## License

MIT License — see the [LICENSE](LICENSE) file.

---

## Citation

```bibtex
@software{flowagent2025,
  title        = {FlowAgent: A Multi-Agent Framework for Bioinformatics Workflows},
  author       = {Cribbs Lab},
  year         = {2025},
  url          = {https://github.com/cribbslab/flowagent}
}
```

---

## Suggested prompts (shell workflows)

**Paired-end RNA-seq (Kallisto), with checkpointing:**

```bash
flowagent prompt "Analyze RNA-seq: paired reads named *.fastq.1.gz and *.fastq.2.gz in the current directory; reference Homo_sapiens.GRCh38.cdna.all.fa; Kallisto; QC and outputs under results/." \
  --checkpoint-dir workflow_state
```

**Single-end RNA-seq:**

```bash
flowagent prompt "Analyze single-end RNA-seq: *.fastq.gz here; reference Homo_sapiens.GRCh38.cdna.all.fa; Kallisto; QC; save under results/rna_seq_analysis." \
  --checkpoint-dir workflow_state
```

**Single-nuclei (example with kb-python):**

```bash
flowagent prompt "Single-nuclei RNA-seq: paired *.fastq.1.gz / *.fastq.2.gz; genome Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz; GTF Homo_sapiens.GRCh38.105.gtf.gz; use kb-python for nuclei; QC; results/; kb-python via pip if needed." \
  --checkpoint-dir workflow_state
```

**Resume after failure:**

```bash
flowagent prompt "Continue RNA-seq analysis" --checkpoint-dir workflow_state --resume
```

**Force full re-run from checkpoint metadata:**

```bash
flowagent prompt "Re-run everything" --checkpoint-dir workflow_state --resume --force-resume
```

---

## Version compatibility and environment updates

- Kallisto index version checks and workflow metadata are described in code paths that call Kallisto; keep tool versions aligned with your indices.
- Refresh conda:

```bash
conda env update -f conda/environment/environment.yml
```

---

*For the full environment variable reference, always consult [`.env.example`](.env.example) alongside this README.*
