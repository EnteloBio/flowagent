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
7. [Python Session API](#python-session-api)
8. [Samplesheets](#samplesheets)
9. [Building on FlowAgent](#building-on-flowagent)
10. [Pipeline formats: shell, Nextflow, Snakemake](#pipeline-formats-shell-nextflow-snakemake)
11. [Execution backends](#execution-backends)
12. [Web interface](#web-interface)
13. [MCP server (editor integration)](#mcp-server-editor-integration)
14. [Workflow presets](#workflow-presets)
15. [Analysis reports](#analysis-reports)
16. [Validate outputs and interpret results](#validate-outputs-and-interpret-results)
17. [Notebook output](#notebook-output)
18. [Checkpoints and resume](#checkpoints-and-resume)
19. [Custom scripts](#custom-scripts)
20. [HPC and cluster notes](#hpc-and-cluster-notes)
21. [Architecture](#architecture)
22. [Development](#development)
23. [Benchmarking](#benchmarking)
24. [Documentation (MkDocs)](#documentation-mkdocs)
25. [Contributing and license](#contributing-and-license)

---

## Features

- **Natural-language workflows** — Describe RNA-seq, ChIP-seq, ATAC-seq, Hi-C, single-cell, and other analyses; the LLM proposes structured steps (commands, dependencies, resources).
- **Multiple LLM backends** — OpenAI, Anthropic Claude, Google Gemini, and local models via **Ollama** (OpenAI-compatible API), selected with `LLM_PROVIDER` and related env vars.
- **Plan artifacts** — `flowagent plan` writes a reviewable `workflow.json` + `workflow.md`; `flowagent run` executes a frozen plan without re-prompting the LLM. Plans carry full provenance metadata (model, prompt, timestamp).
- **Inspectable CLI** — `flowagent init`, `plan`, `run`, `status`, `logs`, `diff`, `validate-output`, `interpret` — feels like `samtools` or `nextflow`, not a chat demo.
- **Python Session API** — `from flowagent import Session`; structured `PlanResult` / `RunResult` return types; sync wrapper for Jupyter; event hooks for widgets.
- **Samplesheet-native** — Pass `--samplesheet samplesheet.csv` (nf-core format) and preset commands are auto-expanded to per-sample loops.
- **Tool-calling agent loop** — The model can list files, check binaries, run commands, and read/write files before finalizing a plan.
- **Portable pipelines** — Generate **Nextflow** DSL2 (`main.nf` + `nextflow.config`) or **Snakemake** (`Snakefile` + `config.yaml`) from the same workflow plan, with optional validation and execution.
- **Flexible execution** — Local subprocess execution, **SLURM** via raw `sbatch`, **cgat-core** cluster submission, **DRMAA** (SGE/TORQUE), **Kubernetes** jobs, or delegated **Nextflow** / **Snakemake** runs.
- **DAG-aware scheduling** — Workflow steps are organized as a DAG; independent steps can run in parallel where the executor supports it.
- **Smart resume** — Skip steps that already completed based on outputs and logs.
- **Assay-agnostic reports** — Auto-detects RNA-seq, ChIP/ATAC, and variant-calling outputs and generates assay-appropriate QC summaries (mapping rate, peak counts, Ti/Tv).
- **Output fidelity scoring** — `flowagent validate-output` scores your results against published reference tables using Benchmark F (Spearman ρ, Jaccard, F1).
- **MCP server** — `flowagent mcp serve` exposes all 18 FlowAgent tools as a Model Context Protocol server for Cursor, VS Code, and other editors.
- **Presets** — Curated workflow templates (Kallisto RNA-seq, STAR, ChIP-seq, ATAC-seq) under `flowagent/presets/` for reproducible plans without an LLM call.
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
| **anthropic** | `claude-sonnet-4-6`, `claude-opus-4-8` | `ANTHROPIC_API_KEY`; install `anthropic`. |
| **google** | `gemini-2.5-flash`, `gemini-2.5-pro` | `GOOGLE_API_KEY`; install `google-genai`. |
| **ollama** | `llama4`, `mistral`, `qwen`, `deepseek` | Usually `LLM_BASE_URL=http://localhost:11434/v1`; key can be a placeholder. |

Install optional provider packages if needed:

```bash
pip install anthropic google-genai
```

Workflow planning and analysis still expect **JSON-shaped** answers from the model; stronger models (e.g. GPT-4 class, Claude Sonnet, Gemini Pro/Flash) generally give more reliable pipeline JSON than small local models.

---

## Command-line interface

The entry point is **`flowagent`**. All subcommands are listed below.

```
flowagent {init,plan,run,status,logs,diff,validate-output,interpret,prompt,serve,mcp} ...
```

### Project setup

```bash
# Scaffold a new project (creates data/, reference/, results/, workflow_state/, .env, samplesheet.csv)
flowagent init my_rnaseq
flowagent init my_rnaseq --preset rnaseq-kallisto   # also writes workflow.json + workflow.md
```

### Plan → review → run (recommended)

```bash
# 1. Generate a plan — writes workflow.json + workflow.md, no execution
flowagent plan "RNA-seq with kallisto on data/*.fastq.gz"
flowagent plan "ChIP-seq hg38" --preset chipseq --out chipseq_plan/
flowagent plan "RNA-seq" --samplesheet samplesheet.csv   # per-sample expansion

# 2. Review the plan
cat workflow.md

# 3. Execute the frozen plan (no LLM call)
flowagent run workflow.json
flowagent run workflow.json --executor hpc --hpc-system slurm
flowagent run workflow.json --checkpoint-dir workflow_state --resume
```

### Monitor a run

```bash
flowagent status                                  # reads workflow_state/
flowagent status --checkpoint-dir my_state/

flowagent logs                                    # list available step logs
flowagent logs kallisto_quant                     # full log
flowagent logs kallisto_quant --tail 50           # last 50 lines
```

### Diff two plans

```bash
flowagent diff workflow_v1.json workflow_v2.json
```

### Score and interpret outputs

```bash
# Score outputs against a published fidelity reference (Benchmark F)
flowagent validate-output results/ --case gse52778_dex_de

# LLM interpretation report on any results directory
flowagent interpret results/
flowagent interpret results/ --model gpt-4.1 --no-save
```

### Freeform prompt (smart routing)

```bash
# Smart routing: LLM decides whether to use workflow or agent mode
flowagent prompt "Run RNA-seq analysis on FASTQ files in data/"
flowagent prompt "What bioinformatics tools are installed?" --agent
flowagent prompt "Run kallisto" --workflow --preset rnaseq-kallisto

# Nextflow / Snakemake export
flowagent prompt "RNA-seq QC and quantification" --pipeline-format nextflow --profile docker
flowagent prompt "variant calling outline" --pipeline-format snakemake --no-execute

# Resume
flowagent prompt "same analysis" --checkpoint-dir workflow_state --resume
```

### `prompt` options reference

| Option | Description |
|--------|-------------|
| `--checkpoint-dir DIR` | Store / load checkpoints for resume. |
| `--resume` | Resume from checkpoint. |
| `--force-resume` | Run all steps even if some appear complete. |
| `--analysis-dir DIR` | Analyze existing results instead of running a new workflow. |
| `--pipeline-format {nextflow,snakemake}` | Emit Nextflow/Snakemake under `flowagent_pipeline_output/`. |
| `--profile NAME` | Nextflow profile (e.g. `local`, `docker`, `slurm`). |
| `--no-execute` | With Nextflow/Snakemake: write files only. |
| `--executor {local,cgat,hpc,kubernetes,nextflow,snakemake}` | Override `EXECUTOR_TYPE` for this process. |
| `--hpc-system {slurm,sge,torque}` | HPC scheduler. |
| `--preset ID` | Use a preset workflow (e.g. `rnaseq-kallisto`). |
| `--samplesheet PATH` | nf-core-style CSV; expands preset to per-sample commands. |
| `--model MODEL` | LLM model override (e.g. `gpt-4.1`, `claude-sonnet-4-6`). |
| `--non-interactive` | Skip interactive questions; use defaults. |
| `--no-dag` | DAG-blind planner ablation (Benchmark H). |
| `--validate-output CASE` | After run, score outputs against a fidelity reference. |
| `--interpret` | After run, run LLM interpretation on outputs. |

### Web interface and MCP server

```bash
flowagent serve --host 0.0.0.0 --port 8000    # FastAPI + SSE web UI
flowagent mcp serve                            # MCP tool server (HTTP, port 8765)
flowagent mcp serve --stdio                   # stdio transport for Cursor / VS Code
```

---

## Python Session API

`from flowagent import Session` provides a first-class async API with structured return objects — no stdout parsing required.

```python
from flowagent import Session

async with Session(model="gpt-4.1", data_dir=".", executor="local") as fa:
    ctx  = await fa.inspect()                        # discovers files, organism, pairing
    plan = await fa.plan("RNA-seq → DESeq2", context=ctx)  # PlanResult
    plan = await fa.validate(plan)                   # completeness check
    run  = await fa.run(plan, checkpoint="workflow_state/") # RunResult
    print(run.succeeded, run.output_dir)
    report = await fa.interpret(run.output_dir)      # LLM analysis report
```

**Sync wrapper** for Jupyter notebooks and scripts:

```python
from flowagent import Session

run = Session.run_sync("RNA-seq with kallisto", executor="local")
print(run.output_dir)
```

**Return types:**

| Object | Key attributes |
|--------|----------------|
| `PlanResult` | `.plan` (dict), `.steps` (list), `.plan_path`, `.markdown_path` |
| `RunResult`  | `.status`, `.succeeded`, `.output_dir`, `.steps`, `.raw` |

**Event hooks** (for Jupyter widgets or CI logging):

```python
Session(
    model="gpt-4.1",
    on_step_start=lambda name: print(f"▶ {name}"),
    on_recovery=lambda name, err: print(f"⚠ recovering {name}: {err}"),
    on_token=lambda tok: print(tok, end="", flush=True),
)
```

**Save plan to disk** from the API:

```python
plan = await fa.plan("ChIP-seq", save_to="chipseq_plan/")
# writes chipseq_plan/workflow.json + chipseq_plan/workflow.md
```

**Export to Nextflow / Snakemake:**

```python
nf_path = fa.to_nextflow(plan, output_dir="pipeline/")
sm_path = fa.to_snakemake(plan, output_dir="pipeline/")
```

The legacy `FlowAgent` class is still available as a backwards-compatible alias.

---

## Samplesheets

FlowAgent accepts **nf-core-style samplesheet CSVs**:

```csv
sample,fastq_1,fastq_2,condition
SRR1234,data/SRR1234_R1.fastq.gz,data/SRR1234_R2.fastq.gz,treated
SRR5678,data/SRR5678_R1.fastq.gz,data/SRR5678_R2.fastq.gz,untreated
```

Pass it to any command with `--samplesheet`:

```bash
flowagent plan "RNA-seq with kallisto" --samplesheet samplesheet.csv
flowagent prompt "run RNA-seq" --preset rnaseq-kallisto --samplesheet samplesheet.csv
```

When a samplesheet is provided:
- Input files and paired-end detection come from the sheet rather than filesystem globbing.
- Preset commands that contain glob patterns (`data/*.fastq.gz`) are rewritten to per-sample commands or loops automatically.
- The planner prompt includes a summary of samples and conditions so the LLM doesn't have to guess contrasts.

Single-end samples (empty `fastq_2` column) and tab-delimited files are both handled. Extra columns beyond `sample`, `fastq_1`, `fastq_2`, `condition` are stored and available for downstream use.

**From Python:**

```python
from flowagent.core.samplesheet import load_samplesheet, expand_preset_for_samples
from flowagent.presets.catalog import get_preset

sheet = load_samplesheet("samplesheet.csv")
print(sheet.summary())         # "4 samples (paired-end), conditions: treated, untreated"
plan  = get_preset("rnaseq-kallisto")
plan  = expand_preset_for_samples(plan, sheet)  # per-sample commands
```

---

## Building on FlowAgent

FlowAgent exposes three integration surfaces, so it's straightforward to embed in your own tools, pipelines, or lab infrastructure without coupling to the CLI.

### 1. Python library (`pip install flowagent`)

The cleanest option for Python projects — import directly, no subprocess or HTTP required:

```python
from flowagent import Session

async with Session(model="gpt-4.1") as fa:
    plan = await fa.plan("RNA-seq with kallisto")
    run  = await fa.run(plan)
```

Works in Jupyter, CI scripts, Snakemake `run:` blocks, Nextflow `script:` blocks, Django/Flask views, or any async Python context. The sync wrapper `Session.run_sync(...)` requires no `async`/`await` at all.

### 2. MCP protocol (any MCP-compatible client)

`flowagent mcp serve` starts a [Model Context Protocol](https://modelcontextprotocol.io/) server that any MCP client can call. This includes:

- **Cursor / VS Code** — add to `.cursor/mcp.json` and the agent inside the editor gains bioinformatics planning and execution tools
- **Claude Desktop** — add to `claude_desktop_config.json`
- **Custom clients** — send JSON-RPC 2.0 `tools/call` requests over HTTP

```bash
flowagent mcp serve               # HTTP at http://127.0.0.1:8765/mcp
flowagent mcp serve --stdio       # stdio for native MCP editor wiring
```

The tool list (`GET /mcp/tools`) and a browser-friendly landing page are served at the same port.

### 3. HTTP REST (`POST /mcp`)

The MCP server's HTTP transport is plain JSON-RPC 2.0 — callable from any language without an MCP library:

```bash
# List tools
curl http://127.0.0.1:8765/mcp/tools

# Call a tool
curl -X POST http://127.0.0.1:8765/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"plan_workflow","arguments":{"prompt":"RNA-seq with kallisto"}}}'
```

```python
import httpx, json

r = httpx.post("http://127.0.0.1:8765/mcp", json={
    "jsonrpc": "2.0", "id": 1, "method": "tools/call",
    "params": {"name": "check_tool", "arguments": {"name": "kallisto"}},
})
print(r.json()["result"]["content"][0]["text"])
```

The FastAPI web interface (`flowagent serve`) also auto-generates interactive API docs at `http://127.0.0.1:8000/docs` (Swagger UI) and `http://127.0.0.1:8000/redoc`.

### Comparison

| Surface | Language | Protocol | When to use |
|---------|----------|----------|-------------|
| `Session` Python API | Python only | direct import | Jupyter, CI, Python pipelines |
| MCP server (stdio) | Any MCP client | MCP / JSON-RPC | Cursor, VS Code, Claude Desktop |
| MCP server (HTTP) | Any | plain HTTP + JSON | R scripts, Nextflow, bash, web apps |
| FastAPI web UI | Browser / HTTP | REST + SSE | Interactive sessions, job monitor |

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

## Web interface

```bash
export USER_EXECUTION_DIR="$(pwd)"   # required: working directory for the web session
flowagent serve --host 0.0.0.0 --port 8000
```

Opens a **FastAPI + Server-Sent Events** web UI. Use the same port you passed to `--port` (default 8000). Commands exposed in the UI:

- **`/Run`** — Parse intent, then run `run_workflow` (checkpoint/resume aware).
- **`/Analyse`** — Point at a results directory; runs `analyze_workflow`.
- **`/Agent`** — Interactive loop with **tool calling** (list files, check tools, run commands, read/write files) before answering.

Requires the `web` optional dependency group:

```bash
pip install -e ".[web]"   # fastapi, uvicorn, sse-starlette
```

---

## MCP server (editor integration)

FlowAgent can act as a **Model Context Protocol tool server** so editors like Cursor and VS Code can call it directly — no separate app or chat window required.

```bash
# HTTP transport (default) — browse to http://127.0.0.1:8765 to see the tool list
flowagent mcp serve

# Custom port
flowagent mcp serve --port 9000

# stdio transport (for native MCP editor integration)
flowagent mcp serve --stdio
```

The server at startup prints the `.cursor/mcp.json` snippet to paste:

```json
{
  "mcpServers": {
    "flowagent": {
      "url": "http://127.0.0.1:8765/mcp"
    }
  }
}
```

**Exposed tools (18 total):**

| Category | Tools |
|----------|-------|
| Workflow | `plan_workflow`, `run_workflow`, `export_pipeline`, `analyze_results`, `load_preset`, `check_workflow_status` |
| Agent | `list_files`, `check_tool`, `install_dependency`, `execute_command`, `read_file`, `write_file`, `search_literature`, `download_data`, `plan_workflow` (agent variant), `run_workflow`, `search_files`, `get_file_info` |

Requires the `web` optional dependency group: `pip install -e ".[web]"`.

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

FlowAgent auto-detects the assay type from output files and generates an appropriate report:

| Assay detected | Key metrics reported |
|----------------|---------------------|
| **RNA-seq** | n samples, mean mapping rate, expressed transcripts per sample |
| **ChIP/ATAC** | peak file count, total peaks, FRiP proxy from flagstat |
| **Variant calling** | VCF count, variant count, Ti/Tv ratio |
| **Generic** | file inventory by extension |

```bash
flowagent interpret results/                          # dedicated command
flowagent interpret results/ --model gpt-4.1
flowagent prompt "analyze" --analysis-dir results/   # via prompt
```

Assay-specific recommendations are included (e.g. low mapping rate warning, Ti/Tv out-of-range alert, low peak count warning). Reports are saved to `analysis_report.md` and `agentic_analysis.md` in the results directory unless `--no-save` is passed.

---

## Validate outputs and interpret results

```bash
# Benchmark F: score outputs against a published reference
flowagent validate-output results/ --case gse52778_dex_de

# Benchmark G: LLM interpretation of outputs
flowagent interpret results/
flowagent interpret results/ --model gpt-4.1 --no-save
```

`validate-output` delegates to `benchmarks/bench_fidelity.py` and requires:
1. A finished FlowAgent run with outputs in `results/`.
2. Reference files under `benchmarks/references/` (see `make references` in `benchmarks/`).
3. A case ID from `benchmarks/config/fidelity_cases.yaml`.

`interpret` runs `AgenticAnalysisSystem` and prints an LLM-generated narrative alongside rule-based QC metrics.

---

## Notebook output

Every workflow run emits **three** sibling artefacts next to
`workflow.json` in the output directory:

| File | Role |
|---|---|
| `notebook.ipynb` | Jupyter notebook for re-execution / sharing on GitHub. |
| `notebook.html`  | Rendered HTML of the `.ipynb` — opens in any browser, no Jupyter install needed. |
| `notebook.Rmd`   | R Markdown for users who prefer RStudio. Inline `Rscript -e '...'` invocations are extracted into native `{r}` chunks; everything else is `{bash}`. Knit to HTML/PDF/Word with `rmarkdown::render("notebook.Rmd")`. |

All three share the same content but cater to different downstream
audiences (Jupyter, browser-only, RStudio). The `.ipynb` and `.Rmd` are
built in two layers — a deterministic scaffold and an optional
LLM-generated assay-aware narrative that's woven into the cells.

**Deterministic scaffold (always present):**

- Title cell with run metadata (model, provider, executor, git SHA).
- Original natural-language prompt verbatim.
- Workflow plan rendered as a markdown table.
- One markdown header + `%%bash` code cell per step, with status badges
  (🟢 completed, 🟡 recovered, 🔴 failed, ⚪ skipped) and pre-populated
  stdout/stderr from the actual run.
- Run summary block with counts of completed / recovered / failed steps.
- Trailing "load results" cells auto-detected from disk (DESeq2 results
  CSV, MACS2 narrowPeak, GATK VCF, MultiQC HTML).

**LLM-generated narrative (added on top, when an LLM is configured):**

After execution, FlowAgent makes one additional LLM call that examines
the prompt, the executed plan, and the captured results, and returns
structured JSON containing:

1. **Analysis overview** — 2-3 paragraphs scientific introduction tailored
   to the assay (RNA-seq, ChIP-seq, methylation, variant calling, etc.),
   the dataset accession, and the reference genome.
2. **Per-step rationale** — 1-2 sentences below each step header
   explaining *why* this step exists in this specific pipeline (e.g. why
   kallisto pseudo-alignment was chosen over STAR for this RNA-seq run).
3. **Results interpretation** — 2-3 paragraphs citing specific outputs
   that completed successfully, naming filenames, exit codes, and
   anything that warrants follow-up.
4. **Suggested follow-up analyses** — a short bullet list of 2-3 plausible
   next steps appropriate to the assay, using only the data already in
   hand (or marked otherwise).

This makes the notebook content tailored to the *specific* pipeline that
ran — an RNA-seq notebook reads about transcript quantification and
glucocorticoid signalling; a ChIP-seq notebook reads about peak calling
and motif enrichment.

The narrative call is best-effort: any failure logs a warning and ships
the deterministic notebook unchanged. To disable it (for `--mock`, CI, or
offline use), set:

```bash
export FLOWAGENT_NOTEBOOK_NARRATIVE=0
```

This format is the FlowAgent equivalent of a BixBench capsule —
self-contained, re-executable, and reviewable in JupyterLab without
re-running FlowAgent.

You can also export an existing run on demand (always deterministic;
the narrative path is currently only available via `WorkflowManager`):

```bash
# Jupyter notebook (.ipynb), with .html rendered alongside automatically
python -m flowagent.utils.export_notebook \
    --workflow flowagent_output/Unnamed_Workflow/workflow.json \
    --results  flowagent_output/Unnamed_Workflow/run_results.json \
    --prompt   "Download GSE52778 …" \
    --out      flowagent_output/Unnamed_Workflow/notebook.ipynb

# R Markdown (.Rmd) — knit with `rmarkdown::render()` in R
python -m flowagent.utils.export_rmarkdown \
    --workflow flowagent_output/Unnamed_Workflow/workflow.json \
    --results  flowagent_output/Unnamed_Workflow/run_results.json \
    --prompt   "Download GSE52778 …" \
    --out      flowagent_output/Unnamed_Workflow/notebook.Rmd
```

`--results` is optional in both; without it the notebook is emitted
with empty output sections and is ready to re-execute.

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

```
flowagent/
├── __init__.py               Session + FlowAgent API (PlanResult, RunResult)
├── cli.py                    init / plan / run / status / logs / diff /
│                             validate-output / interpret / prompt / serve / mcp
├── mcp_server.py             JSON-RPC 2.0 MCP server (HTTP + stdio)
├── workflow.py               run_workflow / analyze_workflow helpers
├── core/
│   ├── llm.py                Domain prompts, workflow-type heuristics
│   ├── workflow_manager.py   DAG scheduling, execution, recovery, reports
│   ├── plan_store.py         Save/load/diff plan artifacts (workflow.json)
│   ├── samplesheet.py        nf-core CSV parser + per-sample preset expansion
│   ├── pipeline_planner.py   gather_pipeline_context (files, organism, refs)
│   ├── pipeline_generator/   Nextflow DSL2 + Snakemake codegen
│   ├── completeness.py       Structural plan validator
│   ├── agent_loop.py         Tool-calling agent loop
│   ├── tool_definitions.py   18 agent tools (AGENT_TOOLS + WORKFLOW_TOOLS)
│   ├── executor.py           Local / SLURM subprocess execution
│   ├── executors.py          LocalExecutor, CGATExecutor, HPCExecutor, K8s
│   ├── executor_factory.py   Backend selection from EXECUTOR_TYPE / CLI
│   ├── providers/            OpenAI / Anthropic / Google / Ollama adapters
│   └── schemas.py            Pydantic models for all LLM I/O contracts
├── agents/agentic/
│   ├── analysis_system.py    AgenticAnalysisSystem (assay-aware reports)
│   └── assay_detector.py     Auto-detect rna_seq / chip_atac / variant / generic
├── presets/catalog.py        Preset plans + apply_context + samplesheet expansion
└── web.py                    FastAPI + SSE web UI
```

FlowAgent is designed as a **decomposable workflow operating system**: each agentic layer (DAG planning, completeness reflection, command validation, CoVe verification, execution, recovery, interpretation) is independently toggleable and benchmarked (Benchmarks A–L), and exposes the same capabilities through CLI plan artifacts, a Python `Session` API, and optional MCP integration for editor-native use.
---

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Optional: `mypy`, `ruff`, `black`, `isort` as in your team conventions.

---

## Benchmarking

FlowAgent ships with a reproducible benchmark suite under [`benchmarks/`](benchmarks/)
that measures the seven core claims of the tool: planning correctness,
LLM-driven error recovery, generator fidelity, executor coverage,
head-to-head performance against other agentic bioinformatics systems,
output fidelity against published references, and biological-interpretation
quality. Each benchmark writes timestamped results (JSON + CSV +
`manifest.json`) and the `make report` target renders publication-ready
figures. See [`benchmarks/README.md`](benchmarks/README.md) for the full
reference.

### The seven benchmarks

| ID | Claim tested | API key? | Extra infra? |
|---|---|---|---|
| **A** | FlowAgent generates valid plans from natural-language prompts | required | none |
| **B** | FlowAgent self-heals realistic pipeline faults (28 faults × 3 tiers) | required | none |
| **C** | Generated Nextflow/Snakemake code is valid and preserves plan intent | no | `nextflow`/`snakemake` improve validation |
| **D** | All six execution backends (local, cgat, hpc, kubernetes, nextflow, snakemake) function | no | best-effort; uses mocks when infra absent |
| **E** | FlowAgent is competitive with other agentic bio systems (BioMaster, AutoBA, Biomni) on the same prompt corpus | required | clones of BioMaster + AutoBA (+ Biomni) on disk |
| **F** | FlowAgent's *outputs* match published references (DE tables, peaks, VCFs) — Spearman ρ / Jaccard / F1 | no (pure scorer) | end-to-end FlowAgent run + reference files |
| **G** | The reporting agent answers biological-interpretation MCQs and open-ended questions correctly, and abstains when evidence is insufficient | required | analysis outputs from a prior FlowAgent run |

### Install benchmark-only dependencies

```bash
pip install pandas matplotlib pyyaml networkx tabulate python-dotenv
```

### API keys (auto-loaded from `.env`)

The harness reads a `.env` file from the repo root automatically — no
`source` or `export` needed. Put your keys there:

```bash
# .env (already in .gitignore)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

Shell-set env vars still win over `.env` (standard dotenv semantics), so you
can override a single key per run without editing the file.

### Quick start — smoke test (no API key, ~3 s)

```bash
cd benchmarks
make smoke          # runs every benchmark in --mock mode
```

### Full runs

Every target runs inside `benchmarks/`; results land in
`benchmarks/results/<benchmark>/<timestamp>/`.

```bash
cd benchmarks

# ── Benchmark A — planning correctness & cost ─────────────
make plan       MODEL=gpt-4.1 REPLICATES=3      # one model
make plan-all   REPLICATES=3                     # every model in config/models.yaml

# ── Benchmark B — error recovery (28 faults × N seeds) ────
make recovery   MODEL=gpt-4.1 SEEDS=5
python bench_recovery.py --tier unrecoverable --seeds 10 --model gpt-4.1
python recovery_taxonomy.py                      # classify recovery outcomes

# ── Benchmark C — generator fidelity (no API key) ────────
make gen

# ── Benchmark D — executor coverage (no API key) ─────────
make exec

# ── Benchmark E — head-to-head vs competitors ─────────────
# Requires BIOMASTER_DIR / AUTOBA_DIR / BIOMNI_DIR in .env as needed; see benchmarks/README.md
make competitors MODEL=gpt-4.1 REPLICATES=3

# ── Benchmark F — output fidelity (no LLM calls; pure scorer) ─────
# Score the OUTPUTS of a finished FlowAgent run against a published
# reference table. See config/fidelity_cases.yaml for the case schema;
# drop reference files under references/ before invoking.
python bench_fidelity.py \
    --case gse52778_dex_de \
    --candidate-dir results/realworld_GSE52778 \
    --model gpt-4.1 --replicate 0
# Or bulk-score a directory of <case>__<model>__rep<N> subdirs:
python bench_fidelity.py --bulk-dir results/fidelity_runs

# ── Benchmark G — biological-interpretation quality ──────────
# 5 MCQs + 3 open-ended per dataset (LLM-as-judge for open-ended).
# --mock skips all LLM calls (deterministic stub answers, exercises
# the scoring path; useful for CI smoke tests).
python bench_interpretation.py --model gpt-4.1 --judge gpt-5.4
python bench_interpretation.py --model gpt-4.1 --mock        # smoke

# ── Post-processing (no API calls) ────────────────────────
make rescore    # re-evaluate latest planning run with current scoring logic
make merge      # combine all planning runs into a single deduplicated CSV

# ── Figures ───────────────────────────────────────────────
make report     # auto-uses merged/rescored data when present
make all        # single MODEL end-to-end: plan + recovery + gen + exec + report
make all-sweep  # full sweep: plan-all + recovery + gen + exec + competitors
                # + rescore + merge + report
```

Figures are written to `benchmarks/results/figures/` as both `.pdf`
(for manuscripts) and `.png` (for READMEs).

### Subset runs

You rarely want to sweep every model in one go — the default `plan-all` sweep
is 30 models (40 in the registry; 10 deprecated IDs are skipped):
× 41 prompts × 3 replicates ≈ 4,300 cells. Narrow the sweep with `--models`:

```bash
cd benchmarks

# A cheap pilot across one model per family
python bench_planning.py --replicates=1 \
  --models=gpt-5.4-mini,gpt-4.1-mini,o3-mini,claude-haiku-4-5,claude-sonnet-4-6,gemini-2.5-flash

# A specific prompt against a specific model (fast debugging loop)
python bench_planning.py --prompts=rnaseq_kallisto_basic --model=gpt-4.1 --replicates=1
```

### Typical multi-model workflow

```bash
cd benchmarks

make plan-all REPLICATES=3      # sweep every model in config/models.yaml
make rescore                     # (optional) re-apply current scoring logic
make merge                       # collate into one CSV under results/planning/_merged/
make report                      # figures → results/figures/
```

### Switching LLM providers and models

[`benchmarks/config/models.yaml`](benchmarks/config/models.yaml) ships with
35 non-deprecated models across three families: OpenAI (GPT-5.4, GPT-4.1,
o-series reasoning, legacy 4o/4-turbo/3.5), Anthropic (Opus 4.5/4.6/4.7,
Sonnet 4.5/4.6, Haiku 4.5), Google (Gemini 3.x, 2.5, 1.5 legacy). Each model
lists its provider, API-key environment variable, and per-1k-token pricing.
Make sure the matching key is set (via `.env` or `export`) before sweeping.

To add a local Ollama model or a new provider entry, append to that YAML —
see the comment at the top of the file.

### Python interpreter selection

The Makefile auto-detects `python` or `python3`. Pin a specific interpreter
(e.g. a conda env) with:

```bash
make plan PY=$(which python) MODEL=gpt-4.1
```

### End-to-end real-world case study (Figure 5 of the manuscript)

The suite under [`benchmarks/case_study/`](benchmarks/case_study/) generates
the three-panel case-study figure from a single `flowagent prompt` run
(execution trace + PCA + DESeq2 volcano):

```bash
# Panel 5a — execution trace
python benchmarks/case_study/timeline.py \
    --run-dir benchmarks/results/realworld_GSE52778 \
    --out figures/fig5a_timeline

# Panels 5b + 5c — PCA + volcano (requires R + DESeq2 + patchwork)
cd benchmarks/results/realworld_GSE52778
Rscript ../../case_study/pca_volcano.R \
    --txi results/rna_seq_kallisto/deseq2/txi.rds \
    --sample-sheet sample_conditions.tsv \
    --de-csv results/rna_seq_kallisto/deseq2/deseq2_results.csv \
    --counts results/rna_seq_kallisto/deseq2/gene_counts.csv \
    --out ../../../figures/fig5bc_biology \
    --reference untreated \
    --highlight FKBP5,DUSP1,KLF15,PER1,CRISPLD2,TSC22D3
```

See [`benchmarks/case_study/README.md`](benchmarks/case_study/README.md) for
a LaTeX figure block and full reproduction instructions.

### Rough cost and wall-time estimates

At April-2026 provider rates. Cost scales roughly linearly with the number
of models; `plan-all` is dominated by the premium reasoning tier
(`gpt-5.4-pro`, `o3-pro`, `o1-pro`) — drop those three from `models.yaml`
for a 10× cheaper sweep.

| Target | Cells | Wall time | API cost |
|---|---|---|---|
| `make smoke` | 0 | ~3 s | $0 |
| `make plan` (1 model) | 123 | ~5–15 min | $0.05–$5 (depends on tier) |
| `make plan-all` (30 models) | ~3,700 | ~2–4 h (concurrent) | $150–300 full, $20–40 without premium reasoning |
| `make recovery` (1 model × 5 seeds) | 140 | ~35–45 min | $2–3 |
| `make competitors` (3 systems × 1 model) | ~125 | ~30 min | $3–5 |
| `make gen` | — | <1 min | $0 |
| `make exec` | — | <1 min | $0 |
| `make rescore` | — | ~5 s | $0 |
| `make merge` | — | ~1 s | $0 |
| `make report` | — | ~3 s | $0 |

### Reproducibility

Every run emits a `manifest.json` with:

- Git SHA at run time
- Python version + full `pip freeze` snapshot
- Model IDs used (redacted env var names)
- Random seeds per cell
- Timestamp

See [`benchmarks/README.md`](benchmarks/README.md) for the full catalogue of
prompts (`corpus/prompts.yaml`) and faults (`config/faults.yaml`).

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

GPL-3.0 License — see the [LICENSE](LICENSE) file.

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
