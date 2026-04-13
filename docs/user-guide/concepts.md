# Core Concepts

This page covers the abstractions you'll see throughout the FlowAgent
codebase and CLI. If you've used Nextflow or Snakemake, much of this will
be familiar; the FlowAgent twist is **LLM-driven planning + recovery**.

## Workflow plan

A **workflow plan** is a JSON dict that fully describes what to run. It
matches `WorkflowPlanSchema`:

```json
{
  "workflow_type": "rna_seq_kallisto",
  "steps": [
    {
      "name": "fastqc",
      "command": "fastqc -t 4 *.fastq.gz -o results/fastqc",
      "dependencies": ["create_results_dirs"],
      "outputs": ["results/fastqc/*_fastqc.html"],
      "description": "QC of raw reads"
    },
    ...
  ]
}
```

Plans come from one of three sources:

| Source | When | LLM calls |
|---|---|---|
| `LLMInterface.generate_workflow_plan(prompt)` | natural-language prompt | yes |
| `flowagent.presets.catalog.get_preset(id)` | known pipeline (rnaseq-kallisto, etc.) | no |
| User-supplied JSON | programmatic / advanced | no |

## Pipeline context

Before planning, FlowAgent gathers a **`PipelineContext`** describing what's
on disk: input files, paired-end status, organism, genome build, local
references, and download URLs for missing references. The context is
passed to the LLM so generated plans use real paths instead of guesses.
See `flowagent/core/pipeline_planner.py`.

## Generators

A **generator** converts a plan into a portable pipeline file:

| Generator | Output | Validates with |
|---|---|---|
| `NextflowGenerator` | `main.nf` (DSL2) + `nextflow.config` | `nextflow run -preview` |
| `SnakemakeGenerator` | `Snakefile` + `config.yaml` + `envs/*.yaml` | `snakemake --lint`, `snakemake -n` |

Both follow the same `PipelineGenerator` abstract interface
(`generate(plan)`, `validate(code)`, `default_filename()`).

CGAT-core, HPC, Kubernetes, and local pipelines do **not** go through a
generator — they consume the plan dict directly via `execute_step()`.

## Executors

An **executor** runs a step (or a generated pipeline). The `ExecutorFactory`
selects one of six backends:

| Type | Backed by | Use case |
|---|---|---|
| `local` | `asyncio.subprocess` | development, small datasets |
| `cgat` | cgatcore | clusters via CGAT-core's queue layer |
| `hpc` | DRMAA / cgatcore | direct SLURM/SGE/TORQUE |
| `kubernetes` | `kubernetes.client.BatchV1Api` | K8s clusters |
| `nextflow` | `nextflow run` subprocess | distributable Nextflow pipelines |
| `snakemake` | `snakemake` subprocess | distributable Snakemake pipelines |

See [Execution Backends](executors.md) for the full matrix and selection
guidance.

## LLM interface and providers

`LLMInterface` is the entry point for all LLM calls. It delegates to a
provider abstraction (`flowagent/core/providers/`) supporting:

- `openai` — GPT-4.1, GPT-4o, o1/o3/o4 series
- `anthropic` — Claude Sonnet/Opus
- `google` — Gemini 2.5
- `ollama` — any local model

Switch providers with `LLM_PROVIDER` + `LLM_MODEL` env vars (or `--model`
on the CLI). See [LLM Providers](llm-providers.md).

## Error recovery loop

When a step fails, `WorkflowManager._attempt_error_recovery` packages the
failure context (exit code, stderr, command, tool availability, platform)
and asks the LLM for a fixed command:

```
{"diagnosis": "wget not on macOS", "fixed_command": "curl -fSL -o ...",
 "explanation": "substituted curl"}
```

The fixed command is re-executed; if it still fails, recovery recurses up
to 3 attempts. The same loop also wraps `--pipeline-format` runs, where
the LLM can patch the entire workflow plan and regenerate the pipeline
file. See [Error Recovery](error-recovery.md).

## Smart resume + checkpoints

After every step, FlowAgent writes `<output>/.checkpoint/checkpoint.json`
with completed step names. On `--resume`, `smart_resume.py` does *output
detection* — it inspects each step's expected outputs and skips steps
whose outputs already exist. This is faster than the checkpoint-only model
and survives lost checkpoints.

Pass `--force-resume` to re-run everything regardless of completion.

## Workflow presets

A **preset** is a fully-formed workflow plan stored in
`flowagent/presets/catalog.py`. Presets bypass the LLM entirely, are
deterministic, and serve as gold references in benchmarks. See
[Workflow Presets](presets.md).

## Two execution modes

The CLI routes prompts through one of two modes:

| Mode | When | What it does |
|---|---|---|
| **Workflow** | preset, `--pipeline-format`, or workflow keywords ("run", "execute", ...) | plan → generate → run |
| **Agent** | exploratory / conversational prompts | tool-calling loop with `list_files`, `check_tool`, `read_file`, etc. |

Force one with `--workflow` or `--agent`.

## Reproducibility manifest

Every workflow run writes `output_dir/workflow_manifest.json` with: git
SHA, FlowAgent version, LLM provider + model, executor type, full plan,
SHA-256 of every output file (first 10 MB). This is the primary artefact
for reviewer reproduction.
