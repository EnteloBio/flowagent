# FlowAgent

**FlowAgent** is a multi-agent framework for automating bioinformatics
workflows. It uses large language models (LLMs) to plan pipelines from
plain-English prompts, generates **Nextflow** or **Snakemake** code,
executes shell steps across **six backends** (local, cgat-core, HPC,
Kubernetes, Nextflow, Snakemake), and **self-heals** failures by feeding
the error back to the LLM and patching the workflow on the fly.

---

## What FlowAgent does

| Stage | Capability |
|---|---|
| **Plan** | Natural-language prompt → structured workflow DAG via OpenAI / Anthropic / Google Gemini / Ollama |
| **Generate** | DAG → DSL2 `main.nf` (Nextflow) or `Snakefile` (Snakemake) |
| **Execute** | Plan dict → run via `local`, `cgat`, `hpc` (SLURM/SGE/TORQUE), `kubernetes`, `nextflow`, or `snakemake` |
| **Recover** | On step failure, ask the LLM to diagnose + fix, then retry (up to 3 attempts) |
| **Report** | LLM-driven analysis report linking outputs to biological interpretation |

---

## Key features

- **Natural-language workflows** — Describe the analysis; the LLM proposes structured steps with commands, dependencies, and resources.
- **Multi-provider LLMs** — Switch between OpenAI, Anthropic Claude, Google Gemini, and local models via Ollama with one env var.
- **Pipeline code generation** — Emit valid, container-aware Nextflow DSL2 or Snakemake `Snakefile` for reproducibility and HPC submission.
- **Six execution backends** — Local subprocess, CGAT-core, native SLURM/SGE/TORQUE, Kubernetes Jobs, Nextflow runtime, Snakemake runtime.
- **LLM-driven error recovery** — When a step fails, FlowAgent automatically asks the LLM to diagnose the error and produce a fixed command. Recovers from missing tools, wrong flags, output-collision bugs, and shell-escaping issues.
- **Workflow presets** — Curated, version-controlled plans for common pipelines (`rnaseq-kallisto`, `rnaseq-star`, `chipseq`, `atacseq`).
- **Smart resume + checkpoints** — Skip completed steps on re-run via output detection; full checkpoint/resume from any failure point.
- **Web UI** — `flowagent serve` launches a Chainlit-based chat interface.
- **Reproducible benchmarks** — A `benchmarks/` suite with prompt corpus, fault catalogue, and figure generation.

---

## Quick install

```bash
git clone https://github.com/cribbslab/flowagent.git
cd flowagent
pip install -e .

# Optional dependency groups
pip install -e ".[hpc]"          # cgatcore, DRMAA
pip install -e ".[kubernetes]"   # kubernetes client
pip install -e ".[dev]"          # pytest, mypy, ruff
```

See [Installation](getting-started/installation.md) for full instructions
and [LLM Providers](user-guide/llm-providers.md) for API-key configuration.

---

## Quick start (CLI)

```bash
# Set your provider + key
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...

# Generate + run a Kallisto RNA-seq pipeline as Nextflow
flowagent prompt "RNA-seq analysis with Kallisto" --pipeline-format nextflow

# Use a preset (skips LLM planning entirely — deterministic + free)
flowagent prompt "run it" --preset rnaseq-kallisto

# Generate the pipeline file but don't execute it
flowagent prompt "ChIP-seq with Bowtie2 + MACS2" \
    --pipeline-format snakemake --no-execute

# Resume from a checkpoint after a previous failure
flowagent prompt "Continue RNA-seq" --checkpoint-dir workflow_state --resume
```

The full flag reference is in the [CLI Reference](user-guide/cli-reference.md).

---

## How it fits together

```
                  ┌──────────────────────────┐
   user prompt ──▶│  PipelinePlanner +       │
                  │  LLMInterface (any LLM)  │
                  └────────────┬─────────────┘
                               │  WorkflowPlan dict
            ┌──────────────────┼──────────────────────┐
            ▼                  ▼                      ▼
   ┌─────────────────┐  ┌────────────────┐  ┌──────────────────┐
   │ NextflowGen.    │  │ SnakemakeGen.  │  │ Direct execution │
   │ → main.nf       │  │ → Snakefile    │  │ via plan dict    │
   └────────┬────────┘  └────────┬───────┘  └────────┬─────────┘
            │                    │                    │
            ▼                    ▼                    ▼
   ┌─────────────────────────────────────────────────────────┐
   │       ExecutorFactory.create(executor_type)             │
   │  local | cgat | hpc | kubernetes | nextflow | snakemake │
   └────────────────────────┬────────────────────────────────┘
                            │  step result
                            ▼
                  ┌─────────────────────┐
                  │  Error recovery     │
                  │  loop (LLM-driven)  │
                  └─────────────────────┘
```

---

## Where to next

- **Just trying it out?** → [Quick Start](getting-started/quickstart.md)
- **Switching LLM providers?** → [LLM Providers](user-guide/llm-providers.md)
- **Need an HPC pipeline?** → [HPC Configuration](user-guide/hpc.md) + [Execution Backends](user-guide/executors.md)
- **Want to skip the LLM?** → [Workflow Presets](user-guide/presets.md)
- **Generating Nextflow / Snakemake?** → [Pipeline Generation](user-guide/pipeline-generation.md)
- **Worried about reliability?** → [Error Recovery](user-guide/error-recovery.md)
- **Reproducing the manuscript figures?** → [Benchmarking](benchmarking.md)
- **Adding custom analysis steps?** → [Custom Scripts](custom_scripts/index.md)

---

## License

GPL-3.0 — see [LICENSE](https://github.com/cribbslab/flowagent/blob/main/LICENSE).

## Citation

```bibtex
@software{flowagent2025,
  title  = {FlowAgent: A Multi-Agent Framework for Bioinformatics Workflows},
  author = {Cribbs Lab},
  year   = {2025},
  url    = {https://github.com/cribbslab/flowagent}
}
```
