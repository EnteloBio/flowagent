# Quick Start

This walkthrough takes you from "I just installed FlowAgent" to "I have a
working pipeline" in five minutes.

## 1. Set your LLM provider

```bash
export LLM_PROVIDER=openai           # or anthropic, google, ollama
export LLM_MODEL=gpt-4.1
export OPENAI_API_KEY=sk-...
```

See [LLM Providers](../user-guide/llm-providers.md) for all options
(Claude, Gemini, local Ollama).

## 2. Drop your input files in the working directory

```
my_analysis/
├── HBR_Rep1_R1.fastq.gz
├── HBR_Rep1_R2.fastq.gz
├── UHR_Rep1_R1.fastq.gz
└── UHR_Rep1_R2.fastq.gz
```

FlowAgent auto-detects paired-end naming (`_R1`/`_R2` or `_1`/`_2`).

## 3. Generate and run a pipeline

```bash
cd my_analysis
flowagent prompt "RNA-seq analysis with Kallisto" --pipeline-format nextflow
```

What happens, in order:

1. `gather_pipeline_context` scans the cwd for FASTQ files and asks any
   missing questions interactively (organism, reference source).
2. The LLM produces a structured workflow plan (download reference, FastQC,
   kallisto index, kallisto quant, MultiQC).
3. `NextflowGenerator` writes `flowagent_pipeline_output/main.nf` and
   `nextflow.config`.
4. `nextflow run main.nf -profile local -resume` executes; live progress
   appears in your terminal.
5. If any step fails, the **error recovery loop** sends the failure back to
   the LLM, gets a fixed command, regenerates `main.nf`, and retries (up to
   3 attempts). See [Error Recovery](../user-guide/error-recovery.md).

## 4. Skip the LLM with a preset

If you don't want any LLM calls (offline, deterministic, free), use a
preset:

```bash
flowagent prompt "run it" --preset rnaseq-kallisto --pipeline-format nextflow
```

Available presets: `rnaseq-kallisto`, `rnaseq-star`, `chipseq`, `atacseq`.
See [Workflow Presets](../user-guide/presets.md).

## 5. Just generate the pipeline file (no execution)

Useful for inspection or HPC submission:

```bash
flowagent prompt "ChIP-seq with Bowtie2 + MACS2" \
    --pipeline-format snakemake --no-execute
```

The generated `flowagent_pipeline_output/Snakefile` plus `config.yaml`
and `envs/*.yaml` can be committed to a repo and re-run anywhere
Snakemake is installed.

## 6. Resume after a failure

```bash
flowagent prompt "RNA-seq analysis" --checkpoint-dir wf_state
# ... if it fails, fix the issue and:
flowagent prompt "Continue RNA-seq" --checkpoint-dir wf_state --resume
```

Smart resume detects existing outputs and skips completed steps automatically.

## 7. Run the web interface

```bash
flowagent serve
# then open http://localhost:8000
```

A Chainlit-based chat UI for interactive workflow design. See
[Web Interface](../user-guide/web-interface.md).

---

## Where to next

- The complete CLI flag list: [CLI Reference](../user-guide/cli-reference.md)
- Worked examples for each workflow type: [RNA-seq](../workflows/rna-seq.md), [ChIP-seq](../workflows/chip-seq.md), [ATAC-seq](../workflows/atac-seq.md), [Hi-C](../workflows/hi-c.md)
- Picking the right execution backend: [Executors](../user-guide/executors.md)
- Adding your own analysis scripts: [Custom Scripts](../custom_scripts/index.md)
