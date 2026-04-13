# Workflow Presets

Presets are **fully-formed workflow plans** that bypass the LLM entirely.
Use them when you want determinism, zero API cost, or a known-good
baseline.

## Available presets

| Preset ID | Pipeline | Reference needs |
|---|---|---|
| `rnaseq-kallisto` | FastQC → Kallisto index → Kallisto quant → MultiQC | cDNA FASTA |
| `rnaseq-star` | FastQC → STAR index → STAR align → featureCounts → MultiQC | Genome FASTA + GTF |
| `chipseq` | FastQC → Trim Galore → Bowtie2 align → MACS2 → MultiQC | Genome FASTA |
| `atacseq` | FastQC → Trim Galore → Bowtie2 → filter/dedup → MACS2 → MultiQC | Genome FASTA |

The full canonical definitions live in
[`flowagent/presets/catalog.py`](https://github.com/cribbslab/flowagent/blob/main/flowagent/presets/catalog.py).

## Using a preset

```bash
flowagent prompt "go" --preset rnaseq-kallisto --pipeline-format nextflow
```

The prompt text is essentially ignored — the preset's `steps` are loaded
verbatim. You can still combine `--preset` with:

- `--pipeline-format nextflow` / `snakemake` to emit pipeline files
- `--executor hpc` / `kubernetes` / etc. to choose the backend
- `--no-execute` to just generate the file
- `--checkpoint-dir` and `--resume` for restart semantics

## Reference handling

If a preset declares it needs a reference (`reference_needs.cdna: true`),
FlowAgent inspects the cwd. If a local file is found
(`reference/transcriptome.fa`), it's used directly. Otherwise,
`apply_context_to_preset()` **prepends a download step** using the
correct URL for the chosen organism + source (Ensembl by default,
Gencode optional).

You can override organism/source non-interactively:

```bash
flowagent prompt "go" --preset rnaseq-kallisto --non-interactive
# uses defaults: human, GRCh38, Ensembl
```

## Programmatic access

```python
from flowagent.presets.catalog import (
    list_presets, get_preset, apply_context_to_preset,
)
from flowagent.core.schemas import PipelineContext

# List
for p in list_presets():
    print(p["id"], "-", p["description"])

# Fetch
plan = get_preset("rnaseq-kallisto")
print(plan["workflow_type"], len(plan["steps"]))

# Inject reference download steps
ctx = PipelineContext(
    organism="mouse", genome_build="GRCm39",
    reference_url="https://ftp.ensembl.org/pub/release-113/fasta/mus_musculus/cdna/Mus_musculus.GRCm39.cdna.all.fa.gz",
)
plan = apply_context_to_preset(get_preset("rnaseq-kallisto"), ctx)
```

## When to use a preset vs. an LLM prompt

| Use a preset | Use an LLM prompt |
|---|---|
| Standard pipeline, no special tweaks | Custom analysis (different tool, custom flag) |
| Demo / teaching / CI | Exploratory work |
| Want deterministic, byte-identical re-runs | Don't know the canonical commands |
| Want $0 cost | Have an API key and want flexibility |

Presets are also the ground-truth references for FlowAgent's
[Benchmarking](../benchmarking.md) suite — Benchmark C tests that the
generators reproduce them faithfully.

## Adding a custom preset

Add an entry to `flowagent/presets/catalog.py::PRESET_CATALOG`:

```python
PRESET_CATALOG["my-pipeline"] = {
    "name": "My pipeline",
    "description": "What it does",
    "workflow_type": "custom",
    "reference_needs": {"genome": True, "gtf": False, "cdna": False},
    "steps": [
        {
            "name": "create_dirs",
            "command": "mkdir -p results/qc",
            "dependencies": [],
            "outputs": ["results/"],
            "resources": {"cpus": 1, "memory": "1G", "time_min": 5},
        },
        # ...
    ],
}
```

The preset is immediately available via `--preset my-pipeline` and
`get_preset("my-pipeline")`.

## Limitations

- Presets are **plan-level** only. They don't currently support
  per-sample parametrisation — if you have N samples, you'll need to
  expand the steps yourself (or use the LLM to do it from a prompt).
- Presets are NOT validated by `WorkflowPlanSchema` at load time; an
  invalid preset will fail at execution. Run
  `pytest tests/test_workflow.py` after adding a preset.
- Adding a new preset requires a code change. We may move presets to
  YAML in a future release.
