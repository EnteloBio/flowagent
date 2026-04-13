# FlowAgent Benchmarks

Reproducible benchmarks that measure FlowAgent's core claims: natural-language
**planning correctness**, **adaptive error recovery**, **generator fidelity**,
and **executor coverage**. Drive the manuscript figures.

## Layout

```
benchmarks/
├── config/            # Model list + fault catalogue
├── corpus/            # Benchmark prompts and gold plans
├── harness/           # Shared helpers (runner, metrics, fault injection, plots)
├── bench_planning.py  # Benchmark A: planning correctness across LLMs
├── bench_recovery.py  # Benchmark B: error recovery (key manuscript figure)
├── bench_generation.py# Benchmark C: Nextflow / Snakemake codegen fidelity
├── bench_executors.py # Benchmark D: executor-coverage matrix
├── Makefile           # Convenience orchestration
└── results/           # Gitignored outputs (CSV, JSON, PDF)
```

## The four benchmarks

| ID | Claim | Needs API key | Needs infra |
|---|---|---|---|
| **A** | FlowAgent generates valid plans from natural language | yes | no |
| **B** | FlowAgent self-heals faults that break traditional WMS | yes | no |
| **C** | Generated Nextflow / Snakemake is valid and preserves plan intent | no (preset path) | `nextflow` + `snakemake` for `.validate()` |
| **D** | All six execution backends function | no | best-effort — runs in mock mode if infra absent |

### Backend-capability reality

FlowAgent currently **generates** two pipeline formats (Nextflow, Snakemake) but
**executes** through six backends (local, cgat, hpc, kubernetes, nextflow,
snakemake). Benchmark C tests only the two generators; Benchmark D tests all
six executors.

## Running

```bash
# No API key needed for C or D (smoke test everything)
make smoke

# Needs OPENAI_API_KEY (or matching env var for your chosen provider)
make plan      MODEL=gpt-4.1 REPLICATES=3
make recovery  MODEL=gpt-4.1 SEEDS=5

# Deterministic — no API key required
make gen
make exec

# Run every benchmark + build figures into results/figures/
make all
make report
```

## Cost + wall-clock estimates

Rough guide (GPT-4.1 as of 2026-04; rates as in `config/models.yaml`).

| Benchmark | Wall time | API cost |
|---|---|---|
| Planning (24 prompts × 1 model × 3 replicates) | ~20 min | ~$2 |
| Recovery (10 faults × 5 seeds × 1 model) | ~15 min | ~$1 |
| Generation | <1 min | $0 |
| Executors (mock mode) | <1 min | $0 |

## Reproducibility

Every run writes a `manifest.json` with git SHA, Python version, installed
package versions, model IDs, and a redacted env-var snapshot. Prompt corpus
(`corpus/prompts.yaml`) and fault catalogue (`config/faults.yaml`) are version
controlled and immutable per-release. Random seeds are logged per-cell.

## Out of scope (documented explicitly)

- **User study** (wet-lab vs. Nextflow-tutorial): requires IRB + human
  participants; noted as a future extension.
- **Concordance with nf-core published pipelines**: supported by Benchmark C's
  API but requires GB-scale test data; see `extras/concordance.md`.
- **Full live HPC / Kubernetes execution**: Benchmark D uses mocks when infra
  is absent and does live runs when present; production cluster behaviour
  is documented as a reviewer-reproduction path.
- **CGAT-core / WDL / CWL generators**: not implemented in FlowAgent today.
