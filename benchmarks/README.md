# FlowAgent Benchmarks

Reproducible benchmarks that measure FlowAgent's core claims: natural-language
**planning correctness**, **adaptive error recovery**, **generator fidelity**,
and **executor coverage**. Drive the manuscript figures.

## Layout

```
benchmarks/
├── config/
│   ├── models.yaml         # LLMs to sweep (OpenAI, Anthropic, Google, ...)
│   └── faults.yaml         # Fault catalogue (for Benchmark B)
├── corpus/
│   └── prompts.yaml        # 23 benchmark prompts with expected properties
├── harness/                # Shared helpers
│   ├── runner.py           # Provider switching, sweep, .env loader
│   ├── metrics.py          # Scoring functions
│   ├── fault_inject.py     # Fault implementations
│   ├── env_detect.py       # Which executor backends are live-testable
│   ├── executor_probes.py  # Per-backend probes for Benchmark D
│   └── plot.py             # Publication-ready figures
├── bench_planning.py       # Benchmark A: planning correctness
├── bench_recovery.py       # Benchmark B: error recovery (key manuscript figure)
├── bench_generation.py     # Benchmark C: Nextflow/Snakemake codegen fidelity
├── bench_executors.py      # Benchmark D: executor-coverage matrix
├── rescore_planning.py     # Re-evaluate existing plans with updated metrics (no API calls)
├── merge_runs.py           # Combine runs across models/sessions into one CSV
├── Makefile                # Convenience orchestration
└── results/                # Gitignored outputs (CSV, JSON, PDF)
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

## API keys

The harness auto-loads a `.env` file from the repo root (walks up from
`benchmarks/`). Put your keys in there:

```bash
# .env (repo root, already gitignored)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

No need to `source` or `export` — the harness picks them up automatically.
Keys set in the shell win over `.env` (standard dotenv semantics).

## Running

### No API key needed (smoke test everything in ~3 s)

```bash
make smoke
```

### Single model

```bash
make plan     MODEL=gpt-4.1 REPLICATES=3    # Benchmark A, 1 model
make recovery MODEL=gpt-4.1 SEEDS=5         # Benchmark B, 1 model
```

### All models at once

```bash
make plan-all REPLICATES=3
```

Sweeps every model in `config/models.yaml` concurrently (4 at a time).
For the default config that's **4 models × 23 prompts × 3 replicates = 276 cells**,
~30 min wall time, ~$6–8 total.

### Deterministic benchmarks (no API key)

```bash
make gen      # Benchmark C: generator fidelity
make exec     # Benchmark D: executor coverage
```

### Rescore without re-calling the LLM

If the scoring code changes (e.g. after loosening a metric or adding synonym
lists to `prompts.yaml`), you can re-evaluate an existing run without
spending more API budget:

```bash
make rescore
```

This writes `results/planning/<run>/rescored_<ts>/{metrics.csv, results.json}`.

### Combine runs from multiple sessions

If you ran models incrementally (one at a time, different days, etc.),
merge every `results/planning/<ts>/` into a single deduplicated CSV:

```bash
make merge
```

This writes `results/planning/_merged/<ts>/metrics.csv`. The merger prefers
rescored subdirectories over the original results and deduplicates by
`(model, input_id, replicate)` keeping the most recent value — so re-running
a model to fix something cleanly replaces the stale rows.

### Figures

```bash
make report
```

`harness/plot.py::_latest()` looks in this priority order when picking the
data to plot:

1. `results/planning/_merged/<latest>/` (multi-model aggregated)
2. `results/planning/<latest>/rescored_<latest>/` (single run, rescored)
3. `results/planning/<latest>/` (raw single run)

Outputs go to `results/figures/{planning,recovery,generation,executors}.{pdf,png}`.

### Everything at once

```bash
make all      # plan + recovery + gen + exec + report
```

## Typical multi-model workflow

```bash
# 1. Drop keys in .env (or export them)
cat > ../.env <<'EOF'
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
EOF

# 2. Sweep all LLMs
make plan-all REPLICATES=3      # ~30 min, ~$6–8

# 3. (Optional) re-evaluate with current scoring logic
make rescore

# 4. Aggregate into a single CSV
make merge

# 5. Render the comparison figure
make report

# Figure lands at results/figures/planning.pdf
```

## Cost + wall-clock estimates

Rough guide (GPT-4.1 rates per 1k tokens from `config/models.yaml`).

| Target | Models | Wall time | API cost |
|---|---|---|---|
| `make plan` | 1 | ~20 min | ~$2 |
| `make plan-all` | 4 | ~30 min (concurrent) | ~$6–8 |
| `make recovery` | 1 | ~15 min | ~$1 |
| `make gen` | — | <1 min | $0 |
| `make exec` | — | <1 min | $0 |
| `make rescore` | — | ~5 s | $0 |
| `make merge` | — | ~1 s | $0 |

## Reproducibility

Every run writes a `manifest.json` with: git SHA, Python version, installed
package versions, model IDs, timestamp, and a redacted env-var snapshot
(API keys show as `<redacted>`). Prompt corpus (`corpus/prompts.yaml`) and
fault catalogue (`config/faults.yaml`) are version-controlled and immutable
per release. Random seeds are logged per-cell.

## Troubleshooting

**"All cells errored"** — check `metrics.csv`; if `error` contains
`Environment variable ... is required`, the harness didn't find your `.env`.
Verify with:

```bash
python -c "from harness.runner import _DOTENV_PATH; print(_DOTENV_PATH)"
```

If it prints `None`, add a `.env` file to the repo root with your keys.

**`make report` crashes with "No objects to concatenate"** — fixed in the
current version; if you're on an older checkout, pull latest.

**Only 1 row per (model, prompt) in merged CSV even though I ran 3 replicates** — the deduplicator collapses rows only when `(model, input_id, replicate)` are identical. If your replicate numbers got reset, it's rerunning fresh. Check `_source_run` in `metrics.csv` to trace.

## Out of scope (documented explicitly)

- **User study** (wet-lab vs. Nextflow-tutorial): requires IRB + human
  participants; noted as a future extension.
- **Concordance with nf-core published pipelines**: supported by Benchmark C's
  API but requires GB-scale test data; see `extras/concordance.md`.
- **Full live HPC / Kubernetes execution**: Benchmark D uses mocks when infra
  is absent and does live runs when present; production cluster behaviour
  is documented as a reviewer-reproduction path.
- **CGAT-core / WDL / CWL generators**: not implemented in FlowAgent today.
