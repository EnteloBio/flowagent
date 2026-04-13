# Benchmarking

FlowAgent ships a reproducible benchmark suite under
[`benchmarks/`](https://github.com/cribbslab/flowagent/tree/main/benchmarks)
that produces the figures cited in the manuscript. Four benchmarks
measure four distinct claims.

## The four benchmarks

| ID | Claim | Needs API key? | Needs infra? |
|---|---|---|---|
| **A — Planning** | FlowAgent generates valid plans from natural-language prompts | yes | no |
| **B — Recovery** | FlowAgent self-heals realistic faults | yes | no |
| **C — Generation** | Generated Nextflow / Snakemake is valid + preserves plan intent | no | `nextflow` / `snakemake` improve validation |
| **D — Executors** | All six execution backends function | no | best-effort; mocks when infra absent |

## Quick start (no API key, ~3 s)

```bash
cd benchmarks
make smoke
```

This runs every benchmark in `--mock` mode and verifies the harness is
wired correctly. Useful for CI.

## Full runs

```bash
cd benchmarks

# Benchmark A — needs OPENAI_API_KEY (or other provider key)
make plan MODEL=gpt-4.1 REPLICATES=3

# Benchmark B — needs API key
make recovery MODEL=gpt-4.1 SEEDS=5

# Benchmark C — deterministic, no API key
make gen

# Benchmark D — no API key; live mode is best-effort
make exec

# All four + figures
make all
make report
```

Output lands in `benchmarks/results/<benchmark>/<timestamp>/` with a
`metrics.csv`, `results.json`, and `manifest.json`. The `report` target
renders `benchmarks/results/figures/{planning,recovery,generation,executors}.{pdf,png}`.

## Switching LLM providers

The `MODEL` Make variable accepts any model from
`benchmarks/config/models.yaml`. Set the matching API key:

```bash
export OPENAI_API_KEY=sk-...
make plan MODEL=gpt-4.1

export ANTHROPIC_API_KEY=sk-ant-...
make plan MODEL=claude-sonnet-4-20250514
```

See [LLM Providers](user-guide/llm-providers.md).

## Cost + wall-time estimates

Approximate per-run cost (GPT-4.1 pricing, single model):

| Benchmark | Wall time | API cost |
|---|---|---|
| `plan` (24 prompts × 3 replicates) | ~20 min | ~$2 |
| `recovery` (10 faults × 5 seeds) | ~15 min | ~$1 |
| `gen` | <1 min | $0 |
| `exec` (mock mode) | <1 min | $0 |

## What each benchmark measures

### Benchmark A — Planning correctness

For each prompt × model × replicate, scores:

- `plan_valid` — passes `WorkflowPlanSchema.model_validate()`
- `dag_valid` — dependency graph is acyclic + closed
- `type_correct` — `plan["workflow_type"]` matches expected
- `tools_present_fraction` — required tools appear in commands
- `no_forbidden_tools` — banned tools (e.g. `wget` on macOS) absent
- `preset_concordance` — Jaccard / token-F1 vs. gold preset (when available)
- `wall_seconds`, `cost_usd`

Corpus: `benchmarks/corpus/prompts.yaml` (23 prompts across 7 workflow
families, including 3 adversarial cases).

Figure: grouped bar chart, models on x-axis, metrics on y-axis.

### Benchmark B — Error recovery (the manuscript's anchor figure)

For each fault × seed:

1. Provoke the failure with a real `execute_step()` call.
2. Invoke `_attempt_error_recovery` with `max_attempts=3`.
3. Record: `recovered`, `attempts_to_success`, `diagnosis_relevant`,
   `command_changed`, `wall_seconds`.

Fault catalogue: `benchmarks/config/faults.yaml`
(missing binary, typo, wrong flag, missing path, paired/single mismatch,
corrupt FASTQ, permission, shell escaping, multiqc collision, conda pin).

Figure: stacked bars per fault class showing recovered@1 / @2 / @3 / failed.

### Benchmark C — Generation fidelity

For each preset × generator (Nextflow, Snakemake):

- `validation_ok` — passes `nextflow -preview` / `snakemake --lint`
- `step_count_matches` — `len(parsed) == len(plan.steps)`
  (Snakemake's implicit `rule all` is discounted)
- `dag_isomorphic` — generated dependency graph isomorphic to plan's
- `tools_preserved` — every CLI tool from the plan appears in the code
- `regression_launchdir_quoted` (Nextflow) — regression test for the
  path-with-spaces fix

Figure: heatmap (presets × generators × checks).

### Benchmark D — Executor coverage

For each of the 6 executors (local, cgat, hpc, kubernetes, nextflow, snakemake),
graded at three levels:

- **interface_ok** — class instantiates, `execute_step` is async, returns
  the right dict shape
- **mock_ok** — job-spec construction passes with external APIs mocked
  (`unittest.mock.patch` of `kubernetes.client`, DRMAA, etc.)
- **live_ok** — trivial echo step runs end-to-end (only when infra is
  available — `harness/env_detect.py` decides)

This produces a useful matrix on any host: a laptop without SLURM gets
`live_ok=null` for HPC but still tests interface + mock.

Figure: 6-row × 3-column heatmap.

## Reproducibility

Every run writes a `manifest.json`:

```json
{
  "benchmark": "planning",
  "git_sha": "5d9ecbc...",
  "python": "3.13.0",
  "platform": "macOS-14.6-arm64",
  "models": [{"id": "gpt-4.1", "provider": "openai", ...}],
  "packages": {"flowagent": "0.2.0", "openai": "1.50.0", ...},
  "env_snapshot": {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "<redacted>"},
  "timestamp": "2026-04-13T09:30:00+00:00"
}
```

API keys are redacted automatically. The full prompt corpus
(`corpus/prompts.yaml`) and fault catalogue (`config/faults.yaml`) are
under version control — pin the commit SHA in your paper.

## Out of scope (documented in `benchmarks/README.md`)

- **User study** (wet-lab vs. Nextflow tutorial): requires IRB + human
  participants.
- **Concordance with nf-core published pipelines**: supported by
  Benchmark C's framework but requires GB-scale truth data. See
  `benchmarks/extras/concordance.md` for the manual workflow.
- **Production live HPC / K8s execution**: Benchmark D mocks
  scheduler/cluster APIs; full end-to-end submission is a reviewer-
  reproduction path documented in the README.
- **CGAT-core / WDL / CWL generators**: not implemented today; adding
  one is a feature, not a benchmark.

## Where to learn more

- [`benchmarks/README.md`](https://github.com/cribbslab/flowagent/blob/main/benchmarks/README.md) — full Make targets + cost estimates
- [Error Recovery](user-guide/error-recovery.md) — the system Benchmark B measures
- [Pipeline Generation](user-guide/pipeline-generation.md) — what Benchmark C tests
- [Executors](user-guide/executors.md) — the matrix Benchmark D covers
