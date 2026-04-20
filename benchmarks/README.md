# FlowAgent Benchmarks

Reproducible benchmarks that measure FlowAgent's core claims: natural-language
**planning correctness**, **per-model cost**, **adaptive error recovery**,
**generator fidelity**, and **executor coverage**. Drive the manuscript figures.

## Layout

```
benchmarks/
├── config/
│   ├── models.yaml          # LLMs to sweep (OpenAI, Anthropic, Google, ...)
│   └── faults.yaml          # Fault catalogue (for Benchmark B)
├── corpus/
│   └── prompts.yaml         # 41 prompts (23 standard + 18 hard) with expected properties
├── harness/
│   ├── runner.py            # Provider switching, sweep, .env loader
│   ├── metrics.py           # Scoring + cost helpers
│   ├── fault_inject.py      # Fault implementations
│   ├── env_detect.py        # Which executor backends are live-testable
│   ├── executor_probes.py   # Per-backend probes for Benchmark D
│   ├── competitors.py       # Competitor interface + FlowAgent/BioMaster/AutoBA adapters
│   ├── biomaster_shim.py    # Subprocess shim driving upstream BioMaster
│   ├── autoba_shim.py       # Subprocess shim driving upstream AutoBA
│   └── plot.py              # Publication-ready figures (colour-blind safe)
├── bench_planning.py        # Benchmark A: planning correctness + cost
├── bench_recovery.py        # Benchmark B: error recovery
├── bench_generation.py      # Benchmark C: Nextflow/Snakemake codegen fidelity
├── bench_executors.py       # Benchmark D: executor-coverage matrix
├── bench_competitors.py     # Benchmark E: head-to-head vs other agentic systems
├── rescore_planning.py      # Re-evaluate existing plans with updated metrics (no API calls)
├── merge_runs.py            # Combine runs across models/sessions into one CSV
├── Makefile                 # Convenience orchestration
└── results/                 # Gitignored outputs (CSV, JSON, PDF)
```

## The five benchmarks

| ID | Claim | Needs API key | Needs infra |
|---|---|---|---|
| **A** | FlowAgent generates valid plans from natural language | yes | no |
| **B** | FlowAgent self-heals faults that break traditional WMS (28 faults, 3 tiers) | yes | no |
| **C** | Generated Nextflow / Snakemake is valid and preserves plan intent | no (preset path) | `nextflow` + `snakemake` for `.validate()` |
| **D** | All six execution backends function | no | best-effort — mock mode if infra absent |
| **E** | FlowAgent is competitive with other agentic bio systems on the same corpus | yes | BioMaster + AutoBA clones on disk (see below) |

### Prompt corpus

`corpus/prompts.yaml` contains **41 prompts** across two difficulty tiers:

- **23 standard prompts** — covering common RNA-seq, ChIP-seq, ATAC-seq,
  variant calling, scRNA-seq, and QC workflows. Designed to probe whether the
  LLM produces a sensible, tool-correct, stepwise plan.
- **18 hard prompts** (IDs prefixed `hard_`) — designed to stress one or more
  LLM failure modes: niche domains (bisulfite sequencing, metagenomics,
  miRNA, Hi-C), long end-to-end chains (8+ steps), modern tool selection
  (hifiasm vs spades, Mutect2 vs HaplotypeCaller), forbidden shortcuts
  (kallisto when STAR is required), and R-package wrappers (DADA2, DiffBind,
  QDNAseq, tximport).

### Fault catalogue (Benchmark B)

`config/faults.yaml` + `harness/fault_inject.py` contain **28 faults** across
three tiers:

- **15 easy faults** — surface-level fixes: a flag typo, a missing output
  directory, an abbreviated ambiguous flag. A competent LLM should recover
  on the first attempt. Examples: `missing_wget`, `tool_typo`,
  `samtools_subcommand_typo`, `ambiguous_flag`, `cp_source_missing`,
  `missing_python_module`.
- **9 hard faults** — require semantic reasoning or a multi-step fix
  (insert a new step, not just edit the failing one). Examples:
  `bam_unsorted_indexing` (needs a `samtools sort` prepended),
  `missing_bwa_index` (needs `bwa index` run first),
  `chromosome_naming_mismatch` (detect chr1-vs-1 prefix),
  `java_heap_oom` (raise `-Xmx`), `missing_sequence_dict` (create GATK dict).
- **4 unrecoverable faults** — data / environment problems where the
  *desired* outcome is *failure*. A correct LLM should refuse to "fix"
  these: `corrupt_fastq`, `empty_input_file`, `binary_as_fastq`,
  `paired_single_mismatch`.

Each fault produces a real failure signature (genuine exit code + stderr
via shell stubs or real tools), so recovery is judged on the LLM's ability
to read and fix an authentic error.

### Models

`config/models.yaml` defines 20 models across three tiers:

| Tier | OpenAI | Anthropic | Google |
|---|---|---|---|
| legacy | `gpt-3.5-turbo`, `gpt-4` | `claude-3-haiku`, `claude-3-sonnet` | `gemini-1.5-flash` |
| mid | `gpt-4-turbo`, `gpt-4o-mini` | `claude-3-5-haiku`, `claude-3-5-sonnet` | `gemini-1.5-pro` |
| frontier | `gpt-4o`, `gpt-4.1`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano` | `claude-sonnet-4`, `claude-opus-4-5`, `claude-opus-4-6`, `claude-opus-4-7` | `gemini-2.5-flash` |

Add or remove a model by editing `config/models.yaml` — the harness, scoring,
and plot code pick up new IDs automatically (so long as the short name is
registered in [`harness/plot.py`](harness/plot.py) for axis labels).

## API keys

The harness auto-loads a `.env` file from the repo root (walks up from
`benchmarks/`). Put your keys in there:

```bash
# .env (repo root, already gitignored)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Optional — only needed for Benchmark E (head-to-head)
BIOMASTER_DIR=/absolute/path/to/BioMaster
AUTOBA_DIR=/absolute/path/to/AutoBA
```

No need to `source` or `export` — the harness picks them up automatically.
Keys set in the shell win over `.env` (standard dotenv semantics).

## Running

### Smoke test (no API key, ~3 s)

```bash
make smoke
```

Runs every benchmark in mock mode to verify the harness imports cleanly and
the scoring pipeline is sound.

### Single model — Benchmark A

```bash
make plan MODEL=claude-opus-4-7 REPLICATES=3
```

### All models — Benchmark A

```bash
make plan-all REPLICATES=3
```

Sweeps every model in `config/models.yaml` **concurrently (4 at a time)**.
With 20 models × 41 prompts × 3 replicates this is **2 460 cells**, roughly
30–60 min wall clock depending on API latency. See the cost table below.

### Benchmark B — error recovery

```bash
make recovery MODEL=claude-opus-4-7 SEEDS=5
```

### Deterministic benchmarks (no API key)

```bash
make gen      # Benchmark C: generator fidelity
make exec     # Benchmark D: executor coverage
```

### Benchmark E — head-to-head against other agents

```bash
make competitors MODEL=gpt-4.1 REPLICATES=3
```

Runs every registered competitor (currently `flowagent`, `biomaster`, and
`autoba`) on the same prompt corpus, scored with the same `score_plan`
metrics so the comparison is apples-to-apples. Results land in
`results/competitors/<ts>/` with per-row `competitor`, `plan`,
`prompt_tokens`, `completion_tokens`, `cost_usd`, `wall_seconds`, and the
standard scoring columns. At the end of each run, the driver prints a
per-competitor **pass / fail / crash** rollup and writes it to
`summary.tsv`:

```
Head-to-head rollup (pass / fail / crash per competitor):
  Competitor        Pass   Fail  Crash   Pass%    $/cell    Wall
  --------------------------------------------------------------
  FlowAgent         8/10      2      0   80.0%   $0.0123   15.4s
  BioMaster         4/10      4      2   40.0%   $0.0087   18.2s
  AutoBA            5/10      5      0   50.0%   $0.0195   22.1s
```

Where:
- **Pass** = plan produced and scored True on every `score_plan` gate.
- **Fail** = plan produced but missed at least one scoring gate
  (workflow type, expected tools, forbidden tools, min step count).
- **Crash** = the competitor raised before producing any scorable plan.
  Broken out separately so robustness shows up as its own column rather
  than silently dragging down the pass rate.

**Subsetting:**

```bash
python bench_competitors.py \
  --competitors=flowagent,biomaster \
  --prompts=rnaseq_kallisto_basic,hard_full_germline_pipeline \
  --replicates=2
```

`--mock` runs offline with canned plans derived from each prompt's
`gold_preset` / `expected_tools`, useful for smoke-testing the harness.

#### BioMaster setup (one-off, ~5 min)

BioMaster (Su et al., 2025) is a multi-agent bioinformatics workflow system.
Upstream is a script project — no pip packaging — so we drive it via a
subprocess shim ([`harness/biomaster_shim.py`](harness/biomaster_shim.py))
that synthesises a BioMaster-native `config.yaml` and invokes upstream's
own `run.py config.yaml` entrypoint via `runpy` (the same code path as
`python run.py config.yaml` would execute after a fresh `git clone`).
The shim wraps that invocation in a LangChain OpenAI callback to capture
token usage, then reads the on-disk `output/<id>_PLAN.json` and maps it
into FlowAgent's plan schema.

```bash
# 1. Clone the upstream repo
git clone <biomaster-repo-url> /path/to/BioMaster
cd /path/to/BioMaster

# 2. Install its pinned deps (skip PySide6 — unused leftover, saves ~500 MB)
pip install -r <(grep -ivE '^(pyside6|shiboken6)' requirements.txt)

# 3. Point the harness at the clone
echo 'BIOMASTER_DIR=/path/to/BioMaster' >> /path/to/flowagent/.env
```

Smoke-test the shim directly before wiring it into the sweep:

```bash
python benchmarks/harness/biomaster_shim.py \
  --prompt "Run a kallisto RNA-seq quantification on paired-end FASTQs"
```

Expect a JSON envelope on stdout with `plan`, `prompt_tokens`,
`completion_tokens`, `cost_usd`, `wall_seconds`. First run is slow — BioMaster
indexes its `doc/` RAG into a scratch Chroma store on cold start; that's
discarded when the subprocess exits.

**Notes:**

- The config has `executor: false` so BioMaster plans without actually
  running bioinformatics tools (the default `true` would try to `conda
  install` + execute each generated shell script per cell, which is
  infeasible in a benchmark context).
- Upstream `execute_TASK` has an `UnboundLocalError` when `executor: false`
  (a bare `DEBUG_output_dict` reference inside an `if self.excutor:`
  branch). The shim captures that as a soft error and still scores the
  `PLAN.json` that `execute_PLAN` wrote before the crash. Step `command`
  fields are populated from each PLAN step's `tools` metadata — BioMaster's
  own tool-name text — so `score_plan`'s matcher still has something to
  work with.
- `workflow_type` is inferred post-hoc by a deterministic classifier
  (`biomaster_shim._classify_workflow_type`) because BioMaster has no
  workflow taxonomy. Asymmetric wildcard semantics in
  `harness/metrics.py::type_matches` mean an "actual=custom" would
  auto-fail strictly typed prompts (e.g. `rnaseq_kallisto_basic`,
  `chipseq_macs2`) even if the plan is perfect — the classifier gives
  BioMaster the equivalent benefit FlowAgent earns by labelling its plan.
- Each cell uses a fresh `uuid`-derived `id` and a temp working dir, so
  concurrent cells don't collide on BioMaster's `output/<id>_PLAN.json`.
  The BioMaster clone itself stays clean (no `./output`, `./chroma_db`,
  `./token.txt` pollution).
- Token and cost accounting come from
  `langchain_community.callbacks.get_openai_callback` — directly comparable
  to FlowAgent's `_TokenTracker` numbers.
- Drop-in for other agents: subclass `Competitor` in `harness/competitors.py`
  and register it in `build_registry()`.

#### AutoBA setup (one-off, ~10 min)

AutoBA / Auto-BioinfoGPT (Zhou et al., 2023) is a second-generation multi-agent
bioinformatics planner. Upstream is also a script project — the shim
([`harness/autoba_shim.py`](harness/autoba_shim.py)) invokes AutoBA's own
`app.py --config cfg.yaml --openai KEY --model MODEL --execute False`
entrypoint via `runpy`, and monkey-patches
`openai.resources.chat.completions.Completions.create` to capture token
usage (AutoBA uses the raw `openai` SDK, not LangChain — so the BioMaster
callback trick doesn't apply, but an SDK-level patch gives the same
accounting).

```bash
# 1. Clone the repo
git clone https://github.com/JoshuaChou2018/Auto-BioinfoGPT /path/to/AutoBA

# 2. Install its deps. AutoBA imports torch.cuda at module load time and
#    llama_index (for RAG), even when we don't use those paths.
pip install openai pyyaml torch \
    llama-index-core \
    llama-index-embeddings-openai \
    llama-index-embeddings-huggingface

# 3. Point the harness at the clone
echo 'AUTOBA_DIR=/path/to/AutoBA' >> /path/to/flowagent/.env
```

Smoke-test the shim directly:

```bash
python benchmarks/harness/autoba_shim.py \
  --prompt "Run a kallisto RNA-seq quantification on paired-end FASTQs"
```

**Notes:**

- AutoBA runs with `--execute False` so it plans + writes per-task shell
  scripts without executing them. Each task becomes one FlowAgent-schema
  step; `<output_dir>/<N>.sh` bodies become the step `command` (with a
  fallback to the task description string if the shell is missing).
- The `workflow_type` is inferred by the same classifier used for
  BioMaster (`biomaster_shim._classify_workflow_type`) — AutoBA's plan is
  just a list of task strings, so we project them through the shared
  tool-signature mapping.
- AutoBA's `app.py` has a top-level `import torch.cuda`, so `torch` must
  be installed even if you never invoke its GPU paths.

### Everything at once

```bash
make all         # single MODEL (default gpt-4.1): plan + recovery + gen + exec + report
make all-sweep   # full 20-model sweep: plan-all + recovery + gen + exec + rescore + merge + report
```

Use `make all` for a quick end-to-end smoke of one model (fast, cheap). Use
`make all-sweep` for the multi-model manuscript run — it automatically chains
`rescore → merge → report` in the right order so all models appear in the
final figures.

## Post-processing (important order)

When scoring logic or `prompts.yaml` is updated, you can re-evaluate existing
runs without spending more API budget. **The order matters**:

```bash
make rescore    # 1. Rescore each run with the current metrics code
make merge      # 2. Combine rescored runs into one deduplicated CSV
make report     # 3. Render figures from the merged CSV
```

- `rescore` reads each run's `results.json`, re-applies `score_plan` with the
  current `metrics.py`, preserves token counts, and re-computes `cost_usd`
  using the current `models.yaml` pricing. Outputs land under
  `results/planning/<run>/rescored_<ts>/`.
- `merge` prefers the latest `rescored_*` subdir inside each run and
  deduplicates by `(model, input_id, replicate)` so re-running a single model
  cleanly replaces stale rows.
- `report` generates figures from `results/planning/_merged/<latest>` if
  present, falling back to the newest single run.

**If you run `merge` before `rescore`**, the merged CSV captures the
pre-rescore numbers. Always rescore first.

## Cost tracking

`bench_planning.py` records the true per-plan token usage (including every
internal LLM call — pattern extraction, planning, optional JSON repair) by
wrapping the provider. Each results row has:

| Column | Meaning |
|---|---|
| `prompt_tokens` | Total input tokens across all internal LLM calls for this plan |
| `completion_tokens` | Total output tokens across all internal LLM calls |
| `llm_calls` | How many provider calls were made to produce the plan |
| `cost_usd` | Dollar cost computed from `models.yaml` pricing |
| `wall_seconds` | End-to-end generation time |

Two publication-ready cost figures are emitted by `make report`:

- **`planning_cost_summary.pdf`** — two-panel bar chart: cost per 100 plans
  and cost per **successful** plan (the latter penalises cheap-but-flaky
  models).
- **`planning_cost_quality.pdf`** — scatter of pass-rate vs. cost on a log
  x-axis, with Pareto-frontier models annotated.
- **`planning_cost_summary.tsv`** — a plaintext per-model table
  (`model`, `mean_cost`, `cost_per_pass`, `cost_per_100_plans`, `pass_rate`,
  mean input/output tokens) for dropping straight into a manuscript.

**Updating pricing** — if a provider lowers their rates, edit `models.yaml`
and run `make rescore && make merge && make report`. No re-bench needed.

## Figures

```bash
make report
```

Writes PDF + 300 DPI PNG to `results/figures/`. Outputs:

| File | Content |
|---|---|
| `planning.pdf` | Pass rate by model, split into standard vs. hard prompts |
| `planning_heatmap.pdf` | Per-prompt × per-model pass-rate heatmap (hard prompts) |
| `planning_cost_summary.pdf` | Per-model cost bar chart (two panels) |
| `planning_cost_quality.pdf` | Pass-rate vs. cost scatter (log x-axis) |
| `recovery.pdf` | Benchmark B per-fault recovery, grouped into Easy / Hard / Unrecoverable |
| `recovery_tier_summary.pdf` | Compact per-tier summary: recovery rate for Easy + Hard, correct-rejection rate for Unrecoverable |
| `generation.pdf` | Benchmark C generator-fidelity heatmap |
| `executors.pdf` | Benchmark D executor-coverage matrix |

Pass `--svg` to also emit editable SVGs for Illustrator / Inkscape:

```bash
python -m harness.plot --results=results --svg
```

Style notes:
- Colour palette is **Okabe-Ito** (colour-blind safe at the 8% deuteranope
  level). Provider colours: Anthropic amber, OpenAI blue, Google green.
- Fonts are Arial / Helvetica / DejaVu Sans (fallback chain). PDFs embed
  TrueType so they remain editable.
- All heatmaps use a calibrated red→amber→green colormap that stays
  interpretable in greyscale.

## Typical multi-model workflow

```bash
# 1. Drop keys in .env
cat > ../.env <<'EOF'
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
EOF

# 2. Sweep all LLMs for Benchmark A (~30–60 min)
make plan-all REPLICATES=3

# 3. Apply the current scoring logic
make rescore

# 4. Aggregate into a single CSV
make merge

# 5. Render all figures + cost tables
make report

# Figures land in results/figures/
```

### Incremental additions (new model, re-run a flaky one)

```bash
make plan MODEL=claude-opus-4-7 REPLICATES=3
make rescore     # rescore everything, including the new run
make merge       # merge picks up the new rescored output automatically
make report
```

## Cost + wall-clock estimates

Rough guide at current (Apr 2026) rates across the full 20-model sweep.

| Target | Models | Wall time | API cost |
|---|---|---|---|
| `make plan` | 1 | ~5–15 min | ~$0.05–$2 (depends on model tier) |
| `make plan-all` | 20 | ~30–60 min (concurrent) | ~$15–30 |
| `make recovery` | 1 | ~35–45 min (28 faults × 5 seeds) | ~$2–3 |
| `make gen` | — | <1 min | $0 |
| `make exec` | — | <1 min | $0 |
| `make rescore` | — | ~5 s | $0 |
| `make merge` | — | ~1 s | $0 |
| `make report` | — | ~3 s | $0 |

Cheapest frontier model for `plan-all`: roughly $0.05 for the full 123-cell
sweep (`gpt-5.4-nano` or `gemini-1.5-flash`). Most expensive: ~$10+ for
`claude-opus-4-7` or `gpt-4` alone. The exact per-model breakdown is in
`planning_cost_summary.tsv` after the first real run.

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

**`make rescore` fails with `results/planning/_merged/results.json does not exist`** —
the rescorer used to pick up the special `_merged` directory. Pull latest; fixed
by skipping `_`-prefixed dirs.

**Only 1 row per (model, prompt) in merged CSV even though I ran 3 replicates** —
the deduplicator collapses rows only when `(model, input_id, replicate)` are
identical. If your replicate numbers got reset it's rerunning fresh. Check
`_source_run` in `metrics.csv` to trace.

**Cost columns are all zero** — you ran against pre-token-tracking data.
Re-run `make plan` / `make plan-all` to populate `prompt_tokens`,
`completion_tokens`, `cost_usd`. Tokens can't be recovered after the fact.

**A model I added errors immediately** — check that its short name is
registered in `harness/plot.py::_short_name` (axis labels only; the harness
itself accepts any model ID the provider accepts). A typo'd model ID will
surface as an API-side 404 or 400 in the `error` column.

## Out of scope (documented explicitly)

- **User study** (wet-lab vs. Nextflow-tutorial): requires IRB + human
  participants; noted as a future extension.
- **Concordance with nf-core published pipelines**: supported by Benchmark C's
  API but requires GB-scale test data; see `extras/concordance.md`.
- **Full live HPC / Kubernetes execution**: Benchmark D uses mocks when infra
  is absent and does live runs when present; production cluster behaviour
  is documented as a reviewer-reproduction path.
- **CGAT-core / WDL / CWL generators**: not implemented in FlowAgent today.
