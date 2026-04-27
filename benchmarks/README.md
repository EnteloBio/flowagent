# FlowAgent Benchmarks

Reproducible benchmarks that measure FlowAgent's seven core claims: natural-language
**planning correctness**, **per-model cost**, **adaptive error recovery**,
**generator fidelity**, **executor coverage**, **output fidelity** against
published references, and **biological-interpretation quality**. Drive the
manuscript figures.

## Layout

```
benchmarks/
├── config/
│   ├── models.yaml                     # LLMs to sweep (OpenAI, Anthropic, Google)
│   ├── faults.yaml                     # Fault catalogue (Benchmark B)
│   ├── fidelity_cases.yaml             # Output-fidelity cases (Benchmark F)
│   └── interpretation_questions.yaml   # MCQ + open-ended questions (Benchmark G)
├── corpus/
│   └── prompts.yaml                    # 41 prompts (23 standard + 18 hard)
├── references/                         # Materialised gold-standard outputs (gitignored)
│   ├── download_references.py          # Orchestrator — fetches each Benchmark F reference
│   ├── install_r_deps.R                # Installs Bioconductor packages used by R recipes
│   ├── make_reference_*.R              # Frozen Bioconductor recipes (DE-table cases)
│   ├── macs_txt_to_bed.py              # MACS .txt → 3-col BED post-processor
│   ├── counts_tsv_to_bed.py            # Corces counts-matrix → consensus BED
│   ├── subset_giab_chr20.sh            # bcftools/zgrep chr20 subset of GIAB v4.2.1
│   └── README.md                       # What each reference is + license notes
├── harness/
│   ├── runner.py                       # Provider switching, sweep, .env loader
│   ├── metrics.py                      # Scoring + cost helpers
│   ├── fault_inject.py                 # Fault implementations (Benchmark B)
│   ├── env_detect.py                   # Which executor backends are live-testable
│   ├── executor_probes.py              # Per-backend probes (Benchmark D)
│   ├── competitors.py                  # Competitor interface + adapters
│   ├── biomaster_shim.py               # Subprocess shim driving upstream BioMaster
│   ├── autoba_shim.py                  # Subprocess shim driving upstream AutoBA
│   ├── fidelity_metrics.py             # de_table / peak_bed / vcf comparators (Benchmark F)
│   └── plot.py                         # Publication-ready figures (colour-blind safe)
├── bench_planning.py                   # A — planning correctness + cost
├── bench_recovery.py                   # B — error recovery
├── bench_generation.py                 # C — Nextflow/Snakemake codegen fidelity
├── bench_executors.py                  # D — executor-coverage matrix
├── bench_competitors.py                # E — head-to-head vs other agentic systems
├── bench_fidelity.py                   # F — pure scoring layer for output fidelity
├── bench_fidelity_run.py               # F — end-to-end driver (runs flowagent then scores)
├── bench_interpretation.py             # G — MCQ + open-ended interpretation
├── rescore_planning.py                 # Re-evaluate existing plans with updated metrics
├── recovery_taxonomy.py                # Classify Benchmark B responses
├── merge_runs.py                       # Combine runs across models/sessions
├── supp_table_models.py                # Supplementary Table 2 (model registry × empirical stats)
├── Makefile                            # Convenience orchestration
└── results/                            # Gitignored outputs (CSV, JSON, PDF)
```

## The seven benchmarks

| ID | Claim | Needs API key | Needs infra |
|---|---|---|---|
| **A** | FlowAgent generates valid plans from natural language | yes | no |
| **B** | FlowAgent self-heals faults that break traditional WMS (28 faults, 3 tiers) | yes | no |
| **C** | Generated Nextflow / Snakemake is valid and preserves plan intent | no (preset path) | `nextflow` + `snakemake` for `.validate()` |
| **D** | All six execution backends function | no | best-effort — mock mode if infra absent |
| **E** | FlowAgent is competitive with other agentic bio systems on the same corpus | yes | BioMaster + AutoBA clones on disk |
| **F** | FlowAgent's *outputs* match published references (Spearman ρ / Jaccard / F1) | no — pure scorer | network for first-run reference download |
| **G** | LLMs interpret bioinformatics outputs correctly + abstain when evidence is insufficient | yes | reference files materialised by F |

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

### Fidelity cases (Benchmark F)

`config/fidelity_cases.yaml` declares **7 cases** spanning three assay
families. Each case has a `comparison` key (`de_table`, `peak_bed`, or
`vcf`) and a `reference_source` block that tells the orchestrator how to
materialise the gold-standard file (`direct_url` HTTPS download or
`r_script` Bioconductor recipe).

| Case ID | Assay | Reference build | Comparator |
|---|---|---|---|
| `gse52778_dex_de` | RNA-seq DE (DEX/airway) | R script: airway pkg + DESeq2 | de_table |
| `gse60450_mammary_de` | RNA-seq DE (mouse mammary) | R script: edgeR/limma-voom on NCBI counts | de_table |
| `gse152418_covid_blood_de` | RNA-seq DE (COVID-19 vs healthy) | R script: DESeq2 on GEO counts | de_table |
| `encsr000euq_suz12_h1` | ChIP-seq peaks (SUZ12) | Direct URL: ENCODE IDR-thresholded peaks | peak_bed |
| `gse32222_er_chip` | ChIP-seq peaks (ER-α) | Direct URL: GSM-deposited MACS peaks | peak_bed |
| `gse74912_atac_immune` | ATAC-seq peaks (immune atlas) | Direct URL: GEO counts → consensus BED | peak_bed |
| `giab_na12878_chr20` | Germline variants | Direct URL: GIAB v4.2.1 → bcftools chr20 | vcf |

The runner is a **pure scoring layer** — it does not invoke FlowAgent. Run
FlowAgent end-to-end on each prompt yourself, then score against the
materialised references with `bench_fidelity.py`.

### Interpretation questions (Benchmark G)

`config/interpretation_questions.yaml` contains **32 questions across the
same 7 datasets** as Benchmark F (24 MCQ + 8 open-ended), with a
calibrated refusal-correct question per dataset to test whether models
abstain when the supplied evidence is insufficient. Open-ended responses
are graded by an LLM judge (default `gpt-5.4`) against a per-question
rubric and reference answer.

The benchmark feeds each dataset's reference file (DE table, peak BED,
truth VCF) directly to the model under test — FlowAgent itself is not in
the loop. This makes Benchmark G a model-vs-model comparison on
deterministic inputs, in the spirit of BixBench.

### Reference data

Reference files for Benchmarks F and G are **not committed**; materialise
them once before scoring:

```bash
make install-r-deps    # one-time: BiocManager::install for the R recipes
make references        # fetches every reference declared in fidelity_cases.yaml
```

See [`references/README.md`](references/README.md) for per-file source
notes, sizes, and license caveats. CI / no-R-install runs:
`make references SKIP_R=1`.

### Models

`config/models.yaml` defines **30 models** across three tiers, plus
several legacy/preview aliases for back-compatibility with archived runs:

| Tier | OpenAI | Anthropic | Google |
|---|---|---|---|
| current | `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `o3`, `o3-mini`, `o4-mini` | `claude-opus-4-5/6/7`, `claude-sonnet-4-5/6`, `claude-haiku-4-5` | `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite` |
| preview | — | — | `gemini-3.1-pro-preview`, `gemini-3.1-flash-lite-preview`, `gemini-3-flash-preview` |
| legacy | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`, `o1` | `claude-opus-4`, `claude-opus-4-1`, `claude-sonnet-4`, `claude-haiku-3-5` | `gemini-1.5-pro`, `gemini-1.5-flash` |

Add or remove a model by editing `config/models.yaml` — the harness,
scoring, and plot code pick up new IDs automatically (so long as the
short name is registered in [`harness/plot.py`](harness/plot.py) for
axis labels).

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

Scope to a single tier of the fault catalogue with `--tier`, useful when
iterating on the unrecoverable sub-story (4 faults × N seeds × M models
runs cheaply):

```bash
python bench_recovery.py --tier unrecoverable --seeds 10 --model gpt-4.1
```

Each row in the output `results.json` captures the LLM's full response,
regardless of outcome — `recovery_diagnosis`, `rejection_reason`,
`fixed_command`, `llm_raw_response`, and a `recovery_outcome` label
(`success` / `proposed` / `rejected` / `silent`) so post-hoc
classification doesn't need to re-call the model.

#### Recovery taxonomy (unrecoverable tier)

`recovery_taxonomy.py` clusters every cell into one of five buckets so
the sub-story on "how do agents respond to unfixable faults?" becomes
a manuscript-ready figure:

```bash
# Aggregate every recovery run under results/recovery/
python recovery_taxonomy.py

# Or a specific model sweep
python recovery_taxonomy.py --runs 'results/recovery/2026-04-20T*'
```

| Category | Meaning |
|---|---|
| `correct_refusal` | Refused AND the diagnosis names the real data issue (e.g. "truncated gzip" for a `corrupt_fastq` fault). |
| `misdiagnosed_refusal` | Refused but blamed the wrong thing (e.g. "FASTA missing" when the actual fault is `paired_single_mismatch`). The refusal is coincidentally correct — a systematic blind spot where the LLM pattern-matches to a different failure class than the one that fired. |
| `unsafe_repair` | Proposed a fix that ran clean on an unrecoverable fault. Most dangerous class — downstream pipeline believes everything worked but the data is compromised. |
| `attempted_repair` | Proposed a fix that still failed. At least the pipeline surfaces the failure. |
| `silent_failure` | No dict returned (max-attempts hit, LLM timeout, parse error). |

Outputs (written to `results/recovery/_taxonomy/<ts>/`):

- `taxonomy.tsv` — wide per-(model × fault) × category count + pct
  table, paste-ready for a manuscript figure.
- `per_cell.csv` — flat table with each cell's category + matched
  keyword signal; spot-checkable by hand.
- `examples.md` — 2–3 representative diagnosis quotes per (category ×
  fault), for supplementary captions.

Keyword signals used to distinguish `correct_refusal` from
`misdiagnosed_refusal` live in `recovery_taxonomy._FAULT_SIGNALS`; tune
them if `examples.md` shows false positives or missed refusals.

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

### Benchmark F — output fidelity

Two pieces: a **driver** (`bench_fidelity_run.py`) that invokes FlowAgent
on every case's prompt, and a **scorer** (`bench_fidelity.py`) that
compares each candidate's outputs against the materialised reference.
The driver auto-invokes the scorer after all cells finish.

```bash
# Run every case at gpt-4.1, 1 replicate; auto-scores at the end
make fidelity-run MODEL=gpt-4.1

# Multi-model + multi-replicate
make fidelity-run MODELS=gpt-4.1,claude-sonnet-4-6,gemini-2.5-flash REPLICATES=2

# Just one case (e.g. iterating on the GSE52778 pipeline)
python bench_fidelity_run.py --model gpt-4.1 --case gse52778_dex_de

# Already have outputs on disk — just score them
python bench_fidelity_run.py --score-only

# Score one existing FlowAgent run dir directly (legacy single-case path)
python bench_fidelity.py \
    --case gse52778_dex_de \
    --candidate-dir results/realworld_GSE52778 \
    --model gpt-4.1 --replicate 0
```

**Directory layout.** Each cell lives at:

```
results/fidelity_runs/
├── _driver.log                                  # cross-cell progress
├── _driver_summary.json                         # per-cell status JSON
├── gse52778_dex_de__gpt-4.1__rep0/
│   ├── prompt.txt                               # exact prompt that was run
│   ├── run.log                                  # FlowAgent stdout/stderr
│   ├── flowagent_output/Unnamed_Workflow/...
│   └── results/rna_seq_kallisto/deseq2/deseq2_results.csv
├── gse52778_dex_de__claude-sonnet-4-6__rep0/
│   └── ...
└── encsr000euq_suz12_h1__gpt-4.1__rep0/
    └── ...
```

**Checkpointing.** A cell is considered complete when its
`output_relpath` (declared in `fidelity_cases.yaml`) exists and is
non-empty. Re-running the driver skips completed cells automatically.
Pass `--force` to bypass the skip and re-run everything. Per-cell
timeouts default to 24 h (`--timeout-hours`).

**Logging.** Top-level driver log streams to both stdout *and*
`_driver.log`; per-cell logs are at `<cell>/run.log`. Tail one in
another terminal to watch a single pipeline:

```bash
tail -f results/fidelity_runs/gse52778_dex_de__gpt-4.1__rep0/run.log
```

**Concurrency.** `--concurrency N` runs N cells in flight at once;
default 1 because pipelines are bandwidth- and disk-heavy (RNA-seq cases
download 10s of GB of FASTQ data per cell). Be conservative on a
laptop; safe to bump on a workstation.

**Cost.** Each cell is a real bioinformatics pipeline — downloads FASTQs,
runs kallisto / MACS2 / GATK, calls the LLM dozens of times for plan +
recovery. Realistic per-cell budgets: $0.50–$5 in API spend, 2–8 h
wall time, 5–50 GB disk. Six cases × 10 models × 1 replicate is
**not** something to fire off lightly.

**Score-only mode.** `--score-only` skips all FlowAgent invocations and
runs the scorer over whatever cells already exist. Useful when iterating
on the comparator code:

```bash
python bench_fidelity_run.py --score-only
```

The `de_table` comparator strips Ensembl version suffixes and tolerates
common gene-ID column aliases (`gene_id`, `Gene`, `Unnamed: 0`, etc.) so
candidate outputs from kallisto + tximport, STAR + featureCounts, or
salmon all join correctly to the reference. Output rows include
`spearman_lfc`, `jaccard_top_n`, `n_overlap`; for peak/VCF cases:
`jaccard_peak`, `precision`, `recall`, `f1`.

**Typical numbers** for `gse52778_dex_de` against the `airway` package
canonical reference:

```
spearman_lfc=0.75   jaccard_top_n=0.45   n_overlap=20933
```

Spearman 0.75 is in the "different quantifier, same biology" range
(kallisto vs STAR+HTSeq); Jaccard top-200 captures the moderate drift in
which genes sit just above the |log2FC|>1 cutoff.

### Benchmark G — biological-interpretation quality

```bash
# Single model
make interpretation MODEL=gpt-4.1 JUDGE=gpt-5.4

# Multi-model sweep (recommended for the manuscript figure)
python bench_interpretation.py \
  --models gpt-5.4,gpt-5.4-mini,o3,gpt-4.1,claude-opus-4-7,claude-sonnet-4-6,claude-haiku-4-5,gemini-2.5-pro,gemini-2.5-flash,gemini-3.1-flash-lite-preview \
  --judge gpt-5.4

# Every model in models.yaml
python bench_interpretation.py --all-models --judge gpt-5.4

# Mock mode (no LLM calls; deterministic stub answers — CI smoke)
python bench_interpretation.py --models gpt-4.1 --mock
```

Each row in the output `metrics.csv` carries `(model, dataset,
question_id, question_type, correct, judge_score,
judge_justification, raw_response, candidate_answer)` so the
interpretation figure can split MCQ accuracy from open-ended judge
scores and refusal calibration. The schema is stable even when LLM calls
error out — every row gets `correct=False` as a default so a partial
sweep still produces a plottable CSV.

The `interpretation_figure` in `harness/plot.py` renders three panels:

- **Per-model overall MCQ accuracy** with Wilson 95% CIs.
- **Model × dataset MCQ-accuracy heatmap** (grey cells = no data).
- **Per-model open-ended judge mean** ± 1 SD.

### Everything at once

```bash
make all         # single MODEL (default gpt-4.1): plan + recovery + gen + exec + report
make all-sweep   # full 30-model sweep: plan-all + recovery + gen + exec + competitors + rescore + merge + report
```

Use `make all` for a quick end-to-end smoke of one model (fast, cheap). Use
`make all-sweep` for the multi-model manuscript run — it automatically chains
`rescore → merge → report` in the right order so all models appear in the
final figures.

Benchmarks F and G are **not** included in `all-sweep` because they have
distinct workflow shapes (F needs prior FlowAgent runs to score; G is an
LLM-only sweep against fixed reference inputs). Run them separately:

```bash
make references                                # one-time, materialises Benchmark F refs
make fidelity                                  # bulk-score a fidelity_runs/ tree
make interpretation MODEL=gpt-4.1 JUDGE=gpt-5.4
```

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
- `merge` collates runs across all benchmarks (planning, competitors,
  recovery, interpretation, fidelity), preferring the latest `rescored_*`
  subdir for each. Dedup keys per benchmark:

  | Benchmark | Dedup key |
  |---|---|
  | planning | `(model, input_id, replicate)` |
  | competitors | `(competitor, input_id, replicate)` |
  | recovery | `(model, fault_id, seed)` |
  | interpretation | `(model, dataset, question_id)` |
  | fidelity | `(case_id, model, replicate)` |

  Re-running a single model cleanly replaces stale rows. Schema-incomplete
  rows from runs that errored out (no `correct` column for interpretation,
  no comparator metric for fidelity) are dropped at merge time so they
  don't bias per-model rollups.

- `report` generates figures from `results/<bench>/_merged/<latest>` if
  present, falling back to the newest single run per benchmark.

**If you run `merge` before `rescore`**, the merged CSV captures the
pre-rescore numbers. Always rescore first.

**Force a clean re-merge:** `make merge REFRESH=1` (or
`python merge_runs.py --refresh`). Useful after the question YAML or
dedup keys change.

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
| `planning_heatmap.pdf` | Per-prompt × per-model pass-rate heatmap |
| `planning_heatmap_by_tier.pdf` | Heatmap split into current vs legacy model panels |
| `planning_cost_summary.pdf` | Per-model cost bar chart (two panels) |
| `planning_cost_quality.pdf` | Pass-rate vs cost scatter (log x-axis), Pareto frontier |
| `planning_latency.pdf` | Per-model wall-clock + speed-vs-quality trade-off |
| `planning_turns.pdf` | Mean LLM calls per plan (turns to completion) |
| `planning_consistency.pdf` | Inter-replicate unanimity per model |
| `planning_hallucination.pdf` | Hallucinated-tool fraction per model |
| `planning_tokens.pdf` | Mean prompt + completion tokens per plan |
| `recovery.pdf` | Benchmark B per-fault recovery, grouped Easy / Hard / Unrecoverable |
| `recovery_tier_summary.pdf` | Compact per-tier summary |
| `recovery_per_fault_heatmap.pdf` | Cross-model per-fault recovery heatmap |
| `recovery_taxonomy.pdf` | 5-outcome taxonomy on the unrecoverable tier |
| `recovery_reasoning_split.pdf` | Reasoning vs non-reasoning model recovery comparison |
| `recovery_per_model/recovery_<model>.pdf` | Per-model breakdown across all faults |
| `generation.pdf` | Benchmark C generator-fidelity heatmap |
| `executors.pdf` | Benchmark D executor-coverage matrix |
| `competitors.pdf` | Benchmark E pass / fail / crash per competitor |
| `competitors_perprompt.pdf` | Competitor × prompt outcome heatmap |
| `competitors_agentic.pdf` | FlowAgent vs BioMaster vs AutoBA focused comparison |
| `interpretation.pdf` | Benchmark G three-panel: MCQ accuracy + heatmap + open-ended judge mean |
| `planning_cost_summary.tsv` | Per-model cost / pass-rate / token table for the manuscript |
| `supp_table2_models.tsv` | Supplementary Table 2: model registry × empirical token / cost / latency stats |

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

Rough guide at current (Apr 2026) rates across the full 30-model registry.

| Target | Models | Wall time | API cost |
|---|---|---|---|
| `make plan` | 1 | ~5–15 min | ~$0.05–$2 (depends on model tier) |
| `make plan-all` | 30 | ~45–90 min (concurrent) | ~$20–40 |
| `make recovery` | 1 | ~35–45 min (28 faults × 5 seeds) | ~$2–3 |
| `make gen` | — | <1 min | $0 |
| `make exec` | — | <1 min | $0 |
| `make competitors` | 1 | ~10–30 min (depends on BioMaster RAG) | ~$1–4 |
| `make references SKIP_R=1` | — | ~30 s (one-off) | $0 (network only) |
| `make references` | — | ~5–10 min (R-script cases) | $0 |
| `make fidelity --bulk-dir=…` | — | <1 s per case | $0 (pure scoring) |
| `make fidelity-run` (1 model) | 1 | ~12–24 h sequential, ~6–10 h at CONCURRENCY=3 | ~$3–15 (7 cases × 1 model) |
| `make fidelity-run MODELS=a,b,c` | 3 | ~30+ h sequential | ~$10–45 (21 cells) |
| `make interpretation` | 1 | ~5–10 min (32 questions) | ~$0.50–$2 |
| `bench_interpretation.py --models=…` | 10 | ~30–60 min | ~$5–15 |
| `make rescore` / `merge` / `report` | — | ~5 s | $0 |
| `make install-r-deps` | — | ~5–10 min (one-off) | $0 |

Cheapest frontier model for `plan-all`: roughly $0.05 for the full 123-cell
sweep (`gpt-5.4-nano` or `gemini-2.5-flash-lite`). Most expensive: ~$10+ for
`claude-opus-4-7` or `o1` alone. The exact per-model breakdown is in
`planning_cost_summary.tsv` after the first real run; the consolidated
manuscript-ready table sits in `supp_table2_models.tsv` (joined with
registry metadata via `python supp_table_models.py`).

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

**Benchmark F: `candidate missing column: gene_id`** — the comparator now
recognises common aliases (`Unnamed: 0`, `Gene`, `gene`, `feature_id`,
`ensembl_id`, `ensembl_gene_id`, `GeneID`). If your candidate uses a
different column name, declare it via `params.gene_id_column` in
`config/fidelity_cases.yaml`.

**Benchmark F: `candidate not found: …`** — the path is built as
`<candidate-dir>/<output_relpath>`. From inside `benchmarks/`, drop the
leading `benchmarks/` from `--candidate-dir` (e.g. `--candidate-dir
results/realworld_GSE52778`).

**Benchmark F R script: `cannot open URL '…bioconductor.org/…'`** — older
script versions pointed at retired Bioc course-materials URLs. Pull
latest of this directory; URLs were migrated to NCBI mirrors.

**Benchmark F R script: `there is no package called 'recount3' / 'DiffBind'`** —
older recipes depended on these heavy packages. Current recipes use only
`airway`, `DESeq2`, `edgeR`, `limma`, `Glimma`, `SummarizedExperiment`.
Run `make install-r-deps` to install the current set.

**Benchmark G: `ValueError: cannot convert float NaN to integer`** in
`make report` — the merge picked up a partial CSV from a run that errored
out before the schema-stable fix. Refresh: `make merge REFRESH=1` then
`make report`. Schema-incomplete rows are now dropped at merge time so
this only affects archived data.

**Benchmark G: `Event loop is closed` warnings during multi-model sweep** —
cosmetic only; data still written correctly. The patched runner uses one
event loop for the whole sweep so these no longer appear after pulling
latest.

**Benchmark G: `AuthenticationError: Incorrect API key provided`** — your
shell has a stale `OPENAI_API_KEY` overriding `.env` (the harness's
dotenv loader uses `setdefault`, so shell wins). Fix: `unset
OPENAI_API_KEY` and re-run; the working key in `.env` will then take
effect.

## Out of scope (documented explicitly)

- **User study** (wet-lab vs. Nextflow-tutorial): requires IRB + human
  participants; noted as a future extension.
- **Concordance with nf-core published pipelines**: supported by Benchmark C's
  API but requires GB-scale test data; see `extras/concordance.md`.
- **Full live HPC / Kubernetes execution**: Benchmark D uses mocks when infra
  is absent and does live runs when present; production cluster behaviour
  is documented as a reviewer-reproduction path.
- **CGAT-core / WDL / CWL generators**: not implemented in FlowAgent today.
