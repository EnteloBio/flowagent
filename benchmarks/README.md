# FlowAgent Benchmarks

Reproducible benchmarks that measure FlowAgent's ten core claims:
natural-language **planning correctness**, **per-model cost**, **adaptive
error recovery**, **generator fidelity**, **executor coverage**, **output
fidelity** against published references, **biological-interpretation
quality**, and three ablation studies -- FlowAgent's **DAG-awareness**
(does telling the LLM about the dependency graph help?), FlowAgent's
**completeness reflection** (does a DAG-Plan-style "regenerate if
structurally incomplete" loop help?), and a competitor-side
**DAG-prompt** ablation that asks the same DAG-vs-no-DAG question of an
external framework (Claude Code) over which we have only prompt-level
control. Drive the manuscript figures.

## Layout

```
benchmarks/
├── config/
│   ├── models.yaml                     # LLMs to sweep (OpenAI, Anthropic, Google)
│   ├── faults.yaml                     # Fault catalogue (Benchmark B)
│   ├── fidelity_cases.yaml             # Output-fidelity cases (Benchmark F)
│   └── interpretation_questions.yaml   # MCQ + open-ended questions (Benchmark G)
├── corpus/
│   └── prompts.yaml                    # 66 prompts (23 standard + 18 hard transcription + 25 inference)
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
│   ├── biomni_shim.py                  # Subprocess shim driving upstream Biomni
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
├── bench_ablation.py                   # H — DAG-aware vs DAG-blind planner ablation
├── make_ablation_figure.py             # H — figure + paired stats for Benchmark H
├── bench_reflection_ablation.py        # I — completeness-reflection ablation
├── make_reflection_figure.py           # I — figure + paired stats for Benchmark I
├── bench_competitor_dag_ablation.py    # J — competitor DAG-prompt ablation (Claude Code)
├── make_competitor_dag_figure.py       # J — figure + paired stats for Benchmark J
├── rescore_planning.py                 # Re-evaluate existing plans with updated metrics
├── recovery_taxonomy.py                # Classify Benchmark B responses
├── merge_runs.py                       # Combine runs across models/sessions
├── supp_table_models.py                # Supplementary Table 2 (model registry × empirical stats)
├── Makefile                            # Convenience orchestration
└── results/                            # Gitignored outputs (CSV, JSON, PDF)
```

## The ten benchmarks

| ID | Claim | Needs API key | Needs infra | Make target |
|---|---|---|---|---|
| **A** | FlowAgent generates valid plans from natural language | yes | no | `make plan` / `make plan-all` |
| **B** | FlowAgent self-heals faults that break traditional WMS (28 faults, 3 tiers) | yes | no | `make recovery` |
| **C** | Generated Nextflow / Snakemake is valid and preserves plan intent | no (preset path) | `nextflow` + `snakemake` for `.validate()` | `make gen` |
| **D** | All six execution backends function | no | best-effort — mock mode if infra absent | `make exec` |
| **E** | FlowAgent is competitive with other agentic bio systems on the same corpus | yes | BioMaster + AutoBA + Biomni + Claude Code clones / CLIs (Edison opt-in) | `make competitors` / `make competitors-all` |
| **F** | FlowAgent's *outputs* match published references (Spearman ρ / Jaccard / F1) | no — pure scorer | network for first-run reference download | `make fidelity-run` (live) / `make fidelity` (score-only) |
| **G** | LLMs interpret bioinformatics outputs correctly + abstain when evidence is insufficient | yes | reference files materialised by F | `make interpretation` |
| **H** | Telling FlowAgent's planner about the dependency DAG improves bioinformatics plan quality | yes | no | `make ablation` / `make ablation-pilot` |
| **I** | DAG-Plan-style completeness validator + LLM reflection retry improves plan quality | yes | no | `make reflection` / `make reflection-pilot` |
| **J** | Telling a *competitor* framework (Claude Code) about the DAG via prompt-only intervention improves its plan quality | yes (Claude Code CLI auth) | Claude Code CLI installed | `make competitor-dag-ablation` / `make competitor-dag-ablation-pilot` |

### Prompt corpus

`corpus/prompts.yaml` contains **66 prompts** across two scoring tiers:

- **41 transcription prompts** (`tier: transcription`, the default) —
  the historical corpus. Each prompt names the canonical tools to use,
  so the score measures whether the LLM faithfully turns a tool list
  into a structured plan with valid commands, dependencies, and forbidden-tool
  exclusions. 23 are "standard" everyday workflows; the remaining 18
  (`hard_*`) stress niche domains (bisulfite, metagenomics, Hi-C),
  long chains, modern tool selection, and R-package wrappers.
- **25 inference prompts** (`tier: inference`, IDs prefixed `inf_`) —
  the new tier, scored by [`score_plan_inference`](harness/metrics.py).
  Each prompt describes the *goal* and *input data only*, never tool
  names ("Quantify transcript abundance from paired-end RNA-seq for
  downstream DESeq2"). Each prompt declares an
  `acceptable_tool_sets` list (e.g. `[[salmon, multiqc],
  [kallisto, multiqc], [star, featurecounts, multiqc]]`); the plan
  passes only if **at least one** of those sets is fully covered using
  *strict* tool matching (the prose fallback is disabled). Hallucinated
  tools, malformed commands (per
  [`harness/command_validator.py`](harness/command_validator.py)) and
  forbidden tools all gate `overall_pass`.

**What this benchmark does NOT test.** Whether the LLM picks the
*best* toolchain among the acceptable set, runtime/memory profile of
the resulting pipeline, scientific correctness of downstream
parameters (e.g. DESeq2 `lfcShrink` flavour). Those are evaluated end-
to-end by Benchmark F, not by plan inspection.

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

**Recovery contract.** A recovery proposal can either patch a single
command (`patch_command`), restructure the DAG via a structured
`plan_patch` operation (`insert_before`, `insert_after`,
`replace_step`, `remove_step`), or refuse (`refuse`). The LLM is
prompted in two phases (diagnose → respond) so it commits to a fault
class before being shown the patching guidance. DAG patches preserve
acyclicity and are rolled back on failure (see
[`flowagent/core/workflow_dag.py`](../flowagent/core/workflow_dag.py)
`apply_plan_patch`).

**Antipatterns rejected up front.** Bare no-ops (`true` / `:` /
`exit 0` / `test 0`), trailing failure-suppression operators
(`|| true`, `|| continue`, `|| :`, `|| exit 0`), leading `set +e`,
and "fixes" that drop the original tool family and leave only shell
builtins are detected by
[`_is_recovery_antipattern`](../flowagent/core/workflow_manager.py)
before execution. Every fault in
[`harness/fault_inject.py`](harness/fault_inject.py) declares
non-empty `outputs`, so the verifier engages on every cell — a
"successful" recovery that produces no artifacts is failed
explicitly. Recovery outcomes that pass exit-code-wise but match a
no-op shape are bucketed as `cheat_repair` by
[`recovery_taxonomy.py`](recovery_taxonomy.py), separately from
`unsafe_repair` (silent corruption of valid data).

**What this benchmark does NOT test.** Recovery from genuine data
corruption (Tier 3, where the right answer is `refuse`), recovery
across multiple sequential failures in one run (each fault is
isolated), or the wall-clock cost of recovery (we report attempts and
prompt size, not minutes-to-fix).

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

`config/interpretation_questions.yaml` contains MCQ + open-ended
questions across the same 7 datasets as Benchmark F. Every question
is tagged with an `evidence_class`:

- `data_required` — answer is derivable only from the supplied input
  files. Each dataset contributes **≥3** of these.
- `internal_knowledge` — answer comes from textbook biology /
  experimental design (e.g. "what does an SUZ12 ChIP-seq target?");
  retained as a control to separate "the model knows the
  field" from "the model can read the supplied data".
- `calibrated_refusal` — exactly one option says "the supplied data
  cannot answer this"; correctness rewards the refusal. Tests
  abstention.

Open-ended questions declare `depends_on_inputs: [...]` listing which
input fields they require, and the rubric only credits claims
derivable from those files (out-of-evidence speculation is penalised
explicitly). Responses are graded by an LLM judge (default `gpt-5.4`)
against five anchored score bands (0-20, 21-40, 41-59, 60-79, 80-100)
with the **pass mark stated explicitly (≥60)**. The judge returns
structured JSON with `score`, `hits[]`, `misses[]`, `fabrications[]`,
`grounding_quote`, and `justification` so each judgment is auditable
in the metrics CSV.

MCQ responses are extracted by a tag-aware parser (the prompt asks
for `<answer>X</answer><explain>...</explain>`) with a tiered regex
fallback for models that emit prose, and a final pass that prefers
the *last* in-set capital letter (so "I believe the answer is B"
returns `B`, not `I`).

Inter-judge calibration is a separate harness:

```bash
python benchmarks/judge_calibration.py \
    --metrics results/interpretation/<run>/metrics.csv \
    --judge-a gpt-5.4 --judge-b claude-opus-4-7 --n 30
```

emitting Pearson r, Cohen's κ on pass/fail, and mean score delta —
report in the manuscript supplement.

The benchmark feeds each dataset's reference file (DE table, peak BED,
truth VCF) directly to the model under test — FlowAgent itself is not in
the loop. This makes Benchmark G a model-vs-model comparison on
deterministic inputs, in the spirit of BixBench.

**What this benchmark does NOT test.** Multi-turn dialogue,
follow-up clarification, or the ability to *generate* analysis code
(only to interpret existing outputs). Open-ended grading is by LLM
judge, calibrated against a second judge but not against
field-expert annotation — see the calibration harness output for the
inter-judge κ and treat scores accordingly.

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

**Reasoning capability (`reasoning: bool`, `reasoning_default: low|
medium|high|none`)** is declared per model in `models.yaml`, sourced
from each provider's documented model card (Anthropic extended-
thinking, Google Gemini thinking-budget, OpenAI reasoning-effort).
[`harness/plot.py`](harness/plot.py) reads this YAML at figure-
generation time, so the recovery reasoning-vs-non-reasoning split
panel is always grounded in the latest provider documentation rather
than a hand-curated list.

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

Runs Benchmarks A–D in mock mode to verify the harness imports cleanly and
the scoring pipeline is sound. The three ablations (H, I, J) and the
live-run benchmarks (E, F, G) are not in `make smoke` because they
require API keys (or a Claude Code login for J); pilot them instead with:

```bash
make ablation-pilot              MODEL=claude-haiku-4-5         # ~$0.05, 5 prompts × 2 arms
make reflection-pilot            MODEL=claude-haiku-4-5         # ~$0.20, 8 prompts × 2 arms
make competitor-dag-ablation-pilot CDAG_MODEL=claude-haiku-4-5  # ~$0.10-0.50, 3 prompts × 2 arms (Claude Code)
```

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

Runs the default-sweep competitors (currently `flowagent`,
`biomaster`, `autoba`, `biomni`, `claude_code`, plus optional
zero-shot `raw_<model_id>` baselines) on the same prompt corpus,
scored with the same `score_plan` metrics so the comparison is
apples-to-apples.

**Opt-in lanes excluded from the default sweep.** `edison` is
registered but **not** included in `make competitors`. Edison
Analysis is structurally different from the other competitors -- it
runs the full bioinformatics pipeline end-to-end (3-15 min/task and
real credits) instead of just generating a plan, so it is not
directly comparable on wall-clock or pass-rate. Each timed-out cell
also still consumes credits because the analysis continues on
Edison's servers after the harness kills the local subprocess. To
include it, name it explicitly:

```bash
# Edison-only sweep (use a long timeout so polling can finish):
python bench_competitors.py --competitors=edison \
    --model=gpt-4.1 --replicates=3 --timeout=1800 --out=results

# Manuscript-grade four-way comparison (FlowAgent + Claude Code +
# Biomni + Edison) — uses --competitors=...,edison explicitly and
# enforces a credit cap:
make competitors-all MODEL=claude-haiku-4-5 REPLICATES=3 EDISON_BUDGET=50
```

The opt-in set is defined as `_OPT_IN_COMPETITORS` in
[`bench_competitors.py`](bench_competitors.py) and pinned by
`tests/test_competitors.py::TestOptInFilter`.

**Fairness convention -- universal no-DAG default for competitors.**
Every non-FlowAgent competitor in Benchmark E runs in **DAG-blind**
mode by default. FlowAgent's differentiator is its DAG-aware planner
(prompt + retry loop + structured-output schema), and any
DAG-related signal in a competitor's plan would silently hand them
part of FlowAgent's contribution. The flip is enforced at two
layers, depending on whether the competitor exposes a prompt knob:

| Competitor | Mechanism | Default |
|---|---|---|
| `claude_code` | Shim has two prompt templates; `--with-dag-instruction` flag toggles. `ClaudeCodeCompetitor(with_dag=...)` propagates. | `with_dag=False` (DAG-blind prompt) |
| `edison` | Shim has two `_PLAN_GUIDELINES` system-prompt variants; `--with-dag-instruction` flag toggles. `EdisonCompetitor(with_dag=...)` propagates. | `with_dag=False` (DAG-blind prompt) |
| `raw_<model>` | Shim has two `_RAW_LLM_SYSTEM_PROMPT` variants. `RawLLMCompetitor(with_dag=...)` propagates. | `with_dag=False` (DAG-blind prompt) |
| `biomni` | Agent has no DAG-aware mode upstream (LangChain ReAct loop). Shim **does not synthesise** linear `[step_N-1]` deps from tool-call order. | Always empty `dependencies` |
| `biomaster` | PLAN.json has no `dependencies` field. Shim **does not synthesise** `[step_N-1]` from `step_number` ordering. | Always empty `dependencies` |
| `autoba` | Plan is a flat list of task sentences. Shim **does not synthesise** `[step_N-1]` from enumeration order. | Always empty `dependencies` |
| `flowagent` | The system under test. Uses its DAG-aware planner. | `LLM_DAG_AWARE=true` (real DAG) |

The combined effect: in Benchmark E, only `flowagent` plans carry
non-trivial `dependencies`, so `dag_edge_density` /
`parallel_width` / `stage_efficiency` become a clean signal of "did
the planner actually think about a DAG?" rather than a parsing
artefact. The metric pipeline normalises flat-list plans
(`parallel_width=1`, `stage_efficiency=1.0`) so flat competitors
aren't unfairly inflated either way. See [Benchmark
J](#benchmark-j--prompt-level-dag-instruction-for-competitors) for
the paired ablation that turns the prompt-level DAG instruction back
on for competitors only, in isolation. The DAG-aware opt-in lanes
live under `*_dag_aware`-suffixed slugs and are exercised only by
Benchmark J.

**Pinned by tests.** The convention is enforced by:
- `tests/test_competitors.py::TestRegistry::test_default_competitors_are_dag_blind` -- registry-level black-box check.
- `tests/test_shim_no_dag_synthesis.py` -- per-shim unit tests for Biomni / BioMaster / AutoBA parsers.
- `tests/test_claude_code_shim.py`, `tests/test_edison_shim_dag_toggle.py`, `tests/test_raw_llm_dag_toggle.py` -- prompt-template defaults + slug rename for the toggleable competitors.

A future PR that re-introduces a DAG instruction or a synthetic
`[step_N-1]` chain in any default Benchmark E competitor will turn
at least one of these red.

The four-way ablation comparison (FlowAgent vs Claude Code vs Biomni vs
Edison Analysis) requested by the manuscript can be launched as:

```bash
make competitors-all MODEL=claude-haiku-4-5 REPLICATES=3 EDISON_BUDGET=50
```

`EDISON_BUDGET` is forwarded as `--edison-budget-credits`; once the
shared budget file (default `$TMPDIR/edison_budget.json`) crosses the
cap, further Edison cells short-circuit with a clear error envelope. Results land in
`results/competitors/<ts>/` with per-row `competitor`, `plan`,
`prompt_tokens`, `completion_tokens`, `cost_usd`, `wall_seconds`, and the
standard scoring columns. At the end of each run, the driver prints a
per-competitor rollup with **two co-primary outcomes** plus cost, and
writes it to `summary.tsv`:

```
Head-to-head rollup (two co-primary metrics + cost):
  Pass% = strict overall_pass rate.  Tools% = mean expected-tool fraction (partial credit, crashes excluded).
  Competitor        Pass   Fail  Crash   Pass%  Tools%    $/cell    Wall
  -----------------------------------------------------------------------
  FlowAgent         8/10      2      0   80.0%   95.0%   $0.0123   15.4s
  BioMaster         4/10      4      2   40.0%   62.5%   $0.0087   18.2s
  AutoBA            5/10      5      0   50.0%   71.0%   $0.0195   22.1s
  Biomni            6/10      3      1   60.0%   83.3%   $0.0210   25.0s
```

Where:
- **Pass** (strict) = plan produced and scored True on every `score_plan`
  gate. The headline single-number ranking.
- **Tools** (partial credit) = mean `tools_present_fraction` over scored
  cells (crashes excluded). Pre-registered as a co-primary outcome so a
  5-of-6 plan no longer scores identically to a 0-of-6 plan, and so
  narrative-style competitors aren't erased by a single missed gate.
- **Fail** = plan produced but missed at least one scoring gate
  (workflow type, expected tools, forbidden tools, min step count).
- **Crash** = the competitor raised before producing any scorable plan.
  Broken out separately so robustness shows up as its own column rather
  than silently dragging down the pass rate.

Both headline metrics also appear as side-by-side panels in
`results/figures/competitors.pdf`.

**Subsetting:**

```bash
python bench_competitors.py \
  --competitors=flowagent,biomaster \
  --prompts=rnaseq_kallisto_basic,hard_full_germline_pipeline \
  --replicates=2
```

`--mock` runs offline with canned plans derived from each prompt's
`gold_preset` / `expected_tools`, useful for smoke-testing the harness.

**If you see `cli-adapter: RuntimeError: … no JSON envelope on stdout`:**
the subprocess shim did not print a parseable JSON line on stdout (often
upstream crashed before the shim’s final `print`, or the wrong `python`
/env was used). Progress lines only show a short status — open the run’s
`results/competitors/<ts>/results.json` or `metrics.csv` for the full error,
or run the shim by hand, e.g.
`python harness/autoba_shim.py --prompt "…" --model gpt-4.1 --autoba-dir "$AUTOBA_DIR"`
and read stderr.

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
- **macOS:** if the shim dies with OpenMP / `libomp.dylib already initialized`
  and exit code **-6**, the harness sets `KMP_DUPLICATE_LIB_OK=TRUE` for the
  AutoBA subprocess (and `autoba_shim.py` does the same when run manually).
  You can also export it in your shell for other tools.
- **Exit -11 (`SIGSEGV`) with empty stdout/stderr:** the child crashed in native
  code (typically PyTorch / Accelerate / BLAS) before Python could print or
  flush. The harness now runs the shim with `python -u`, `PYTHONFAULTHANDLER=1`,
  single-threaded BLAS/OMP defaults (`OMP_NUM_THREADS=1`, etc.), and the shared
  helpers in [`harness/autoba_child_env.py`](harness/autoba_child_env.py). If it
  still segfaults: reinstall `torch`/`numpy` from the **same** conda channel,
  try `conda install pytorch cpuonly -c pytorch`, or run Benchmark E on Linux.

#### Biomni setup (one-off, ~15+ min for full upstream env)

Biomni (biorxiv [10.1101/2025.05.30.656746](https://www.biorxiv.org/content/10.1101/2025.05.30.656746v1))
is a LangGraph ReAct biomedical agent. The harness drives it through
[`harness/biomni_shim.py`](harness/biomni_shim.py), which sets
`react.configure(plan=True, …)` and a **bounded** LangGraph
`recursion_limit` (default 15, overridable via `BIOMNI_RECURSION_LIMIT`) so
runs stay closer in cost to the other competitors than Biomni’s interactive
default. Tool calls in the trace become `steps[]`; if the model only emits
a narrative plan, the shim falls back to text-derived steps. `workflow_type`
uses the same `biomaster_shim._classify_workflow_type` post-hoc mapper as
AutoBA / BioMaster.

```bash
# 1. Clone
git clone https://github.com/snap-stanford/Biomni.git /path/to/Biomni
cd /path/to/Biomni

# 2. Install the package (full scientific stack: see biomni_env/README.md)
pip install -e .

# 3. API keys & provider — follow upstream .env.example (ANTHROPIC_API_KEY,
#    OPENAI_API_KEY, LLM_SOURCE, …). Align the model with your sweep:
#    export BIOMNI_LLM=gpt-4.1   # or match OPENAI_MODEL / --model

# 4. Point the harness at the repo root (directory that contains biomni/)
echo 'BIOMNI_DIR=/path/to/Biomni' >> /path/to/flowagent/.env
```

Smoke-test:

```bash
python benchmarks/harness/biomni_shim.py \
  --prompt "Run a kallisto RNA-seq quantification on paired-end FASTQs" \
  --model gpt-4.1
```

**Notes:**

- By default the shim sets `BIOMNI_USE_TOOL_RETRIEVER=true` and, after
  `configure()`, rebuilds the LangGraph app with **prompt-based retrieval**
  (mirroring Biomni’s own `go()`), because **OpenAI caps the `tools` array at
  128** while the full Biomni registry is much larger. Set
  `BIOMNI_USE_TOOL_RETRIEVER=false` only for providers without that limit
  (and use `BIOMNI_MAX_TOOLS_PER_REQUEST` if a different cap applies).
- Token / cost: OpenAI models use `get_openai_callback`; other providers
  use LangChain’s `UsageMetadataCallbackHandler` when available — otherwise
  counts may be zero while the plan is still scored.
- If Biomni crashes with ``AttributeError: module 'biomni.tool.…' has no attribute '…'``,
  the tool registry is out of sync with the Python modules (upstream drift).
  ``biomni_shim.py`` patches ``api_schema_to_langchain_tool`` to register a
  small placeholder for missing APIs so the agent can still run in Benchmark E.
- **ImportError: zarr-python major version > 2 is not supported** (often while
  importing ``scanpy`` / ``anndata``): your env has **Zarr 3.x**, but the
  installed **anndata** build only supports **Zarr 2.x**. In the same conda env
  as Biomni, pin Zarr v2, then retry:

  ```bash
  pip install "zarr>=2.18,<3"
  # or: conda install "zarr<3"
  ```

  If conflicts persist, use Biomni’s documented ``biomni_env`` setup or a
  dedicated conda env for Benchmark E competitors.
- **ImportError: cannot import name ``ZarrRuntimeWarning`` from ``zarr.errors``**
  (or other broken imports under ``site-packages/zarr/``): the install is
  **mixed or half-upgraded** (v2 and v3 files together). Remove Zarr completely,
  then install a single v2 line:

  ```bash
  pip uninstall zarr zarr-python -y   # both names can exist
  pip install "zarr>=2.18,<3"
  python -c "import zarr; print(zarr.__version__)"
  ```

  With conda: ``conda remove zarr --yes`` then ``conda install -c conda-forge "zarr>=2.18,<3"``.

#### Claude Code setup (one-off, ~2 min)

Claude Code is Anthropic's general-purpose CLI coding agent. The
adapter ([`harness/competitors.py:ClaudeCodeCompetitor`](harness/competitors.py))
drives it via [`harness/claude_code_shim.py`](harness/claude_code_shim.py)
with `--print --output-format json --permission-mode plan` so the
agent emits its full reply as a single JSON object on stdout and never
edits files.

```bash
# 1. Install the Claude Code CLI per Anthropic's docs:
#    https://docs.claude.com/en/docs/claude-code/overview
# 2. Authenticate
claude /login

# 3. (optional) pin the binary if it isn't on PATH
echo 'CLAUDE_CODE_BIN=/path/to/claude' >> /path/to/flowagent/.env
# 4. (optional) pin a model; default is whatever Claude Code chose
echo 'CLAUDE_CODE_MODEL=claude-sonnet-4-5' >> /path/to/flowagent/.env
```

Smoke-test the shim directly before wiring it into the sweep:

```bash
python benchmarks/harness/claude_code_shim.py \
  --prompt "Run a kallisto RNA-seq quantification on paired-end FASTQs"
```

Expect a JSON envelope on stdout with `plan`, `prompt_tokens`,
`completion_tokens`, `cost_usd`, `wall_seconds`. Cost is read from
Claude Code's own `total_cost_usd` field.

#### Edison Analysis setup (one-off, ~3 min)

Edison Scientific's Edison Analysis (FutureHouse spinout) is a hosted,
execution-oriented bioinformatics agent. The adapter
([`harness/competitors.py:EdisonCompetitor`](harness/competitors.py))
drives it via [`harness/edison_shim.py`](harness/edison_shim.py),
overriding the system prompt so Edison emits a JSON workflow plan
without actually running any code.

```bash
# 1. Install the SDK
pip install edison-client

# 2. Sign up at https://platform.edisonscientific.com  (academic .edu
#    accounts get a free monthly credit allocation), generate an API key.
echo 'EDISON_API_KEY=...'              >> /path/to/flowagent/.env

# 3. (recommended) cap cumulative spend across this process
echo 'EDISON_BUDGET_CREDITS=100'        >> /path/to/flowagent/.env
```

Edison Analysis runs are slow (3–10 min/task) and cost credits per
task. Always pilot first:

```bash
make competitors-all MODEL=claude-haiku-4-5 REPLICATES=1 \
    EDISON_BUDGET=20  # hard cap
```

The shim writes a shared budget file at `$EDISON_BUDGET_FILE` (default
`$TMPDIR/edison_budget.json`) so parallel cells share the running
total; once consumed exceeds `EDISON_BUDGET_CREDITS`, subsequent
cells short-circuit with a clear error envelope.

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

**Disk cleanup.** Pass `CLEANUP=1` to delete heavy intermediates
(`raw_data/`, kallisto index, FastQC HTML) immediately after each
*successful* cell. Recovers ~50–80 GB per RNA-seq cell while preserving
the candidate output file the scorer needs:

```bash
make fidelity-run MODEL=gpt-4.1 CONCURRENCY=3 CLEANUP=1
```

Three safety properties:
- Cleanup only fires when the declared `output_relpath` exists and is
  non-empty — never on a failed or partial cell, so you can always
  re-run cells that died.
- Cleanup is opt-in (default off) so existing workflows don't lose
  data unexpectedly.
- The driver's skip-path *also* runs cleanup, so you can recover space
  from already-completed cells mid-sweep without re-running them:

```bash
# Already finished some cells — now want the disk back without re-running
make fidelity-run MODEL=gpt-4.1 CLEANUP=1
# logs:
#   [skip]    gse52778_dex_de__gpt-4.1__rep0: output already exists
#   [cleanup] gse52778_dex_de__gpt-4.1__rep0: freed 78.3 GB
```

What gets deleted vs preserved per cell:

| Deleted on success | Preserved (always) |
|---|---|
| `raw_data/` (FASTQs + SRA cache + reference genome + GEO metadata) | `prompt.txt`, `run.log`, `flowagent_output/` (workflow.json, notebook.ipynb) |
| `results/rna_seq_kallisto/kallisto_index/` (~3 GB index) | `results/rna_seq_kallisto/deseq2/` (DE table, gene counts) |
| `results/rna_seq_kallisto/fastqc/` (HTML reports) | `results/rna_seq_kallisto/kallisto_quant/<sample>/abundance.h5` |
| `results/macs2/bowtie2_alignments/` (ChIP/ATAC intermediates) | `results/macs2/peaks.narrowPeak` |
| `results/gatk/aligned/` (variant-calling intermediates) | `results/gatk/filtered.vcf.gz` |

**Caveat after cleanup:** re-running a cleaned cell from scratch
(via `FORCE=1` plus a deleted output file) will re-download `raw_data/`
— costing 50+ GB and several hours. Only enable cleanup when you're
confident the run is final. Smart-resume's normal "output exists →
skip" path is unaffected: cleaned cells re-run instantly because the
scorer-relevant output is preserved.

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

### Benchmark H — DAG-awareness ablation

```bash
# 5-prompt smoke (validates the whole pipeline; ~$0.05 on Claude Haiku)
make ablation-pilot MODEL=claude-haiku-4-5

# Full 66-prompt × MODEL × REPLICATES × 2 arms sweep
make ablation MODEL=claude-haiku-4-5 REPLICATES=3

# Render figure_ablation.pdf + figure_ablation__stats.tsv (uses the most recent run)
make ablation-figure
# Or point at a specific run
make ablation-figure ABLATION_DIR=results/ablation/2026-05-06T21-33-42
```

Tests whether telling the LLM about the dependency DAG -- the standard
"Dependencies must form a valid DAG (no cycles)" rule plus the
`dependencies` field in the structured-output schema -- changes
bioinformatics plan quality. Same model, same prompt, two planner
configurations:

| Arm | `LLM_DAG_AWARE` | Schema | Prompt rule |
|---|---|---|---|
| `dag_aware` | `true` (default) | [`WorkflowPlanSchema`](../flowagent/core/schemas.py) (with `dependencies`) | "Dependencies must form a valid DAG (no cycles)" |
| `dag_blind` | `false` | [`WorkflowPlanSchemaNoDAG`](../flowagent/core/schemas.py) (no `dependencies`) | Flat ordered list, dependencies never mentioned |

Empty dependency lists are injected post-parse on the DAG-blind side
so the resulting plan is a trivially valid DAG with zero edges -- this
keeps the existing `dag_valid` gate green for both arms and makes the
delta show up only in metrics that actually reflect plan quality:

| Metric | Where | What it measures |
|---|---|---|
| `dag_edge_density` | [`harness/metrics.dag_shape`](harness/metrics.py) | edges / max(steps - 1, 1). Sanity: should be 0 for `dag_blind`. |
| `parallel_width` | same | max width of a topological layer. Sanity: 1 for `dag_blind`. |
| `stage_efficiency` | same | `num_steps / num_dag_layers` (DAG-Plan analogue, Gao & Mu 2025). 1.0 for linear; >1 for parallel; normalised to 1.0 when there are zero edges. |
| `completeness_pass` | [`harness/metrics.completeness_metrics`](harness/metrics.py) | does the plan satisfy the four DAG-Plan-style structural rules (every `align` has an `index`/`download` ancestor; every `download` has a consumer; every `quantify`/`call`/`de` reaches an informative sink; weakly connected with terminal sink)? |
| `tools_present_fraction` | [`score_plan`](harness/metrics.py) | did DAG awareness change tool selection? |
| `hallucination_rate` | same | did it suppress unknown / made-up tool names? |
| `preset_command_f1` | same | did it improve adherence to the gold preset commands? |
| `overall_pass` | same | did it move the gating outcome? (paired McNemar) |

`stage_efficiency` and `completeness_pass` were added to support
Benchmark I (see below) but are recorded for *every* run, so they
appear as new columns in Benchmark H's `paired_metrics.csv` too.

Output:

* `results/ablation/<ts>/dag_aware/results.jsonl` + `metrics.csv`
* `results/ablation/<ts>/dag_blind/results.jsonl` + `metrics.csv`
* `results/ablation/<ts>/paired_metrics.csv` -- single CSV with an `arm`
  column, joined by `(model, input_id, replicate)` so paired tests
  ([`make_ablation_figure.py`](make_ablation_figure.py)) line up the
  same prompt across the two planner configurations.
* `figure_ablation.pdf` / `.png` -- per-metric arm means with bootstrap
  95% CIs.
* `figure_ablation__stats.tsv` -- paired Wilcoxon (continuous) and
  McNemar (`overall_pass`) per metric (default basename `<out>__stats.tsv`,
  override with `--stats-out`).

**Pilot results (3 prompts, Claude Haiku 4.5, single replicate)
already in the repo** confirm the ablation is working end-to-end:
`dag_edge_density` 1.33 vs 0.0 and `parallel_width` 3.7 vs 1.0 between
the two arms; an early `hallucination_rate` signal (0.0 vs 0.21) hints
that DAG awareness keeps the LLM more disciplined, but a full 66-prompt
sweep is needed to call statistical significance. See
[`figures/figure_ablation_pilot.pdf`](../figures/figure_ablation_pilot.pdf).

### Benchmark I — completeness-reflection ablation

```bash
# 8-prompt smoke (validates the whole pipeline; ~$0.20 on Claude Haiku)
make reflection-pilot MODEL=claude-haiku-4-5

# Full 66-prompt × MODEL × REPLICATES × 2 arms sweep
make reflection MODEL=claude-haiku-4-5 REPLICATES=3

# Same sweep but cap retries (default 2) — useful to study cost/benefit
make reflection MODEL=claude-haiku-4-5 REPLICATES=3 MAX_RETRIES=1

# Render figure_reflection.pdf + stats_reflection.tsv (uses the most recent run)
make reflection-figure
# Or point at a specific run
make reflection-figure REFLECTION_DIR=results/reflection/2026-05-06T22-09-17
```

Tests whether the DAG-Plan-style structural completeness validator
(introduced in [`flowagent/core/completeness.py`](../flowagent/core/completeness.py))
plus an LLM reflection retry loop improves plan quality. Both arms keep
`LLM_DAG_AWARE=true` so this is a clean A/B of the reflection retry
alone (not a confound with the Benchmark H DAG-prompt toggle):

| Arm | `LLM_COMPLETENESS_REFLECT` | Behaviour |
|---|---|---|
| `reflect_on` | `true` (default) | After each plan, run [`validate_workflow_completeness`](../flowagent/core/completeness.py). On failure, append the failure list to the conversation as a reflection prompt and re-query the LLM, up to `LLM_COMPLETENESS_MAX_RETRIES` times (default 2 → max 3 LLM calls). The most recent plan wins regardless. |
| `reflect_off` | `false` | The validator still runs at score time (so `completeness_pass` and `num_completeness_failures` are reported for both arms), but the planner accepts the first draft without retry. |

Both arms share the rest of the planner stack: the typed-node `kind`
field on each step, the post-hoc heuristic `fill_missing_kinds` fallback,
the reference-download wiring, and the structured-output schema. The
delta is the retry loop alone.

The four structural rules the validator enforces (each contributes at
most one failure message; see
[`validate_workflow_completeness`](../flowagent/core/completeness.py)):

1. Every `align` step needs an `index` or `download` ancestor.
2. Every `download` step must have a downstream consumer (no
   "fetched but never used" references).
3. Every `quantify`/`call`/`de` step must reach an informative sink
   (`report`, `terminal`, or another `quantify`/`call`/`de`) — chains
   ending in a glue `other` sink fail.
4. The DAG must be weakly connected with at least one terminal sink
   and no cycles.

Step `kind` is one of `download | index | qc | trim | align | sort |
dedup | call | quantify | de | report | terminal | other` and is
emitted by the LLM (asked for in the prompt) or, on the
`WorkflowPlanSchemaNoDAG` ablation arm + JSON-repair retries that drop
fields, inferred heuristically from `command + name`.

Headline metrics for this benchmark:

| Metric | What it measures |
|---|---|
| `completeness_pass` | binary pass/fail per plan against the four rules. Headline McNemar test in `stats_reflection.tsv`. |
| `num_completeness_failures` | count of rule violations per plan. |
| `completeness_attempts` | number of LLM calls used to reach the final plan (1 = no retries; 2-3 = reflection fired). |
| `stage_efficiency` | `num_steps / num_dag_layers`. Reflected here because reflection can change graph topology. |
| `cost_usd` | the cost overhead of reflection — directly comparable across arms. |
| `overall_pass`, `tools_present_fraction`, `hallucination_rate` | inherited from `score_plan`; should be neutral in a clean reflection-only ablation. |

Output (mirrors Benchmark H's layout):

* `results/reflection/<ts>/reflect_on/results.jsonl` + `metrics.csv`
* `results/reflection/<ts>/reflect_off/results.jsonl` + `metrics.csv`
* `results/reflection/<ts>/paired_metrics.csv` -- joined by
  `(model, input_id, replicate)`.
* `figure_reflection.pdf` / `.png` -- per-metric arm means with
  bootstrap 95% CIs (panels: `completeness_pass`, `stage_efficiency`,
  `overall_pass`, `dag_edge_density`, `parallel_width`,
  `completeness_failures`, `hallucination_rate`,
  `tools_present_fraction`, `num_steps`).
* `stats_reflection.tsv` -- paired Wilcoxon (continuous) + McNemar
  (binary) per metric, with `mean_reflect_on`, `mean_reflect_off`,
  `mean_diff`, `p_value`, `test`, and `(b_only, c_only)` discordant
  counts for the McNemar tests.

**Pilot results (8 RNA-seq prompts, Claude Haiku 4.5, single
replicate)** in `results/reflection/2026-05-06T22-09-17/`:
`completeness_pass` rises from 0.75 to 1.00 (`c_only=2, b_only=0` —
2 plans recovered by reflection, none degraded). The two recovered
plans were `rnaseq_kallisto_basic` and `rnaseq_geo`, both with
`download_*` steps that lacked a downstream consumer in the first
draft. Cost overhead is concentrated on the cells that actually
retry: per-cell mean cost rose from $0.013 to $0.018 (~40%) but only
2/8 cells issued retries.

Use the new `MAX_RETRIES` knob to study cost/benefit:

```bash
# 1 retry only (max 2 LLM calls per plan)
make reflection MAX_RETRIES=1

# Validation runs but no retries — completeness_pass still reported
make reflection MAX_RETRIES=0
```

`MAX_RETRIES=0` is also useful as a third arm: it keeps the validator
in the scoring loop (so the metric is comparable) but disables the
retry, isolating the cost of *running the validator* from the cost of
*acting on it*.

### Benchmark J — competitor DAG-prompt ablation

```bash
# 3-prompt smoke (~$0.10-0.50 on Claude Haiku 4.5)
make competitor-dag-ablation-pilot CDAG_MODEL=claude-haiku-4-5

# Full 66-prompt × REPLICATES × 2 arms sweep, Claude Code only
make competitor-dag-ablation CDAG_MODEL=claude-haiku-4-5 REPLICATES=3

# Render figure_competitor_dag__claude_code.pdf + per-competitor stats
make competitor-dag-figure
# Or point at a specific run
make competitor-dag-figure CDAG_DIR=results/competitor_dag_ablation/2026-05-07T15-06-15
```

#### What it tests

Benchmark H toggles FlowAgent's **own** DAG awareness, which flips
both the planner prompt **and** the structured-output schema
(`WorkflowPlanSchemaNoDAG` removes the `dependencies` field entirely).
That ablation conflates "prompt-level DAG instruction" with
"schema-level DAG enforcement".

For *competitor* frameworks (Claude Code, Biomni, Edison) we don't
control the schema. Benchmark J asks the cleanest question we can:
**does prompt-level DAG instruction alone change a competitor's plan
quality?** If yes, prompt engineering is enough. If no — and FlowAgent's
H ablation shows a positive delta — schema-level enforcement is the
necessary intervention, not just any mention of the word "dependency"
in a prompt. That's the manuscript story for FlowAgent's contribution
beyond raw LLM prompting.

#### Arms (Claude Code)

The Claude Code shim
([`harness/claude_code_shim.py`](harness/claude_code_shim.py)) ships
with two prompt templates and a `--with-dag-instruction` CLI flag.
`ClaudeCodeCompetitor(with_dag=...)` propagates the flag.

| Arm | Prompt template | Slug |
|---|---|---|
| `dag_blind` | Schema example has **no** `dependencies` field; no topological-order rule. **This is the default** -- the slug `claude_code` in every other benchmark refers to this DAG-blind variant. | `claude_code` |
| `dag_aware` | Schema includes `dependencies: [<prior step>]`, plus rules: *"Steps must be in topological order"* and *"`dependencies` must reference names of prior steps exactly."* Every other rule (tool-first command, no side effects, no markdown fences, etc.) is byte-identical to `dag_blind`, so the only experimental variable is the DAG instruction. Pinned by the unit test `TestSelectTemplate.test_non_dag_rules_unchanged_between_arms`. | `claude_code_dag_aware` |

The asymmetry between the two slugs is intentional: it lets the same
ablation arms co-exist as separate competitors in the registry (so
Benchmark J can run them paired) while keeping `claude_code` --
the slug Benchmark E uses -- pointing at the *fair head-to-head*
DAG-blind variant. The figure script knows which arm each row came
from via the `competitor_arm` column. The same convention applies to
`EdisonCompetitor` (default `edison` = DAG-blind, opt-in
`edison_dag_aware`) and `RawLLMCompetitor` (default `raw_<model>` =
DAG-blind, opt-in `raw_<model>_dag_aware`).

#### Why DAG-blind is the default for competitors

FlowAgent's contribution is a **DAG-aware planner** (LLM prompt asks
for `dependencies`, retry loop validates them against
`networkx.is_directed_acyclic_graph`, and the structured-output
schema reserves a slot for them). Other competitors don't have any
of that scaffolding; their only knob is the prompt template the
shim wraps around the user's request.

If we ran Benchmark E's head-to-head with the competitor shims
**also** asking for `dependencies`, the comparison would be
contaminated: any "FlowAgent wins" delta could just as easily come
from "FlowAgent's prompt happens to ask for the right field". By
defaulting every competitor (including the raw-LLM lane) to the
DAG-blind template, Benchmark E isolates the value of FlowAgent's
*full* DAG-aware stack -- planner + retry loop + schema -- against
agents that get a vanilla bioinformatics-pipeline prompt with no
graph instructions. Benchmark J then re-introduces the DAG
instruction *only on the competitor side* to measure how much of
that gap closes from prompt engineering alone.

#### Currently supported competitors

* **Claude Code** — wired up. Default driver model
  `claude-haiku-4-5` (override with `CDAG_MODEL=claude-sonnet-4-5`
  for the manuscript figure).
* **Edison Analysis** — shim has the
  `--with-dag-instruction` toggle and `EdisonCompetitor(with_dag=...)`
  is supported (the default `edison` slug in Benchmark E is
  DAG-blind). Not yet enabled in `_build_arm_competitors`; flipping
  it on is a one-line change once the manuscript run gets a budget
  allocation for paired Edison cells (each task is 3-10 min and
  costs credits per call).
* **Biomni** — not wired up. Biomni's LangGraph ReAct loop has its
  own upstream system prompt and may ignore user-level DAG
  instructions entirely; adding it requires a shim-level prompt
  template the harness can toggle, plus an availability sanity-check
  that the framework actually emits `dependencies` fields when asked.
* **Raw LLM** — `RawLLMCompetitor(model_id, with_dag=...)` is
  supported (default `raw_<model>` slug is DAG-blind). Not currently
  in `_build_arm_competitors` because the manuscript story for
  Benchmark J focuses on agentic competitors with their own
  scaffolding; raw-LLM ablations against a DAG instruction
  duplicate Benchmark H's question more directly.

#### Headline metrics (same set as Benchmark H, plotted side by side)

| Metric | What it measures |
|---|---|
| `dag_edge_density` | Sanity: did the DAG-aware arm actually emit dependencies? On Claude Code we expect ≫0 in `dag_aware` and ≈0 in `dag_blind`. |
| `parallel_width` | Sanity: max width of a topological layer. Should rise with `dag_edge_density`. |
| `stage_efficiency` | `num_steps / num_dag_layers`. Higher = more parallelism exposed. |
| `overall_pass` | Strict gating outcome. Headline McNemar test in `figure_competitor_dag__claude_code__stats.tsv`. |
| `tools_present_fraction` | Partial-credit tool coverage. |
| `hallucination_rate` | Fraction of plan tools we don't recognise. |
| `preset_command_f1` | Token-F1 vs the gold preset command (subset metric — ~5 of 66 prompts have a gold preset). |
| `preset_name_jaccard` | Step-name Jaccard vs the gold preset (same subset). |
| `num_steps` | Raw step count. |

#### Output layout

```
results/competitor_dag_ablation/<ts>/
├── claude_code/
│   ├── dag_aware/
│   │   └── results.jsonl + results.json + metrics.csv + manifest.json
│   ├── dag_blind/
│   │   └── results.jsonl + ...
│   └── paired_metrics.csv          # joined by (model, input_id, replicate)
├── paired_metrics.csv              # cross-competitor — one row per cell
└── manifest.json
```

The figure script renders one PDF per competitor:
`figure_competitor_dag__claude_code.pdf` + the matching
`__stats.tsv`. When future competitors are added, each gets its own
PDF in the same run output.

#### Decision tree for interpreting the result

| dag_edge_density (aware) | overall_pass delta | What it tells you |
|---|---|---|
| ≈ 0 | any | Claude Code ignored the `dependencies` instruction. Prompt-level DAG instruction is insufficient for this framework. **Null result is itself a finding.** |
| > 0, sane | aware ≫ blind | Prompt engineering alone helps Claude Code. (FlowAgent's H delta should still be larger if schema-level enforcement adds value.) |
| > 0, sane | aware ≈ blind | Claude Code can emit DAGs but it doesn't help its plan quality. Suggests other quality determinants (tool selection, command syntax) dominate over graph structure. |
| > 0, sane | aware < blind | Adding the DAG instruction *hurts* — likely confuses the planner. (Unlikely but possible — useful signal that prompt engineering is fragile.) |

### Everything at once

```bash
make all         # single MODEL (default gpt-4.1): plan + recovery + gen + exec + report
make all-sweep   # full 30-model sweep: plan-all + recovery + gen + exec + competitors + rescore + merge + report
```

Use `make all` for a quick end-to-end smoke of one model (fast, cheap). Use
`make all-sweep` for the multi-model manuscript run — it automatically chains
`rescore → merge → report` in the right order so all models appear in the
final figures.

Benchmarks F, G, H, I, and J are **not** included in `all-sweep` because
they have distinct workflow shapes:
- **F** needs prior FlowAgent runs to score (or `fidelity-run` to drive
  end-to-end pipelines that take hours).
- **G** is an LLM-only sweep against fixed reference inputs.
- **H** and **I** are paired ablations (two arms per model) and would
  double the planning cost of `all-sweep`.
- **J** drives an external CLI (Claude Code) and uses different
  authentication / cost accounting than FlowAgent's own benchmarks.

Run them separately:

```bash
make references                                # one-time, materialises Benchmark F refs
make fidelity                                  # bulk-score a fidelity_runs/ tree
make interpretation MODEL=gpt-4.1 JUDGE=gpt-5.4
make ablation MODEL=claude-haiku-4-5 REPLICATES=3 && make ablation-figure
make reflection MODEL=claude-haiku-4-5 REPLICATES=3 && make reflection-figure
make competitor-dag-ablation CDAG_MODEL=claude-haiku-4-5 REPLICATES=3 && make competitor-dag-figure
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
| `competitors_agentic.pdf` | FlowAgent vs BioMaster vs AutoBA vs Biomni focused comparison |
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
| `make fidelity-run` (1 model) | 1 | ~12–24 h sequential, ~6–10 h at CONCURRENCY=3 | ~$3–15 (7 cases × 1 model); ~50–80 GB/RNA-seq cell on disk |
| `make fidelity-run CLEANUP=1` | 1 | same wall time | same cost; only output files retained, ~5 MB total |
| `make fidelity-run MODELS=a,b,c` | 3 | ~30+ h sequential | ~$10–45 (21 cells); without `CLEANUP=1`, **>1 TB peak disk** |
| `make interpretation` | 1 | ~5–10 min (32 questions) | ~$0.50–$2 |
| `bench_interpretation.py --models=…` | 10 | ~30–60 min | ~$5–15 |
| `make ablation-pilot` | 1 | ~3–5 min (5 prompts × 2 arms) | ~$0.05 on Claude Haiku |
| `make ablation` | 1 | ~30–60 min (66 prompts × 3 reps × 2 arms) | ~$1–4 on Claude Haiku, ~$10+ on flagship |
| `make reflection-pilot` | 1 | ~3–5 min (8 prompts × 2 arms) | ~$0.20 on Claude Haiku |
| `make reflection` | 1 | ~30–60 min (66 prompts × 3 reps × 2 arms) | ~$1.5–6 on Claude Haiku, ~$15+ on flagship (~40% overhead vs Benchmark H from retries) |
| `make competitor-dag-ablation-pilot` | Claude Code (1) | ~2–4 min (3 prompts × 2 arms) | ~$0.10–$0.50 on Claude Haiku 4.5 |
| `make competitor-dag-ablation` | Claude Code (1) | ~30–60 min (66 prompts × 3 reps × 2 arms, conc=2) | ~$5–15 on Claude Haiku 4.5; multiply by ~3-5× for Sonnet |
| `make ablation-figure` / `make reflection-figure` / `make competitor-dag-figure` | — | ~10–20 s | $0 |
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

**Benchmark F: `No .env file found` from cells four directories deep** —
FlowAgent's own dotenv loader (in `flowagent/config/settings.py`) only
checks `./env` and `$USER_EXECUTION_DIR/.env`, neither of which resolve
from a cell at `results/fidelity_runs/<cell>/`. The driver
(`bench_fidelity_run.py`) loads `.env` at startup via the harness's
walk-up loader and propagates the keys + `USER_EXECUTION_DIR` to every
subprocess, so this only manifests if you invoke `flowagent prompt`
directly inside a cell dir without the driver. Use `make fidelity-run`
or set `OPENAI_API_KEY` in your shell first.

**Benchmark F: pipeline died in <30 s with `rc=0 produced=False`** —
nine times out of ten this is the planner emitting a truncated
4-step "GEO download only" workflow because the prompt didn't trigger
the kallisto/DE expansion. Verify by inspecting the cell's
`flowagent_output/Unnamed_Workflow/workflow.json` — if it has 4 steps,
the prompt didn't trigger the analysis branch. The case prompts in
`config/fidelity_cases.yaml` are deliberately worded with
"RNA-seq kallisto pipeline" + "DESeq2 differential expression" to
get score ≥ 2 in `_detect_workflow_type`'s keyword scoring; if the
plan still truncates, check that the case's `prompt:` field hasn't
been edited to drop those keywords.

**Benchmark F: filling up disk** — pass `CLEANUP=1` to delete `raw_data/`
+ kallisto index + FastQC HTML after each successful cell. Recovers
~50–80 GB per RNA-seq cell. The driver's skip-path also runs cleanup,
so re-running with `CLEANUP=1` recovers space from already-completed
cells without re-running their pipelines. See "Disk cleanup" under
the Benchmark F section.

**Benchmark F: `pipefail` errors on Debian-based images** — `/bin/sh =
dash` on Debian/Ubuntu doesn't support `set -o pipefail`. The recovery
loop fires automatically and patches the command on first attempt;
no action needed. If you want to skip the recovery overhead entirely,
symlink `/bin/bash` over `/bin/sh` in your container.

## Out of scope (documented explicitly)

- **User study** (wet-lab vs. Nextflow-tutorial): requires IRB + human
  participants; noted as a future extension.
- **Concordance with nf-core published pipelines**: supported by Benchmark C's
  API but requires GB-scale test data; see `extras/concordance.md`.
- **Full live HPC / Kubernetes execution**: Benchmark D uses mocks when infra
  is absent and does live runs when present; production cluster behaviour
  is documented as a reviewer-reproduction path.
- **CGAT-core / WDL / CWL generators**: not implemented in FlowAgent today.
