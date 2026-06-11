# Benchmark CHANGELOG

This file records changes to the FlowAgent benchmarks corpus, scoring,
and harness, with an explicit note on which prior runs each change
**invalidates** for direct comparison. Anything below the watermark
should be re-run before being plotted alongside fresh numbers.

The format is based on [Keep a Changelog](https://keepachangelog.com/).
Versioning is benchmark-only and independent of the FlowAgent core
release cadence.

## [v2.1] — 2026-05-08 — Hardened hallucination detector

### Benchmark A — Planning (`bench_planning.py`)

#### `hallucinated_tools` column — schema change (v1 → v2)

| Version | Format | Example |
|---------|--------|---------|
| v1 (≤ v2.0) | `"name;name;..."` | `"kalsito;fakeblast"` |
| v2 (≥ v2.1) | `"name:category[:correction];..."` | `"kalsito:typo:kallisto;fakeblast:unknown"` |

Parsers that only need the names can split on `;` and take the substring
before the first `:`. Old archived CSVs (v1 schema) are still plotted
correctly; the figure auto-detects the schema version.

Migration: re-score an existing results directory to upgrade the column:

```
make rescore RESULTS_DIR=results/planning/<timestamp>
```

#### New columns (both `score_plan` and `score_plan_inference`)

| Column | Type | Description |
|--------|------|-------------|
| `num_hallucinated_typos` | int | Count of flagged tokens classified as `typo` |
| `hallucinated_typos` | str | `"token->correction;..."` for each typo |

#### New known-tools source

The hallucination check now uses a checked-in snapshot
(`data/known_tools.yaml`, ~16 k entries) built from:

- **Bioconda** channel repodata (`noarch` + `linux-64`)
- **Bioconductor** Software, Annotation, and Experiment package lists
- **Hand-curated runtime / infra list** (~60 entries: `bash`, `conda`,
  `nextflow`, `docker`, `aws`, `kubectl`, etc.)

The snapshot replaces the ~150-entry `_BIOINFO_TOOLS` hand-curated
whitelist.  `_BIOINFO_TOOLS` is retained as a fallback for environments
where `data/known_tools.yaml` has not been generated.

Refresh the snapshot (maintainer-only, ~6-month cadence):

```
make refresh-tools
```

#### Typo detection

Tokens not in the snapshot are now run through a Damerau-Levenshtein
fuzzy matcher (≥5 chars, distance ≤ 2) against the full known-tools set.
Near-matches are classified as `typo:<correction>` rather than `unknown`.

#### Four-category classification

Each unrecognised token is assigned one of:

| Category | Meaning |
|----------|---------|
| `typo` | Close Levenshtein match to a known tool |
| `filename` | Has a path separator or biodata file extension |
| `runtime_glue` | Common shell / infra command (not a bioinfo tool) |
| `unknown` | Genuinely unknown — true hallucination candidate |

#### Figure update

`hallucination_figure` gains a third panel **(c) Category breakdown**:
a per-model stacked bar showing the share of each category among all
flagged tokens.  Falls back to the original 2-panel layout for CSVs
that pre-date the v2 column schema.

**Invalidates.** The `hallucinated_tools` column in all v2.0 and older
`metrics.csv` files has changed schema.  `overall_pass`, `hallucination_rate`,
and `num_hallucinated_tools` are **not** affected.  Re-score with
`make rescore` to populate `num_hallucinated_typos` and `hallucinated_typos`.

---

## [v2.0] — 2026-05-06 — Reviewer fix matrix

This release implements the deep-fix plan addressing the reviewer's
critique of FlowAgent's four LLM benchmarks (planning, recovery,
reasoning, interpretation). It is a backwards-incompatible release of
the benchmark scoring layer; **all v1.x metrics must be regenerated
before being mixed with v2.0 results**.

### Benchmark A — Planning (`bench_planning.py`)

- **Added.** New `inference` prompt tier with 25 prompts that describe
  the analysis goal and input data only, never the tool names. Each
  prompt declares `acceptable_tool_sets` listing every valid
  toolchain. Existing 41 prompts are now `tier: transcription`.
  *([`corpus/prompts.yaml`](corpus/prompts.yaml))*
- **Added.** `score_plan_inference` in
  [`harness/metrics.py`](harness/metrics.py): the plan passes if at
  least one acceptable toolchain is fully covered using strict tool
  matching (prose fallback off), commands are well-formed
  (per-tool whitelists), no forbidden tools, and `expected_min_steps`
  is met.
- **Added.** `harness/command_validator.py` with per-tool flag
  whitelists (kallisto, salmon, STAR, hisat2, bwa, samtools, bcftools,
  macs2, featurecounts, htseq-count, multiqc, …) and a
  `commands_well_formed_fraction` metric that gates `overall_pass` in
  the inference tier.
- **Changed.** `tool_covered` now takes a `prose_fallback: bool = True`
  flag. The transcription scorer keeps the legacy fallback for back-
  compat; the inference scorer disables it so an LLM cannot earn
  credit for naming a tool in narrative without invoking it.
- **Changed.** `score_plan` accepts `strict_hallucinations: bool =
  False`. When set, `overall_pass` requires `hallucination_rate ==
  0`. Hallucinated tools are reported in every run regardless.

**Invalidates.** All v1.x Benchmark A `metrics.csv` files. Re-run
`make plan-all REPLICATES=3`. Plans collected at v1 can be re-scored
in place with `python benchmarks/rescore_planning.py` (the raw plan
JSONs are stable; only scoring changed).

### Benchmark B — Recovery (`bench_recovery.py`)

- **Added.** Pipeline-level recovery contract. The LLM may now
  return a `plan_patch` with one of `insert_before`, `insert_after`,
  `replace_step`, `remove_step` operations.
  [`WorkflowDAG.apply_plan_patch`](../flowagent/core/workflow_dag.py)
  applies the patch under an acyclicity guard, with rollback on any
  validation failure. *(Closes the "DAG immutability" gap.)*
- **Added.** `cheat_repair` recovery taxonomy bucket
  ([`recovery_taxonomy.py`](recovery_taxonomy.py)). A success-on-exit-
  code response whose `fixed_command` parses to a no-op shape is now
  flagged separately from `unsafe_repair` (silent corruption of valid
  data).
- **Changed.** `_is_recovery_antipattern`
  ([`flowagent/core/workflow_manager.py`](../flowagent/core/workflow_manager.py))
  now rejects bare `true` / `:` / `exit 0` / `test 0`, trailing
  `|| true` / `|| continue` / `|| :` / `|| exit 0` failure
  suppression, leading `set +e`, and "fixes" that drop the original
  tool family and leave only shell builtins. Each rejected shape is
  pinned by [`tests/test_recovery_antipatterns.py`](tests/test_recovery_antipatterns.py).
- **Changed.** Every fault in
  [`harness/fault_inject.py`](harness/fault_inject.py) now declares
  non-empty `outputs`, so `_verify_recovery_outputs` engages on every
  cell — a "successful" recovery that produces no artifacts is
  failed explicitly.
- **Changed.** Recovery prompt restructured into two phases:
  *diagnose* (commit to a `failure_class`) → *respond* (one of
  `patch_command`, `patch_pipeline`, `refuse`). This removes the
  patching bias the reviewer flagged, where the LLM was nudged to
  produce a fixed command before considering whether to refuse.

**Invalidates.** All v1.x Benchmark B `metrics.csv` and
`recovery_taxonomy/per_cell.csv` files: the contract, antipattern
list, fault outputs, and recovery prompt all changed. Re-run
`make recovery-all SEEDS=5`.

### Benchmark C — Reasoning split (`harness/plot.py` panel)

- **Added.** `reasoning: bool` and `reasoning_default: low|medium|high|
  none` per model in [`config/models.yaml`](config/models.yaml). Values
  sourced from each provider's documented model card (Anthropic
  extended-thinking, Google Gemini thinking-budget, OpenAI reasoning-
  effort).
- **Changed.** `recovery_reasoning_split_figure` now reads the YAML
  rather than the previous hand-curated `_REASONING_MODEL_IDS` set,
  and stratifies by `tier` (`current` / `legacy` / `deprecated`) so
  the manuscript figure makes generation differences explicit
  (e.g. OpenAI's tested reasoning models `o3` / `o4-mini` are older
  than Anthropic's `claude-opus-4-7`).

**Invalidates.** Just the figure. The metrics CSV underneath is the
v2.0 Benchmark B output and does not need re-collection — re-run
`make figs` to regenerate the split panel from existing data.

### Benchmark G — Interpretation (`bench_interpretation.py`)

- **Added.** `evidence_class` tag on every question
  (`data_required`, `internal_knowledge`, or `calibrated_refusal`)
  in [`config/interpretation_questions.yaml`](config/interpretation_questions.yaml).
  Every dataset now contributes **≥3 `data_required` MCQs**. New
  panel `interpretation_evidence_class_figure` in
  [`harness/plot.py`](harness/plot.py) plots accuracy stratified by
  evidence class with a 25%-chance baseline.
- **Added.** Structured-output MCQ contract: prompts ask for
  `<answer>X</answer><explain>...</explain>`. The new
  `_extract_letter` is tag-aware with a tiered regex fallback for
  models that emit prose. Final fallback prefers the *last* in-set
  capital letter, so "I believe the answer is B" returns `B`, not
  `I`. Tests in [`tests/test_letter_extractor.py`](tests/test_letter_extractor.py).
- **Added.** Anchored-band judge prompt with explicit pass mark
  (≥60) and structured JSON output (`score`, `hits[]`, `misses[]`,
  `fabrications[]`, `grounding_quote`, `justification`). The metrics
  row now carries the rubric hits / misses / fabrications, so each
  judgment is auditable.
- **Added.** Inter-judge calibration harness
  [`benchmarks/judge_calibration.py`](judge_calibration.py): re-scores
  N sampled responses with a second judge model, reports Pearson r,
  Cohen's κ on pass/fail, and mean score delta.
- **Changed.** Every open-ended question now declares
  `depends_on_inputs: [...]` and was rewritten where the previous
  rubric required information not derivable from the listed inputs
  (notably `gse32222_q04_summary`, `encsr000euq_q04_summary`,
  `gse74912_q04_summary`, `giab_q04_summary`,
  `gse52778_q06_summary`, `gse52778_q07_caveats`,
  `gse52778_q08_followup`, `gse60450_q04_summary`,
  `gse152418_q06_caveats`).

**Invalidates.** All v1.x Benchmark G `metrics.csv` files. The
question YAML changed (new MCQs added, open-ended questions
rewritten), the judge prompt changed (different anchors), and the
parser changed (different MCQs may now resolve). Re-run
`make interp-all`. Existing raw model outputs CAN be partially re-
used by the judge harness (the per-row `candidate_answer` field is
stable for open-ended questions whose text did not change), but the
MCQ side must be re-collected.

### Cross-cutting

- **Added.** Five new test files in
  [`benchmarks/tests/`](tests/) pinning the inference scorer, recovery
  antipatterns, DAG patches, letter extractor, and judge prompt.
- **Added.** Per-benchmark "What this benchmark does NOT test"
  caveats in [`README.md`](README.md).
