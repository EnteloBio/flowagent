# FlowAgent benchmark orchestration.
#
# All targets are thin wrappers that forward to the bench_*.py entrypoints.
# Override defaults with make variables, e.g.
#     make plan MODEL=claude-sonnet-4-6 REPLICATES=5

# Prefer `python` if it exists, else fall back to `python3`. Override by
# setting PY on the command line, e.g. `make plan PY=/path/to/venv/bin/python`.
PY          ?= $(shell command -v python 2>/dev/null || command -v python3)
MODEL       ?= gpt-4.1
JUDGE       ?= gpt-5.4
REPLICATES  ?= 3
SEEDS       ?= 5
RESULTS_DIR ?= results

ifeq ($(strip $(PY)),)
$(error No python interpreter found on PATH. Set PY=/path/to/python)
endif

.PHONY: all all-sweep smoke plan plan-all recovery gen exec competitors competitors-all ablation ablation-pilot ablation-figure reflection reflection-pilot reflection-figure competitor-dag-ablation competitor-dag-ablation-pilot competitor-dag-figure validator-ablation validator-ablation-pilot validator-figure cove-ablation cove-ablation-pilot cove-figure fidelity fidelity-run references install-r-deps interpretation rescore merge report test clean help refresh-tools

help:
	@echo "FlowAgent benchmark targets:"
	@echo "  make smoke        - run every benchmark in --mock mode (no API key)"
	@echo "  make plan         - Benchmark A for one MODEL (default: $(MODEL))"
	@echo "  make plan-all     - Benchmark A for ALL models in config/models.yaml"
	@echo "  make recovery     - Benchmark B: error recovery (needs API key)"
	@echo "  make gen          - Benchmark C: generator fidelity (no API key)"
	@echo "  make exec         - Benchmark D: executor coverage (no API key)"
	@echo "  make competitors  - Benchmark E: head-to-head vs BioMaster / AutoBA / Biomni / Claude Code / Edison / raw-LLM"
	@echo "  make competitors-all - Benchmark E: full 4-way comparison (FlowAgent / Claude Code / Biomni / Edison)"
	@echo "  make ablation     - Benchmark H: DAG-aware vs DAG-blind planner ablation (66 prompts × replicates × 2 arms)"
	@echo "  make ablation-pilot - Benchmark H smoke: 5 prompts × 1 replicate × 2 arms (validates the ablation pipeline)"
	@echo "  make ablation-figure ABLATION_DIR=results/ablation/<ts> - render figure_ablation.pdf + stats_ablation.tsv"
	@echo "  make reflection   - Benchmark I: completeness-reflection ablation (66 prompts × replicates × 2 arms)"
	@echo "  make reflection-pilot - Benchmark I smoke: 8 prompts × 1 replicate × 2 arms"
	@echo "  make reflection-figure REFLECTION_DIR=results/reflection/<ts> - render figure_reflection.pdf + stats_reflection.tsv"
	@echo "  make competitor-dag-ablation - Benchmark J: competitor DAG-prompt ablation (Claude Code, 66 prompts × replicates × 2 arms)"
	@echo "  make competitor-dag-ablation-pilot - Benchmark J smoke: 3 prompts × 1 replicate × 2 arms"
	@echo "  make competitor-dag-figure CDAG_DIR=results/competitor_dag_ablation/<ts> - render figure_competitor_dag__<comp>.pdf + stats"
	@echo "  make validator-ablation - T0: command-level validator on/off ablation (66 prompts × replicates × 2 arms)"
	@echo "  make validator-ablation-pilot - T0 smoke: 5 prompts × 1 replicate × 2 arms"
	@echo "  make validator-figure VALIDATOR_DIR=results/validator_ablation/<ts> - render figure_validator.pdf + stats"
	@echo "  make cove-ablation - T4: CoVe verifier on/off ablation, annotation-only mode (66 prompts × replicates × 2 arms)"
	@echo "  make cove-ablation-pilot - T4 smoke: 5 prompts × 1 replicate × 2 arms"
	@echo "  make cove-figure COVE_DIR=results/cove_ablation/<ts> - render figure_cove.pdf + stats + signal table"
	@echo "  make rescore      - Re-evaluate the latest planning run with current scoring"
	@echo "  make merge        - Combine all planning runs into one CSV under _merged/"
	@echo "  make test         - run the benchmark test suite (pytest)"
	@echo "  make all          - single MODEL end-to-end: plan + recovery + gen + exec + report"
	@echo "  make all-sweep    - full multi-model sweep: plan-all + recovery + gen + exec + competitors + rescore + merge + report"
	@echo "  make report       - regenerate figures (uses _merged if present)"
	@echo "  make clean        - delete results/"
	@echo "  make refresh-tools - regenerate data/known_tools.yaml from Bioconda/Bioconductor (maintainer-only)"

smoke:
	$(PY) bench_planning.py   --mock --models=$(MODEL) --replicates=1 --out=$(RESULTS_DIR)
	$(PY) bench_recovery.py   --mock --seeds=1 --out=$(RESULTS_DIR)
	$(PY) bench_generation.py --out=$(RESULTS_DIR)
	$(PY) bench_executors.py  --out=$(RESULTS_DIR)

plan:
	$(PY) bench_planning.py --models=$(MODEL) --replicates=$(REPLICATES) --out=$(RESULTS_DIR)

plan-all:
	$(PY) bench_planning.py --replicates=$(REPLICATES) --out=$(RESULTS_DIR)

rescore:
	$(PY) rescore_planning.py --results-base=$(RESULTS_DIR)

rescore-all:
	$(PY) rescore_planning.py --all --results-base=$(RESULTS_DIR)

merge:
	$(PY) merge_runs.py --results-base=$(RESULTS_DIR) $(if $(REFRESH),--refresh,)

recovery:
	$(PY) bench_recovery.py --model=$(MODEL) --seeds=$(SEEDS) --out=$(RESULTS_DIR)

gen:
	$(PY) bench_generation.py --out=$(RESULTS_DIR)

exec:
	$(PY) bench_executors.py --out=$(RESULTS_DIR)

# Optional: narrow the sweep to specific competitors and / or override the
# raw-LLM baselines. Defaults preserve historical behaviour (all default-
# sweep competitors + the three frontier raw lanes). Examples::
#
#   make competitors COMPETITORS=flowagent RAW_MODELS=""
#       # FlowAgent only -- no third-party shims, no raw baselines.
#   make competitors COMPETITORS=flowagent,claude_code RAW_MODELS=""
#       # Plan-only head-to-head, no raw lanes.
#   make competitors RAW_MODELS="gpt-4.1,claude-haiku-4-5"
#       # Default competitors + a custom raw-LLM lane set.
#
# The opt-in lane ``edison`` is excluded from the default sweep; name it
# explicitly via COMPETITORS=...,edison or use ``make competitors-all``.
RAW_MODELS ?= gpt-5.4,claude-opus-4-7,gemini-2.5-pro
COMPETITORS_FLAG := $(if $(COMPETITORS),--competitors=$(COMPETITORS),)
competitors:
	$(PY) bench_competitors.py --model=$(MODEL) --replicates=$(REPLICATES) \
	    $(COMPETITORS_FLAG) \
	    --raw-models="$(RAW_MODELS)" \
	    --out=$(RESULTS_DIR)

# Four-way comparison covering FlowAgent + the three competitor frameworks
# requested in the ablation study (Claude Code, Biomni, Edison Analysis).
# Soft-skips any system whose env vars / CLIs are not set up.
# Override CC_BUDGET / EDISON_BUDGET to cap external-API spend.
EDISON_BUDGET ?= 50
competitors-all:
	$(PY) bench_competitors.py --model=$(MODEL) --replicates=$(REPLICATES) \
	    --competitors=flowagent,claude_code,biomni,edison \
	    --raw-models="" \
	    --edison-budget-credits=$(EDISON_BUDGET) \
	    --out=$(RESULTS_DIR)

# Benchmark H -- DAG-awareness ablation.
# Full sweep: 66 prompts × MODEL × REPLICATES × 2 arms; ~1 hour and a few
# dollars on Claude Haiku, more on flagship models. Override MODEL or
# pass MODELS=a,b,c to sweep multiple models.
ablation:
	$(PY) bench_ablation.py \
	    $(if $(MODELS),--models=$(MODELS),--models=$(MODEL)) \
	    --replicates=$(REPLICATES) \
	    --out=$(RESULTS_DIR)

# Quick smoke for the ablation: 5 prompts × 1 replicate × 2 arms.
ablation-pilot:
	$(PY) bench_ablation.py \
	    --models=$(MODEL) \
	    --replicates=1 \
	    --limit=5 \
	    --out=$(RESULTS_DIR)

# Render figure_ablation.pdf + stats_ablation.tsv from a finished run.
# Pass ABLATION_DIR=results/ablation/<timestamp>/ to point at a specific run.
ABLATION_DIR ?= $(shell ls -td $(RESULTS_DIR)/ablation/*/ 2>/dev/null | head -1)
ablation-figure:
	@if [ -z "$(ABLATION_DIR)" ]; then \
	    echo "No ablation runs found under $(RESULTS_DIR)/ablation. "; \
	    echo "Run \`make ablation\` first or pass ABLATION_DIR=...."; \
	    exit 1; \
	fi
	$(PY) make_ablation_figure.py \
	    --paired=$(ABLATION_DIR)/paired_metrics.csv \
	    --out=figure_ablation

# Benchmark I -- completeness-reflection ablation.
# Both arms keep LLM_DAG_AWARE=true; only LLM_COMPLETENESS_REFLECT is
# toggled. Tests whether the DAG-Plan-style completeness validator + LLM
# retry loop introduced in flowagent/core/completeness.py improves plan
# quality (measured via the new completeness_pass + stage_efficiency
# columns from harness/metrics.py).
reflection:
	$(PY) bench_reflection_ablation.py \
	    $(if $(MODELS),--models=$(MODELS),--models=$(MODEL)) \
	    --replicates=$(REPLICATES) \
	    $(if $(MAX_RETRIES),--max-retries=$(MAX_RETRIES),) \
	    --out=$(RESULTS_DIR)

# Quick smoke for Benchmark I: 8 prompts × 1 replicate × 2 arms.
# ~2-5 min, ~$0.20 on Claude Haiku 4.5.
reflection-pilot:
	$(PY) bench_reflection_ablation.py \
	    --models=$(MODEL) \
	    --replicates=1 \
	    --limit=8 \
	    --out=$(RESULTS_DIR)

# Render figure_reflection.pdf + stats_reflection.tsv from a finished run.
# Pass REFLECTION_DIR=results/reflection/<timestamp>/ to point at a specific run.
REFLECTION_DIR ?= $(shell ls -td $(RESULTS_DIR)/reflection/*/ 2>/dev/null | head -1)
reflection-figure:
	@if [ -z "$(REFLECTION_DIR)" ]; then \
	    echo "No reflection runs found under $(RESULTS_DIR)/reflection. "; \
	    echo "Run \`make reflection\` first or pass REFLECTION_DIR=...."; \
	    exit 1; \
	fi
	$(PY) make_reflection_figure.py \
	    --paired=$(REFLECTION_DIR)/paired_metrics.csv \
	    --out=figure_reflection

# Benchmark J -- competitor DAG-prompt ablation.
# Toggles whether the COMPETITOR's prompt template asks for a DAG
# (`dependencies` field + topological-order rule) vs a flat list. Currently
# wired up for Claude Code only; Biomni / Edison need the same shim-level
# template toggle before they can be added (see bench_competitor_dag_ablation.py).
#
# COMPETITORS  -- comma-separated competitor slugs (default: claude_code).
# CDAG_MODEL   -- driver model for the competitor's shim
#                  (default: claude-haiku-4-5).
# REPLICATES   -- replicates per (prompt, arm) cell.
COMPETITORS ?= claude_code
CDAG_MODEL  ?= claude-haiku-4-5
competitor-dag-ablation:
	$(PY) bench_competitor_dag_ablation.py \
	    --competitors=$(COMPETITORS) \
	    --model=$(CDAG_MODEL) \
	    --replicates=$(REPLICATES) \
	    --out=$(RESULTS_DIR)

# Quick smoke for Benchmark J: 3 prompts × 1 replicate × 2 arms.
# ~$0.10-0.50 on Claude Haiku 4.5 (depends on prompt length).
competitor-dag-ablation-pilot:
	$(PY) bench_competitor_dag_ablation.py \
	    --competitors=$(COMPETITORS) \
	    --model=$(CDAG_MODEL) \
	    --replicates=1 \
	    --limit=3 \
	    --out=$(RESULTS_DIR)

# Render figure_competitor_dag__<comp>.pdf + per-competitor stats from a
# finished run. Pass CDAG_DIR=results/competitor_dag_ablation/<timestamp>/
# to point at a specific run; defaults to the most recent.
CDAG_DIR ?= $(shell ls -td $(RESULTS_DIR)/competitor_dag_ablation/*/ 2>/dev/null | head -1)
competitor-dag-figure:
	@if [ -z "$(CDAG_DIR)" ]; then \
	    echo "No competitor DAG-ablation runs found under "; \
	    echo "$(RESULTS_DIR)/competitor_dag_ablation. "; \
	    echo "Run \`make competitor-dag-ablation\` first or pass CDAG_DIR=...."; \
	    exit 1; \
	fi
	$(PY) make_competitor_dag_figure.py \
	    --paired=$(CDAG_DIR)/paired_metrics.csv \
	    --out=figure_competitor_dag

# Todo T0 -- command-level validator on/off ablation.
# Toggles FLOWAGENT_VALIDATOR_ENABLED; both arms keep LLM_DAG_AWARE=true and
# LLM_COMPLETENESS_REFLECT=true. The on-arm runs _autofix_generated_steps +
# _validate_generated_steps in the primary planning loop and triggers a retry
# on any command-level violation; the off-arm bypasses both. Structural
# completeness reflection stays on in both arms — we're ablating the
# command-level layer only. See bench_validator_ablation.py docstring for
# rationale + recovery-taxonomy framing.
validator-ablation:
	$(PY) bench_validator_ablation.py \
	    $(if $(MODELS),--models=$(MODELS),--models=$(MODEL)) \
	    --replicates=$(REPLICATES) \
	    --out=$(RESULTS_DIR)

# Quick smoke: 5 prompts × 1 replicate × 2 arms.
validator-ablation-pilot:
	$(PY) bench_validator_ablation.py \
	    --models=$(MODEL) \
	    --replicates=1 \
	    --limit=5 \
	    --out=$(RESULTS_DIR)

# Render figure_validator.pdf + stats_validator.tsv from a finished run.
# Pass VALIDATOR_DIR=results/validator_ablation/<timestamp>/ to point at a
# specific run; defaults to the most recent.
VALIDATOR_DIR ?= $(shell ls -td $(RESULTS_DIR)/validator_ablation/*/ 2>/dev/null | head -1)
validator-figure:
	@if [ -z "$(VALIDATOR_DIR)" ]; then \
	    echo "No validator-ablation runs found under $(RESULTS_DIR)/validator_ablation. "; \
	    echo "Run \`make validator-ablation\` first or pass VALIDATOR_DIR=...."; \
	    exit 1; \
	fi
	$(PY) make_validator_figure.py \
	    --paired=$(VALIDATOR_DIR)/paired_metrics.csv \
	    --out=figure_validator

# Todo T4 -- CoVe verifier on/off ablation (annotation-only mode).
# Toggles FLOWAGENT_COVE_VERIFY; pins FLOWAGENT_COVE_ABSTAIN=false in both
# arms so the verifier's concerns are recorded on the plan envelope but the
# plan still ships. This is a SIGNAL-MEASUREMENT run, not a behaviour-change
# run: a follow-up with FLOWAGENT_COVE_ABSTAIN=true measures the abstention
# behaviour change once the signal is shown to be useful.
# Both arms keep LLM_DAG_AWARE=true and FLOWAGENT_VALIDATOR_ENABLED=true.
cove-ablation:
	$(PY) bench_cove_ablation.py \
	    $(if $(MODELS),--models=$(MODELS),--models=$(MODEL)) \
	    --replicates=$(REPLICATES) \
	    --out=$(RESULTS_DIR)

# Quick smoke: 5 prompts × 1 replicate × 2 arms.
cove-ablation-pilot:
	$(PY) bench_cove_ablation.py \
	    --models=$(MODEL) \
	    --replicates=1 \
	    --limit=5 \
	    --out=$(RESULTS_DIR)

# Render figure_cove.pdf + stats_cove.tsv + stats_cove_signal.tsv from a
# finished run. The signal table reports verifier precision / recall as a
# failure predictor (built from the verifier_on JSONL's _verifier envelope).
# Pass COVE_DIR=results/cove_ablation/<timestamp>/ to point at a specific
# run; defaults to the most recent.
COVE_DIR ?= $(shell ls -td $(RESULTS_DIR)/cove_ablation/*/ 2>/dev/null | head -1)
cove-figure:
	@if [ -z "$(COVE_DIR)" ]; then \
	    echo "No cove-ablation runs found under $(RESULTS_DIR)/cove_ablation. "; \
	    echo "Run \`make cove-ablation\` first or pass COVE_DIR=...."; \
	    exit 1; \
	fi
	$(PY) make_cove_figure.py \
	    --paired=$(COVE_DIR)/paired_metrics.csv \
	    --out=figure_cove

# Benchmark F — output fidelity (pure scoring; no LLM calls).
# Use this when the candidate dirs already exist on disk.
fidelity:
	$(PY) bench_fidelity.py --bulk-dir=$(RESULTS_DIR)/fidelity_runs

# Benchmark F end-to-end driver: runs flowagent on every fidelity case,
# checkpoints by output-existence, then auto-scores. Each cell can be
# hours; pipelines download FASTQs and run kallisto/MACS2/etc.
fidelity-run:
	$(PY) bench_fidelity_run.py \
	    $(if $(MODELS),--models $(MODELS),--model $(MODEL)) \
	    --replicates=$(REPLICATES) \
	    --concurrency=$(if $(CONCURRENCY),$(CONCURRENCY),1) \
	    --results-base=$(RESULTS_DIR) \
	    $(if $(CASE),--case $(CASE),) \
	    $(if $(FORCE),--force,) \
	    $(if $(CLEANUP),--cleanup,)

# Materialise the reference outputs that bench_fidelity.py scores against.
# Most direct-URL fetches need only network access. The R-script builds
# additionally need an R install with the Bioconductor packages listed in
# benchmarks/references/README.md; pass SKIP_R=1 to skip those (CI use).
references:
	$(PY) references/download_references.py $(if $(SKIP_R),--skip-r-scripts,)

# Install all R / Bioconductor packages required by the R-script
# reference builds (DESeq2, edgeR, limma, recount3, ...). Idempotent.
install-r-deps:
	Rscript references/install_r_deps.R

# Benchmark G — biological-interpretation quality
interpretation:
	$(PY) bench_interpretation.py --model=$(MODEL) --judge=$(JUDGE) $(EXTRA)

test:
	$(PY) -m pytest tests -v

# Refresh the Bioconda + Bioconductor + runtime snapshot used by the
# hallucination detector.  Run occasionally (every ~6 months or before a
# manuscript sweep) — the resulting data/known_tools.yaml is checked in and
# used by CI without any network access.
refresh-tools:
	$(PY) refresh_known_tools.py $(if $(OUT),--out $(OUT),)

all: plan recovery gen exec report

# Full multi-model sweep end-to-end with rescore + merge in the right order.
all-sweep: plan-all recovery gen exec competitors rescore merge report

report:
	$(PY) -m harness.plot --results=$(RESULTS_DIR)

clean:
	rm -rf $(RESULTS_DIR)
