# FlowAgent benchmark orchestration.
#
# All targets are thin wrappers that forward to the bench_*.py entrypoints.
# Override defaults with make variables, e.g.
#     make plan MODEL=claude-sonnet-4-20250514 REPLICATES=5

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

.PHONY: all all-sweep smoke plan plan-all recovery gen exec competitors fidelity fidelity-run references install-r-deps interpretation rescore merge report test clean help

help:
	@echo "FlowAgent benchmark targets:"
	@echo "  make smoke        - run every benchmark in --mock mode (no API key)"
	@echo "  make plan         - Benchmark A for one MODEL (default: $(MODEL))"
	@echo "  make plan-all     - Benchmark A for ALL models in config/models.yaml"
	@echo "  make recovery     - Benchmark B: error recovery (needs API key)"
	@echo "  make gen          - Benchmark C: generator fidelity (no API key)"
	@echo "  make exec         - Benchmark D: executor coverage (no API key)"
	@echo "  make competitors  - Benchmark E: head-to-head vs BioMaster / other agents"
	@echo "  make rescore      - Re-evaluate the latest planning run with current scoring"
	@echo "  make merge        - Combine all planning runs into one CSV under _merged/"
	@echo "  make test         - run the benchmark test suite (pytest)"
	@echo "  make all          - single MODEL end-to-end: plan + recovery + gen + exec + report"
	@echo "  make all-sweep    - full multi-model sweep: plan-all + recovery + gen + exec + competitors + rescore + merge + report"
	@echo "  make report       - regenerate figures (uses _merged if present)"
	@echo "  make clean        - delete results/"

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

merge:
	$(PY) merge_runs.py --results-base=$(RESULTS_DIR) $(if $(REFRESH),--refresh,)

recovery:
	$(PY) bench_recovery.py --model=$(MODEL) --seeds=$(SEEDS) --out=$(RESULTS_DIR)

gen:
	$(PY) bench_generation.py --out=$(RESULTS_DIR)

exec:
	$(PY) bench_executors.py --out=$(RESULTS_DIR)

competitors:
	$(PY) bench_competitors.py --model=$(MODEL) --replicates=$(REPLICATES) --out=$(RESULTS_DIR)

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
	    $(if $(FORCE),--force,)

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

all: plan recovery gen exec report

# Full multi-model sweep end-to-end with rescore + merge in the right order.
all-sweep: plan-all recovery gen exec competitors rescore merge report

report:
	$(PY) -m harness.plot --results=$(RESULTS_DIR)

clean:
	rm -rf $(RESULTS_DIR)
