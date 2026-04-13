# FlowAgent benchmark orchestration.
#
# All targets are thin wrappers that forward to the bench_*.py entrypoints.
# Override defaults with make variables, e.g.
#     make plan MODEL=claude-sonnet-4-20250514 REPLICATES=5

# Prefer `python` if it exists, else fall back to `python3`. Override by
# setting PY on the command line, e.g. `make plan PY=/path/to/venv/bin/python`.
PY          ?= $(shell command -v python 2>/dev/null || command -v python3)
MODEL       ?= gpt-4.1
REPLICATES  ?= 3
SEEDS       ?= 5
RESULTS_DIR ?= results

ifeq ($(strip $(PY)),)
$(error No python interpreter found on PATH. Set PY=/path/to/python)
endif

.PHONY: all smoke plan recovery gen exec report clean help

help:
	@echo "FlowAgent benchmark targets:"
	@echo "  make smoke      - run every benchmark in --mock mode (no API key)"
	@echo "  make plan       - Benchmark A: planning correctness (needs API key)"
	@echo "  make recovery   - Benchmark B: error recovery (needs API key)"
	@echo "  make gen        - Benchmark C: generator fidelity (no API key)"
	@echo "  make exec       - Benchmark D: executor coverage (no API key)"
	@echo "  make all        - run all four benchmarks"
	@echo "  make report     - regenerate figures from latest results/"
	@echo "  make clean      - delete results/"

smoke:
	$(PY) bench_planning.py   --mock --models=$(MODEL) --replicates=1 --out=$(RESULTS_DIR)
	$(PY) bench_recovery.py   --mock --seeds=1 --out=$(RESULTS_DIR)
	$(PY) bench_generation.py --out=$(RESULTS_DIR)
	$(PY) bench_executors.py  --out=$(RESULTS_DIR)

plan:
	$(PY) bench_planning.py --models=$(MODEL) --replicates=$(REPLICATES) --out=$(RESULTS_DIR)

recovery:
	$(PY) bench_recovery.py --model=$(MODEL) --seeds=$(SEEDS) --out=$(RESULTS_DIR)

gen:
	$(PY) bench_generation.py --out=$(RESULTS_DIR)

exec:
	$(PY) bench_executors.py --out=$(RESULTS_DIR)

all: plan recovery gen exec report

report:
	$(PY) -m harness.plot --results=$(RESULTS_DIR)

clean:
	rm -rf $(RESULTS_DIR)
