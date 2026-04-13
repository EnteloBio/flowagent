# Changelog

All notable changes to FlowAgent are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **LLM-driven error recovery loop** (`WorkflowManager._attempt_error_recovery`)
  with up to 3 retry attempts per failing step. Wired into both the
  sequential execution path and the DAG-parallel path via a new
  `recovery_fn` callback in `WorkflowDAG.execute_parallel`.
- **Pipeline-format error recovery**: when `--pipeline-format
  nextflow|snakemake` runs fail, the entire workflow plan is sent back
  to the LLM, regenerated, and re-executed (up to 3 attempts).
- **Workflow presets** (`flowagent/presets/catalog.py`):
  `rnaseq-kallisto`, `rnaseq-star`, `chipseq`, `atacseq` — fully-formed
  plans that bypass the LLM. Use with `--preset <id>`.
- **Six execution backends** unified under
  `flowagent.core.executor_factory.ExecutorFactory`: `local`, `cgat`,
  `hpc` (SLURM/SGE/TORQUE via DRMAA), `kubernetes`, `nextflow`, `snakemake`.
- **Multi-provider LLM layer** (`flowagent.core.providers`): OpenAI,
  Anthropic Claude, Google Gemini, Ollama. Switch with `LLM_PROVIDER` +
  `LLM_MODEL`; provider auto-detected from model string when omitted.
- **Reproducible benchmark suite** under [`benchmarks/`](https://github.com/cribbslab/flowagent/tree/main/benchmarks)
  covering planning correctness (A), error recovery (B), generator
  fidelity (C), and executor coverage (D). See [Benchmarking](benchmarking.md).
- **Live progress display** in the Nextflow + Snakemake executors
  (terminal stdout inheritance with ANSI passthrough; logs preserved
  to `.nextflow.log` / `snakemake.log` for post-mortem).
- **Smart resume** (`flowagent.core.smart_resume`): on `--resume`,
  detects existing outputs per step (tool-specific validators for
  Kallisto, FastQC, MultiQC, BWA, Samtools, etc.) and skips completed
  work without consulting the checkpoint.
- **Workflow manifest** (`<output>/workflow_manifest.json`) emitted on
  every run with git SHA, LLM provider/model, executor, plan, and
  SHA-256 of every output file.

### Changed

- **`wget` → `curl`** in all hardcoded download steps (preset catalogue,
  `pipeline_planner.build_reference_download_steps`, GEO downloader)
  for macOS portability.
- **LLM system prompt** now explicitly forbids `wget` on macOS and
  requires `multiqc -f -n multiqc_report` to avoid output-file
  collisions on re-runs.
- **`_primary_tool()` helper** in both Nextflow and Snakemake generators
  now correctly identifies the primary bioinformatics tool in chained
  commands like `mkdir -p foo && fastqc ...` (was previously routing
  to the `mkdir` / `base` conda env, breaking recovery cascades).
- **`cd '${launchDir}'`** is now single-quoted in generated Nextflow
  processes so paths containing spaces or parentheses don't break bash.
- **Snakemake conda envs** no longer pin exact versions
  (`bioconda::kallisto` instead of `bioconda::kallisto=0.50.1`) to allow
  conda to solve on `osx-arm64` where many bioconda builds lag.
- **Snakemake executor** uses `set -o pipefail` + `tee` so the actual
  snakemake exit code propagates through the log pipe (previously
  always reported success when `tee` succeeded).
- **`flowagent` CLI rebrand** (was `cognomic`); all imports and
  references updated.

### Fixed

- Output collision when `multiqc_report.html` already exists (now
  prompted to use `-f -n multiqc_report`).
- Path-with-spaces (`Mac (3)/...`) breaking unquoted `cd` in generated
  Nextflow processes.
- DAG executor failing fast on first step error before recovery could
  run (added `recovery_fn` parameter).
- Stale conda pin for `kallisto=0.50.1` failing on `osx-arm64`.

### Documentation

- Documentation overhauled to cover all new features:
  [Concepts](user-guide/concepts.md),
  [CLI Reference](user-guide/cli-reference.md),
  [LLM Providers](user-guide/llm-providers.md),
  [Workflow Presets](user-guide/presets.md),
  [Pipeline Generation](user-guide/pipeline-generation.md),
  [Execution Backends](user-guide/executors.md),
  [Error Recovery](user-guide/error-recovery.md),
  [Web Interface](user-guide/web-interface.md),
  [Benchmarking](benchmarking.md).
- HPC docs rewritten to cover all three HPC paths (`--executor hpc`,
  `--executor cgat`, Nextflow with SLURM profile).

---

## [0.2.0] — 2025

Initial public release under the **FlowAgent** name.

### Added

- Plan-then-execute architecture with `WorkflowManager`.
- Nextflow + Snakemake pipeline generators.
- Local + CGAT-core + HPC + Kubernetes executors.
- Chainlit-based web UI (`flowagent serve`).
- Custom-script integration system.

### Changed

- Renamed project from **Cognomic** to **FlowAgent**.
- All imports and module paths updated from `cognomic.*` to `flowagent.*`.

### Fixed

- Documentation build process.
