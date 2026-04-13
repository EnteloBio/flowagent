# Error Recovery

When a pipeline step fails, FlowAgent doesn't just log the error and
exit — it sends the failure back to the LLM, asks for a fixed command,
and retries. This page explains what's recovered, how, and what's not.

## The recovery loop

Every executor returns a step result with `status == "error"` or
`"failed"` when something goes wrong. `WorkflowManager._attempt_error_recovery`
then:

1. **Builds context**: failing command, exit code, stderr (first 2000 chars),
   stdout (first 1000 chars), platform (macOS/Linux), and a tool-availability
   map computed from `shutil.which()` over every executable in the command.

2. **Asks the LLM** for a fix:
   ```
   Diagnose the error and return a corrected shell command.
   Common fixes:
   - Exit 127 (command not found): substitute equivalent tool (curl for wget, ...)
   - 'No such file or directory': add mkdir -p
   - Permission denied: check paths
   ...
   Return JSON: {"diagnosis": ..., "fixed_command": ..., "explanation": ...}
   ```

3. **Re-executes** the patched step.

4. **Recurses** on continued failure, up to `max_attempts=3`. Each
   attempt sees the previous fix's stderr.

If the LLM returns `fixed_command: null`, recovery is abandoned and the
failure is reported with the LLM's diagnosis.

## Which paths have recovery?

| Execution path | Recovery loop |
|---|---|
| `WorkflowManager.execute_workflow` (sequential) | ✅ per-step |
| `WorkflowDAG.execute_parallel` | ✅ per-step (via `recovery_fn` callback) |
| `--pipeline-format nextflow` / `snakemake` (CLI) | ✅ regenerates the entire pipeline file |
| Direct `executor.execute_step()` calls | ❌ (caller must invoke recovery manually) |

For the `--pipeline-format` path, the prompt is slightly different — the
LLM gets the FULL workflow plan plus the error and returns a corrected
plan. The pipeline file is regenerated and the runtime
(`nextflow run -resume` / `snakemake`) re-executes only the failed +
downstream steps.

## What gets recovered well

These fault classes are reliably fixed in one or two attempts:

| Fault | LLM fix |
|---|---|
| `wget: command not found` (macOS) | Substitute `curl -fSL -o ...` |
| Tool typo (`fastq_c`, `kalisto`) | Correct the spelling |
| `multiqc_report.html` already exists | Add `-f -n multiqc_report` flags |
| Wrong flag (`kallisto index -x` instead of `-i`) | Fix the flag |
| Path-with-spaces shell error | Add quoting |
| Missing output directory | Insert `mkdir -p` (in the same shell command) |

## What doesn't recover

The LLM is only changing the **command string**. It cannot fix:

- Genuine input data corruption (truncated FASTQ from a bad download)
- Out-of-memory errors that need a different node
- Network outages
- Wrong assumptions about your data shape (paired-end → single-end is
  fixable; missing files for a sample are not)

For these, recovery either (a) hits the 3-attempt cap, or (b) returns
`fixed_command: null` with a diagnosis pointing at the underlying issue.

## Cost

Each recovery attempt is one extra LLM call (~$0.01–0.02 with GPT-4.1).
With `max_attempts=3` the worst case adds ~$0.06 per failed step.

For pipelines with many steps, you can disable recovery by patching
`max_attempts=1` in your fork — but the cost/benefit is overwhelmingly
in favour of leaving it on.

## Programmatic invocation

```python
from flowagent.core.workflow_manager import WorkflowManager

mgr = WorkflowManager(executor_type="local")
bad_step = {"name": "x", "command": "wget -q -O f.gz https://...", ...}
bad = await mgr._step_executor.execute_step(bad_step, "out/", cwd=".")
recovered = await mgr._attempt_error_recovery(bad_step, bad, "out/", max_attempts=3)
if recovered and recovered["status"] not in ("error", "failed"):
    print("fixed:", recovered["fixed_command"])
    print("diagnosis:", recovered["recovery_diagnosis"])
```

The returned dict carries `recovery_attempt`, `recovery_diagnosis`,
`fixed_command`, and `original_command` for traceability.

## Logging

Every recovery attempt is logged at INFO level:

```
INFO flowagent.cli - Attempting LLM error recovery...
INFO flowagent.core.workflow_manager - LLM recovery attempt 1 for 'download_reference': substituted curl
INFO flowagent.core.workflow_manager - Fixed command: mkdir -p reference && curl -fSL -o ...
INFO flowagent.core.workflow_manager - Step 'download_reference' recovered successfully on attempt 1
```

Set `LOG_LEVEL=DEBUG` to also see the JSON payload sent to the LLM.

## Benchmarking recovery

Benchmark B (`benchmarks/bench_recovery.py`) measures recovery rate
across 10 fault classes:

| Fault | Class |
|---|---|
| `missing_wget` | missing binary |
| `tool_typo` | typo |
| `wrong_flag` | wrong flag |
| `missing_output_dir` | missing path |
| `paired_single_mismatch` | data-shape mismatch |
| `corrupt_fastq` | corrupt input |
| `readonly_output` | permission |
| `path_with_spaces` | shell escaping |
| `multiqc_collision` | output collision |
| `stale_conda_pin` | env mismatch |

See [Benchmarking](../benchmarking.md) for how to reproduce the recovery
figure that anchors the manuscript.

## Disabling recovery

Pass `max_attempts=1` in code, or use `--pipeline-format` + `--no-execute`
to defer execution entirely (no recovery without execution). At the CLI
level there's currently no flag to disable recovery; if you need this,
file an issue.
