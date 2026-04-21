"""Benchmark B — LLM-driven error recovery.

For each fault in ``harness.fault_inject.FAULTS``, seed a failure, invoke
``WorkflowManager._attempt_error_recovery``, and record whether recovery
succeeded, after how many attempts, and whether the diagnosis was relevant.

Mock mode returns synthetic "recovered@1 for missing_binary/missing_path
faults, failed otherwise" so the harness is exercisable without API keys.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE_DIR = Path(__file__).parent
sys.path.insert(0, str(_HERE_DIR))
sys.path.insert(0, str(_HERE_DIR.parent))

from harness.fault_inject import FAULTS, Fault           # noqa: E402
from harness.runner import load_yaml, set_provider, timestamped_dir, write_manifest  # noqa: E402

HERE = Path(__file__).parent


# ── Real recovery run ────────────────────────────────────────────

async def _run_real(fault: Fault, seed: int, tmp: Path) -> Dict[str, Any]:
    from flowagent.core.workflow_manager import WorkflowManager

    step, env_ctx = fault.apply(preset=None, seed=seed, workdir=tmp)
    mgr = WorkflowManager(executor_type="local")

    # Wrap the command with a cd so it runs in the expected working directory.
    # LocalExecutor.execute_step() only accepts (step,) — no cwd kwarg.
    cwd = env_ctx.cwd or str(tmp)
    step["command"] = f"cd {shlex.quote(cwd)} && {step['command']}"
    step["timeout"] = 60  # Kill hung commands after 60s

    # Apply any env-var overrides requested by the fault.
    env_backup = {}
    for k, v in (env_ctx.env or {}).items():
        env_backup[k] = os.environ.get(k)
        os.environ[k] = v

    try:
        # 1. Provoke a real failure.
        bad = await mgr._step_executor.execute_step(step)
        bad_status = bad.get("status")
        provoked_failure = bad_status in ("error", "failed")

        # 2. Recovery attempt.
        recovered = None
        t0 = time.perf_counter()
        if provoked_failure:
            recovered = await mgr._attempt_error_recovery(
                step, bad, env_ctx.out_dir, max_attempts=3,
            )
        wall = time.perf_counter() - t0

        # Capture the LLM's reasoning regardless of whether it claimed a
        # fix — on ``unrecoverable`` faults the agent's refusal text is
        # often the only datum worth scoring, but the old code only
        # persisted these fields when ``recovered`` was truthy. See
        # Benchmark B sub-analysis (hallucinated-fix taxonomy): we need
        # the raw diagnosis + any attempted command even when the fix was
        # refused, so we can categorise rejection styles post-hoc.
        r = recovered or {}
        status = r.get("status") if recovered is not None else None

        def _pick(*keys):
            for k in keys:
                v = r.get(k)
                if v:
                    return v
            return None

        diagnosis = _pick(
            "recovery_diagnosis", "diagnosis", "reasoning", "explanation",
        )
        fixed_cmd = _pick("fixed_command", "proposed_command", "command")
        rejection = _pick(
            "rejection_reason", "no_fix_reason", "reason", "decline_reason",
        )
        raw_response = _pick(
            "raw_response", "llm_response", "response_text", "message",
        )
        # Classify *what the agent did* even when it didn't "recover":
        #   success     — proposed+executed a fix that cleared the error
        #   proposed    — proposed a fix but it still failed or wasn't run
        #   rejected    — explicitly declined (returned rejection_reason)
        #   silent      — no dict returned at all (bug / empty output)
        if recovered is None:
            outcome = "silent"
        elif status == "rejected":
            # Workflow manager now returns a structured refusal with
            # status="rejected" instead of None. MUST be checked before
            # the "not in (error,failed)" branch below or a correct
            # refusal would be mis-classified as a successful recovery.
            outcome = "rejected"
        elif status not in ("error", "failed"):
            outcome = "success"
        elif rejection or (diagnosis and not fixed_cmd):
            outcome = "rejected"
        elif fixed_cmd:
            outcome = "proposed"
        else:
            outcome = "silent"

        return {
            "fault": fault.id,
            "fault_class": fault.cls,
            "fault_tier": fault.tier,
            "seed": seed,
            "provoked_failure": provoked_failure,
            "original_command": step["command"],
            "original_exit": bad.get("returncode") or bad.get("exit_code"),
            "original_stderr_excerpt": str(bad.get("stderr", ""))[:2000],
            "recovered": outcome == "success",
            "recovery_outcome": outcome,
            "recovery_status": status,
            "attempts": r.get("recovery_attempt"),
            "recovery_diagnosis": diagnosis,
            "fixed_command": fixed_cmd,
            "rejection_reason": rejection,
            "llm_raw_response": raw_response,
            "diagnosis_relevant": fault.diagnosis_matches(r or None),
            "wall_seconds": wall,
        }
    finally:
        # Restore env.
        for k, v in env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if env_ctx.cleanup:
            try:
                env_ctx.cleanup()
            except Exception:
                pass


# ── Mock recovery run ────────────────────────────────────────────

def _run_mock(fault: Fault, seed: int) -> Dict[str, Any]:
    """Synthetic response without touching the filesystem or the LLM."""
    # Simulate tier-dependent recovery: easy always recovers; hard recovers
    # 70% of the time; unrecoverable faults correctly stay failed.
    if fault.tier == "easy":
        recovered = True
    elif fault.tier == "hard":
        # Pseudo-deterministic ~70% recovery per (fault, seed)
        recovered = ((hash((fault.id, seed)) & 0xFFFF) / 0xFFFF) < 0.7
    else:
        recovered = False
    diag = "Simulated diagnosis for %s" % fault.id
    if recovered:
        outcome = "success"
        rejection = None
    elif fault.tier == "unrecoverable":
        outcome = "rejected"
        rejection = f"Simulated refusal: {fault.id} data is not repairable"
    else:
        outcome = "proposed"
        rejection = None
    return {
        "fault": fault.id,
        "fault_class": fault.cls,
        "fault_tier": fault.tier,
        "seed": seed,
        "provoked_failure": True,
        "original_command": "<mock>",
        "original_exit": 1,
        "original_stderr_excerpt": "<mock stderr>",
        "recovered": recovered,
        "recovery_outcome": outcome,
        "recovery_status": "success" if recovered else "failed",
        "attempts": 1 if recovered else None,
        "recovery_diagnosis": diag,
        "fixed_command": "<mock fix>" if recovered else None,
        "rejection_reason": rejection,
        "llm_raw_response": None,
        "diagnosis_relevant": recovered,
        "wall_seconds": 0.0,
    }


# ── Driver ───────────────────────────────────────────────────────

async def _drive(fault_ids: List[str], seeds: int, *,
                 model_cfg: Optional[Dict[str, Any]], mock: bool,
                 out_dir: Path) -> List[Dict[str, Any]]:
    results = []
    if not mock and model_cfg:
        set_provider(model_cfg)

    total = len(fault_ids) * seeds
    done = 0
    for fid in fault_ids:
        fault = FAULTS[fid]
        for seed in range(seeds):
            done += 1
            print(f"[{done}/{total}] fault={fid}  seed={seed} ...",
                  flush=True, end="")
            t0 = time.perf_counter()
            with tempfile.TemporaryDirectory(prefix=f"rec_{fid}_") as td:
                try:
                    if mock:
                        row = _run_mock(fault, seed)
                    else:
                        # Hard cap: 180s per fault/seed (covers LLM + cmd timeouts
                        # even if asyncio cancellation doesn't propagate cleanly).
                        row = await asyncio.wait_for(
                            _run_real(fault, seed, Path(td)),
                            timeout=180,
                        )
                except (Exception, asyncio.CancelledError) as exc:
                    row = {
                        "fault": fid, "seed": seed,
                        "error": f"{type(exc).__name__}: {exc}",
                        "recovered": False, "provoked_failure": False,
                    }
                elapsed = time.perf_counter() - t0
                status = "recovered" if row.get("recovered") else (
                    row.get("error", "failed"))
                print(f" {status} ({elapsed:.1f}s)", flush=True)
                results.append(row)

    # Persist
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    # Flat CSV
    from harness.runner import _write_csv
    _write_csv(out_dir / "metrics.csv", results)
    write_manifest(
        out_dir, benchmark="recovery",
        models=[model_cfg] if model_cfg else [],
        extra={"num_faults": len(fault_ids), "seeds": seeds, "mock": mock},
    )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4.1",
                    help="Model ID from config/models.yaml (for real mode)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--faults",
                    help="Comma-separated fault IDs (default: all)")
    ap.add_argument("--tier",
                    choices=["easy", "hard", "unrecoverable"],
                    help="Restrict to one fault tier (alternative to --faults)")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--config", default=str(HERE / "config" / "models.yaml"))
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.faults:
        fault_ids = args.faults.split(",")
    elif args.tier:
        fault_ids = [fid for fid, f in FAULTS.items() if f.tier == args.tier]
    else:
        fault_ids = list(FAULTS.keys())

    model_cfg = None
    if not args.mock:
        models = load_yaml(Path(args.config))["models"]
        matches = [m for m in models if m["id"] == args.model]
        if not matches:
            raise SystemExit(f"Model {args.model!r} not in {args.config}")
        model_cfg = matches[0]

    out_dir = timestamped_dir(Path(args.out), "recovery")
    results = asyncio.run(_drive(
        fault_ids, args.seeds,
        model_cfg=model_cfg, mock=args.mock, out_dir=out_dir,
    ))
    n_recovered = sum(1 for r in results if r.get("recovered"))
    print(f"[ok] {n_recovered}/{len(results)} recovered → {out_dir}")


if __name__ == "__main__":
    main()
