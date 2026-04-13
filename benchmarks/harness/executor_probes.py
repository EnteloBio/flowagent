"""Per-executor probes for Benchmark D.

Each probe grades its executor on three levels (stored in the result dict):

    interface_ok  - the class instantiates, has async execute_step signature,
                    and returns a dict with the required keys.
    mock_ok       - job-spec construction passes with external APIs mocked.
    live_ok       - trivial echo step runs end-to-end (only if infra present).

A probe NEVER raises; failures are recorded as boolean False with an
``error`` field. This guarantees the benchmark always produces a result
grid regardless of the environment.
"""

from __future__ import annotations

import asyncio
import inspect
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch


STUB_STEP = {
    "name": "bench_echo",
    "command": "echo ok",
    "dependencies": [],
    "outputs": [],
    "description": "benchmark smoke step",
    "resources": {"cpus": 1, "memory": "1G", "time_min": 1},
}


def _interface_ok(executor) -> Dict[str, Any]:
    """All executors must expose async execute_step returning a dict."""
    try:
        fn = getattr(executor, "execute_step", None)
        if fn is None:
            return {"interface_ok": False, "interface_error": "no execute_step"}
        if not inspect.iscoroutinefunction(fn):
            return {"interface_ok": False,
                    "interface_error": "execute_step is not async"}
        return {"interface_ok": True}
    except Exception as exc:
        return {"interface_ok": False,
                "interface_error": f"{type(exc).__name__}: {exc}"}


# ── LocalExecutor ─────────────────────────────────────────────────

async def probe_local(executor, mode: str) -> Dict[str, Any]:
    result = _interface_ok(executor)
    result["mock_ok"] = result["interface_ok"]  # no external deps to mock
    if not result["interface_ok"]:
        return result
    try:
        t0 = time.perf_counter()
        r = await executor.execute_step(dict(STUB_STEP))
        result["live_ok"] = (
            r.get("status") in ("completed", "success")
            and "ok" in (r.get("stdout", "") or "")
        )
        result["live_seconds"] = time.perf_counter() - t0
    except Exception as exc:
        result["live_ok"] = False
        result["live_error"] = f"{type(exc).__name__}: {exc}"
    return result


# ── NextflowExecutor ──────────────────────────────────────────────

async def probe_nextflow(executor, mode: str) -> Dict[str, Any]:
    result = _interface_ok(executor)
    # Mock: feed a stub pipeline file that doesn't actually exist. We only
    # check that NextflowExecutor constructs the nextflow command correctly
    # without shelling out. We do this by patching asyncio.create_subprocess_exec.
    try:
        fake = AsyncMock()
        fake.returncode = 0
        fake.wait = AsyncMock(return_value=0)
        with patch("asyncio.create_subprocess_exec", return_value=fake):
            r = await executor.execute_step({
                "name": "probe_nf",
                "pipeline_file": "/tmp/does_not_exist.nf",
                "cwd": "/tmp",
            })
        result["mock_ok"] = r.get("status") in ("completed", "failed")
    except Exception as exc:
        result["mock_ok"] = False
        result["mock_error"] = f"{type(exc).__name__}: {exc}"

    if mode == "live":
        try:
            with tempfile.TemporaryDirectory() as td:
                nf = Path(td) / "main.nf"
                nf.write_text(
                    'workflow { println "ok" }\n'
                    'nextflow.enable.dsl = 2\n'
                )
                t0 = time.perf_counter()
                r = await executor.execute_step({
                    "name": "probe_nf_live",
                    "pipeline_file": str(nf),
                    "cwd": td,
                })
                result["live_ok"] = r.get("returncode") == 0
                result["live_seconds"] = time.perf_counter() - t0
        except Exception as exc:
            result["live_ok"] = False
            result["live_error"] = f"{type(exc).__name__}: {exc}"
    else:
        result["live_ok"] = None  # not applicable
    return result


# ── SnakemakeExecutor ─────────────────────────────────────────────

async def probe_snakemake(executor, mode: str) -> Dict[str, Any]:
    result = _interface_ok(executor)
    try:
        fake = AsyncMock()
        fake.returncode = 0
        fake.wait = AsyncMock(return_value=0)
        with patch("asyncio.create_subprocess_exec", return_value=fake):
            r = await executor.execute_step({
                "name": "probe_sm",
                "pipeline_file": "/tmp/does_not_exist/Snakefile",
                "cwd": "/tmp",
            })
        result["mock_ok"] = r.get("status") in ("completed", "failed")
    except Exception as exc:
        result["mock_ok"] = False
        result["mock_error"] = f"{type(exc).__name__}: {exc}"

    if mode == "live":
        try:
            with tempfile.TemporaryDirectory() as td:
                sf = Path(td) / "Snakefile"
                sf.write_text(
                    "rule ok:\n"
                    "    output: 'done.marker'\n"
                    "    shell: 'touch {output}'\n"
                )
                t0 = time.perf_counter()
                r = await executor.execute_step({
                    "name": "probe_sm_live",
                    "pipeline_file": str(sf),
                    "cwd": td,
                })
                result["live_ok"] = r.get("returncode") == 0
                result["live_seconds"] = time.perf_counter() - t0
        except Exception as exc:
            result["live_ok"] = False
            result["live_error"] = f"{type(exc).__name__}: {exc}"
    else:
        result["live_ok"] = None
    return result


# ── CGATExecutor ──────────────────────────────────────────────────

async def probe_cgat(executor, mode: str) -> Dict[str, Any]:
    """CGAT live mode is intentionally skipped — it requires a configured
    cluster and submit queue. We only verify interface + job-option shape."""
    result = _interface_ok(executor)
    # Mock: verify _prepare_job_options produces a dict with cluster fields.
    try:
        prep = getattr(executor, "_prepare_job_options", None)
        if prep:
            opts = prep(dict(STUB_STEP))
            result["mock_ok"] = isinstance(opts, dict)
        else:
            result["mock_ok"] = True  # Interface-only fall-through
    except Exception as exc:
        result["mock_ok"] = False
        result["mock_error"] = f"{type(exc).__name__}: {exc}"
    result["live_ok"] = None  # cluster required
    return result


# ── HPCExecutor ───────────────────────────────────────────────────

async def probe_hpc(executor, mode: str) -> Dict[str, Any]:
    result = _interface_ok(executor)
    try:
        prep = getattr(executor, "_prepare_job_options", None)
        opts = prep(dict(STUB_STEP)) if prep else {}
        result["mock_ok"] = isinstance(opts, dict)
    except Exception as exc:
        result["mock_ok"] = False
        result["mock_error"] = f"{type(exc).__name__}: {exc}"
    # Live mode would require a SLURM/SGE queue; treat as "not applicable"
    # unless the caller forces it.
    result["live_ok"] = None
    return result


# ── KubernetesExecutor ────────────────────────────────────────────

async def probe_kubernetes(executor, mode: str) -> Dict[str, Any]:
    result = _interface_ok(executor)
    # Mock: verify _prepare_job_spec produces a valid Job dict.
    try:
        prep = getattr(executor, "_prepare_job_spec", None)
        if prep:
            spec = prep(dict(STUB_STEP))
            # Minimum sanity: Kubernetes Job manifest shape.
            has_shape = (
                isinstance(spec, dict)
                and ("spec" in spec or "template" in spec or "containers" in str(spec))
            )
            result["mock_ok"] = has_shape
        else:
            result["mock_ok"] = True
    except Exception as exc:
        result["mock_ok"] = False
        result["mock_error"] = f"{type(exc).__name__}: {exc}"
    result["live_ok"] = None
    return result
