"""Detect which execution backends are available on this host.

Drives Benchmark D's ``mode`` selection: executors get graded live if their
infrastructure is present, mock-only otherwise.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
from typing import Set


def _has_binary(name: str) -> bool:
    return shutil.which(name) is not None


def _has_module(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def _kubectl_reachable() -> bool:
    """Best-effort check that a K8s API server is configured and reachable."""
    if not _has_binary("kubectl"):
        return False
    try:
        result = subprocess.run(
            ["kubectl", "cluster-info", "--request-timeout=2s"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def available_executors() -> Set[str]:
    """Return the subset of executor backends that can be live-tested."""
    available = {"local"}  # Always available.

    if _has_binary("nextflow"):
        available.add("nextflow")
    if _has_binary("snakemake"):
        available.add("snakemake")
    if _has_module("cgatcore"):
        available.add("cgat")
    if _has_binary("sbatch") or _has_binary("qsub"):
        available.add("hpc")
    if _kubectl_reachable():
        available.add("kubernetes")
    return available


if __name__ == "__main__":
    av = available_executors()
    for name in ("local", "nextflow", "snakemake", "cgat", "hpc", "kubernetes"):
        status = "live" if name in av else "mock"
        print(f"  {name:10s}  {status}")
