"""Benchmark D — Executor coverage matrix.

Exercises every execution backend advertised by FlowAgent
(local, nextflow, snakemake, cgat, hpc, kubernetes) across three levels:

    interface_ok  - class instantiates, execute_step is async returning a dict
    mock_ok       - job-spec construction passes with external APIs mocked
    live_ok       - trivial echo step runs end-to-end (gated by env detection)

Always produces a full 6-row × 3-column result grid so the figure exists
even on hosts without any schedulers installed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_HERE_DIR = Path(__file__).parent
sys.path.insert(0, str(_HERE_DIR))
sys.path.insert(0, str(_HERE_DIR.parent))

from harness.env_detect import available_executors           # noqa: E402
from harness.executor_probes import (                        # noqa: E402
    probe_local, probe_nextflow, probe_snakemake,
    probe_cgat, probe_hpc, probe_kubernetes,
)
from harness.runner import timestamped_dir, write_manifest  # noqa: E402


PROBES = {
    "local":      probe_local,
    "nextflow":   probe_nextflow,
    "snakemake":  probe_snakemake,
    "cgat":       probe_cgat,
    "hpc":        probe_hpc,
    "kubernetes": probe_kubernetes,
}


async def _drive() -> List[Dict[str, Any]]:
    from flowagent.core.executor_factory import ExecutorFactory

    live = available_executors()
    results: List[Dict[str, Any]] = []
    for name, probe in PROBES.items():
        row: Dict[str, Any] = {
            "executor": name,
            "mode": "live" if name in live else "mock",
        }
        try:
            executor = ExecutorFactory.create(name)
        except Exception as exc:
            row.update({
                "interface_ok": False,
                "mock_ok": False,
                "live_ok": False,
                "instantiation_error": f"{type(exc).__name__}: {exc}",
            })
            results.append(row)
            continue
        try:
            probe_result = await probe(executor, row["mode"])
            row.update(probe_result)
        except Exception as exc:
            row["probe_error"] = f"{type(exc).__name__}: {exc}"
        results.append(row)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out_dir = timestamped_dir(Path(args.out), "executors")
    results = asyncio.run(_drive())

    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    from harness.runner import _write_csv
    _write_csv(out_dir / "metrics.csv", results)
    write_manifest(out_dir, benchmark="executors", models=[],
                   extra={"available_live": list(available_executors())})
    n_iface = sum(1 for r in results if r.get("interface_ok"))
    n_mock  = sum(1 for r in results if r.get("mock_ok"))
    n_live  = sum(1 for r in results if r.get("live_ok") is True)
    print(f"[ok] interface {n_iface}/{len(results)}  "
          f"mock {n_mock}/{len(results)}  live {n_live}/{len(results)} "
          f"→ {out_dir}")


if __name__ == "__main__":
    main()
