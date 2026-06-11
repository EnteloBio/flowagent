"""Benchmark H -- DAG-awareness ablation for FlowAgent's planner.

Runs the same prompt corpus + model set under two FlowAgent planner
configurations and produces paired-prompt results so the question
"does telling the LLM about the dependency DAG improve plan quality?"
can be answered with a Wilcoxon signed-rank test on identical inputs.

Arms
----
* ``dag_aware`` -- default planner (``LLM_DAG_AWARE=true``). The LLM sees
  the standard DAG instruction ("Dependencies must form a valid DAG, no
  cycles") and emits a ``dependencies`` field per step.
* ``dag_blind`` -- ablation planner (``LLM_DAG_AWARE=false``). The LLM
  sees a flat-list prompt with no mention of dependencies, and the
  planner schema (``WorkflowPlanSchemaNoDAG``) has no ``dependencies``
  field. Empty dependency lists are injected post-parse so the
  resulting plan is a trivially valid DAG with zero edges.

Both arms reuse :func:`bench_planning.run_one` so token / cost
accounting and scoring are identical to the existing planning
benchmark (Benchmark A); the only difference is the env-var flip
performed by ``set_provider_with_dag``.

Output
------
``results/ablation/<timestamp>/`` (created via ``timestamped_dir``):

  * ``dag_aware/results.jsonl``  + ``results.json`` + ``metrics.csv`` + ``manifest.json``
  * ``dag_blind/results.jsonl``  + ``results.json`` + ``metrics.csv`` + ``manifest.json``
  * ``paired_metrics.csv``   -- per (input, model, replicate, arm)
                                row, ready for ``make_ablation_figure.py``.

Usage::

    python bench_ablation.py \\
        --models gpt-5.4-mini,claude-sonnet-4 \\
        --replicates 3 \\
        --prompts corpus/prompts.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make sibling harness/ + flowagent/ importable when run directly.
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from harness.runner import (  # noqa: E402
    load_yaml, parse_models_filter, set_provider, sweep, timestamped_dir, write_manifest,
)
from bench_planning import run_one as _planning_run_one  # noqa: E402


# ── Per-cell runner that flips LLM_DAG_AWARE before delegating ─────

def _make_runner(*, dag_aware: bool, mock: bool):
    """Return a coroutine ``runner(model_cfg, entry, replicate)`` that
    runs ``bench_planning.run_one`` with ``LLM_DAG_AWARE`` set to the
    requested value before each cell.

    The env-var write is done inside the runner (not once at startup)
    because the harness sweeps multiple cells in parallel; setting the
    var once would race with arms running concurrently. Today both arms
    are run sequentially (one ``sweep`` call after the other) but
    setting it per-cell makes the runner safe regardless.
    """
    arm_label = "dag_aware" if dag_aware else "dag_blind"

    async def _runner(model_cfg: Dict[str, Any], entry: Dict[str, Any], rep: int):
        os.environ["LLM_DAG_AWARE"] = "true" if dag_aware else "false"
        result = await _planning_run_one(model_cfg, entry, rep, mock=mock)
        result["arm"] = arm_label
        result["dag_aware"] = dag_aware
        return result

    return _runner


# ── Paired-results consolidator ────────────────────────────────────

def _write_paired_csv(out_dir: Path,
                      arms: Dict[str, List[Dict[str, Any]]]) -> Path:
    """Write a single CSV joining both arms by (model, input_id, replicate)."""
    target = out_dir / "paired_metrics.csv"

    # Union of metric columns across both arms (excludes nested fields).
    keys: List[str] = ["arm"]
    seen = {"arm"}
    for rows in arms.values():
        for r in rows:
            for k, v in r.items():
                if k in seen:
                    continue
                if isinstance(v, (str, int, float, bool)) or v is None:
                    keys.append(k); seen.add(k)

    with target.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for arm, rows in arms.items():
            for r in rows:
                w.writerow({k: r.get(k) for k in keys})
    return target


# ── CLI ────────────────────────────────────────────────────────────

def _load_models(cfg_path: Path, only: List[str]) -> List[Dict[str, Any]]:
    cfg = load_yaml(cfg_path)
    models = cfg["models"]
    if only:
        wanted = set(only)
        models = [m for m in models if m["id"] in wanted]
        if not models:
            raise SystemExit(f"No models in {cfg_path} match: {only}")
    return models


def main():
    ap = argparse.ArgumentParser(
        description="DAG-awareness ablation (Benchmark H).")
    ap.add_argument("--models", default=None,
                    help="Comma-separated model IDs to run (default: all in config).")
    ap.add_argument("--replicates", type=int, default=3,
                    help="Replicates per (model, prompt, arm) cell. Default 3.")
    ap.add_argument("--prompts", default=str(_HERE / "corpus" / "prompts.yaml"))
    ap.add_argument("--config", default=str(_HERE / "config" / "models.yaml"))
    ap.add_argument("--out", default="results")
    ap.add_argument("--mock", action="store_true",
                    help="Skip real LLM calls; use canned mock plans.")
    ap.add_argument("--arms", default="dag_aware,dag_blind",
                    help="Which arms to run (default: both, comma-separated).")
    ap.add_argument("--limit", type=int, default=None,
                    help="If set, only the first N prompts are run "
                         "(useful for the smoke / pilot run).")
    ap.add_argument("--resume", default=None,
                    help="Path to an existing results/ablation/<ts>/ dir; "
                         "skips cells already in each arm's results.jsonl.")
    args = ap.parse_args()

    only = parse_models_filter(args.models)
    models = _load_models(Path(args.config), only)
    inputs = load_yaml(Path(args.prompts))["prompts"]
    if args.limit:
        inputs = inputs[: args.limit]

    arms_to_run = [a.strip() for a in args.arms.split(",") if a.strip()]
    valid = {"dag_aware", "dag_blind"}
    bad = [a for a in arms_to_run if a not in valid]
    if bad:
        raise SystemExit(f"Unknown arm(s): {bad}; valid: {sorted(valid)}")

    if args.resume:
        out_dir = Path(args.resume)
        if not out_dir.is_dir():
            raise SystemExit(f"--resume dir does not exist: {out_dir}")
        print(f"[resume] reusing {out_dir}")
    else:
        out_dir = timestamped_dir(Path(args.out), "ablation")

    # Snapshot the pre-run env so we can restore LLM_DAG_AWARE on exit
    # and not leak the toggle into a follow-up benchmark in the same shell.
    prev_dag_aware = os.environ.get("LLM_DAG_AWARE")

    arms_results: Dict[str, List[Dict[str, Any]]] = {}
    try:
        async def _run_all_arms() -> Dict[str, List[Dict[str, Any]]]:
            results: Dict[str, List[Dict[str, Any]]] = {}
            for arm in arms_to_run:
                arm_dir = out_dir / arm
                arm_dir.mkdir(parents=True, exist_ok=True)
                dag_aware = (arm == "dag_aware")
                print(f"\n=== Arm: {arm} (LLM_DAG_AWARE={'true' if dag_aware else 'false'}) ===")

                runner = _make_runner(dag_aware=dag_aware, mock=args.mock)

                async def _wrapped(m, e, r, _runner=runner, _mock=args.mock):
                    if not _mock:
                        set_provider(m)
                    return await _runner(m, e, r)

                sweep_result = await sweep(
                    _wrapped,
                    models=models, inputs=inputs,
                    replicates=args.replicates,
                    out_dir=arm_dir,
                    benchmark_name=f"ablation:{arm}",
                )
                print(f"[ok] arm={arm} wrote {len(sweep_result.results)} rows -> {arm_dir}")
                results[arm] = sweep_result.results
            return results

        arms_results = asyncio.run(_run_all_arms())
    finally:
        # Restore env so subsequent runs in the same shell aren't surprised.
        if prev_dag_aware is None:
            os.environ.pop("LLM_DAG_AWARE", None)
        else:
            os.environ["LLM_DAG_AWARE"] = prev_dag_aware

    if len(arms_results) >= 2:
        paired = _write_paired_csv(out_dir, arms_results)
        print(f"\n[paired] joined CSV written to {paired}")

    # Top-level manifest summarising both arms in one place.
    write_manifest(out_dir, benchmark="ablation", models=models, extra={
        "arms": arms_to_run,
        "num_inputs": len(inputs),
        "replicates": args.replicates,
        "rows_per_arm": {k: len(v) for k, v in arms_results.items()},
    })


if __name__ == "__main__":
    main()
