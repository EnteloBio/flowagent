"""Benchmark I -- DAG-completeness reflection ablation.

Companion to ``bench_ablation.py`` (Benchmark H, DAG-aware vs DAG-blind).
Both arms here keep ``LLM_DAG_AWARE=true``; the only difference is
whether the planner runs the new DAG-Plan-style structural completeness
check + LLM reflection loop introduced in
``flowagent/core/completeness.py``:

* ``reflect_on``  -- ``LLM_COMPLETENESS_REFLECT=true`` (default). After
  each plan the validator runs four structural rules
  (every align needs an index/download ancestor; every download must
  be consumed; quantify/call/de must reach an informative sink;
  graph weakly connected with a terminal sink). On failure, the
  failure list is appended to the conversation and the LLM is
  re-queried, up to ``LLM_COMPLETENESS_MAX_RETRIES`` times.
* ``reflect_off`` -- ``LLM_COMPLETENESS_REFLECT=false``. The validator
  is still applied at score time (so we get ``completeness_pass`` as
  a non-gating per-row metric for both arms), but the planner does
  not retry on failure -- it accepts the first draft.

Both arms share the rest of the planner stack: the ``kind`` field on
each step, the post-hoc ``fill_missing_kinds`` heuristic, the
reference-download wiring, the structured-output schema. So this is
a clean A/B of the reflection retry loop alone, not a confound with
typed nodes.

Outputs follow the same layout as ``bench_ablation.py``::

    results/reflection/<timestamp>/
      reflect_on/results.jsonl  + results.json + metrics.csv + manifest.json
      reflect_off/results.jsonl + results.json + metrics.csv + manifest.json
      paired_metrics.csv  -- one row per (input, model, replicate, arm)

Use ``make_reflection_figure.py`` to render bar charts + paired stats
on ``paired_metrics.csv``.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Make sibling harness/ + flowagent/ importable when run directly.
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from harness.runner import (  # noqa: E402
    load_yaml, set_provider, sweep, timestamped_dir, write_manifest,
)
from bench_planning import run_one as _planning_run_one  # noqa: E402


def _make_runner(*, reflect: bool, mock: bool):
    """Per-cell runner that flips ``LLM_COMPLETENESS_REFLECT`` before
    delegating to ``bench_planning.run_one``.

    The env-var write happens inside the runner so concurrent cells in
    the harness sweep don't race. Today both arms run sequentially
    (one ``sweep`` call after the other) but per-cell setting is the
    safe default.
    """
    arm_label = "reflect_on" if reflect else "reflect_off"

    async def _runner(model_cfg: Dict[str, Any], entry: Dict[str, Any], rep: int):
        os.environ["LLM_COMPLETENESS_REFLECT"] = "true" if reflect else "false"
        # Keep DAG-aware enabled for both arms so we isolate the
        # reflection loop's effect, not the DAG prompt itself. (The
        # prior Benchmark H studies dag_aware vs dag_blind separately.)
        os.environ["LLM_DAG_AWARE"] = "true"
        result = await _planning_run_one(model_cfg, entry, rep, mock=mock)
        result["arm"] = arm_label
        result["reflect_enabled"] = reflect
        return result

    return _runner


def _write_paired_csv(out_dir: Path,
                      arms: Dict[str, List[Dict[str, Any]]]) -> Path:
    target = out_dir / "paired_metrics.csv"

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
        description="Completeness-reflection ablation (Benchmark I).")
    ap.add_argument("--models", default=None,
                    help="Comma-separated model IDs (default: all in config).")
    ap.add_argument("--replicates", type=int, default=3,
                    help="Replicates per (model, prompt, arm) cell. Default 3.")
    ap.add_argument("--prompts", default=str(_HERE / "corpus" / "prompts.yaml"))
    ap.add_argument("--config", default=str(_HERE / "config" / "models.yaml"))
    ap.add_argument("--out", default="results")
    ap.add_argument("--mock", action="store_true",
                    help="Skip real LLM calls; use canned mock plans.")
    ap.add_argument("--arms", default="reflect_on,reflect_off",
                    help="Which arms to run (comma-separated).")
    ap.add_argument("--limit", type=int, default=None,
                    help="If set, only the first N prompts are run.")
    ap.add_argument("--max-retries", type=int, default=None,
                    help="Override LLM_COMPLETENESS_MAX_RETRIES for the "
                         "reflect_on arm. Defaults to the package default (2).")
    ap.add_argument("--resume", default=None,
                    help="Path to an existing results/reflection/<ts>/ dir.")
    args = ap.parse_args()

    only = args.models.split(",") if args.models else []
    models = _load_models(Path(args.config), only)
    inputs = load_yaml(Path(args.prompts))["prompts"]
    if args.limit:
        inputs = inputs[: args.limit]

    arms_to_run = [a.strip() for a in args.arms.split(",") if a.strip()]
    valid = {"reflect_on", "reflect_off"}
    bad = [a for a in arms_to_run if a not in valid]
    if bad:
        raise SystemExit(f"Unknown arm(s): {bad}; valid: {sorted(valid)}")

    if args.resume:
        out_dir = Path(args.resume)
        if not out_dir.is_dir():
            raise SystemExit(f"--resume dir does not exist: {out_dir}")
        print(f"[resume] reusing {out_dir}")
    else:
        out_dir = timestamped_dir(Path(args.out), "reflection")

    # Snapshot env so we restore on exit and don't leak the toggle.
    prev_reflect = os.environ.get("LLM_COMPLETENESS_REFLECT")
    prev_dag = os.environ.get("LLM_DAG_AWARE")
    prev_retries = os.environ.get("LLM_COMPLETENESS_MAX_RETRIES")
    if args.max_retries is not None:
        os.environ["LLM_COMPLETENESS_MAX_RETRIES"] = str(args.max_retries)

    arms_results: Dict[str, List[Dict[str, Any]]] = {}
    try:
        for arm in arms_to_run:
            arm_dir = out_dir / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            reflect = (arm == "reflect_on")
            print(f"\n=== Arm: {arm} (LLM_COMPLETENESS_REFLECT={'true' if reflect else 'false'}) ===")

            runner = _make_runner(reflect=reflect, mock=args.mock)

            async def _wrapped(m, e, r, _runner=runner, _mock=args.mock):
                if not _mock:
                    set_provider(m)
                return await _runner(m, e, r)

            sweep_result = asyncio.run(sweep(
                _wrapped,
                models=models, inputs=inputs,
                replicates=args.replicates,
                out_dir=arm_dir,
                benchmark_name=f"reflection:{arm}",
            ))
            print(f"[ok] arm={arm} wrote {len(sweep_result.results)} rows -> {arm_dir}")
            arms_results[arm] = sweep_result.results
    finally:
        # Restore env.
        for var, prev in (
            ("LLM_COMPLETENESS_REFLECT", prev_reflect),
            ("LLM_DAG_AWARE",            prev_dag),
            ("LLM_COMPLETENESS_MAX_RETRIES", prev_retries),
        ):
            if prev is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = prev

    if len(arms_results) >= 2:
        paired = _write_paired_csv(out_dir, arms_results)
        print(f"\n[paired] joined CSV written to {paired}")

    write_manifest(out_dir, benchmark="reflection", models=models, extra={
        "arms": arms_to_run,
        "num_inputs": len(inputs),
        "replicates": args.replicates,
        "rows_per_arm": {k: len(v) for k, v in arms_results.items()},
        "max_retries_env": os.environ.get("LLM_COMPLETENESS_MAX_RETRIES", ""),
    })


if __name__ == "__main__":
    main()
