"""Benchmark M -- workflow tool-hint ablation for FlowAgent's planner.

Runs the same prompt corpus + model set under two planner configurations
to answer: does the per-workflow tool allowlist injected into the
planning prompt improve plan quality?

Arms
----
* ``hint_on``  -- ``FLOWAGENT_TOOL_HINT=true`` (default). The planner
  prompt includes "Valid tool names for this workflow…" from
  ``LLMInterface._tool_hint_for_workflow_type``.
* ``hint_off`` -- ``FLOWAGENT_TOOL_HINT=false``. No tool catalogue in
  the prompt; the LLM chooses tools from training data alone.

Both arms keep ``LLM_DAG_AWARE=true`` and the rest of the default
planner stack so this isolates the tool-hint layer.

Output
------
``results/tool_hint_ablation/<timestamp>/``:

  * ``hint_on/results.jsonl``  + ``results.json`` + ``metrics.csv`` + ``manifest.json``
  * ``hint_off/results.jsonl`` + ``results.json`` + ``metrics.csv`` + ``manifest.json``
  * ``paired_metrics.csv``

Usage::

    python bench_tool_hint_ablation.py \\
        --models gpt-5.4-mini \\
        --replicates 3 \\
        --prompts corpus/prompts.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from harness.runner import (  # noqa: E402
    load_yaml, parse_models_filter, set_provider, sweep, timestamped_dir, write_manifest,
)
from bench_planning import run_one as _planning_run_one  # noqa: E402


def _make_runner(*, tool_hint: bool, mock: bool):
    arm_label = "hint_on" if tool_hint else "hint_off"

    async def _runner(model_cfg: Dict[str, Any], entry: Dict[str, Any], rep: int):
        os.environ["FLOWAGENT_TOOL_HINT"] = "true" if tool_hint else "false"
        os.environ["LLM_DAG_AWARE"] = "true"
        result = await _planning_run_one(model_cfg, entry, rep, mock=mock)
        result["arm"] = arm_label
        result["tool_hint_enabled"] = tool_hint
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
                    keys.append(k)
                    seen.add(k)

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
        description="Workflow tool-hint ablation (Benchmark M).")
    ap.add_argument("--models", default=None,
                    help="Comma-separated model IDs (default: all in config).")
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--prompts", default=str(_HERE / "corpus" / "prompts.yaml"))
    ap.add_argument("--config", default=str(_HERE / "config" / "models.yaml"))
    ap.add_argument("--out", default="results")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--arms", default="hint_on,hint_off")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    only = parse_models_filter(args.models)
    models = _load_models(Path(args.config), only)
    inputs = load_yaml(Path(args.prompts))["prompts"]
    if args.limit:
        inputs = inputs[: args.limit]

    arms_to_run = [a.strip() for a in args.arms.split(",") if a.strip()]
    valid = {"hint_on", "hint_off"}
    bad = [a for a in arms_to_run if a not in valid]
    if bad:
        raise SystemExit(f"Unknown arm(s): {bad}; valid: {sorted(valid)}")

    if args.resume:
        out_dir = Path(args.resume)
        if not out_dir.is_dir():
            raise SystemExit(f"--resume dir does not exist: {out_dir}")
        print(f"[resume] reusing {out_dir}")
    else:
        out_dir = timestamped_dir(Path(args.out), "tool_hint_ablation")

    prev_hint = os.environ.get("FLOWAGENT_TOOL_HINT")
    prev_dag = os.environ.get("LLM_DAG_AWARE")

    arms_results: Dict[str, List[Dict[str, Any]]] = {}
    try:
        async def _run_all_arms() -> Dict[str, List[Dict[str, Any]]]:
            results: Dict[str, List[Dict[str, Any]]] = {}
            for arm in arms_to_run:
                arm_dir = out_dir / arm
                arm_dir.mkdir(parents=True, exist_ok=True)
                tool_hint = (arm == "hint_on")
                print(
                    f"\n=== Arm: {arm} "
                    f"(FLOWAGENT_TOOL_HINT={'true' if tool_hint else 'false'}) ==="
                )

                runner = _make_runner(tool_hint=tool_hint, mock=args.mock)

                async def _wrapped(m, e, r, _runner=runner, _mock=args.mock):
                    if not _mock:
                        set_provider(m)
                    return await _runner(m, e, r)

                sweep_result = await sweep(
                    _wrapped,
                    models=models, inputs=inputs,
                    replicates=args.replicates,
                    out_dir=arm_dir,
                    benchmark_name=f"tool_hint_ablation:{arm}",
                )
                print(f"[ok] arm={arm} wrote {len(sweep_result.results)} rows -> {arm_dir}")
                results[arm] = sweep_result.results
            return results

        arms_results = asyncio.run(_run_all_arms())
    finally:
        for var, prev in (
            ("FLOWAGENT_TOOL_HINT", prev_hint),
            ("LLM_DAG_AWARE", prev_dag),
        ):
            if prev is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = prev

    if len(arms_results) >= 2:
        paired = _write_paired_csv(out_dir, arms_results)
        print(f"\n[paired] joined CSV written to {paired}")

    write_manifest(out_dir, benchmark="tool_hint_ablation", models=models, extra={
        "arms": arms_to_run,
        "num_inputs": len(inputs),
        "replicates": args.replicates,
        "rows_per_arm": {k: len(v) for k, v in arms_results.items()},
    })


if __name__ == "__main__":
    main()
