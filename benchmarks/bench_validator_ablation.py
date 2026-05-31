"""Benchmark — validator-on-vs-off ablation for FlowAgent's planner.

Mirrors :mod:`bench_ablation` (DAG-awareness ablation) but flips
``FLOWAGENT_VALIDATOR_ENABLED`` instead of ``LLM_DAG_AWARE``. Answers todo
T0 from the FlowAgent architecture review: how much of the recent
robustness gain comes from the post-generation validator + auto-fix
layer (``flowagent.core.llm.LLMInterface._autofix_generated_steps`` and
``_validate_generated_steps``) versus the underlying LLM?

Arms
----
* ``validator_on``  — default planner. Validator + autofix + retry-on-fail
  active (``FLOWAGENT_VALIDATOR_ENABLED=true``).
* ``validator_off`` — ablation. Both call sites are skipped and the LLM's
  first emission ships unchanged. The retry path is also skipped so the
  off arm doesn't lose to the on arm on extra inference rounds rather than
  on validator quality.

Both arms reuse :func:`bench_planning.run_one` so token / cost accounting
and scoring are identical to the existing planning benchmark; the only
difference is the env-var flip performed per cell by the runner.

Output
------
``results/validator_ablation/<timestamp>/`` (created via ``timestamped_dir``):

  * ``validator_on/results.jsonl``  + ``results.json`` + ``metrics.csv`` + ``manifest.json``
  * ``validator_off/results.jsonl`` + ``results.json`` + ``metrics.csv`` + ``manifest.json``
  * ``paired_metrics.csv``          — per (input, model, replicate, arm)
                                      row, ready for ``make_validator_figure.py``.

Usage::

    python bench_validator_ablation.py \\
        --models gpt-4.1 \\
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

# Make sibling harness/ + flowagent/ importable when run directly.
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from harness.runner import (  # noqa: E402
    load_yaml, parse_models_filter, set_provider, sweep, timestamped_dir, write_manifest,
)
from bench_planning import run_one as _planning_run_one  # noqa: E402


# ── Per-cell runner that flips FLOWAGENT_VALIDATOR_ENABLED before delegating ─

def _make_runner(*, validator_enabled: bool, mock: bool):
    """Return a coroutine ``runner(model_cfg, entry, replicate)`` that runs
    ``bench_planning.run_one`` with ``FLOWAGENT_VALIDATOR_ENABLED`` set to
    the requested value before each cell.

    The env-var write is done inside the runner (not once at startup)
    because the harness sweeps cells; setting it per-cell makes the runner
    safe regardless of concurrency or run ordering.
    """
    arm_label = "validator_on" if validator_enabled else "validator_off"

    async def _runner(model_cfg: Dict[str, Any], entry: Dict[str, Any], rep: int):
        os.environ["FLOWAGENT_VALIDATOR_ENABLED"] = (
            "true" if validator_enabled else "false"
        )
        result = await _planning_run_one(model_cfg, entry, rep, mock=mock)
        result["arm"] = arm_label
        result["validator_enabled"] = validator_enabled
        return result

    return _runner


# ── Paired-results consolidator ────────────────────────────────────

def _write_paired_csv(out_dir: Path,
                      arms: Dict[str, List[Dict[str, Any]]]) -> Path:
    """Write a single CSV joining both arms by (model, input_id, replicate)."""
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
        description="Validator-on-vs-off ablation (todo T0).")
    ap.add_argument("--models", default=None,
                    help="Comma-separated model IDs to run (default: all in config).")
    ap.add_argument("--replicates", type=int, default=3,
                    help="Replicates per (model, prompt, arm) cell. Default 3.")
    ap.add_argument("--prompts", default=str(_HERE / "corpus" / "prompts.yaml"))
    ap.add_argument("--config", default=str(_HERE / "config" / "models.yaml"))
    ap.add_argument("--out", default="results")
    ap.add_argument("--mock", action="store_true",
                    help="Skip real LLM calls; use canned mock plans.")
    ap.add_argument("--arms", default="validator_on,validator_off",
                    help="Which arms to run (default: both, comma-separated).")
    ap.add_argument("--limit", type=int, default=None,
                    help="If set, only the first N prompts are run "
                         "(useful for the smoke / pilot run).")
    ap.add_argument("--resume", default=None,
                    help="Path to an existing results/validator_ablation/<ts>/ "
                         "dir; skips cells already in each arm's results.jsonl.")
    args = ap.parse_args()

    only = parse_models_filter(args.models)
    models = _load_models(Path(args.config), only)
    inputs = load_yaml(Path(args.prompts))["prompts"]
    if args.limit:
        inputs = inputs[: args.limit]

    arms_to_run = [a.strip() for a in args.arms.split(",") if a.strip()]
    valid = {"validator_on", "validator_off"}
    bad = [a for a in arms_to_run if a not in valid]
    if bad:
        raise SystemExit(f"Unknown arm(s): {bad}; valid: {sorted(valid)}")

    if args.resume:
        out_dir = Path(args.resume)
        if not out_dir.is_dir():
            raise SystemExit(f"--resume dir does not exist: {out_dir}")
        print(f"[resume] reusing {out_dir}")
    else:
        out_dir = timestamped_dir(Path(args.out), "validator_ablation")

    # Snapshot the pre-run env so we can restore the flag on exit and not
    # leak the toggle into a follow-up benchmark run in the same shell.
    prev_flag = os.environ.get("FLOWAGENT_VALIDATOR_ENABLED")

    arms_results: Dict[str, List[Dict[str, Any]]] = {}
    try:
        async def _run_all_arms() -> Dict[str, List[Dict[str, Any]]]:
            results: Dict[str, List[Dict[str, Any]]] = {}
            for arm in arms_to_run:
                arm_dir = out_dir / arm
                arm_dir.mkdir(parents=True, exist_ok=True)
                validator_enabled = (arm == "validator_on")
                print(
                    f"\n=== Arm: {arm} "
                    f"(FLOWAGENT_VALIDATOR_ENABLED="
                    f"{'true' if validator_enabled else 'false'}) ==="
                )

                runner = _make_runner(
                    validator_enabled=validator_enabled, mock=args.mock,
                )

                async def _wrapped(m, e, r, _runner=runner, _mock=args.mock):
                    if not _mock:
                        set_provider(m)
                    return await _runner(m, e, r)

                sweep_result = await sweep(
                    _wrapped,
                    models=models, inputs=inputs,
                    replicates=args.replicates,
                    out_dir=arm_dir,
                    benchmark_name=f"validator_ablation:{arm}",
                )
                print(f"[ok] arm={arm} wrote {len(sweep_result.results)} rows -> {arm_dir}")
                results[arm] = sweep_result.results
            return results

        arms_results = asyncio.run(_run_all_arms())
    finally:
        if prev_flag is None:
            os.environ.pop("FLOWAGENT_VALIDATOR_ENABLED", None)
        else:
            os.environ["FLOWAGENT_VALIDATOR_ENABLED"] = prev_flag

    if len(arms_results) >= 2:
        paired = _write_paired_csv(out_dir, arms_results)
        print(f"\n[paired] joined CSV written to {paired}")

    write_manifest(out_dir, benchmark="validator_ablation", models=models, extra={
        "arms": arms_to_run,
        "num_inputs": len(inputs),
        "replicates": args.replicates,
        "rows_per_arm": {k: len(v) for k, v in arms_results.items()},
    })


if __name__ == "__main__":
    main()
