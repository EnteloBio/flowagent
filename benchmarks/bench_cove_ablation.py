"""Benchmark — CoVe verifier on/off ablation for FlowAgent's planner.

Companion to ``bench_validator_ablation.py`` (T0). Mirrors its shape but
flips ``FLOWAGENT_COVE_VERIFY`` instead of ``FLOWAGENT_VALIDATOR_ENABLED``.
Answers todo T4 from the FlowAgent architecture review: does an
independently-prompted verifier (Chain-of-Verification, Dhuliawala et
al. 2024) provide signal that correlates with downstream plan failure?

Important design choice: this benchmark runs the verifier in
**annotation mode** only (``FLOWAGENT_COVE_ABSTAIN=false``). The plan
ships regardless of what the verifier finds; the verifier's concerns
are recorded on the plan envelope (``_verifier.concerns``) and the
benchmark scorer can be re-run after the fact to measure correlation.

This is a deliberate choice given the recovery-taxonomy and T0
findings: we don't want to introduce *behaviour* changes (abstention)
until we've measured *signal* (correlation between verifier concerns
and overall_pass). If the signal is good, a follow-up run with
``FLOWAGENT_COVE_ABSTAIN=true`` measures the abstention behaviour
change with a known prior on the false-abstention rate.

Arms
----
* ``verifier_off`` — baseline. ``FLOWAGENT_COVE_VERIFY=false``.
  No verifier call, no overhead beyond today's planner.
* ``verifier_on``  — CoVe verifier active in annotation mode.
  ``FLOWAGENT_COVE_VERIFY=true``, ``FLOWAGENT_COVE_ABSTAIN=false``.
  Adds one structured LLM call per plan attempt.

Both arms share ``LLM_DAG_AWARE=true`` and production validator defaults
(autofix on, retry off unless overridden). Per-cell env-var writes mirror
``bench_validator_ablation.py``.

Output
------
``results/cove_ablation/<timestamp>/`` (created via ``timestamped_dir``):

  * ``verifier_off/results.jsonl`` + ``results.json`` + ``metrics.csv``
    + ``manifest.json``
  * ``verifier_on/results.jsonl``  + ``results.json`` + ``metrics.csv``
    + ``manifest.json``
  * ``paired_metrics.csv``         — per (input, model, replicate, arm)
                                     row, ready for figure generation.

The ``_verifier`` envelope (concerns, severities, weighted count) is
preserved on every cell's ``plan`` field in ``results.jsonl`` so a
post-hoc analysis can correlate verifier signals with metrics like
``overall_pass`` / ``hallucination_rate`` / ``completeness_pass``
without re-running the LLM.

Usage::

    python bench_cove_ablation.py \\
        --models claude-haiku-4-5 \\
        --replicates 3
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


# ── Per-cell runner that flips FLOWAGENT_COVE_VERIFY before delegating ─

def _make_runner(*, verifier_enabled: bool, mock: bool):
    """Return ``runner(model_cfg, entry, replicate)`` that flips
    ``FLOWAGENT_COVE_VERIFY`` before each cell.

    Always pins ``FLOWAGENT_COVE_ABSTAIN=false`` regardless of arm —
    this benchmark is for signal measurement, not for testing the
    abstention behaviour. A separate run can flip abstention on once
    we know the signal is worth acting on.
    """
    arm_label = "verifier_on" if verifier_enabled else "verifier_off"

    async def _runner(model_cfg: Dict[str, Any], entry: Dict[str, Any], rep: int):
        os.environ["FLOWAGENT_COVE_VERIFY"] = (
            "true" if verifier_enabled else "false"
        )
        # Annotation mode only. Future runs can flip this to "true" to
        # measure the abstention behaviour change.
        os.environ["FLOWAGENT_COVE_ABSTAIN"] = "false"
        result = await _planning_run_one(model_cfg, entry, rep, mock=mock)
        result["arm"] = arm_label
        result["verifier_enabled"] = verifier_enabled
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
        description="CoVe verifier on/off ablation (todo T4).")
    ap.add_argument("--models", default=None,
                    help="Comma-separated model IDs to run "
                         "(default: all in config).")
    ap.add_argument("--replicates", type=int, default=3,
                    help="Replicates per (model, prompt, arm) cell. Default 3.")
    ap.add_argument("--prompts", default=str(_HERE / "corpus" / "prompts.yaml"))
    ap.add_argument("--config", default=str(_HERE / "config" / "models.yaml"))
    ap.add_argument("--out", default="results")
    ap.add_argument("--mock", action="store_true",
                    help="Skip real LLM calls; use canned mock plans.")
    ap.add_argument("--arms", default="verifier_on,verifier_off",
                    help="Which arms to run (default: both, comma-separated).")
    ap.add_argument("--limit", type=int, default=None,
                    help="If set, only the first N prompts are run "
                         "(useful for the smoke / pilot run).")
    ap.add_argument("--resume", default=None,
                    help="Path to an existing results/cove_ablation/<ts>/ "
                         "dir; skips cells already in each arm's "
                         "results.jsonl.")
    args = ap.parse_args()

    only = parse_models_filter(args.models)
    models = _load_models(Path(args.config), only)
    inputs = load_yaml(Path(args.prompts))["prompts"]
    if args.limit:
        inputs = inputs[: args.limit]

    arms_to_run = [a.strip() for a in args.arms.split(",") if a.strip()]
    valid = {"verifier_on", "verifier_off"}
    bad = [a for a in arms_to_run if a not in valid]
    if bad:
        raise SystemExit(f"Unknown arm(s): {bad}; valid: {sorted(valid)}")

    if args.resume:
        out_dir = Path(args.resume)
        if not out_dir.is_dir():
            raise SystemExit(f"--resume dir does not exist: {out_dir}")
        print(f"[resume] reusing {out_dir}")
    else:
        out_dir = timestamped_dir(Path(args.out), "cove_ablation")

    # Snapshot the pre-run env so we can restore on exit and not leak
    # the toggles into a follow-up benchmark run in the same shell.
    prev_verify = os.environ.get("FLOWAGENT_COVE_VERIFY")
    prev_abstain = os.environ.get("FLOWAGENT_COVE_ABSTAIN")

    arms_results: Dict[str, List[Dict[str, Any]]] = {}
    try:
        async def _run_all_arms() -> Dict[str, List[Dict[str, Any]]]:
            results: Dict[str, List[Dict[str, Any]]] = {}
            for arm in arms_to_run:
                arm_dir = out_dir / arm
                arm_dir.mkdir(parents=True, exist_ok=True)
                verifier_enabled = (arm == "verifier_on")
                print(
                    f"\n=== Arm: {arm} "
                    f"(FLOWAGENT_COVE_VERIFY="
                    f"{'true' if verifier_enabled else 'false'}, "
                    f"FLOWAGENT_COVE_ABSTAIN=false) ==="
                )

                runner = _make_runner(
                    verifier_enabled=verifier_enabled, mock=args.mock,
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
                    benchmark_name=f"cove_ablation:{arm}",
                )
                print(f"[ok] arm={arm} wrote {len(sweep_result.results)} rows -> {arm_dir}")
                results[arm] = sweep_result.results
            return results

        arms_results = asyncio.run(_run_all_arms())
    finally:
        for var, prev in (
            ("FLOWAGENT_COVE_VERIFY", prev_verify),
            ("FLOWAGENT_COVE_ABSTAIN", prev_abstain),
        ):
            if prev is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = prev

    if len(arms_results) >= 2:
        paired = _write_paired_csv(out_dir, arms_results)
        print(f"\n[paired] joined CSV written to {paired}")

    write_manifest(out_dir, benchmark="cove_ablation", models=models, extra={
        "arms": arms_to_run,
        "num_inputs": len(inputs),
        "replicates": args.replicates,
        "rows_per_arm": {k: len(v) for k, v in arms_results.items()},
        "abstention_mode": "annotation-only (FLOWAGENT_COVE_ABSTAIN=false)",
    })


if __name__ == "__main__":
    main()
