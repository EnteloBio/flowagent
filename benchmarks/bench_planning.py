"""Benchmark A — Planning correctness across LLM providers.

Feeds each prompt in ``corpus/prompts.yaml`` through FlowAgent's LLM
planner for each model in ``config/models.yaml``, with N replicates. Scores
every plan with ``harness.metrics.score_plan`` and writes results + figure.

Mock mode (``--mock``): the LLM is replaced with a canned response for each
prompt (pulled from ``PRESET_CATALOG`` when a ``gold_preset`` is declared),
so the harness can be exercised with no API key.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Allow ``python bench_planning.py`` to find both the harness package
# and the flowagent package installed as a sibling of benchmarks/.
_HERE_DIR = Path(__file__).parent
sys.path.insert(0, str(_HERE_DIR))
sys.path.insert(0, str(_HERE_DIR.parent))

from harness.metrics import score_plan, cost_usd   # noqa: E402
from harness.runner import (                       # noqa: E402
    load_yaml, set_provider, sweep, timestamped_dir,
)

HERE = Path(__file__).parent


# ── Real LLM call ────────────────────────────────────────────────

async def _real_plan(prompt_entry: Dict[str, Any]) -> Dict[str, Any]:
    from flowagent.core.llm import LLMInterface
    from flowagent.core.schemas import PipelineContext

    # Minimal context: lets the LLM skip interactive reference resolution.
    ctx = PipelineContext(
        input_files=["HBR_Rep1_R1.fastq.gz", "HBR_Rep1_R2.fastq.gz"],
        paired_end=True,
        organism="human",
        genome_build="GRCh38",
        workflow_type=prompt_entry.get("expected_workflow_type", ""),
    )
    llm = LLMInterface()
    return await llm.generate_workflow_plan(prompt_entry["prompt"], context=ctx)


# ── Mock LLM response ────────────────────────────────────────────

def _mock_plan(prompt_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Return a plausible plan without calling any LLM.

    If the prompt has a ``gold_preset``, use that preset verbatim so the
    metrics pipeline exercises a well-formed plan. Otherwise synthesise a
    minimal plan from ``expected_workflow_type`` + ``expected_tools``.
    """
    gold = prompt_entry.get("gold_preset")
    if gold:
        try:
            from flowagent.presets.catalog import get_preset
            preset = get_preset(gold)
            if preset:
                return {
                    "workflow_type": preset["workflow_type"],
                    "steps": copy.deepcopy(preset["steps"]),
                }
        except Exception:
            pass
    tools = prompt_entry.get("expected_tools") or ["fastqc"]
    steps = [
        {"name": f"step_{i}", "command": f"{tool} input.fastq.gz",
         "dependencies": [] if i == 0 else [f"step_{i-1}"],
         "outputs": [], "description": ""}
        for i, tool in enumerate(tools)
    ]
    return {
        "workflow_type": prompt_entry.get("expected_workflow_type", "custom"),
        "steps": steps,
    }


# ── Per-cell runner ──────────────────────────────────────────────

async def run_one(model_cfg: Dict[str, Any], entry: Dict[str, Any],
                  replicate: int, *, mock: bool = False) -> Dict[str, Any]:
    if not mock:
        set_provider(model_cfg)
        plan = await _real_plan(entry)
    else:
        plan = _mock_plan(entry)

    metrics = score_plan(plan, entry)
    return {
        "model": model_cfg["id"],
        "provider": model_cfg.get("provider"),
        "input_id": entry["id"],
        "prompt": entry["prompt"],
        "replicate": replicate,
        "plan": plan,
        **metrics,
    }


# ── CLI ──────────────────────────────────────────────────────────

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", help="Comma-separated model IDs to run "
                                     "(default: all in config/models.yaml)")
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--mock", action="store_true",
                    help="Skip real LLM calls (uses canned preset responses)")
    ap.add_argument("--prompts", default=str(HERE / "corpus" / "prompts.yaml"))
    ap.add_argument("--config", default=str(HERE / "config" / "models.yaml"))
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    only = args.models.split(",") if args.models else []
    models = _load_models(Path(args.config), only)
    inputs = load_yaml(Path(args.prompts))["prompts"]

    out_dir = timestamped_dir(Path(args.out), "planning")

    async def _runner(m, e, r):
        return await run_one(m, e, r, mock=args.mock)

    sweep_result = asyncio.run(sweep(
        _runner, models=models, inputs=inputs,
        replicates=args.replicates, out_dir=out_dir,
        benchmark_name="planning",
    ))
    print(f"[ok] wrote {len(sweep_result.results)} rows → {out_dir}")


if __name__ == "__main__":
    main()
