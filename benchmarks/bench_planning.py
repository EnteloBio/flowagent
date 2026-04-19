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
from typing import Any, Dict, List, Tuple

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


# ── Real LLM call with token tracking ────────────────────────────

class _TokenTracker:
    """Wraps a provider and tallies token usage across all calls.

    FlowAgent's ``generate_workflow_plan`` makes several internal LLM
    calls (pattern extraction, plan generation, optional JSON repair).
    We intercept them by wrapping the provider so the benchmark sees
    the true per-plan token cost, not just a single final call.
    """

    def __init__(self, inner):
        self._inner = inner
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.call_count = 0

    def _tally(self, resp):
        u = getattr(resp, "usage", None) or {}
        self.prompt_tokens     += int(u.get("prompt_tokens", 0) or 0)
        self.completion_tokens += int(u.get("completion_tokens", 0) or 0)
        self.call_count        += 1
        return resp

    async def chat(self, *a, **kw):            return self._tally(await self._inner.chat(*a, **kw))
    async def chat_with_tools(self, *a, **kw): return self._tally(await self._inner.chat_with_tools(*a, **kw))
    async def chat_structured(self, *a, **kw): return self._tally(await self._inner.chat_structured(*a, **kw))
    async def stream(self, *a, **kw):          return await self._inner.stream(*a, **kw)

    def __getattr__(self, name):  # passthrough for anything we missed
        return getattr(self._inner, name)


async def _real_plan(prompt_entry: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    from flowagent.core.llm import LLMInterface
    from flowagent.core.schemas import PipelineContext

    ctx = PipelineContext(
        input_files=["HBR_Rep1_R1.fastq.gz", "HBR_Rep1_R2.fastq.gz"],
        paired_end=True,
        organism="human",
        genome_build="GRCh38",
        workflow_type=prompt_entry.get("expected_workflow_type", ""),
    )
    llm = LLMInterface()
    tracker = _TokenTracker(llm.provider)
    llm.provider = tracker
    plan = await llm.generate_workflow_plan(prompt_entry["prompt"], context=ctx)
    usage = {
        "prompt_tokens":     tracker.prompt_tokens,
        "completion_tokens": tracker.completion_tokens,
        "llm_calls":         tracker.call_count,
    }
    return plan, usage


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
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0}
    if not mock:
        set_provider(model_cfg)
        plan, usage = await _real_plan(entry)
    else:
        plan = _mock_plan(entry)

    metrics = score_plan(plan, entry)
    cost = cost_usd(usage["prompt_tokens"], usage["completion_tokens"], model_cfg)
    return {
        "model": model_cfg["id"],
        "provider": model_cfg.get("provider"),
        "input_id": entry["id"],
        "prompt": entry["prompt"],
        "replicate": replicate,
        "plan": plan,
        "prompt_tokens":     usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "llm_calls":         usage["llm_calls"],
        "cost_usd":          cost,
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
