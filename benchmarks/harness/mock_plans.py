"""Deterministic canned plans for ``--mock`` benchmark runs.

Shared by :mod:`bench_planning` and :mod:`bench_competitors` so offline
smoke tests and CI stay aligned with the scoring rules in
:func:`harness.metrics.score_plan`.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List


def _pad_steps(
    steps: List[Dict[str, Any]],
    *,
    min_steps: int,
    step_name: Callable[[int], str],
    tools: List[str],
) -> None:
    """Extend ``steps`` in-place until ``len(steps) >= min_steps``.

    Extra steps reuse the last expected tool so ``expected_tools`` coverage
    stays satisfied while ``expected_min_steps`` can be met.
    """
    if min_steps <= 0:
        return
    pad_tool = tools[-1] if tools else "fastqc"
    while len(steps) < min_steps:
        i = len(steps)
        dep = step_name(i - 1) if i else None
        steps.append(
            {
                "name": step_name(i),
                "command": f"{pad_tool} input.fastq.gz",
                "dependencies": [dep] if dep else [],
                "outputs": [],
                "description": "",
            }
        )


def mock_plan_from_prompt(
    prompt_entry: Dict[str, Any],
    *,
    step_name: Callable[[int], str],
) -> Dict[str, Any]:
    """Return a deterministic plan without calling any LLM.

    If ``gold_preset`` resolves in the preset catalog, returns that workflow
    verbatim. Otherwise builds one step per ``expected_tools``, padded to
    ``expected_min_steps``.
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

    tools = list(prompt_entry.get("expected_tools") or ["fastqc"])
    wf = prompt_entry.get("expected_workflow_type", "custom")
    if isinstance(wf, list):
        wf = wf[0] if wf else "custom"
    elif not isinstance(wf, str):
        wf = "custom"

    steps = [
        {
            "name": step_name(i),
            "command": f"{tool} input.fastq.gz",
            "dependencies": [] if i == 0 else [step_name(i - 1)],
            "outputs": [],
            "description": "",
        }
        for i, tool in enumerate(tools)
    ]
    min_steps = int(prompt_entry.get("expected_min_steps") or 0)
    _pad_steps(steps, min_steps=min_steps, step_name=step_name, tools=tools)

    return {"workflow_type": wf, "steps": steps}
