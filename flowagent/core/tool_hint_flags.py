"""Env-flag gating for the per-workflow tool allowlist planner hint (todo T3).

``FLOWAGENT_TOOL_HINT`` (default **on**) controls whether
:meth:`flowagent.core.llm.LLMInterface.generate_workflow_plan` injects
the "Valid tool names for this workflow…" block derived from
``_tool_hint_for_workflow_type``. When off, the LLM receives no
workflow-scoped tool catalogue in the planning prompt.

Read fresh on every call so benchmark harnesses can flip the flag per
cell without restarting the process.
"""

from __future__ import annotations

import os

_FALSE = frozenset({"0", "false", "no", "off"})


def tool_hint_enabled() -> bool:
    raw = os.environ.get("FLOWAGENT_TOOL_HINT")
    if raw is None:
        return True
    return raw.strip().lower() not in _FALSE
