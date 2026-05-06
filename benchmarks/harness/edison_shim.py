"""Subprocess shim that drives Edison Scientific's Edison Analysis agent.

Same envelope shape as ``claude_code_shim.py`` / ``biomni_shim.py``. The
shim uses Edison's official Python SDK (``pip install edison-client``)
to submit a planning job, polls until terminal, extracts the model's
JSON workflow plan from the trajectory, and emits the standard JSON
envelope on stdout.

Edison Analysis is execution-oriented (it actually runs Python/R inside
a sandbox), so we do two things to keep the comparison apples-to-apples
with the other planning competitors:

1. Override the system prompt with
   ``system_prompt_additional_guidelines`` that forbids tool use and
   asks the agent to reply with ONLY a JSON workflow plan in the
   FlowAgent schema.
2. Cap ``max_steps`` (default 5) so it can't burn credits doing actual
   data analysis.

A per-process credit budget is enforced via ``EDISON_BUDGET_CREDITS``;
once consumed-credit metadata in any returned trajectory pushes the
running total over the budget, subsequent invocations short-circuit
with an envelope error so a long sweep can't drain a researcher's
account.

Configuration (env vars):

* ``EDISON_API_KEY``        : required; obtained from platform.edisonscientific.com.
* ``EDISON_MAX_STEPS``      : per-task ``max_steps`` cap (default 5).
* ``EDISON_LANGUAGE``       : runtime language ("PYTHON" | "R", default PYTHON).
* ``EDISON_POLL_SECONDS``   : status-poll interval in seconds (default 15).
* ``EDISON_TIMEOUT``        : overall wall-clock cap, seconds (default 1800).
* ``EDISON_BUDGET_CREDITS`` : cumulative credit cap shared across calls
                              in the same process (default: unlimited).
                              Tracked via the path in ``EDISON_BUDGET_FILE``
                              (default ``$TMPDIR/edison_budget.json``) so
                              parallel shim subprocesses share state.

Run as ``python edison_shim.py --prompt <text> [--files JSON]``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Dict, List, Optional


# ── Plan-only system prompt (FlowAgent schema) ─────────────────────

_PLAN_GUIDELINES = """STRICT OUTPUT FORMAT.

Do not execute any tool, do not run any code, do not write any files.
Your entire reply must be a single JSON object describing a
bioinformatics workflow plan, matching this schema:

{
  "workflow_type": "<rna_seq_kallisto | rna_seq_star | rna_seq_hisat | chip_seq | atac_seq | variant_calling | single_cell_10x | single_cell_kb | qc_only | custom>",
  "steps": [
    {
      "name": "<unique_snake_case_id>",
      "command": "<runnable shell pipeline using a real bioinformatics tool>",
      "dependencies": ["<prior step names, in order>"],
      "outputs": ["<expected output paths>"]
    }
  ]
}

Rules:
- Steps must be in topological order.
- ``dependencies`` must reference prior step names exactly.
- The first token of each ``command`` MUST be the bioinformatics tool
  itself (``fastqc``, ``kallisto``, ``samtools``, etc.).
- Cover every step from raw input to the requested final output.
- Return ONLY the JSON object. No markdown fences, no commentary.
"""


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Same brace-balanced extractor as the Claude Code shim."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for m in _FENCE_RE.finditer(text):
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
    starts = [i for i, c in enumerate(text) if c == "{"]
    for start in starts:
        depth = 0
        for end in range(start, len(text)):
            c = text[end]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:end + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    return None


# ── Budget tracking ───────────────────────────────────────────────

def _budget_file_path() -> Path:
    explicit = os.environ.get("EDISON_BUDGET_FILE")
    if explicit:
        return Path(explicit)
    return Path(gettempdir()) / "edison_budget.json"


def _read_budget_state() -> Dict[str, float]:
    path = _budget_file_path()
    if not path.exists():
        return {"credits_used": 0.0}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"credits_used": 0.0}


def _bump_budget(credits_delta: float) -> Dict[str, float]:
    path = _budget_file_path()
    state = _read_budget_state()
    state["credits_used"] = float(state.get("credits_used", 0.0)) + float(credits_delta)
    try:
        path.write_text(json.dumps(state))
    except Exception:
        pass
    return state


def _budget_cap() -> Optional[float]:
    cap = os.environ.get("EDISON_BUDGET_CREDITS")
    if not cap:
        return None
    try:
        return float(cap)
    except ValueError:
        return None


# ── Envelope ───────────────────────────────────────────────────────

def _empty_plan() -> Dict[str, Any]:
    return {"workflow_type": "custom", "steps": []}


def _envelope(
    plan: Dict[str, Any],
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    llm_calls: int = 0,
    cost_usd: float = 0.0,
    wall_seconds: float = 0.0,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "plan":              plan,
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "llm_calls":         llm_calls,
        "cost_usd":          cost_usd,
        "wall_seconds":      wall_seconds,
        "error":             error,
    }


# ── Edison invocation ──────────────────────────────────────────────

async def _run_edison(
    prompt: str,
    files: List[str],
    *,
    api_key: str,
    max_steps: int,
    language: str,
    poll_seconds: float,
    overall_timeout: float,
) -> Dict[str, Any]:
    """Submit one Edison Analysis task and return parsed plan + usage."""
    try:
        from edison_client import EdisonClient
        from edison_client.models import RuntimeConfig, TaskRequest
        from edison_client.models.app import JobNames
    except Exception as ie:
        return _envelope(
            _empty_plan(),
            error=("edison-client not installed. Install with "
                   f"``pip install edison-client``. Underlying error: {ie}"),
        )

    client = EdisonClient(api_key=api_key)

    runtime_config = RuntimeConfig(
        max_steps=max_steps,
        environment_config={
            "language": language,
            "prompting_config": {
                "system_prompt_additional_guidelines": _PLAN_GUIDELINES,
            },
            # Empty data_storage_uris keeps Edison from staging real data;
            # the agent should treat input files in the user prompt as
            # placeholders (they are referenced by name only in the plan).
            "data_storage_uris": [],
            "additional_tools": None,
        },
    )

    user_prompt = (
        f"{prompt}\n\nAvailable input files (treat as already on disk): "
        f"{', '.join(files) if files else '(none provided)'}.\n\n"
        f"Respond with the JSON workflow plan only -- do not execute "
        f"any code, do not call any tools."
    )

    task = TaskRequest(
        name=JobNames.ANALYSIS,
        query=user_prompt,
        runtime_config=runtime_config,
    )

    t0 = time.perf_counter()
    try:
        trajectory_id = client.create_task(task)
    except Exception as e:
        return _envelope(
            _empty_plan(),
            wall_seconds=time.perf_counter() - t0,
            error=f"edison-create-task: {type(e).__name__}: {e}",
        )

    # Poll until terminal or timeout.
    status = "queued"
    last_err: Optional[str] = None
    while True:
        if time.perf_counter() - t0 > overall_timeout:
            return _envelope(
                _empty_plan(),
                wall_seconds=time.perf_counter() - t0,
                error=f"timeout after {overall_timeout:.0f}s "
                      f"(trajectory_id={trajectory_id})",
            )
        try:
            task_state = client.get_task(trajectory_id)
            status = getattr(task_state, "status", "")
        except Exception as e:
            last_err = f"poll: {type(e).__name__}: {e}"
            await asyncio.sleep(poll_seconds)
            continue
        if status in ("completed", "succeeded", "success", "failed", "cancelled"):
            break
        await asyncio.sleep(poll_seconds)

    wall = time.perf_counter() - t0

    if status not in ("completed", "succeeded", "success"):
        return _envelope(
            _empty_plan(),
            wall_seconds=wall,
            error=f"task ended with status={status!r} ({last_err or 'no error detail'})",
        )

    # Pull the final state. Edison stores the agent's textual answer at
    # ``environment_frame['state']['state']['answer']`` per the public
    # cookbook tutorial; structure may vary across SDK versions, so we
    # fall back to scanning the entire frame for an "answer" key.
    try:
        result = client.get_task(trajectory_id, verbose=True)
    except Exception as e:
        return _envelope(
            _empty_plan(), wall_seconds=wall,
            error=f"edison-get-task-verbose: {type(e).__name__}: {e}",
        )

    answer_text = ""
    frame = getattr(result, "environment_frame", None) or {}
    if isinstance(frame, dict):
        try:
            answer_text = frame.get("state", {}).get("state", {}).get("answer") or ""
        except Exception:
            answer_text = ""
        if not answer_text:
            # Walk the whole frame for any ``answer`` value as a last resort.
            stack: List[Any] = [frame]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    if isinstance(node.get("answer"), str):
                        answer_text = node["answer"]
                        break
                    stack.extend(node.values())
                elif isinstance(node, list):
                    stack.extend(node)

    plan_obj = _extract_json_object(answer_text) or {}
    if not isinstance(plan_obj.get("steps"), list):
        for key in ("plan", "workflow", "pipeline"):
            inner = plan_obj.get(key) if isinstance(plan_obj, dict) else None
            if isinstance(inner, dict) and isinstance(inner.get("steps"), list):
                plan_obj = inner
                break

    normalised_steps: List[Dict[str, Any]] = []
    for step in (plan_obj.get("steps") or []):
        if not isinstance(step, dict):
            continue
        normalised_steps.append({
            "name":         str(step.get("name") or step.get("id") or ""),
            "command":      str(step.get("command") or step.get("cmd") or ""),
            "dependencies": list(step.get("dependencies") or step.get("deps") or []),
            "outputs":      list(step.get("outputs") or []),
            "description":  str(step.get("description") or ""),
        })

    plan = {
        "workflow_type": str(plan_obj.get("workflow_type")
                              or plan_obj.get("pipeline_type")
                              or "custom"),
        "steps": normalised_steps,
    }

    # Best-effort credit / token accounting from the trajectory metadata.
    # The SDK doesn't expose a stable schema for this, so we look in a few
    # known places and degrade gracefully.
    credits_used = 0.0
    cost_usd = 0.0
    in_tok = 0
    out_tok = 0
    turns = 0
    try:
        info = frame.get("state", {}).get("info", {}) if isinstance(frame, dict) else {}
        credits_used = float(info.get("credits_consumed", 0.0) or 0.0)
        cost_usd = float(info.get("cost_usd", 0.0) or 0.0)
        in_tok = int(info.get("input_tokens", 0) or 0)
        out_tok = int(info.get("output_tokens", 0) or 0)
        turns = int(info.get("num_steps", 0) or 0)
    except Exception:
        pass
    if credits_used:
        _bump_budget(credits_used)

    err_msg: Optional[str] = None
    if not normalised_steps:
        err_msg = "Edison Analysis returned no parseable workflow plan"

    return _envelope(
        plan,
        prompt_tokens=in_tok,
        completion_tokens=out_tok,
        llm_calls=turns,
        cost_usd=cost_usd,
        wall_seconds=wall,
        error=err_msg,
    )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Edison Analysis competitor shim")
    ap.add_argument("--prompt", required=True, help="Natural-language task")
    ap.add_argument("--files", default="[]",
                    help="JSON-encoded list of 'name: description' strings")
    args = ap.parse_args(argv)

    api_key = os.environ.get("EDISON_API_KEY")
    if not api_key:
        print(json.dumps(_envelope(
            _empty_plan(),
            error=("EDISON_API_KEY is not set. Sign up at "
                   "https://platform.edisonscientific.com (academic .edu "
                   "accounts get a free credit allocation), generate a key, "
                   "and ``export EDISON_API_KEY=...``."),
        )))
        return 0

    cap = _budget_cap()
    if cap is not None:
        used = float(_read_budget_state().get("credits_used", 0.0))
        if used >= cap:
            print(json.dumps(_envelope(
                _empty_plan(),
                error=(f"EDISON_BUDGET_CREDITS={cap} exhausted "
                       f"(used={used:.2f}). Reset the budget file at "
                       f"{_budget_file_path()} or raise the cap."),
            )))
            return 0

    try:
        files = json.loads(args.files) if args.files else []
        if not isinstance(files, list):
            files = []
    except json.JSONDecodeError:
        files = []

    try:
        max_steps = int(os.environ.get("EDISON_MAX_STEPS", "5"))
    except ValueError:
        max_steps = 5
    language = os.environ.get("EDISON_LANGUAGE", "PYTHON").upper()
    try:
        poll_seconds = float(os.environ.get("EDISON_POLL_SECONDS", "15"))
    except ValueError:
        poll_seconds = 15.0
    try:
        overall_timeout = float(os.environ.get("EDISON_TIMEOUT", "1800"))
    except ValueError:
        overall_timeout = 1800.0

    envelope = asyncio.run(_run_edison(
        args.prompt, files,
        api_key=api_key,
        max_steps=max_steps,
        language=language,
        poll_seconds=poll_seconds,
        overall_timeout=overall_timeout,
    ))
    print(json.dumps(envelope))
    return 0


if __name__ == "__main__":
    sys.exit(main())
