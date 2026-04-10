"""Interactive agentic loop with tool calling.

Instead of generating an entire workflow plan in one shot and then blindly
executing it, this module lets the LLM reason, call tools (inspect files,
check dependencies, run commands), observe results, and self-correct.
"""

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..utils.logging import get_logger
from .providers import LLMProvider
from .tool_definitions import AGENT_TOOLS

logger = get_logger(__name__)

MAX_ITERATIONS = 25


# ── Tool implementations ──────────────────────────────────────

async def _tool_list_files(directory: str, pattern: str = "*") -> str:
    path = Path(directory)
    if not path.exists():
        return json.dumps({"error": f"Directory not found: {directory}"})
    matches = sorted(str(p) for p in path.glob(pattern))
    return json.dumps({"files": matches[:200], "total": len(matches)})


async def _tool_check_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        return json.dumps({"installed": False, "error": f"{name} not found in PATH"})
    try:
        proc = await asyncio.create_subprocess_exec(
            name, "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        version = (stdout or stderr).decode().strip().split("\n")[0]
    except Exception:
        version = "unknown"
    return json.dumps({"installed": True, "path": path, "version": version})


async def _tool_install_dependency(name: str, manager: str = "conda", channel: str = "bioconda") -> str:
    if manager in ("conda", "mamba"):
        cmd = [manager, "install", "-y", "-c", channel, name]
    else:
        cmd = ["pip", "install", name]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return json.dumps({
            "success": proc.returncode == 0,
            "stdout": stdout.decode()[-500:],
            "stderr": stderr.decode()[-500:],
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


async def _tool_execute_command(command: str, cwd: str = ".") -> str:
    try:
        proc = await asyncio.create_subprocess_exec(
            "bash", "-c", command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await proc.communicate()
        return json.dumps({
            "exit_code": proc.returncode,
            "stdout": stdout.decode()[-2000:],
            "stderr": stderr.decode()[-2000:],
        })
    except Exception as e:
        return json.dumps({"exit_code": -1, "error": str(e)})


async def _tool_read_file(path: str, max_lines: int = 200) -> str:
    p = Path(path)
    if not p.exists():
        return json.dumps({"error": f"File not found: {path}"})
    try:
        lines = p.read_text().splitlines()[:max_lines]
        return json.dumps({"lines": len(lines), "content": "\n".join(lines)})
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _tool_write_file(path: str, content: str) -> str:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return json.dumps({"success": True, "path": str(p)})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


TOOL_DISPATCH: Dict[str, Callable] = {
    "list_files": _tool_list_files,
    "check_tool": _tool_check_tool,
    "install_dependency": _tool_install_dependency,
    "execute_command": _tool_execute_command,
    "read_file": _tool_read_file,
    "write_file": _tool_write_file,
}


# ── Agent loop ────────────────────────────────────────────────

async def run_agent_loop(
    provider: LLMProvider,
    user_prompt: str,
    *,
    system_prompt: Optional[str] = None,
    on_token: Optional[Callable[[str], None]] = None,
    max_iterations: int = MAX_ITERATIONS,
) -> Dict[str, Any]:
    """Run an interactive agent loop with tool calling.

    The LLM receives the user prompt plus access to ``AGENT_TOOLS``.
    It can call tools, observe results, and iterate until it produces
    a final text response (no more tool calls).

    Parameters
    ----------
    provider : LLMProvider
        The configured LLM provider.
    user_prompt : str
        The user's natural-language request.
    system_prompt : str, optional
        Override the default system prompt.
    on_token : callable, optional
        Callback for streaming tokens to a UI.
    max_iterations : int
        Safety cap on the number of tool-call rounds.

    Returns
    -------
    dict with ``response`` (final text), ``tool_calls`` (history), ``iterations``.
    """
    if system_prompt is None:
        system_prompt = (
            "You are FlowAgent, an expert bioinformatics workflow assistant. "
            "You can inspect files, check tool availability, install dependencies, "
            "run commands, and read/write files to build and execute analysis pipelines. "
            "Use the provided tools to gather information before generating a workflow. "
            "When you have enough context, describe the plan and execute it step by step."
        )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    all_tool_calls: List[Dict[str, Any]] = []

    for iteration in range(max_iterations):
        resp = await provider.chat_with_tools(messages, AGENT_TOOLS)

        if not resp.tool_calls:
            logger.info("Agent loop finished after %d iterations", iteration + 1)
            return {
                "response": resp.content,
                "tool_calls": all_tool_calls,
                "iterations": iteration + 1,
            }

        # Append the assistant message (with tool calls) to conversation
        messages.append({
            "role": "assistant",
            "content": resp.content or "",
            "tool_calls": [
                {
                    "id": tc.get("id", f"call_{tc.get('name', 'unknown')}_{iteration}"),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", "unknown"),
                        "arguments": json.dumps(tc.get("arguments", {})),
                    },
                }
                for tc in resp.tool_calls
            ],
        })

        # Execute each tool call and append results
        for tc in resp.tool_calls:
            fn_name = tc.get("name", "unknown")
            fn_args = tc.get("arguments", {})
            tc_id = tc.get("id", f"call_{fn_name}_{iteration}")
            logger.info("Tool call: %s(%s)", fn_name, json.dumps(fn_args)[:200])

            handler = TOOL_DISPATCH.get(fn_name)
            if handler:
                try:
                    result = await handler(**fn_args)
                except TypeError as e:
                    result = json.dumps({"error": f"Invalid arguments for {fn_name}: {e}"})
                except Exception as e:
                    result = json.dumps({"error": f"Tool {fn_name} failed: {e}"})
            else:
                result = json.dumps({"error": f"Unknown tool: {fn_name}"})

            all_tool_calls.append({"name": fn_name, "arguments": fn_args, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": result,
            })

    logger.warning("Agent loop hit max iterations (%d)", max_iterations)
    return {
        "response": "Reached maximum iterations without completing.",
        "tool_calls": all_tool_calls,
        "iterations": max_iterations,
    }
