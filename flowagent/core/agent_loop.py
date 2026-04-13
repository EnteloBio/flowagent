"""Interactive agentic loop with tool calling.

Instead of generating an entire workflow plan in one shot and then blindly
executing it, this module lets the LLM reason, call tools (inspect files,
check dependencies, run commands), observe results, and self-correct.
"""

import asyncio
import json
import platform
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..utils.logging import get_logger
from .providers import LLMProvider
from .tool_definitions import AGENT_TOOLS

logger = get_logger(__name__)

MAX_ITERATIONS = 25

# Tools that require user confirmation before execution
DANGEROUS_TOOLS = {"execute_command", "install_dependency", "write_file", "run_workflow"}


async def _default_confirm(tool_name: str, arguments: Dict[str, Any]) -> bool:
    """CLI confirmation gate -- prompts on stdin."""
    summary = json.dumps(arguments, indent=2)[:300]
    print(f"\n[FlowAgent] Tool '{tool_name}' wants to run:\n{summary}")
    try:
        answer = input("Allow? [y/N] ").strip().lower()
        return answer in ("y", "yes")
    except EOFError:
        return True  # non-interactive environments auto-approve


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


async def _tool_plan_workflow(prompt: str) -> str:
    """Generate a workflow plan from natural-language prompt."""
    try:
        from .llm import LLMInterface
        llm = LLMInterface()
        plan = await llm.generate_workflow_plan(prompt)
        return json.dumps(plan, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _tool_run_workflow(prompt: str, checkpoint_dir: str = None, resume: bool = False) -> str:
    """Execute a workflow from a prompt."""
    try:
        from ..workflow import run_workflow as _run
        await _run(prompt, checkpoint_dir, resume)
        return json.dumps({"success": True, "message": "Workflow completed"})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


async def _tool_export_pipeline(prompt: str, format: str = "nextflow", output_dir: str = "flowagent_pipeline_output") -> str:
    """Export a workflow plan as a Nextflow or Snakemake pipeline."""
    try:
        from .llm import LLMInterface
        from .pipeline_generator import NextflowGenerator, SnakemakeGenerator

        llm = LLMInterface()
        plan = await llm.generate_workflow_plan(prompt)
        out = Path(output_dir)
        gen = NextflowGenerator() if format == "nextflow" else SnakemakeGenerator()
        code = gen.generate(plan, output_dir=out)
        filename = gen.default_filename()
        return json.dumps({"success": True, "file": str(out / filename), "format": format})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


async def _tool_analyze_results(directory: str) -> str:
    """Analyze workflow output in a directory."""
    try:
        from ..workflow import analyze_workflow
        result = await analyze_workflow(directory)
        summary = result.get("report", result.get("message", "Analysis complete"))
        if isinstance(summary, str) and len(summary) > 3000:
            summary = summary[:3000] + "...(truncated)"
        return json.dumps({"status": result.get("status", "unknown"), "summary": str(summary)})
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _tool_load_preset(preset_id: str) -> str:
    """Load a workflow preset by ID."""
    try:
        from ..presets.catalog import get_preset, list_presets
        plan = get_preset(preset_id)
        if plan is None:
            available = list_presets()
            return json.dumps({"error": f"Preset '{preset_id}' not found", "available": available})
        return json.dumps(plan, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _tool_check_workflow_status(checkpoint_dir: str) -> str:
    """Read checkpoint.json and return completion status."""
    cp = Path(checkpoint_dir) / "checkpoint.json"
    if not cp.exists():
        return json.dumps({"error": f"No checkpoint found at {cp}"})
    try:
        data = json.loads(cp.read_text())
        return json.dumps({
            "completed_steps": data.get("completed_steps", []),
            "total_steps": len(data.get("workflow_plan", {}).get("steps", [])),
            "timestamp": data.get("timestamp"),
            "output_dir": data.get("output_dir"),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


TOOL_DISPATCH: Dict[str, Callable] = {
    "list_files": _tool_list_files,
    "check_tool": _tool_check_tool,
    "install_dependency": _tool_install_dependency,
    "execute_command": _tool_execute_command,
    "read_file": _tool_read_file,
    "write_file": _tool_write_file,
    "plan_workflow": _tool_plan_workflow,
    "run_workflow": _tool_run_workflow,
    "export_pipeline": _tool_export_pipeline,
    "analyze_results": _tool_analyze_results,
    "load_preset": _tool_load_preset,
    "check_workflow_status": _tool_check_workflow_status,
}


# ── Agent loop ────────────────────────────────────────────────

async def run_agent_loop(
    provider: LLMProvider,
    user_prompt: str,
    *,
    system_prompt: Optional[str] = None,
    on_token: Optional[Callable[[str], None]] = None,
    on_tool_call: Optional[Callable[[str, Dict[str, Any], str], None]] = None,
    confirm_fn: Optional[Callable] = None,
    require_confirmation: bool = False,
    max_iterations: int = MAX_ITERATIONS,
) -> Dict[str, Any]:
    """Run an interactive agent loop with tool calling.

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
    on_tool_call : callable, optional
        Callback ``(name, arguments, result)`` fired after each tool call.
    confirm_fn : callable, optional
        Async callable ``(tool_name, arguments) -> bool`` used to gate
        dangerous tool calls. If None and require_confirmation is True,
        falls back to CLI stdin prompt.
    require_confirmation : bool
        When True, dangerous tools (execute_command, install_dependency,
        write_file, run_workflow) will be gated by confirm_fn.
    max_iterations : int
        Safety cap on the number of tool-call rounds.

    Returns
    -------
    dict with ``response`` (final text), ``tool_calls`` (history), ``iterations``.
    """
    if system_prompt is None:
        os_hint = ""
        sys_name = platform.system()
        if sys_name == "Darwin":
            os_hint = (
                " The host OS is macOS — use BSD-compatible commands "
                "(e.g. 'ls', 'find . -name …' without GNU extensions like -printf)."
            )
        elif sys_name == "Linux":
            os_hint = " The host OS is Linux."

        system_prompt = (
            "You are FlowAgent, an expert bioinformatics workflow assistant. "
            "You can inspect files, check tool availability, install dependencies, "
            "run commands, and read/write files to build and execute analysis pipelines. "
            "Use the provided tools to gather information before generating a workflow. "
            "When you have enough context, describe the plan and execute it step by step."
            f"{os_hint}"
        )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    all_tool_calls: List[Dict[str, Any]] = []

    for iteration in range(max_iterations):
        resp = await provider.chat_with_tools(messages, AGENT_TOOLS)

        if not resp.tool_calls:
            # Stream final response token-by-token if callback is provided
            content = resp.content or ""
            if on_token and content:
                try:
                    async for token in provider.stream(messages):
                        on_token(token)
                    content = resp.content  # keep original for return
                except Exception:
                    # Streaming not supported or failed; push content in chunks
                    for i in range(0, len(content), 40):
                        on_token(content[i:i+40])

            logger.info("Agent loop finished after %d iterations", iteration + 1)
            return {
                "response": content,
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

            # Confirmation gate for dangerous tools
            if require_confirmation and fn_name in DANGEROUS_TOOLS:
                gate = confirm_fn or _default_confirm
                try:
                    approved = await gate(fn_name, fn_args) if asyncio.iscoroutinefunction(gate) else gate(fn_name, fn_args)
                except Exception:
                    approved = False
                if not approved:
                    result = json.dumps({"error": f"User denied execution of {fn_name}"})
                    all_tool_calls.append({"name": fn_name, "arguments": fn_args, "result": result})
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": result})
                    continue

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
            if on_tool_call:
                try:
                    on_tool_call(fn_name, fn_args, result)
                except Exception:
                    pass
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
