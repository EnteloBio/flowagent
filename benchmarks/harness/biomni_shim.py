#!/usr/bin/env python3
"""Subprocess shim that drives Biomni (Stanford Biomni agent) from the
head-to-head benchmark harness.

Biomni exposes a LangGraph ReAct agent (:class:`biomni.agent.react.react`).
This shim runs it in ``plan=True`` mode with a bounded LangGraph recursion
limit, maps tool calls (and fallback narrative) onto FlowAgent's plan schema,
and prints the same JSON envelope as ``biomaster_shim`` / ``autoba_shim``.

Usage::

    python biomni_shim.py --prompt "Run RNA-seq with kallisto"

Options::

    --prompt <str>          Required.
    --files <json-array>    Optional list of "path: description" strings.
    --biomni-dir <path>     Biomni repo root (contains ``biomni/``). Default
                            ``$BIOMNI_DIR``.
    --model <str>           LLM id passed to Biomni (default: harness model or
                            Biomni default from env).
    --recursion-limit <n>   LangGraph ``recursion_limit`` (default: env
                            ``BIOMNI_RECURSION_LIMIT`` or 15).

Environment (see upstream Biomni README): ``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``,
``LLM_SOURCE``, ``BIOMNI_PATH``, ``BIOMNI_USE_TOOL_RETRIEVER``, etc.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Reuse dotenv like other shims
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from harness.runner import _load_dotenv_once  # type: ignore

    _load_dotenv_once()
except Exception:
    pass

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from harness.biomaster_shim import _classify_workflow_type  # type: ignore
except Exception:

    def _classify_workflow_type(plan: Dict[str, Any]) -> str:  # type: ignore
        return "custom"


def _die(msg: str, *, error_type: str = "shim-error", exit_code: int = 2) -> None:
    print(json.dumps({
        "plan": {"workflow_type": "custom", "steps": []},
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "llm_calls": 0,
        "cost_usd": 0.0,
        "wall_seconds": 0.0,
        "error": f"{error_type}: {msg}",
    }))
    sys.exit(exit_code)


def _install_biomni_api_schema_patches() -> None:
    """Patch :func:`biomni.utils.api_schema_to_langchain_tool` before importing
    :class:`biomni.agent.react.react`.

    Upstream ``read_module2api`` can list tools that are not actually defined
    on the module (e.g. ``create_registration_visualization`` removed or
    renamed). The default code path does ``getattr(module, name)`` and
    crashes. We register a no-op :class:`~langchain_core.tools.StructuredTool`
    so the agent can still start; the model rarely needs these in planning
    benchmarks.
    """
    from langchain_core.tools import StructuredTool

    import biomni.utils as bu

    _orig = bu.api_schema_to_langchain_tool

    def _safe(api_schema, mode="generated_tool", module_name=None):
        try:
            return _orig(api_schema, mode=mode, module_name=module_name)
        except (AttributeError, TypeError) as e:
            tool_name = str(api_schema.get("name", "unknown_tool"))
            print(
                f"[biomni_shim] placeholder for missing API {tool_name!r} ({e})",
                file=sys.stderr,
            )
            safe_name = re.sub(r"[^0-9a-zA-Z_]+", "_", tool_name)[:64] or "broken_tool"

            def _stub(**kwargs: Any) -> str:
                return (
                    f"Tool {tool_name} is not available in this Biomni build "
                    f"(schema/module mismatch)."
                )

            return StructuredTool.from_function(
                func=_stub,
                name=safe_name,
                description=(
                    f"Placeholder: Biomni API {tool_name!r} is missing from the "
                    "installed package (benchmark shim)."
                ),
            )

    bu.api_schema_to_langchain_tool = _safe  # type: ignore[assignment]


def _biomni_rebuild_app_with_retrieved_tools(agent: Any, user_prompt: str) -> None:
    """Subset of :meth:`biomni.agent.react.react.go` — prompt-based tool retrieval.

    ``configure()`` binds **all** Biomni tools (~200+) to the LLM; OpenAI Chat
    Completions reject >128 tools. Upstream fixes this inside ``go()`` by
    rebuilding ``agent.app`` **after** configure; our shim streams ``app``
    directly, so we must run the same rebuild here.
    """
    if not getattr(agent, "use_tool_retriever", False):
        return
    if not hasattr(agent, "retriever"):
        return

    data_lake_path = agent.path + "/data_lake"
    data_lake_content = glob.glob(data_lake_path + "/*")
    data_lake_items = [x.split("/")[-1] for x in data_lake_content]
    data_lake_descriptions = []
    for item in data_lake_items:
        description = agent.data_lake_dict.get(item, f"Data lake item: {item}")
        data_lake_descriptions.append({"name": item, "description": description})

    library_descriptions = []
    for lib_name, lib_desc in agent.library_content_dict.items():
        library_descriptions.append({"name": lib_name, "description": lib_desc})

    all_tools = agent.tool_registry.tools if hasattr(agent, "tool_registry") else []
    resources = {
        "tools": all_tools,
        "data_lake": data_lake_descriptions,
        "libraries": library_descriptions,
    }
    selected = agent.retriever.prompt_based_retrieval(
        user_prompt, resources, llm=agent.llm
    )
    tool_names = [
        t["name"] if isinstance(t, dict) else getattr(t, "name", str(t))
        for t in selected["tools"]
    ]
    retrieved: List[Any] = []
    for tool_name in tool_names:
        matching = [t for t in agent.tools if getattr(t, "name", None) == tool_name]
        if matching:
            retrieved.append(matching[0])
    if not any(getattr(t, "name", None) == "run_python_repl" for t in retrieved):
        retrieved.extend([t for t in agent.tools if getattr(t, "name", None) == "run_python_repl"])

    max_tools = int(os.environ.get("BIOMNI_MAX_TOOLS_PER_REQUEST", "128"))
    if len(retrieved) > max_tools:
        print(
            f"[biomni_shim] retrieved {len(retrieved)} tools; truncating to {max_tools}",
            file=sys.stderr,
        )
        repl = [t for t in retrieved if getattr(t, "name", None) == "run_python_repl"]
        others = [t for t in retrieved if getattr(t, "name", None) != "run_python_repl"]
        retrieved = others[: max_tools - (1 if repl else 0)] + (repl[:1] if repl else [])
        retrieved = retrieved[:max_tools]

    agent.app = agent._create_custom_react_agent(agent.llm, retrieved, agent.prompt)


def _biomni_force_tool_cap(agent: Any) -> None:
    """Last resort when retrieval is off: truncate full tool list to provider cap."""
    max_tools = int(os.environ.get("BIOMNI_MAX_TOOLS_PER_REQUEST", "128"))
    if len(agent.tools) <= max_tools:
        return
    tools = list(agent.tools)
    repl = [t for t in tools if getattr(t, "name", None) == "run_python_repl"]
    others = [t for t in tools if getattr(t, "name", None) != "run_python_repl"]
    capped = others[: max_tools - (1 if repl else 0)] + (repl[:1] if repl else [])
    capped = capped[:max_tools]
    print(
        f"[biomni_shim] BIOMNI_USE_TOOL_RETRIEVER=false but {len(agent.tools)} tools "
        f"exceed provider limit; truncating to {len(capped)}",
        file=sys.stderr,
    )
    agent.app = agent._create_custom_react_agent(agent.llm, capped, agent.prompt)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--files", default="[]",
                  help='JSON array of "path: description" strings')
    ap.add_argument("--biomni-dir",
                    default=os.environ.get("BIOMNI_DIR", ""))
    ap.add_argument("--model",
                    default=os.environ.get("OPENAI_MODEL",
                                          os.environ.get("BIOMNI_LLM", "")))
    ap.add_argument(
        "--recursion-limit",
        type=int,
        default=int(os.environ.get("BIOMNI_RECURSION_LIMIT", "15")),
        help="LangGraph recursion_limit (default 15 or $BIOMNI_RECURSION_LIMIT)",
    )
    ap.add_argument("--keep-output", action="store_true",
                    help="Keep scratch data dir on exit")
    return ap.parse_args()


def _compact_json(obj: Any, limit: int = 800) -> str:
    try:
        s = json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:
        s = str(obj)
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _messages_to_steps(messages: Sequence[Any]) -> List[Dict[str, Any]]:
    """Build FlowAgent-shaped steps from LangChain message history."""
    steps: List[Dict[str, Any]] = []
    prev_name: Optional[str] = None

    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            continue
        for tc in tool_calls:
            if isinstance(tc, dict):
                name = tc.get("name") or "tool"
                args = tc.get("args", {})
            else:
                name = getattr(tc, "name", None) or "tool"
                args = getattr(tc, "args", {})
            cmd = f"{name}  # {_compact_json(args)}"
            step_name = f"{name}_{len(steps) + 1}"
            steps.append({
                "name": step_name,
                "command": cmd,
                "dependencies": [prev_name] if prev_name else [],
                "outputs": [],
                "description": f"Biomni tool call: {name}",
            })
            prev_name = step_name

    return steps


def _fallback_steps_from_text(text: str) -> List[Dict[str, Any]]:
    """If there were no tool calls, derive coarse steps from assistant text."""
    text = (text or "").strip()
    if not text:
        return []

    # Split on numbered / bullet lines that look like steps
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    step_like = []
    for ln in lines:
        if re.match(r"^(\d+[\).\]]|[\*\-•])\s+", ln) or re.match(
            r"^(step\s*\d+|^\d+\.)", ln, re.I
        ):
            step_like.append(re.sub(r"^(\d+[\).\]]|[\*\-•])\s+", "", ln))

    bodies = step_like if len(step_like) >= 2 else [text[:6000]]

    steps: List[Dict[str, Any]] = []
    prev: Optional[str] = None
    for i, body in enumerate(bodies):
        name = f"narrative_{i + 1}"
        steps.append({
            "name": name,
            "command": body[:1200],
            "dependencies": [prev] if prev else [],
            "outputs": [],
            "description": body[:8000],
        })
        prev = name
    return steps


def _usage_from_callback(ucb: Any) -> tuple[int, int, int]:
    """Best-effort token totals from LangChain UsageMetadataCallbackHandler."""
    pt = ct = calls = 0
    try:
        md = getattr(ucb, "usage_metadata", None)
        if md is None:
            ag = getattr(ucb, "metadata", None)
            if isinstance(ag, dict):
                md = ag
        if isinstance(md, dict):
            for _model_id, u in md.items():
                if not isinstance(u, dict):
                    continue
                pt += int(u.get("input_tokens", 0) or 0)
                ct += int(u.get("output_tokens", 0) or 0)
                calls += 1
    except Exception:
        pass
    return pt, ct, calls


def main() -> None:
    args = _parse_args()
    t0 = time.perf_counter()

    if not args.biomni_dir:
        _die("Set --biomni-dir or $BIOMNI_DIR to the Biomni repo root",
             error_type="config")
    biomni_dir = Path(args.biomni_dir).expanduser().resolve()
    if not (biomni_dir / "biomni" / "__init__.py").exists():
        _die(f"{biomni_dir} does not look like a Biomni clone "
             "(missing biomni/__init__.py)", error_type="config")

    try:
        files = json.loads(args.files)
        if not isinstance(files, list):
            raise ValueError("not a list")
        files = [str(x) for x in files]
    except Exception as e:
        _die(f"--files must be a JSON array: {e}", error_type="config")

    run_id = uuid.uuid4().hex[:10]
    scratch_dir = Path(tempfile.mkdtemp(prefix=f"biomni_{run_id}_"))
    data_dir = scratch_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    prev_cwd = os.getcwd()
    prompt_tokens = completion_tokens = llm_calls = 0
    cost_usd = 0.0
    error: Optional[str] = None
    plan: Dict[str, Any] = {"workflow_type": "custom", "steps": []}

    try:
        os.environ["BIOMNI_PATH"] = str(data_dir)
        os.environ["BIOMNI_DATA_PATH"] = str(data_dir)
        # OpenAI Chat Completions allow at most 128 tools per request; Biomni's
        # full registry is ~200+. Default to prompt-based retrieval so the
        # model sees a subset. Set BIOMNI_USE_TOOL_RETRIEVER=false only for
        # providers without this cap (and expect failures on OpenAI).
        if "BIOMNI_USE_TOOL_RETRIEVER" not in os.environ:
            os.environ["BIOMNI_USE_TOOL_RETRIEVER"] = "true"

        os.chdir(scratch_dir)
        sys.path.insert(0, str(biomni_dir))

        extra = ""
        if files:
            extra = "\n\nInput files:\n" + "\n".join(files)
        full_prompt = args.prompt + extra

        with contextlib.redirect_stdout(sys.stderr):
            try:
                _install_biomni_api_schema_patches()
                from biomni.agent.react import react  # type: ignore
            except Exception as imp_exc:
                raise RuntimeError(
                    f"Cannot import Biomni (install the clone's deps): {imp_exc}"
                ) from imp_exc

            use_tr = os.environ.get("BIOMNI_USE_TOOL_RETRIEVER", "").lower() == "true"
            model_id = (args.model or "").strip()
            if not model_id:
                _die(
                    "No model: pass --model or set OPENAI_MODEL / BIOMNI_LLM",
                    error_type="config",
                )

            max_req = int(os.environ.get("BIOMNI_MAX_TOOLS_PER_REQUEST", "128"))

            agent = react(
                path=str(data_dir),
                llm=model_id,
                use_tool_retriever=use_tr,
            )
            if len(agent.tools) > max_req and not use_tr:
                print(
                    f"[biomni_shim] More than {max_req} tools loaded; retrying with "
                    "use_tool_retriever=True (OpenAI tool-array limit).",
                    file=sys.stderr,
                )
                agent = react(
                    path=str(data_dir),
                    llm=model_id,
                    use_tool_retriever=True,
                )
            agent.configure(
                plan=True,
                reflect=False,
                data_lake=False,
                react_code_search=False,
                library_access=False,
            )
            # ``configure()`` binds *all* tools; Biomni's ``go()`` rebuilds with
            # retrieved tools — we stream ``app`` directly, so replicate that.
            if getattr(agent, "use_tool_retriever", False):
                _biomni_rebuild_app_with_retrieved_tools(agent, full_prompt)
            elif len(agent.tools) > max_req:
                _biomni_force_tool_cap(agent)

            ucb = None
            try:
                from langchain_core.callbacks import UsageMetadataCallbackHandler

                ucb = UsageMetadataCallbackHandler()
            except Exception:
                pass

            get_openai_callback = None
            try:
                from langchain_community.callbacks import get_openai_callback
            except Exception:
                try:
                    from langchain.callbacks import get_openai_callback  # type: ignore
                except Exception:
                    pass

            lim = max(2, int(args.recursion_limit))
            stream_cfg: Dict[str, Any] = {"recursion_limit": lim}
            if ucb is not None:
                stream_cfg["callbacks"] = [ucb]

            inputs = {"messages": [("user", full_prompt)]}
            last_messages: List[Any] = []
            final_text = ""

            def _run_stream(cfg: Dict[str, Any]) -> None:
                nonlocal last_messages
                for s in agent.app.stream(inputs, stream_mode="values",
                                           config=cfg):
                    last_messages = list(s.get("messages") or [])

            if get_openai_callback is not None:
                with get_openai_callback() as ocb:
                    _run_stream(stream_cfg)
                    prompt_tokens = int(getattr(ocb, "prompt_tokens", 0) or 0)
                    completion_tokens = int(
                        getattr(ocb, "completion_tokens", 0) or 0
                    )
                    llm_calls = int(getattr(ocb, "successful_requests", 0) or 0)
                    cost_usd = float(getattr(ocb, "total_cost", 0.0) or 0.0)
            else:
                _run_stream(stream_cfg)

            if (prompt_tokens == 0 and completion_tokens == 0) and ucb is not None:
                pt, ct, ncall = _usage_from_callback(ucb)
                if pt or ct:
                    prompt_tokens, completion_tokens = pt, ct
                    llm_calls = ncall or llm_calls
            if llm_calls == 0 and last_messages:
                llm_calls = max(
                    1,
                    len([
                        m for m in last_messages
                        if getattr(m, "type", None) in ("ai", "AIMessage")
                        or type(m).__name__ == "AIMessage"
                    ]),
                )

            if last_messages:
                last = last_messages[-1]
                final_text = str(getattr(last, "content", "") or "")

            steps = _messages_to_steps(last_messages)
            if not steps:
                steps = _fallback_steps_from_text(final_text)

            plan = {
                "workflow_type": _classify_workflow_type({"steps": steps}),
                "steps": steps,
            }

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1200:]}"
    finally:
        os.chdir(prev_cwd)
        if not args.keep_output:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    print(json.dumps({
        "plan": plan,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "llm_calls": llm_calls,
        "cost_usd": cost_usd,
        "wall_seconds": time.perf_counter() - t0,
        "error": error,
    }))


if __name__ == "__main__":
    main()
