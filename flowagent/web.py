"""FastAPI web interface for FlowAgent.

Replaces the previous Chainlit-based UI with a modern SSE-streaming
FastAPI backend serving a single-page chat frontend.
"""

import asyncio
import io
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from flowagent.config.settings import Settings
from flowagent.core.agent_loop import run_agent_loop
from flowagent.core.llm import LLMInterface
from flowagent.core.providers import create_provider
from flowagent.presets.catalog import list_presets, get_preset
from flowagent.workflow import analyze_workflow, run_workflow

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "web_static"

app = FastAPI(title="FlowAgent", version="0.2.0")

# ---------------------------------------------------------------------------
# In-memory state (single-user local tool)
# ---------------------------------------------------------------------------

_sessions: Dict[str, Dict[str, Any]] = {}
_chat_queues: Dict[str, asyncio.Queue] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# SSE log capture – replaces the old StringIOHandler/OutputMessage
# ---------------------------------------------------------------------------

class SSELogHandler(logging.Handler):
    """Captures log records and pushes them into an asyncio.Queue as SSE events."""

    def __init__(self, queue: asyncio.Queue):
        super().__init__()
        self._queue = queue
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            data = json.dumps({"timestamp": _now_iso(), "message": msg})
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._queue.put_nowait, ("log", data))
            else:
                self._queue.put_nowait(("log", data))
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------------
# Background task runners
# ---------------------------------------------------------------------------

async def _run_agent(queue: asyncio.Queue, prompt: str):
    """Run the agent loop, pushing tool_call and token events to the queue."""
    s = Settings()
    api_key = s.active_api_key or s.OPENAI_API_KEY
    provider = create_provider(
        s.LLM_PROVIDER, model=s.LLM_MODEL,
        api_key=api_key, base_url=s.LLM_BASE_URL,
    )

    queue.put_nowait(("phase", json.dumps({"label": "Agent is thinking..."})))

    def _on_tool_call(name, arguments, result):
        queue.put_nowait(("tool_call", json.dumps({
            "name": name,
            "arguments": arguments,
            "result": (result or "")[:500],
        })))

    def _on_token(token: str):
        queue.put_nowait(("token", json.dumps({"content": token})))

    streamed = {"sent": False}
    _orig_on_token = _on_token

    def _tracking_on_token(token: str):
        streamed["sent"] = True
        _orig_on_token(token)

    result = await run_agent_loop(
        provider, prompt,
        on_token=_tracking_on_token,
        on_tool_call=_on_tool_call,
    )

    # Fallback: if streaming didn't fire, push the full response in chunks
    if not streamed["sent"]:
        content = result.get("response", "") or ""
        for i in range(0, len(content), 40):
            queue.put_nowait(("token", json.dumps({"content": content[i:i+40]})))
            await asyncio.sleep(0.01)

    queue.put_nowait(("done", json.dumps({"iterations": result.get("iterations", 0)})))


async def _run_workflow_task(queue: asyncio.Queue, prompt: str,
                            checkpoint_dir: Optional[str] = None,
                            resume: bool = False):
    """Run a workflow, capturing logs via SSELogHandler."""
    handler = SSELogHandler(queue)
    handler.set_loop(asyncio.get_event_loop())
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)

    queue.put_nowait(("phase", json.dumps({"label": "Running workflow..."})))

    try:
        await run_workflow(prompt, checkpoint_dir, resume)
        queue.put_nowait(("token", json.dumps({"content": "Workflow completed successfully."})))
    except Exception as exc:
        queue.put_nowait(("error", json.dumps({"message": str(exc)})))
    finally:
        root.removeHandler(handler)
        queue.put_nowait(("done", json.dumps({})))


async def _run_pipeline_task(queue: asyncio.Queue, prompt: str,
                             pipeline_format: str = "nextflow",
                             context_answers: Optional[Dict[str, str]] = None):
    """Run the planning + pipeline generation flow, streaming events."""
    from flowagent.core.pipeline_planner import gather_pipeline_context
    from flowagent.core.pipeline_generator import NextflowGenerator, SnakemakeGenerator

    queue.put_nowait(("phase", json.dumps({"label": "Planning pipeline..."})))

    ctx = await gather_pipeline_context(
        prompt,
        interactive=False,
        answers=context_answers or {},
    )

    queue.put_nowait(("token", json.dumps({
        "content": (
            f"**Pipeline context resolved**\n"
            f"- Organism: {ctx.organism} ({ctx.genome_build})\n"
            f"- Reference source: {ctx.reference_source}\n"
            f"- Input files: {len(ctx.input_files)} found\n"
            f"- Reference: {'local' if ctx.reference_fasta else ('download' if ctx.reference_url else 'none')}\n\n"
        ),
    })))

    queue.put_nowait(("phase", json.dumps({"label": "Generating workflow plan..."})))
    llm = LLMInterface()
    plan = await llm.generate_workflow_plan(prompt, context=ctx)

    gen = NextflowGenerator() if pipeline_format == "nextflow" else SnakemakeGenerator()
    from pathlib import Path
    output_dir = Path(os.environ.get("USER_EXECUTION_DIR", ".")) / "flowagent_pipeline_output"
    code = gen.generate(plan, output_dir=output_dir)

    queue.put_nowait(("token", json.dumps({
        "content": (
            f"Generated `{gen.default_filename()}` in `{output_dir}/`\n\n"
            f"**Steps:** {len(plan.get('steps', []))}\n"
        ),
    })))
    for step in plan.get("steps", []):
        queue.put_nowait(("token", json.dumps({
            "content": f"- `{step.get('name')}`: {step.get('command', '')[:80]}\n",
        })))
        await asyncio.sleep(0.01)

    queue.put_nowait(("done", json.dumps({"pipeline_file": str(output_dir / gen.default_filename())})))


async def _run_analyse_task(queue: asyncio.Queue, analysis_dir: str,
                            save_report: bool = True):
    """Run analysis, capturing logs via SSELogHandler."""
    handler = SSELogHandler(queue)
    handler.set_loop(asyncio.get_event_loop())
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)

    queue.put_nowait(("phase", json.dumps({"label": "Analysing workflow results..."})))

    try:
        result = await analyze_workflow(analysis_dir, save_report)
        summary = result.get("analysis", result.get("message", "Analysis complete."))
        if isinstance(summary, dict):
            summary = json.dumps(summary, indent=2)
        queue.put_nowait(("token", json.dumps({"content": str(summary)})))
    except Exception as exc:
        queue.put_nowait(("error", json.dumps({"message": str(exc)})))
    finally:
        root.removeHandler(handler)
        queue.put_nowait(("done", json.dumps({})))


# ---------------------------------------------------------------------------
# Routes – static
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>FlowAgent</h1><p>Frontend not found.</p>", status_code=500)
    return FileResponse(html_path, media_type="text/html")


@app.get("/api/health")
async def health():
    return {"status": "ok", "cwd": os.getcwd(), "time": _now_iso()}


@app.get("/api/presets")
async def get_presets():
    return {"presets": list_presets()}


@app.get("/api/presets/{preset_id}")
async def get_preset_detail(preset_id: str):
    plan = get_preset(preset_id)
    if not plan:
        return JSONResponse({"error": f"Preset '{preset_id}' not found"}, status_code=404)
    return plan


# ---------------------------------------------------------------------------
# Routes – sessions
# ---------------------------------------------------------------------------

@app.get("/api/sessions")
async def list_sessions():
    sessions = sorted(_sessions.values(), key=lambda s: s["updated_at"], reverse=True)
    return {"sessions": sessions}


@app.post("/api/sessions")
async def create_session(request: Request):
    body = await request.json()
    sid = body.get("uuid") or str(uuid.uuid4())
    name = body.get("name", "New chat")
    now = _now_iso()
    _sessions[sid] = {"uuid": sid, "name": name, "created_at": now, "updated_at": now, "messages": []}
    return {"uuid": sid}


@app.get("/api/sessions/{session_id}/messages")
async def get_messages(session_id: str):
    session = _sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    return {"messages": session.get("messages", [])}


@app.put("/api/sessions/{session_id}/messages")
async def save_messages(session_id: str, request: Request):
    body = await request.json()
    session = _sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    session["messages"] = body.get("messages", [])
    session["updated_at"] = _now_iso()
    return {"ok": True}


@app.patch("/api/sessions/{session_id}")
async def update_session(session_id: str, request: Request):
    body = await request.json()
    session = _sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    if "name" in body:
        session["name"] = body["name"]
    session["updated_at"] = _now_iso()
    return {"ok": True}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Routes – chat + SSE streaming
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def submit_chat(request: Request):
    body = await request.json()
    chat_id = str(uuid.uuid4())
    prompt = body.get("prompt", "")
    mode = body.get("mode", "agent")

    queue: asyncio.Queue = asyncio.Queue()
    _chat_queues[chat_id] = queue

    if mode == "agent":
        asyncio.create_task(_agent_wrapper(chat_id, _run_agent(queue, prompt)))
    elif mode == "run":
        asyncio.create_task(_agent_wrapper(chat_id, _run_workflow_task(
            queue, prompt,
            checkpoint_dir=body.get("checkpoint_dir"),
            resume=body.get("resume", False),
        )))
    elif mode == "pipeline":
        asyncio.create_task(_agent_wrapper(chat_id, _run_pipeline_task(
            queue, prompt,
            pipeline_format=body.get("pipeline_format", "nextflow"),
            context_answers=body.get("context_answers"),
        )))
    elif mode == "analyse":
        asyncio.create_task(_agent_wrapper(chat_id, _run_analyse_task(
            queue, body.get("analysis_dir", prompt),
            save_report=body.get("save_report", True),
        )))
    else:
        queue.put_nowait(("error", json.dumps({"message": f"Unknown mode: {mode}"})))
        queue.put_nowait(("done", json.dumps({})))

    return {"id": chat_id, "status": "started"}


async def _agent_wrapper(chat_id: str, coro):
    """Wrap a task coroutine so exceptions surface as error events."""
    queue = _chat_queues.get(chat_id)
    try:
        await coro
    except Exception as exc:
        logger.exception("Task %s failed", chat_id)
        if queue:
            queue.put_nowait(("error", json.dumps({"message": str(exc)})))
            queue.put_nowait(("done", json.dumps({})))


@app.get("/api/chat/{chat_id}/stream")
async def stream_chat(chat_id: str):
    queue = _chat_queues.get(chat_id)
    if not queue:
        return JSONResponse({"error": "Chat not found"}, status_code=404)

    async def event_generator():
        try:
            while True:
                try:
                    event_type, data = await asyncio.wait_for(queue.get(), timeout=120)
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": "{}"}
                    continue

                yield {"event": event_type, "data": data}

                if event_type == "done":
                    break
        finally:
            _chat_queues.pop(chat_id, None)

    return EventSourceResponse(event_generator())
