"""MCP (Model Context Protocol) server exposing FlowAgent tools.

Supports two transports:
  - HTTP/SSE  — ``flowagent mcp serve --port 8765``  (default)
  - stdio     — ``flowagent mcp serve --stdio``       (Cursor / VS Code)

The MCP JSON-RPC protocol is simple enough to implement without a heavy
framework: we speak JSON-RPC 2.0 over either stdio or HTTP.

Tool catalogue
--------------
All tools from AGENT_TOOLS + WORKFLOW_TOOLS in tool_definitions.py are
exposed.  Each call is dispatched to the matching handler in
``core/agent_loop.py``'s ``_execute_tool`` function so there is no
duplication of implementation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── MCP wire types ────────────────────────────────────────────

_SERVER_INFO = {
    "name": "flowagent",
    "version": "0.2.0",
}

_PROTOCOL_VERSION = "2024-11-05"


def _make_result(id_: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def _make_error(id_: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


# ── Tool registry ─────────────────────────────────────────────

def _build_tool_list() -> List[Dict[str, Any]]:
    """Convert FlowAgent tool schemas to MCP tool format."""
    from .core.tool_definitions import AGENT_TOOLS, WORKFLOW_TOOLS
    mcp_tools = []
    for tool in AGENT_TOOLS + WORKFLOW_TOOLS:
        fn = tool["function"]
        mcp_tools.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "inputSchema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return mcp_tools


async def _dispatch_tool(name: str, arguments: Dict[str, Any]) -> Any:
    """Call a FlowAgent tool by name and return the result."""
    from .core.agent_loop import _execute_tool  # type: ignore[attr-defined]
    try:
        result = await _execute_tool(name, arguments)
        if isinstance(result, (dict, list)):
            return json.dumps(result, indent=2)
        return str(result)
    except Exception as exc:
        logger.warning("Tool %s raised: %s", name, exc)
        return f"Error: {exc}"


# ── Request dispatcher ────────────────────────────────────────

async def handle_request(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Handle one JSON-RPC message and return a response (or None for notifications)."""
    method = msg.get("method", "")
    id_ = msg.get("id")
    params = msg.get("params", {})

    # Notifications have no id — no response needed
    if id_ is None and method not in ("initialize",):
        return None

    if method == "initialize":
        return _make_result(id_, {
            "protocolVersion": _PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": _SERVER_INFO,
        })

    if method == "notifications/initialized":
        return None

    if method == "tools/list":
        return _make_result(id_, {"tools": _build_tool_list()})

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        if not tool_name:
            return _make_error(id_, -32602, "Missing tool name")
        content_text = await _dispatch_tool(tool_name, arguments)
        return _make_result(id_, {
            "content": [{"type": "text", "text": content_text}],
        })

    if method == "ping":
        return _make_result(id_, {})

    return _make_error(id_, -32601, f"Method not found: {method}")


# ── stdio transport ───────────────────────────────────────────

async def _run_stdio() -> None:
    """Read newline-delimited JSON-RPC from stdin, write to stdout."""
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    writer_transport, writer_protocol = await loop.connect_write_pipe(
        asyncio.BaseProtocol, sys.stdout.buffer,
    )
    # Minimal write helper
    def _write(data: bytes) -> None:
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()

    logger.info("MCP stdio server ready")
    while True:
        try:
            line = await reader.readline()
            if not line:
                break
            msg = json.loads(line.decode())
            response = await handle_request(msg)
            if response is not None:
                _write(json.dumps(response).encode() + b"\n")
        except json.JSONDecodeError as exc:
            err = _make_error(None, -32700, f"Parse error: {exc}")
            _write(json.dumps(err).encode() + b"\n")
        except Exception as exc:
            logger.error("stdio handler error: %s", exc)
            break


# ── HTTP/SSE transport ────────────────────────────────────────

def _make_http_app(host: str, port: int):
    """Return a FastAPI ASGI application for the MCP HTTP transport."""
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse as _HTMLResponse
    except ImportError:
        raise ImportError(
            "fastapi required for HTTP MCP transport: "
            "pip install 'flowagent[web]'"
        )

    app = FastAPI(title="FlowAgent MCP Server")

    @app.post("/mcp")
    async def mcp_endpoint(request: Request):
        body = await request.json()
        response = await handle_request(body)
        if response is None:
            return JSONResponse(content={}, status_code=204)
        return JSONResponse(content=response)

    @app.get("/mcp/tools")
    async def list_tools():
        return JSONResponse(content={"tools": _build_tool_list()})

    @app.get("/health")
    async def health():
        return {"status": "ok", "server": _SERVER_INFO}

    @app.get("/", response_class=_HTMLResponse)
    async def root():
        tools = _build_tool_list()
        tool_rows = "\n".join(
            f"<tr><td><code>{t['name']}</code></td>"
            f"<td>{t.get('description','')[:120]}</td></tr>"
            for t in tools
        )
        base_url = f"http://{host}:{port}"
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>FlowAgent MCP Server</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            max-width: 860px; margin: 40px auto; padding: 0 20px; color: #222; }}
    h1   {{ font-size: 1.6rem; margin-bottom: 4px; }}
    .sub {{ color: #666; margin-bottom: 24px; }}
    pre  {{ background: #f4f4f4; padding: 14px 16px; border-radius: 6px;
            font-size: 0.85rem; overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.88rem; }}
    th   {{ text-align: left; border-bottom: 2px solid #ddd; padding: 6px 8px; }}
    td   {{ border-bottom: 1px solid #eee; padding: 6px 8px; vertical-align: top; }}
    td code {{ background: #f0f0f0; padding: 1px 4px; border-radius: 3px; }}
    .pill {{ display:inline-block; background:#0a7; color:#fff;
             border-radius:4px; padding:2px 8px; font-size:0.8rem; }}
  </style>
</head>
<body>
  <h1>FlowAgent MCP Server <span class="pill">running</span></h1>
  <p class="sub">JSON-RPC 2.0 · {len(tools)} tools · protocol {_PROTOCOL_VERSION}</p>

  <h2>Endpoints</h2>
  <table>
    <tr><th>Method</th><th>URL</th><th>Purpose</th></tr>
    <tr><td>POST</td><td><a href="{base_url}/mcp">/mcp</a></td><td>JSON-RPC endpoint (MCP calls)</td></tr>
    <tr><td>GET</td><td><a href="{base_url}/mcp/tools">/mcp/tools</a></td><td>List available tools (JSON)</td></tr>
    <tr><td>GET</td><td><a href="{base_url}/health">/health</a></td><td>Health check</td></tr>
  </table>

  <h2>Configure in Cursor</h2>
  <p>Add to <code>.cursor/mcp.json</code> in your project root:</p>
  <pre>{{
  "mcpServers": {{
    "flowagent": {{
      "url": "{base_url}/mcp"
    }}
  }}
}}</pre>

  <h2>Available tools ({len(tools)})</h2>
  <table>
    <tr><th>Tool</th><th>Description</th></tr>
    {tool_rows}
  </table>
</body>
</html>"""

    return app


# ── Entry point ───────────────────────────────────────────────

def run_mcp_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    stdio: bool = False,
) -> None:
    """Start the MCP server (called by ``flowagent mcp serve``)."""
    if stdio:
        logger.info("Starting FlowAgent MCP stdio server")
        asyncio.run(_run_stdio())
    else:
        try:
            import uvicorn
        except ImportError:
            print(
                "uvicorn required for HTTP MCP server.\n"
                "Install with: pip install 'flowagent[web]'\n"
                "Or use --stdio for stdio transport.",
                file=sys.stderr,
            )
            sys.exit(1)

        app = _make_http_app(host, port)
        print(f"[mcp] FlowAgent MCP server → http://{host}:{port}/mcp")
        print(f"[mcp] Tool list            → http://{host}:{port}/mcp/tools")
        print(
            "\nTo configure in Cursor, add to .cursor/mcp.json:\n"
            '{\n'
            '  "mcpServers": {\n'
            '    "flowagent": {\n'
            f'      "url": "http://{host}:{port}/mcp"\n'
            '    }\n'
            '  }\n'
            '}'
        )
        uvicorn.run(app, host=host, port=port, log_level="info")
