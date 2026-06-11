"""Best-effort async shutdown for provider HTTP clients."""

from __future__ import annotations

import asyncio
from typing import Any


async def aclose_client(client: Any) -> None:
    """Close an SDK client if it exposes ``aclose`` or async ``close``."""
    if client is None:
        return
    aclose = getattr(client, "aclose", None)
    if callable(aclose):
        await aclose()
        return
    close = getattr(client, "close", None)
    if callable(close):
        result = close()
        if asyncio.iscoroutine(result):
            await result


async def aclose_provider(provider: Any) -> None:
    """Close the HTTP client behind a provider (unwraps ``_TokenTracker``)."""
    if provider is None:
        return
    inner = getattr(provider, "_inner", provider)
    client = getattr(inner, "client", None)
    await aclose_client(client)
