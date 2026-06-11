"""Tests for provider HTTP client cleanup."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from flowagent.core.providers.provider_cleanup import aclose_client, aclose_provider


def test_aclose_client_prefers_aclose() -> None:
    client = MagicMock()
    client.aclose = AsyncMock()
    client.close = AsyncMock()
    asyncio.run(aclose_client(client))
    client.aclose.assert_awaited_once()
    client.close.assert_not_called()


def test_aclose_provider_unwraps_token_tracker() -> None:
    inner = MagicMock()
    inner.client = MagicMock(spec=["close"])
    inner.client.close = AsyncMock()
    wrapper = MagicMock()
    wrapper._inner = inner
    asyncio.run(aclose_provider(wrapper))
    inner.client.close.assert_awaited_once()


def test_aclose_client_sync_close() -> None:
    client = MagicMock(spec=["close"])
    client.close = MagicMock(return_value=None)
    asyncio.run(aclose_client(client))
    client.close.assert_called_once()
