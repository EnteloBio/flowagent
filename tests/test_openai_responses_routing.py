"""Tests for OpenAI Pro-model Responses API routing."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flowagent.core.providers.openai_provider import OpenAIProvider


@pytest.mark.asyncio
async def test_gpt_5_5_pro_uses_responses_api():
    provider = OpenAIProvider(api_key="sk-test", model="gpt-5.5-pro")
    mock_response = MagicMock()
    mock_response.output_text = "B"
    mock_response.model = "gpt-5.5-pro"
    mock_response.usage = None

    with patch.object(
        provider, "_call_responses_with_retry", new_callable=AsyncMock,
    ) as mock_responses:
        mock_responses.return_value = mock_response
        resp = await provider.chat(
            [{"role": "user", "content": "Answer B"}],
            model="gpt-5.5-pro",
        )

    mock_responses.assert_awaited_once()
    kwargs = mock_responses.await_args.kwargs
    assert kwargs["model"] == "gpt-5.5-pro"
    assert kwargs["input"] == "Answer B"
    assert kwargs["reasoning"] == {"effort": "high"}
    assert resp.content == "B"
