"""Anthropic provider (Claude 3.5 / 4 family)."""

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from .base import LLMProvider, ProviderResponse

logger = logging.getLogger(__name__)

TOOL_CHOICE_MAP = {
    "auto": {"type": "auto"},
    "required": {"type": "any"},
    "none": {"type": "auto"},
}


class AnthropicProvider(LLMProvider):
    """Wraps the ``anthropic`` async client."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_retries: int = 3,
        timeout: float = 120.0,
    ):
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError(
                "Install the anthropic package: pip install anthropic"
            )
        self.default_model = model
        self._max_retries = max_retries
        self.client = AsyncAnthropic(api_key=api_key, timeout=timeout)

    def _model(self, override: Optional[str]) -> str:
        return override or self.default_model

    @staticmethod
    def _accepts_temperature(model: str) -> bool:
        """Recent Opus thinking-style models (opus-4-5 / 4-6 / 4-7 etc.)
        removed the ``temperature`` parameter and the API returns 400
        if it's included. Detect those by prefix so we can omit it.
        """
        m = (model or "").lower()
        # Known-incompatible families. Keep conservative — only exclude
        # exact prefixes Anthropic has deprecated ``temperature`` on;
        # everything else keeps the usual behaviour.
        for prefix in (
            "claude-opus-4-5",
            "claude-opus-4-6",
            "claude-opus-4-7",
        ):
            if m.startswith(prefix):
                return False
        return True

    @staticmethod
    def _split_system(messages: List[Dict[str, str]]):
        """Separate system messages (Anthropic uses a top-level ``system`` param)."""
        system_parts: List[str] = []
        chat: List[Dict[str, str]] = []
        for m in messages:
            if m["role"] == "system":
                system_parts.append(m["content"])
            else:
                chat.append(m)
        return "\n".join(system_parts) if system_parts else None, chat

    async def _call_with_retry(self, **kwargs) -> Any:
        last_err: Optional[Exception] = None
        for attempt in range(max(self._max_retries, 1)):
            try:
                return await self.client.messages.create(**kwargs)
            except Exception as exc:
                if "rate" in str(exc).lower() or "overloaded" in str(exc).lower():
                    last_err = exc
                    wait = 2 ** attempt
                    logger.warning("Anthropic rate-limited, retrying in %ss…", wait)
                    await asyncio.sleep(wait)
                else:
                    raise
        raise last_err  # type: ignore[misc]

    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> ProviderResponse:
        system, chat_msgs = self._split_system(messages)
        resolved_model = self._model(model)
        kwargs: Dict[str, Any] = {
            "model": resolved_model,
            "messages": chat_msgs,
            "max_tokens": max_tokens if max_tokens is not None else 4096,
        }
        if self._accepts_temperature(resolved_model):
            kwargs["temperature"] = temperature
        if system:
            kwargs["system"] = system

        resp = await self._call_with_retry(**kwargs)
        content = "".join(
            block.text for block in resp.content if hasattr(block, "text")
        )
        return ProviderResponse(
            content=content,
            model=resp.model,
            usage={
                "prompt_tokens": resp.usage.input_tokens,
                "completion_tokens": resp.usage.output_tokens,
            },
            raw=resp,
        )

    async def chat_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.2,
        tool_choice: str = "auto",
    ) -> ProviderResponse:
        system, chat_msgs = self._split_system(messages)
        anthropic_tools = []
        for t in tools:
            fn = t.get("function", t)
            anthropic_tools.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {}),
            })

        resolved_model = self._model(model)
        kwargs: Dict[str, Any] = {
            "model": resolved_model,
            "messages": chat_msgs,
            "tools": anthropic_tools,
            "max_tokens": 4096,
        }
        if self._accepts_temperature(resolved_model):
            kwargs["temperature"] = temperature
        if system:
            kwargs["system"] = system
        tc = TOOL_CHOICE_MAP.get(tool_choice, {"type": tool_choice})
        kwargs["tool_choice"] = tc

        resp = await self._call_with_retry(**kwargs)
        content_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        for block in resp.content:
            if hasattr(block, "text"):
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })

        return ProviderResponse(
            content="".join(content_parts),
            model=resp.model,
            usage={
                "prompt_tokens": resp.usage.input_tokens,
                "completion_tokens": resp.usage.output_tokens,
            },
            tool_calls=tool_calls,
            raw=resp,
        )

    async def chat_structured(
        self,
        messages: List[Dict[str, str]],
        response_schema: Dict[str, Any],
        *,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> ProviderResponse:
        schema_tool = {
            "name": "structured_response",
            "description": "Return a structured JSON response matching the schema.",
            "input_schema": response_schema,
        }
        system, chat_msgs = self._split_system(messages)
        resolved_model = self._model(model)
        kwargs: Dict[str, Any] = {
            "model": resolved_model,
            "messages": chat_msgs,
            "tools": [schema_tool],
            "tool_choice": {"type": "tool", "name": "structured_response"},
            "max_tokens": 4096,
        }
        if self._accepts_temperature(resolved_model):
            kwargs["temperature"] = temperature
        if system:
            kwargs["system"] = system

        resp = await self._call_with_retry(**kwargs)
        content = ""
        for block in resp.content:
            if block.type == "tool_use" and block.name == "structured_response":
                content = json.dumps(block.input)
                break

        return ProviderResponse(
            content=content,
            model=resp.model,
            usage={
                "prompt_tokens": resp.usage.input_tokens,
                "completion_tokens": resp.usage.output_tokens,
            },
            raw=resp,
        )

    async def stream(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> AsyncIterator[str]:
        system, chat_msgs = self._split_system(messages)
        resolved_model = self._model(model)
        kwargs: Dict[str, Any] = {
            "model": resolved_model,
            "messages": chat_msgs,
            "max_tokens": 4096,
        }
        if self._accepts_temperature(resolved_model):
            kwargs["temperature"] = temperature
        if system:
            kwargs["system"] = system

        async with self.client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
