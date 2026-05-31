"""OpenAI provider (GPT-4o, GPT-4o-mini, o1, o3, etc.)."""

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from openai import AsyncOpenAI, RateLimitError

from .base import LLMProvider, ProviderResponse
from .openai_models import requires_responses_api, resolve_openai_model

logger = logging.getLogger(__name__)


def _is_not_chat_model_error(exc: Exception) -> bool:
    """True when OpenAI rejects a model on v1/chat/completions."""
    msg = str(exc).lower()
    return (
        "not a chat model" in msg
        or "v1/completions" in msg
        or "v1/responses" in msg
    )


class OpenAIProvider(LLMProvider):
    """Wraps the official ``openai`` async client."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4.1",
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        self.default_model = model
        self._max_retries = max_retries
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def _model(self, override: Optional[str]) -> str:
        return resolve_openai_model(override or self.default_model)

    # -- core methods ---------------------------------------------------

    # Reasoning models where the server-side default is "medium" — which burns
    # 20–50k hidden-thinking tokens per call, all billed at output rates.
    # Downgrade to "low" unless the caller explicitly overrides. Also cap
    # ``max_completion_tokens`` so one runaway cell can't cost $100.
    _REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")
    _DEFAULT_REASONING_EFFORT = "low"
    _DEFAULT_MAX_COMPLETION_TOKENS = 8000

    @classmethod
    def _is_reasoning_model(cls, model: str) -> bool:
        m = (model or "").lower()
        return any(m.startswith(p) for p in cls._REASONING_MODEL_PREFIXES)

    @staticmethod
    def _reasoning_effort(model: str, override: Optional[str]) -> str:
        """Pick reasoning effort; Pro-tier models require ``high``."""
        if override:
            return override
        m = (model or "").lower()
        if m.endswith("-pro") and m.startswith("gpt-5"):
            return "high"
        return OpenAIProvider._DEFAULT_REASONING_EFFORT

    @staticmethod
    def _messages_for_responses_api(
        messages: List[Dict[str, str]],
    ) -> tuple[Optional[str], Any]:
        """Split chat messages into Responses API ``instructions`` + ``input``."""
        system_parts: List[str] = []
        input_items: List[Dict[str, str]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content") or ""
            if role == "system":
                system_parts.append(content)
            else:
                input_items.append({"role": role, "content": content})
        instructions = "\n\n".join(system_parts) if system_parts else None
        if not input_items:
            return instructions, ""
        if len(input_items) == 1 and input_items[0]["role"] == "user":
            return instructions, input_items[0]["content"]
        return instructions, input_items

    async def _chat_via_responses(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ) -> ProviderResponse:
        """Route through ``v1/responses`` for Pro-tier models."""
        instructions, input_payload = self._messages_for_responses_api(messages)
        kwargs: Dict[str, Any] = {
            "model": model,
            "input": input_payload,
        }
        if instructions:
            kwargs["instructions"] = instructions
        kwargs["max_output_tokens"] = (
            max_tokens if max_tokens is not None else self._DEFAULT_MAX_COMPLETION_TOKENS
        )
        if self._is_reasoning_model(model):
            kwargs["reasoning"] = {
                "effort": self._reasoning_effort(model, reasoning_effort),
            }
        response = await self._call_responses_with_retry(**kwargs)
        usage: Dict[str, Any] = {}
        if getattr(response, "usage", None):
            usage = (
                response.usage.model_dump()
                if hasattr(response.usage, "model_dump")
                else dict(response.usage)
            )
        return ProviderResponse(
            content=response.output_text or "",
            model=getattr(response, "model", model) or model,
            usage=usage,
            raw=response,
        )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ) -> ProviderResponse:
        effective_model = self._model(model)
        if requires_responses_api(effective_model):
            return await self._chat_via_responses(
                messages,
                model=effective_model,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            )
        kwargs: Dict[str, Any] = {
            "model": effective_model,
            "messages": messages,
        }
        if self._is_reasoning_model(effective_model):
            # Reasoning models reject ``temperature``/``max_tokens``; use
            # ``max_completion_tokens`` + ``reasoning_effort``.
            kwargs["max_completion_tokens"] = (
                max_tokens if max_tokens is not None else self._DEFAULT_MAX_COMPLETION_TOKENS
            )
            kwargs["reasoning_effort"] = self._reasoning_effort(
                effective_model, reasoning_effort,
            )
        else:
            kwargs["temperature"] = temperature
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

        try:
            completion = await self._call_with_retry(**kwargs)
        except Exception as exc:
            if _is_not_chat_model_error(exc):
                return await self._chat_via_responses(
                    messages,
                    model=effective_model,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                )
            raise
        if not completion.choices:
            return ProviderResponse(content="", model=completion.model, raw=completion)
        choice = completion.choices[0]
        return ProviderResponse(
            content=choice.message.content or "",
            model=completion.model,
            usage=completion.usage.model_dump() if hasattr(completion.usage, "model_dump") else (dict(completion.usage) if completion.usage else {}),
            raw=completion,
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
        completion = await self._call_with_retry(
            model=self._model(model),
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
        )
        if not completion.choices:
            return ProviderResponse(content="", model=completion.model, raw=completion)
        choice = completion.choices[0]
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args,
                })
        return ProviderResponse(
            content=choice.message.content or "",
            model=completion.model,
            usage=completion.usage.model_dump() if hasattr(completion.usage, "model_dump") else (dict(completion.usage) if completion.usage else {}),
            tool_calls=tool_calls,
            raw=completion,
        )

    async def chat_structured(
        self,
        messages: List[Dict[str, str]],
        response_schema: Dict[str, Any],
        *,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> ProviderResponse:
        completion = await self._call_with_retry(
            model=self._model(model),
            messages=messages,
            temperature=temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema.get("title", "response"),
                    "strict": True,
                    "schema": response_schema,
                },
            },
        )
        if not completion.choices:
            return ProviderResponse(content="", model=completion.model, raw=completion)
        choice = completion.choices[0]
        return ProviderResponse(
            content=choice.message.content or "",
            model=completion.model,
            usage=completion.usage.model_dump() if hasattr(completion.usage, "model_dump") else (dict(completion.usage) if completion.usage else {}),
            raw=completion,
        )

    async def stream(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> AsyncIterator[str]:
        response = await self.client.chat.completions.create(
            model=self._model(model),
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    # -- retry helper ---------------------------------------------------

    async def _call_responses_with_retry(self, **kwargs) -> Any:
        import asyncio
        attempts = max(self._max_retries, 1)
        last_err: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                return await self.client.responses.create(**kwargs)
            except RateLimitError as exc:
                last_err = exc
                wait = 2 ** attempt
                logger.warning("OpenAI rate-limited (responses), retrying in %ss…", wait)
                await asyncio.sleep(wait)
        raise last_err  # type: ignore[misc]

    async def _call_with_retry(self, **kwargs) -> Any:
        import asyncio
        attempts = max(self._max_retries, 1)
        last_err: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                return await self.client.chat.completions.create(**kwargs)
            except RateLimitError as exc:
                last_err = exc
                wait = 2 ** attempt
                logger.warning("OpenAI rate-limited, retrying in %ss…", wait)
                await asyncio.sleep(wait)
        raise last_err  # type: ignore[misc]
