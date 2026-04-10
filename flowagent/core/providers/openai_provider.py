"""OpenAI provider (GPT-4o, GPT-4o-mini, o1, o3, etc.)."""

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from openai import AsyncOpenAI, RateLimitError

from .base import LLMProvider, ProviderResponse

logger = logging.getLogger(__name__)


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
        return override or self.default_model

    # -- core methods ---------------------------------------------------

    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> ProviderResponse:
        kwargs: Dict[str, Any] = {
            "model": self._model(model),
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        completion = await self._call_with_retry(**kwargs)
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
            except Exception as exc:
                if "model_not_found" in str(exc) and kwargs.get("model") != "gpt-4.1-mini":
                    logger.warning("Model %s not found, falling back to gpt-4.1-mini", kwargs["model"])
                    kwargs["model"] = "gpt-4.1-mini"
                    continue
                raise
        raise last_err  # type: ignore[misc]
