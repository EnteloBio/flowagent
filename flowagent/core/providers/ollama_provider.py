"""Ollama provider for local models via the OpenAI-compatible API."""

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from openai import AsyncOpenAI

from .base import LLMProvider, ProviderResponse

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"


class OllamaProvider(LLMProvider):
    """Uses Ollama's OpenAI-compatible endpoint so local models
    (Llama 3, Mistral, Qwen, etc.) work with the same tool-calling
    and structured-output interfaces.
    """

    def __init__(
        self,
        model: str = "llama4",
        base_url: str = DEFAULT_OLLAMA_URL,
    ):
        self.default_model = model
        self.client = AsyncOpenAI(
            api_key="ollama",  # Ollama ignores the key but the SDK requires one
            base_url=base_url,
        )

    def _model(self, override: Optional[str]) -> str:
        return override or self.default_model

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
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        completion = await self.client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        return ProviderResponse(
            content=choice.message.content or "",
            model=completion.model,
            usage=dict(completion.usage) if completion.usage else {},
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
        completion = await self.client.chat.completions.create(
            model=self._model(model),
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
        )
        choice = completion.choices[0]
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                })
        return ProviderResponse(
            content=choice.message.content or "",
            model=completion.model,
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
        completion = await self.client.chat.completions.create(
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
        choice = completion.choices[0]
        return ProviderResponse(
            content=choice.message.content or "",
            model=completion.model,
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
