"""Google Gemini provider (Gemini 2.5 family)."""

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from .base import LLMProvider, ProviderResponse

logger = logging.getLogger(__name__)


class GoogleProvider(LLMProvider):
    """Wraps the ``google-genai`` client for Gemini models."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
    ):
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "Install the google-genai package: pip install google-genai"
            )
        self.default_model = model
        self.genai = genai
        self.client = genai.Client(api_key=api_key)

    def _model(self, override: Optional[str]) -> str:
        return override or self.default_model

    @staticmethod
    def _to_genai_contents(messages: List[Dict[str, str]]):
        """Convert OpenAI-style messages to Gemini contents + system_instruction."""
        system_parts: List[str] = []
        contents: List[Dict[str, Any]] = []
        for m in messages:
            if m["role"] == "system":
                system_parts.append(m["content"])
            else:
                role = "model" if m["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": m["content"]}]})
        return contents, "\n".join(system_parts) if system_parts else None

    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> ProviderResponse:
        contents, system_instruction = self._to_genai_contents(messages)
        config: Dict[str, Any] = {"temperature": temperature}
        if max_tokens:
            config["max_output_tokens"] = max_tokens
        if system_instruction:
            config["system_instruction"] = system_instruction

        response = await self.client.aio.models.generate_content(
            model=self._model(model),
            contents=contents,
            config=self.genai.types.GenerateContentConfig(**config),
        )
        text = response.text or ""
        usage = {}
        if response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                "completion_tokens": response.usage_metadata.candidates_token_count or 0,
            }
        return ProviderResponse(
            content=text,
            model=self._model(model),
            usage=usage,
            raw=response,
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
        contents, system_instruction = self._to_genai_contents(messages)

        gemini_declarations = []
        for t in tools:
            fn = t.get("function", t)
            gemini_declarations.append(self.genai.types.FunctionDeclaration(
                name=fn["name"],
                description=fn.get("description", ""),
                parameters=fn.get("parameters", {}),
            ))

        gemini_tools = [self.genai.types.Tool(function_declarations=gemini_declarations)]
        config: Dict[str, Any] = {
            "temperature": temperature,
            "tools": gemini_tools,
        }
        if system_instruction:
            config["system_instruction"] = system_instruction

        response = await self.client.aio.models.generate_content(
            model=self._model(model),
            contents=contents,
            config=self.genai.types.GenerateContentConfig(**config),
        )
        tool_calls = []
        text_parts = []
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    tool_calls.append({
                        "id": f"call_{part.function_call.name}",
                        "name": part.function_call.name,
                        "arguments": dict(part.function_call.args) if part.function_call.args else {},
                    })
                elif part.text:
                    text_parts.append(part.text)

        return ProviderResponse(
            content="".join(text_parts),
            model=self._model(model),
            tool_calls=tool_calls,
            raw=response,
        )

    async def chat_structured(
        self,
        messages: List[Dict[str, str]],
        response_schema: Dict[str, Any],
        *,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> ProviderResponse:
        contents, system_instruction = self._to_genai_contents(messages)
        config: Dict[str, Any] = {
            "temperature": temperature,
            "response_mime_type": "application/json",
            "response_schema": response_schema,
        }
        if system_instruction:
            config["system_instruction"] = system_instruction

        response = await self.client.aio.models.generate_content(
            model=self._model(model),
            contents=contents,
            config=self.genai.types.GenerateContentConfig(**config),
        )
        usage = {}
        if response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                "completion_tokens": response.usage_metadata.candidates_token_count or 0,
            }
        return ProviderResponse(
            content=response.text or "",
            model=self._model(model),
            usage=usage,
            raw=response,
        )

    async def stream(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> AsyncIterator[str]:
        contents, system_instruction = self._to_genai_contents(messages)
        config: Dict[str, Any] = {"temperature": temperature}
        if system_instruction:
            config["system_instruction"] = system_instruction

        async for chunk in await self.client.aio.models.generate_content_stream(
            model=self._model(model),
            contents=contents,
            config=self.genai.types.GenerateContentConfig(**config),
        ):
            if chunk.text:
                yield chunk.text
