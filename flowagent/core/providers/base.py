"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional


@dataclass
class ProviderResponse:
    """Standardised response from any LLM provider."""

    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    raw: Any = None


class LLMProvider(ABC):
    """Unified async interface that every backend must implement."""

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> ProviderResponse:
        """Send a chat completion request and return the full response."""

    @abstractmethod
    async def chat_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.2,
        tool_choice: str = "auto",
    ) -> ProviderResponse:
        """Chat completion with function / tool calling support."""

    @abstractmethod
    async def chat_structured(
        self,
        messages: List[Dict[str, str]],
        response_schema: Dict[str, Any],
        *,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> ProviderResponse:
        """Chat completion that guarantees JSON conforming to *response_schema*."""

    @abstractmethod
    async def stream(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> AsyncIterator[str]:
        """Yield content tokens as they arrive."""

    # Convenience helpers ------------------------------------------------

    async def complete(self, prompt: str, **kwargs) -> str:
        """One-shot completion from a plain text prompt."""
        messages = [{"role": "user", "content": prompt}]
        resp = await self.chat(messages, **kwargs)
        return resp.content
