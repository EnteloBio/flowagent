"""Provider factory and model-string resolution.

Model strings can be specified as ``provider/model`` (e.g.
``openai/gpt-4.1``, ``anthropic/claude-sonnet-4-20250514``, ``ollama/llama4``)
or as just a provider name when using the settings' default model.
"""

import logging
from typing import Optional

from .base import LLMProvider

logger = logging.getLogger(__name__)

PROVIDER_PREFIXES = {
    "gpt-": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "claude-": "anthropic",
    "gemini-": "google",
    "llama": "ollama",
    "mistral": "ollama",
    "qwen": "ollama",
    "deepseek": "ollama",
    "phi": "ollama",
    "codellama": "ollama",
}


def _infer_provider(model: str) -> str:
    """Best-effort guess of the provider from a bare model name."""
    lower = model.lower()
    for prefix, provider in PROVIDER_PREFIXES.items():
        if lower.startswith(prefix):
            return provider
    return "openai"


def create_provider(
    provider: str,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs,
) -> LLMProvider:
    """Instantiate the correct :class:`LLMProvider` subclass.

    Parameters
    ----------
    provider : str
        One of ``openai``, ``anthropic``, ``google``, ``ollama``.
    model : str, optional
        Override the default model for the provider.
    api_key : str, optional
        API key (not needed for Ollama).
    base_url : str, optional
        Custom endpoint URL.
    """
    provider = provider.lower().strip()

    if provider == "openai":
        from .openai_provider import OpenAIProvider

        kw = {}
        if model:
            kw["model"] = model
        if api_key:
            kw["api_key"] = api_key
        else:
            import os
            kw["api_key"] = os.environ.get("OPENAI_API_KEY", "")
        if base_url:
            kw["base_url"] = base_url
        return OpenAIProvider(**kw, **kwargs)

    if provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        kw = {}
        if model:
            kw["model"] = model
        if api_key:
            kw["api_key"] = api_key
        else:
            import os
            kw["api_key"] = os.environ.get("ANTHROPIC_API_KEY", "")
        return AnthropicProvider(**kw, **kwargs)

    if provider == "google":
        from .google_provider import GoogleProvider

        kw = {}
        if model:
            kw["model"] = model
        if api_key:
            kw["api_key"] = api_key
        else:
            import os
            kw["api_key"] = os.environ.get("GOOGLE_API_KEY", "")
        return GoogleProvider(**kw, **kwargs)

    if provider == "ollama":
        from .ollama_provider import OllamaProvider

        kw = {}
        if model:
            kw["model"] = model
        if base_url:
            kw["base_url"] = base_url
        return OllamaProvider(**kw, **kwargs)

    raise ValueError(
        f"Unknown LLM provider: {provider!r}. "
        f"Supported: openai, anthropic, google, ollama"
    )


def get_provider_for_model(
    model_string: str,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs,
) -> LLMProvider:
    """Resolve a ``provider/model`` string and return a ready provider.

    Examples::

        get_provider_for_model("openai/gpt-4.1")
        get_provider_for_model("anthropic/claude-sonnet-4-20250514")
        get_provider_for_model("gpt-4.1")         # inferred → openai
        get_provider_for_model("ollama/llama4")
    """
    if "/" in model_string:
        provider, model = model_string.split("/", 1)
    else:
        model = model_string
        provider = _infer_provider(model)

    logger.info("Resolved model %r → provider=%s model=%s", model_string, provider, model)
    return create_provider(provider, model=model, api_key=api_key, base_url=base_url, **kwargs)
