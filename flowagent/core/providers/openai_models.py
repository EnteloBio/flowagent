"""Map registry / shorthand OpenAI ids to API model strings.

The benchmark registry uses stable logical ids (``gpt-5.5-mini``) for CSV keys
and figure labels. The OpenAI API only accepts released model slugs; some
registry entries are forward-looking placeholders until OpenAI ships them.
"""

from __future__ import annotations

# Registry id or alias -> API slug accepted by v1/chat/completions.
_OPENAI_API_ALIASES: dict[str, str] = {
    # Not released on the API as of May 2026 — interim mini successor.
    "gpt-5.5-mini": "gpt-5.4-mini",
}

# Pro-tier GPT-5 models are Responses-API-only (not v1/chat/completions).
_RESPONSES_API_MODELS: frozenset[str] = frozenset({
    "gpt-5.5-pro",
    "gpt-5.4-pro",
    "gpt-5-pro",
})


def resolve_openai_model(model: str) -> str:
    """Return the API model string for *model*."""
    name = (model or "").strip()
    return _OPENAI_API_ALIASES.get(name, name)


def requires_responses_api(model: str) -> bool:
    """True when *model* must be called via ``v1/responses`` (not chat completions)."""
    name = resolve_openai_model(model).lower()
    if name in _RESPONSES_API_MODELS:
        return True
    # Catch future ``gpt-5*-pro`` slugs without listing every one.
    return name.startswith("gpt-5") and name.endswith("-pro")
