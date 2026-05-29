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


def resolve_openai_model(model: str) -> str:
    """Return the API model string for *model*."""
    name = (model or "").strip()
    return _OPENAI_API_ALIASES.get(name, name)
