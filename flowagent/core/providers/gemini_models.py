"""Map registry / shorthand Gemini ids to Generative Language API model strings.

The benchmark registry uses short logical ids (``gemini-1.5-pro``) for stable
CSV keys and figure labels. Google's API expects versioned names such as
``gemini-1.5-pro-002`` or stable GA ids like ``gemini-2.5-flash``.
"""

from __future__ import annotations

# Logical id or alias -> API model string accepted by google-genai.
_GEMINI_API_ALIASES: dict[str, str] = {
    # Legacy 1.5 — bare names 404 on v1beta; pin stable snapshots.
    "gemini-1.5-pro": "gemini-1.5-pro-002",
    "gemini-1.5-flash": "gemini-1.5-flash-002",
    # Retired preview ids -> GA successors (for manual re-runs / rescoring).
    "gemini-3-flash-preview": "gemini-3.5-flash",
    "gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite",
}


def resolve_gemini_model(model: str) -> str:
    """Return the API model string for *model* (strips optional ``models/`` prefix)."""
    name = (model or "").strip()
    if name.startswith("models/"):
        name = name[len("models/") :]
    return _GEMINI_API_ALIASES.get(name, name)
