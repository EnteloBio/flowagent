"""Compat patches for ``openai.resources.chat.completions.Completions.create``.

Upstream competitors (AutoBA, Biomni, BioMaster) call OpenAI through
different stacks — raw SDK, LangChain — but all hit the same mismatches
when the harness drives GPT-5.x reasoning models with a slightly older
``openai`` package than ``langchain-openai`` expects.
"""

from __future__ import annotations

from typing import Any, Dict, MutableMapping, Optional

_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")
_DEFAULT_REASONING_EFFORT = "low"
_DEFAULT_MAX_COMPLETION_TOKENS = 8000

# LangChain >=0.3 may pass kwargs the installed openai SDK does not accept.
_UNSUPPORTED_CHAT_KWARGS = frozenset({"use_responses_api"})


def is_reasoning_model(model: str) -> bool:
    m = (model or "").lower()
    return any(m.startswith(p) for p in _REASONING_MODEL_PREFIXES)


def normalize_openai_chat_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite gpt-4-era kwargs for GPT-5 / o-series reasoning models."""
    out = dict(kwargs)
    model = str(out.get("model") or "")
    if not is_reasoning_model(model):
        return out
    if "max_tokens" in out:
        out.setdefault("max_completion_tokens", out.pop("max_tokens"))
    elif "max_completion_tokens" not in out:
        out["max_completion_tokens"] = _DEFAULT_MAX_COMPLETION_TOKENS
    out.pop("temperature", None)
    out.setdefault("reasoning_effort", _DEFAULT_REASONING_EFFORT)
    return out


def prepare_openai_chat_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop unsupported kwargs, then apply reasoning-model normalization."""
    out = dict(kwargs)
    for key in _UNSUPPORTED_CHAT_KWARGS:
        out.pop(key, None)
    return normalize_openai_chat_kwargs(out)


def install_openai_completions_patch(
    *,
    stats: Optional[MutableMapping[str, int]] = None,
) -> None:
    """Idempotently patch ``Completions.create`` for benchmark subprocesses."""
    try:
        from openai.resources.chat.completions import Completions
    except Exception:
        return
    if getattr(Completions.create, "_flowagent_patched", False):
        return

    original_create = Completions.create

    def _patched_create(self, *args, **kwargs):
        kwargs = prepare_openai_chat_kwargs(kwargs)
        resp = original_create(self, *args, **kwargs)
        if stats is not None:
            try:
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    stats["prompt_tokens"] = int(stats.get("prompt_tokens", 0)) + int(
                        getattr(usage, "prompt_tokens", 0) or 0
                    )
                    stats["completion_tokens"] = int(
                        stats.get("completion_tokens", 0)
                    ) + int(getattr(usage, "completion_tokens", 0) or 0)
                    stats["calls"] = int(stats.get("calls", 0)) + 1
            except Exception:
                pass
        return resp

    _patched_create._flowagent_patched = True  # type: ignore[attr-defined]
    Completions.create = _patched_create  # type: ignore[assignment]
