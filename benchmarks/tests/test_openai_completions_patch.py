"""Tests for shared OpenAI Completions.create compat patch."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from harness.openai_completions_patch import prepare_openai_chat_kwargs  # noqa: E402


def test_reasoning_model_rewrites_max_tokens():
    out = prepare_openai_chat_kwargs({
        "model": "gpt-5.4",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1024,
        "temperature": 0,
    })
    assert "max_tokens" not in out
    assert out["max_completion_tokens"] == 1024
    assert "temperature" not in out
    assert out["reasoning_effort"] == "low"


def test_strips_langchain_use_responses_api():
    out = prepare_openai_chat_kwargs({
        "model": "gpt-4.1",
        "use_responses_api": True,
        "max_tokens": 512,
    })
    assert "use_responses_api" not in out
    assert out["max_tokens"] == 512


def test_non_reasoning_model_unchanged_except_strip():
    kw = {
        "model": "gpt-4.1",
        "max_tokens": 512,
        "temperature": 0,
    }
    assert prepare_openai_chat_kwargs(kw) == kw
