"""Tests for Gemini API model id resolution."""

from flowagent.core.providers.gemini_models import resolve_gemini_model


def test_bare_legacy_names_map_to_versioned_api_ids():
    assert resolve_gemini_model("gemini-1.5-pro") == "gemini-1.5-pro-002"
    assert resolve_gemini_model("gemini-1.5-flash") == "gemini-1.5-flash-002"


def test_current_ga_names_pass_through():
    assert resolve_gemini_model("gemini-2.5-flash") == "gemini-2.5-flash"
    assert resolve_gemini_model("gemini-3.5-flash") == "gemini-3.5-flash"


def test_strips_models_prefix():
    assert resolve_gemini_model("models/gemini-2.5-pro") == "gemini-2.5-pro"


def test_set_provider_uses_api_id(monkeypatch):
    import os
    from benchmarks.harness.runner import set_provider

    monkeypatch.delenv("LLM_MODEL", raising=False)
    set_provider({
        "id": "gemini-1.5-pro",
        "provider": "google",
        "env_var": "GOOGLE_API_KEY",
        "api_id": "gemini-1.5-pro-002",
    })
    assert os.environ["LLM_MODEL"] == "gemini-1.5-pro-002"
