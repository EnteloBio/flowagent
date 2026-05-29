"""Tests for OpenAI registry id -> API slug resolution."""

from flowagent.core.providers.openai_models import resolve_openai_model


def test_gpt_5_5_mini_maps_to_gpt_5_4_mini():
    assert resolve_openai_model("gpt-5.5-mini") == "gpt-5.4-mini"


def test_passthrough_for_released_models():
    assert resolve_openai_model("gpt-5.5") == "gpt-5.5"
    assert resolve_openai_model("gpt-5.4-mini") == "gpt-5.4-mini"
