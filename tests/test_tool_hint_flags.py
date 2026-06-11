"""Tests for the workflow tool-hint env flag."""

import os

import pytest

from flowagent.core.tool_hint_flags import tool_hint_enabled


@pytest.fixture(autouse=True)
def _clear_tool_hint_env(monkeypatch):
    monkeypatch.delenv("FLOWAGENT_TOOL_HINT", raising=False)


def test_default_on():
    assert tool_hint_enabled() is True


@pytest.mark.parametrize("val", ["false", "0", "no", "off", "FALSE"])
def test_disabled(val, monkeypatch):
    monkeypatch.setenv("FLOWAGENT_TOOL_HINT", val)
    assert tool_hint_enabled() is False


@pytest.mark.parametrize("val", ["true", "1", "yes"])
def test_enabled(val, monkeypatch):
    monkeypatch.setenv("FLOWAGENT_TOOL_HINT", val)
    assert tool_hint_enabled() is True
