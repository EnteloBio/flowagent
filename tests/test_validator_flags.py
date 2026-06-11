"""Tests for split command-validator env flags."""

import os

import pytest

from flowagent.core.validator_flags import (
    validator_autofix_enabled,
    validator_legacy_enabled,
    validator_retry_enabled,
)


@pytest.fixture(autouse=True)
def _clear_validator_env(monkeypatch):
    for key in (
        "FLOWAGENT_VALIDATOR_ENABLED",
        "FLOWAGENT_VALIDATOR_AUTOFIX",
        "FLOWAGENT_VALIDATOR_RETRY",
    ):
        monkeypatch.delenv(key, raising=False)


def test_defaults_autofix_on_retry_off():
    assert validator_legacy_enabled() is None
    assert validator_autofix_enabled() is True
    assert validator_retry_enabled() is False


def test_legacy_false_disables_both(monkeypatch):
    monkeypatch.setenv("FLOWAGENT_VALIDATOR_ENABLED", "false")
    assert validator_autofix_enabled() is False
    assert validator_retry_enabled() is False


def test_legacy_true_enables_both(monkeypatch):
    monkeypatch.setenv("FLOWAGENT_VALIDATOR_ENABLED", "true")
    assert validator_autofix_enabled() is True
    assert validator_retry_enabled() is True


def test_granular_flags_when_legacy_unset(monkeypatch):
    monkeypatch.setenv("FLOWAGENT_VALIDATOR_AUTOFIX", "false")
    monkeypatch.setenv("FLOWAGENT_VALIDATOR_RETRY", "true")
    assert validator_autofix_enabled() is False
    assert validator_retry_enabled() is True


def test_legacy_overrides_granular(monkeypatch):
    monkeypatch.setenv("FLOWAGENT_VALIDATOR_ENABLED", "true")
    monkeypatch.setenv("FLOWAGENT_VALIDATOR_AUTOFIX", "false")
    monkeypatch.setenv("FLOWAGENT_VALIDATOR_RETRY", "false")
    assert validator_autofix_enabled() is True
    assert validator_retry_enabled() is True
