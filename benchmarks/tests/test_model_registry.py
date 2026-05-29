"""Tests for registry → canonical model id remapping in figures/merges."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE.parent
sys.path.insert(0, str(BENCH_DIR))

from harness.model_registry import canonical_model_id, remap_model_column  # noqa: E402


def test_gpt_5_5_mini_maps_to_gpt_5_4_mini():
    assert canonical_model_id("gpt-5.5-mini") == "gpt-5.4-mini"


def test_remap_collapses_placeholder_rows():
    df = pd.DataFrame({
        "model": ["gpt-5.5-mini", "gpt-5.4-mini"],
        "input_id": ["a", "a"],
        "overall_pass": [1, 0],
    })
    out = remap_model_column(df)
    assert list(out["model"]) == ["gpt-5.4-mini", "gpt-5.4-mini"]
