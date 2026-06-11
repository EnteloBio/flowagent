"""Tests for Benchmark E relative cost-per-successful-plan figure."""

from __future__ import annotations

import pandas as pd

from harness.plot import (
    _per_competitor_cost_table,
    competitors_cost_per_pass_relative_figure,
    competitors_figure,
)


def test_relative_cost_baseline_is_cheapest():
    df = pd.DataFrame([
        {"competitor": "flowagent", "competitor_name": "FlowAgent",
         "overall_pass": True, "cost_usd": 0.061},
        {"competitor": "raw_gpt", "competitor_name": "Raw LLM (gpt)",
         "overall_pass": True, "cost_usd": 0.0041},
        {"competitor": "claude_code", "competitor_name": "Claude Code",
         "overall_pass": True, "cost_usd": 0.083},
    ])
    table = _per_competitor_cost_table(df)
    assert table is not None
    cheapest = table.loc[table["cost_per_pass"].idxmin(), "competitor"]
    assert table.loc[table["competitor"] == cheapest, "rel_cost_per_pass"].iloc[0] == 1.0
    flow_rel = table.loc[table["competitor"] == "flowagent", "rel_cost_per_pass"].iloc[0]
    assert abs(flow_rel - (0.061 / 0.0041)) < 0.01


def test_competitors_figure_includes_relative_cost_panel():
    df = pd.DataFrame([
        {"competitor": "flowagent", "competitor_name": "FlowAgent",
         "overall_pass": True, "cost_usd": 0.05,
         "tools_present_fraction": 1.0},
        {"competitor": "biomni", "competitor_name": "Biomni",
         "overall_pass": True, "cost_usd": 0.10,
         "tools_present_fraction": 0.9},
    ])
    fig = competitors_figure(df)
    assert fig is not None
    assert len(fig.axes) == 4


def test_relative_cost_figure_renders():
    df = pd.DataFrame([
        {"competitor": "flowagent", "competitor_name": "FlowAgent",
         "overall_pass": True, "cost_usd": 0.05},
        {"competitor": "biomni", "competitor_name": "Biomni",
         "overall_pass": True, "cost_usd": 0.10},
    ])
    fig = competitors_cost_per_pass_relative_figure(df)
    assert fig is not None
    assert len(fig.axes) == 1
    assert "relative" in fig.axes[0].get_title(loc="left").lower()
