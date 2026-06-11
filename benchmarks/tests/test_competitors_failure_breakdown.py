"""Tests for rubric-level competitor failure breakdown figures."""

from __future__ import annotations

import pandas as pd

from harness.plot import (
    classify_competitor_failure_reason,
    competitors_failure_breakdown_figure,
)


def _row(**kwargs):
    base = {
        "competitor": "biomni",
        "competitor_name": "Biomni",
        "overall_pass": False,
        "plan_valid": True,
        "dag_valid": True,
        "type_correct": True,
        "tools_present_fraction": 1.0,
        "no_forbidden_tools": True,
        "step_count_ok": True,
        "tier": "transcription",
        "error": None,
    }
    base.update(kwargs)
    return pd.Series(base)


def test_classify_upstream_error():
    r = _row(error="RuntimeError: shim exited 1")
    assert classify_competitor_failure_reason(r) == "upstream_error"


def test_classify_pass():
    r = _row(overall_pass=True)
    assert classify_competitor_failure_reason(r) == "pass"


def test_classify_missing_tools_before_forbidden():
    r = _row(tools_present_fraction=0.8, no_forbidden_tools=False)
    assert classify_competitor_failure_reason(r) == "missing_tools"


def test_classify_wrong_workflow_type():
    r = _row(type_correct=False, tools_present_fraction=0.5)
    assert classify_competitor_failure_reason(r) == "wrong_workflow_type"


def test_classify_inference_missing_tool_set():
    r = _row(
        tier="inference",
        any_tool_set_matched=False,
        type_correct=True,
        tools_present_fraction=None,
    )
    assert classify_competitor_failure_reason(r) == "missing_tools"


def test_failure_breakdown_figure_renders():
    df = pd.DataFrame([
        dict(_row(competitor="flowagent", competitor_name="FlowAgent",
                 overall_pass=True)),
        dict(_row(competitor="biomni", competitor_name="Biomni",
                 tools_present_fraction=0.67)),
        dict(_row(competitor="autoba", competitor_name="AutoBA",
                 type_correct=False)),
        dict(_row(competitor="raw_gpt-5.4-mini",
                 competitor_name="Raw LLM (gpt-5.4-mini)",
                 step_count_ok=False)),
    ])
    fig = competitors_failure_breakdown_figure(df)
    assert fig is not None
    assert len(fig.axes) == 1
    assert fig.axes[0].get_legend() is not None
