"""Tests for judge calibration figure builder."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE.parent
sys.path.insert(0, str(BENCH_DIR))

from make_judge_calibration_figure import build_figure, _point_colors  # noqa: E402


def test_build_figure_smoke():
    df = pd.DataFrame({
        "judge_a_score": [40.0, 55.0, 70.0, 80.0],
        "judge_b_score": [45.0, 50.0, 75.0, 85.0],
    })
    summary = {
        "n": 4,
        "score_correlation": 0.95,
        "cohens_kappa": 0.8,
        "pass_agreement": 0.75,
        "mean_score_delta": 3.75,
        "judge_a": "gpt-5.4",
        "judge_b": "gemini-2.5-pro",
        "pass_mark": 60.0,
    }
    fig = build_figure(df, summary)
    assert fig.axes[0].get_xlabel() == "gpt-5.4 score"
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_point_colors_discordant():
    df = pd.DataFrame({
        "judge_a_score": [70.0, 50.0],
        "judge_b_score": [50.0, 70.0],
    })
    colors = _point_colors(df, pass_mark=60.0)
    assert len(colors) == 2
