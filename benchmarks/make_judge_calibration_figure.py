"""Figure for Benchmark G inter-judge calibration.

Reads ``judge_calibration.csv`` (+ optional ``judge_calibration_summary.json``)
from ``judge_calibration.py`` and renders a two-panel supplement figure:

  a. Paired score scatter (judge A vs judge B) with y=x reference,
     pass threshold at 60, and agreement statistics annotated.
  b. Score-delta histogram (judge B − judge A) showing systematic bias.

Usage::

    python make_judge_calibration_figure.py \\
        --calibration results/judge_calibration/2026-05-31T13-53-40

    python make_judge_calibration_figure.py \\
        --calibration results/judge_calibration/2026-05-31T13-53-40 \\
        --out results/figures/judge_calibration
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from harness.expert_audit_stats import summarise_paired_scores  # noqa: E402

_PASS_MARK = 60.0

# Okabe-Ito
_C_BOTH_PASS = "#009E73"
_C_BOTH_FAIL = "#999999"
_C_DISCORD = "#E69F00"
_C_REF = "#1f2937"
_C_MEAN = "#D55E00"


def _latest_calibration_dir(results_root: Path) -> Optional[Path]:
    cal_root = results_root / "judge_calibration"
    if not cal_root.exists():
        return None
    subs = [
        p for p in cal_root.iterdir()
        if p.is_dir() and (p / "judge_calibration.csv").exists()
    ]
    if not subs:
        return None
    return max(subs, key=lambda p: p.stat().st_mtime)


def load_run(cal_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load calibration CSV and summary JSON from a run directory."""
    csv_path = cal_dir / "judge_calibration.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"judge_calibration.csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    for col in ("judge_a_score", "judge_b_score"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    summary_path = cal_dir / "judge_calibration_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        paired = df.dropna(subset=["judge_a_score", "judge_b_score"])
        summary = summarise_paired_scores(
            paired["judge_a_score"].tolist(),
            paired["judge_b_score"].tolist(),
            pass_mark=_PASS_MARK,
        )
        if "judge_a" in df.columns and df["judge_a"].notna().any():
            summary["judge_a"] = str(df["judge_a"].iloc[0])
        if "judge_b" in df.columns and df["judge_b"].notna().any():
            summary["judge_b"] = str(df["judge_b"].iloc[0])
        summary["pass_mark"] = _PASS_MARK
    return df, summary


def _point_colors(df: pd.DataFrame, *, pass_mark: float) -> np.ndarray:
    a_pass = df["judge_a_score"] >= pass_mark
    b_pass = df["judge_b_score"] >= pass_mark
    colors = np.empty(len(df), dtype=object)
    colors[a_pass & b_pass] = _C_BOTH_PASS
    colors[~a_pass & ~b_pass] = _C_BOTH_FAIL
    colors[a_pass ^ b_pass] = _C_DISCORD
    return colors


def build_figure(
    df: pd.DataFrame,
    summary: Dict[str, Any],
) -> plt.Figure:
    """Build the two-panel judge-calibration figure."""
    df = df.dropna(subset=["judge_a_score", "judge_b_score"]).copy()
    if df.empty:
        raise ValueError("no paired judge scores to plot")

    pass_mark = float(summary.get("pass_mark", _PASS_MARK))
    judge_a = summary.get("judge_a", "judge A")
    judge_b = summary.get("judge_b", "judge B")
    n = int(summary.get("n", len(df)))
    r = summary.get("score_correlation")
    kappa = summary.get("cohens_kappa")
    pass_agree = summary.get("pass_agreement")
    mean_delta = summary.get("mean_score_delta")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.5, 4.6),
        gridspec_kw={"width_ratios": [1.15, 0.85], "wspace": 0.32},
    )

    x = df["judge_a_score"].to_numpy()
    y = df["judge_b_score"].to_numpy()
    delta = y - x
    colors = _point_colors(df, pass_mark=pass_mark)

    # ── Panel a: scatter ─────────────────────────────────────────
    ax1.scatter(x, y, c=colors, s=42, edgecolors="white", linewidths=0.5,
                alpha=0.92, zorder=3)
    lo = max(0, min(x.min(), y.min()) - 5)
    hi = min(105, max(x.max(), y.max()) + 5)
    ax1.plot([lo, hi], [lo, hi], color=_C_REF, linewidth=1.0,
             linestyle="--", alpha=0.55, zorder=1, label="y = x")
    ax1.axhline(pass_mark, color=_C_REF, linewidth=0.8, linestyle=":",
                alpha=0.45, zorder=1)
    ax1.axvline(pass_mark, color=_C_REF, linewidth=0.8, linestyle=":",
                alpha=0.45, zorder=1)
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_xlabel(f"{judge_a} score")
    ax1.set_ylabel(f"{judge_b} score")
    ax1.set_title("a  Paired judge scores", loc="left", fontweight="bold")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_aspect("equal", adjustable="box")

    stats_lines = [f"n = {n}"]
    if r is not None:
        stats_lines.append(f"Pearson r = {r:.2f}")
    if kappa is not None:
        stats_lines.append(f"Cohen's κ = {kappa:.2f}")
    if pass_agree is not None:
        stats_lines.append(f"Pass agreement = {pass_agree:.0%}")
    if mean_delta is not None:
        stats_lines.append(f"Mean Δ (B−A) = {mean_delta:+.1f}")
    ax1.text(
        0.03, 0.97, "\n".join(stats_lines),
        transform=ax1.transAxes, ha="left", va="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#d1d5db", alpha=0.92),
    )

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_C_BOTH_PASS,
               markersize=7, label="Both pass"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_C_BOTH_FAIL,
               markersize=7, label="Both fail"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_C_DISCORD,
               markersize=7, label="Pass discordant"),
    ]
    ax1.legend(handles=legend_handles, loc="lower right", fontsize=7.5,
               framealpha=0.9)

    # ── Panel b: delta histogram ─────────────────────────────────
    bins = np.arange(-40, 65, 5)
    ax2.hist(delta, bins=bins, color="#56B4E9", edgecolor="white",
             linewidth=0.6, alpha=0.85)
    ax2.axvline(0, color=_C_REF, linewidth=1.0, linestyle="--", alpha=0.6)
    if mean_delta is not None:
        ax2.axvline(mean_delta, color=_C_MEAN, linewidth=1.4,
                    label=f"mean Δ = {mean_delta:+.1f}")
    ax2.set_xlabel(f"Score delta ({judge_b} − {judge_a})")
    ax2.set_ylabel("Count")
    ax2.set_title("b  Judge B − judge A", loc="left", fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    if mean_delta is not None:
        ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle(
        "Inter-judge calibration — Benchmark G open-ended responses",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.subplots_adjust(top=0.88, wspace=0.32)
    return fig


def save_figure(fig: plt.Figure, out_base: Path, *, dpi: int = 300) -> Path:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    pdf = out_base.with_suffix(".pdf")
    png = out_base.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    return pdf


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--calibration",
        default="",
        help="Judge-calibration run dir (default: latest under results/)",
    )
    ap.add_argument(
        "--out", default="",
        help="Output basename (default: results/figures/judge_calibration)",
    )
    args = ap.parse_args()

    if args.calibration:
        cal_dir = Path(args.calibration)
        if not cal_dir.is_absolute():
            cal_dir = _HERE / cal_dir
    else:
        cal_dir = _latest_calibration_dir(_HERE / "results")
        if cal_dir is None:
            raise SystemExit(
                "no judge_calibration runs found — run judge_calibration.py first "
                "or pass --calibration"
            )

    df, summary = load_run(cal_dir)
    fig = build_figure(df, summary)

    if args.out:
        out_base = Path(args.out)
        if not out_base.is_absolute():
            out_base = _HERE / out_base
    else:
        out_base = _HERE / "results" / "figures" / "judge_calibration"

    pdf = save_figure(fig, out_base)
    plt.close(fig)
    print(f"[ok] judge_calibration → {pdf}")
    print(f"[ok] judge_calibration → {out_base.with_suffix('.png')}")


if __name__ == "__main__":
    main()
