"""Benchmark G — biological interpretation results figure.

Renders a 4-panel summary from a merged metrics.csv produced by
``bench_interpretation.py``. Designed to expose the headline finding:
multiple-choice and open-ended performance do not co-vary, and
open-ended scores cluster well below the rubric's 60-point pass
threshold.

Panels:
  A. Per-model accuracy: MCQ (with Wilson 95 % CIs) and open-ended
     pass rate (judge ≥ 60), side-by-side bars; chance baseline drawn.
  B. Per-model open-ended judge-score distribution as violins, with
     a strip overlay of individual question scores; rubric's 60-pt
     pass threshold drawn as a horizontal line.
  C. Per-dataset breakdown: side-by-side heatmaps of MCQ accuracy
     and open-ended mean judge score (rows = models, cols = datasets).
  D. MCQ-vs-open-ended scatter — one point per model, Spearman ρ
     and p annotated; exposes the dissociation between the two
     capabilities.

Usage::

    python make_interpretation_figure.py \\
        --metrics results/interpretation/_merged/<TS>/metrics.csv \\
        --out figure_interpretation
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats


OPEN_ENDED_PASS = 60.0  # judge-score threshold for "correct" open-ended


# ── Helpers ──────────────────────────────────────────────────────

def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - margin, centre + margin


def _bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().eq("true")


def _short_dataset(name: str) -> str:
    """Compact dataset names for axis labels."""
    return (name
            .replace("_dex_de",          "")
            .replace("_mammary_de",      "_mammary")
            .replace("_covid_blood_de",  "_covid")
            .replace("_atac_immune",     "_atac")
            .replace("_er_chip",         "_er")
            .replace("_suz12_h1",        "_suz12")
            .replace("_chr20",           "")
            .replace("encsr000euq",      "ENCSR…")
            .replace("giab_na12878",     "GIAB"))


def _summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in sorted(df["model"].unique()):
        g   = df[df["model"] == model]
        mcq = g[g["question_type"] == "mcq"]
        oe  = g[g["question_type"] == "open_ended"]
        mcq_n = len(mcq); mcq_k = int(mcq["correct"].sum())
        oe_n  = len(oe);  oe_k  = int(oe["correct"].sum())
        ci_l, ci_u = _wilson_ci(mcq_k, mcq_n)
        oe_mean = float(oe["judge_score"].mean()) if oe_n else float("nan")
        rows.append({
            "model":   model,
            "mcq_acc": mcq_k / mcq_n if mcq_n else float("nan"),
            "mcq_lo":  ci_l, "mcq_hi": ci_u, "mcq_n": mcq_n,
            "oe_pass": oe_k / oe_n if oe_n else float("nan"),
            "oe_mean": oe_mean, "oe_n": oe_n,
        })
    return (pd.DataFrame(rows)
            .sort_values("mcq_acc", ascending=False)
            .reset_index(drop=True))


# ── Panels ───────────────────────────────────────────────────────

def _panel_a(ax: plt.Axes, summary: pd.DataFrame, chance: float) -> None:
    n = len(summary)
    x = np.arange(n)
    w = 0.4
    mcq_low_err  = summary["mcq_acc"] - summary["mcq_lo"]
    mcq_high_err = summary["mcq_hi"] - summary["mcq_acc"]
    ax.bar(x - w / 2, summary["mcq_acc"], w,
           color="#4a86c4", edgecolor="#1f3a5e", lw=0.6,
           label=f"MCQ (chance ≈ {chance:.0%})")
    ax.errorbar(x - w / 2, summary["mcq_acc"],
                yerr=[mcq_low_err, mcq_high_err],
                fmt="none", color="#222", lw=0.7, capsize=2)
    ax.bar(x + w / 2, summary["oe_pass"], w,
           color="#c46a4a", edgecolor="#702c00", lw=0.6,
           label="Open-ended pass (judge ≥ 60)")
    ax.axhline(chance, color="#888", lw=0.9, linestyle="--", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy / pass rate")
    ax.set_ylim(0, 1)
    ax.set_title("A. Per-model interpretation accuracy",
                 fontsize=11, fontweight="bold", loc="left")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.18, linewidth=0.5)


def _panel_b(ax: plt.Axes, df: pd.DataFrame, model_order: List[str]) -> None:
    oe = df[(df["question_type"] == "open_ended") & df["judge_score"].notna()]
    data = [oe[oe["model"] == m]["judge_score"].to_numpy() for m in model_order]
    parts = ax.violinplot(data, positions=range(len(model_order)),
                          widths=0.78, showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#c46a4a")
        body.set_edgecolor("#702c00")
        body.set_alpha(0.55)
    if "cmedians" in parts:
        parts["cmedians"].set_color("#3a1200"); parts["cmedians"].set_lw(1.4)
    rng = np.random.RandomState(7)
    for i, vals in enumerate(data):
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.13, 0.13, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   s=14, color="#702c00", alpha=0.65,
                   edgecolors="white", linewidths=0.4, zorder=3)
    ax.axhline(OPEN_ENDED_PASS, color="#222", lw=1.0, linestyle="--",
               label="60-pt pass threshold")
    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(model_order, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Judge score (0–100)")
    ax.set_ylim(0, 100)
    ax.set_title("B. Open-ended judge-score distribution",
                 fontsize=11, fontweight="bold", loc="left")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.18, linewidth=0.5)


def _panel_c(fig: plt.Figure, gs_cell: gridspec.SubplotSpec,
             df: pd.DataFrame, model_order: List[str]) -> None:
    inner = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_cell, wspace=0.06)
    ax1 = fig.add_subplot(inner[0])
    ax2 = fig.add_subplot(inner[1])

    mcq = (df[df["question_type"] == "mcq"]
           .pivot_table(index="model", columns="dataset",
                        values="correct", aggfunc="mean"))
    oe = (df[df["question_type"] == "open_ended"]
          .pivot_table(index="model", columns="dataset",
                       values="judge_score", aggfunc="mean"))
    mcq = mcq.reindex(model_order)
    oe  = oe.reindex(model_order)
    cols = [_short_dataset(c) for c in mcq.columns]

    im1 = ax1.imshow(mcq.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax1.set_xticks(range(len(cols)))
    ax1.set_xticklabels(cols, rotation=45, ha="right", fontsize=7.4)
    ax1.set_yticks(range(len(model_order)))
    ax1.set_yticklabels(model_order, fontsize=8)
    ax1.set_title("C. Per-dataset breakdown — MCQ accuracy",
                  fontsize=11, fontweight="bold", loc="left")
    fig.colorbar(im1, ax=ax1, fraction=0.045, pad=0.02)
    for i in range(mcq.shape[0]):
        for j in range(mcq.shape[1]):
            v = mcq.values[i, j]
            if not np.isnan(v):
                ax1.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=6.6,
                         color="#1a1a1a" if v < 0.55 else "white")

    im2 = ax2.imshow(oe.values, aspect="auto", cmap="Reds", vmin=0, vmax=100)
    ax2.set_xticks(range(len(cols)))
    ax2.set_xticklabels(cols, rotation=45, ha="right", fontsize=7.4)
    ax2.set_yticks(range(len(model_order)))
    ax2.set_yticklabels([])
    ax2.set_title("Open-ended mean judge score",
                  fontsize=11, fontweight="bold", loc="left")
    fig.colorbar(im2, ax=ax2, fraction=0.045, pad=0.02)
    for i in range(oe.shape[0]):
        for j in range(oe.shape[1]):
            v = oe.values[i, j]
            if not np.isnan(v):
                ax2.text(j, i, f"{v:.0f}", ha="center", va="center",
                         fontsize=6.6,
                         color="#1a1a1a" if v < 55 else "white")


def _panel_d(ax: plt.Axes, summary: pd.DataFrame) -> None:
    x = summary["mcq_acc"].to_numpy()
    y = summary["oe_mean"].to_numpy()
    keep = ~np.isnan(x) & ~np.isnan(y)
    rho, p = stats.spearmanr(x[keep], y[keep])
    ax.scatter(x, y, s=58, c="#444", edgecolors="white", linewidths=0.7,
               zorder=3)
    for _, r in summary.iterrows():
        ax.annotate(r["model"], (r["mcq_acc"], r["oe_mean"]),
                    fontsize=7.2, alpha=0.85, xytext=(4, 4),
                    textcoords="offset points")
    ax.axhline(OPEN_ENDED_PASS, color="#888", lw=0.9, linestyle="--",
               alpha=0.7, zorder=1)
    ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 1,
            OPEN_ENDED_PASS + 1.0,
            "judge pass threshold",
            fontsize=7.5, ha="right", va="bottom", color="#666", alpha=0.85)
    txt = f"Spearman ρ = {rho:.2f}\np = {p:.3f}\nn = {int(keep.sum())} models"
    ax.text(0.04, 0.96, txt, transform=ax.transAxes,
            ha="left", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#fffae0", edgecolor="#c2a047", lw=0.8),
            zorder=4)
    ax.set_xlabel("MCQ accuracy")
    ax.set_ylabel("Open-ended mean judge score (0–100)")
    ax.set_title("D. MCQ vs open-ended performance per model",
                 fontsize=11, fontweight="bold", loc="left")
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.set_ylim(0, 100)


# ── Driver ───────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--metrics", required=True,
                    help="Path to bench_interpretation merged metrics.csv")
    ap.add_argument("--out", default="figure_interpretation",
                    help="Output basename (.pdf and .png are appended)")
    ap.add_argument("--chance", type=float, default=0.28,
                    help="MCQ chance baseline (default 0.28; range "
                         "depends on per-question option counts, 25–33 %)")
    ap.add_argument("--title",
                    default="Benchmark G — Biological interpretation: "
                            "multiple-choice and open-ended scores do not co-vary")
    args = ap.parse_args()

    df = pd.read_csv(args.metrics)
    df["correct"] = _bool_series(df["correct"])
    df["judge_score"] = pd.to_numeric(df["judge_score"], errors="coerce")

    summary = _summarise(df)
    model_order = summary["model"].tolist()
    judge_models = sorted(
        df.loc[df["question_type"] == "open_ended", "judge_model"]
          .dropna().unique()
    )
    n_datasets = df["dataset"].nunique()
    n_mcq      = (df["question_type"] == "mcq").sum() // max(1, len(model_order))
    n_oe       = (df["question_type"] == "open_ended").sum() // max(1, len(model_order))

    fig = plt.figure(figsize=(15.5, 11.5))
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.55, wspace=0.30,
        left=0.07, right=0.965, top=0.91, bottom=0.10,
    )
    _panel_a(fig.add_subplot(gs[0, 0]), summary, args.chance)
    _panel_b(fig.add_subplot(gs[0, 1]), df, model_order)
    _panel_c(fig, gs[1, 0], df, model_order)
    _panel_d(fig.add_subplot(gs[1, 1]), summary)

    fig.suptitle(args.title, fontsize=13.5, fontweight="bold", y=0.965)
    sub = (f"Open-ended responses scored by "
           f"{', '.join(judge_models) if judge_models else '(no judge model)'}; "
           f"corpus = {n_mcq} MCQ + {n_oe} open-ended across {n_datasets} datasets.")
    fig.text(0.5, 0.935, sub, ha="center", va="top", fontsize=9.2,
             color="#555")

    out = Path(args.out)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {out}.pdf")
    print(f"wrote {out}.png")


if __name__ == "__main__":
    main()
