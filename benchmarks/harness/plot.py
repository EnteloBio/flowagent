"""Figure generation for benchmark results.

Produces the PDFs / PNGs referenced by the manuscript. Each function
consumes a pandas.DataFrame (loaded from ``metrics.csv``) and returns a
matplotlib Figure.

Usage:
    python -m harness.plot --results=results
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# ── Benchmark A: planning correctness ────────────────────────────

def planning_figure(df: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart: models × {type_correct, tools_present, dag_valid}.

    Averaged across prompts and replicates; error bars = standard deviation.
    """
    metrics = ["type_correct", "tools_present_fraction",
               "dag_valid", "no_forbidden_tools", "overall_pass"]
    metrics = [m for m in metrics if m in df.columns]
    agg = (df.groupby("model")[metrics]
             .agg(["mean", "std"])
             .reset_index())

    fig, ax = plt.subplots(figsize=(10, 5))
    models = agg["model"].tolist()
    n_metrics = len(metrics)
    x = range(len(models))
    width = 0.8 / max(n_metrics, 1)

    for i, metric in enumerate(metrics):
        means = agg[(metric, "mean")].astype(float).tolist()
        stds = agg[(metric, "std")].fillna(0).astype(float).tolist()
        offsets = [xi + i * width - 0.4 + width / 2 for xi in x]
        ax.bar(offsets, means, width, yerr=stds, label=metric, capsize=3)

    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0–1)")
    ax.set_title("Benchmark A — Planning correctness across LLMs")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ── Benchmark B: error recovery ──────────────────────────────────

def recovery_figure(df: pd.DataFrame) -> plt.Figure:
    """Stacked bar per fault: recovered@1 | recovered@2 | recovered@3 | failed."""
    if "fault" not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No recovery data", ha="center", va="center")
        return fig

    def _bucket(row):
        if not row.get("recovered"):
            return "failed"
        att = row.get("attempts")
        if att in (1, "1", 1.0): return "recovered@1"
        if att in (2, "2", 2.0): return "recovered@2"
        if att in (3, "3", 3.0): return "recovered@3"
        return "recovered@other"

    df = df.copy()
    df["bucket"] = df.apply(_bucket, axis=1)
    pivot = (df.groupby(["fault", "bucket"]).size()
               .unstack(fill_value=0))
    # Normalise to fractions
    pivot = pivot.div(pivot.sum(axis=1), axis=0)

    order = ["recovered@1", "recovered@2", "recovered@3",
             "recovered@other", "failed"]
    pivot = pivot.reindex(columns=[c for c in order if c in pivot.columns],
                          fill_value=0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax,
               color=["#2ecc71", "#f1c40f", "#e67e22",
                      "#95a5a6", "#e74c3c"][: len(pivot.columns)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of runs")
    ax.set_title("Benchmark B — LLM-driven recovery rate per fault class")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    fig.tight_layout()
    return fig


# ── Benchmark C: generation fidelity ─────────────────────────────

def generation_figure(df: pd.DataFrame) -> plt.Figure:
    """Pass/fail table as a heatmap (preset × generator × check)."""
    if "generator" not in df.columns or "plan_id" not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No generation data", ha="center", va="center")
        return fig

    checks = [c for c in (
        "validation_ok", "step_count_matches", "dag_isomorphic",
        "tools_preserved", "regression_launchdir_quoted"
    ) if c in df.columns]

    df = df.copy()
    df["row"] = df["plan_id"] + " × " + df["generator"]
    heat = df.set_index("row")[checks].astype(float)

    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(heat))))
    im = ax.imshow(heat.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(checks)))
    ax.set_xticklabels(checks, rotation=30, ha="right")
    ax.set_yticks(range(len(heat)))
    ax.set_yticklabels(heat.index, fontsize=8)
    ax.set_title("Benchmark C — Generator fidelity")
    fig.colorbar(im, ax=ax, label="pass (1) / fail (0)")
    fig.tight_layout()
    return fig


# ── Benchmark D: executor coverage ───────────────────────────────

def executor_matrix_figure(df: pd.DataFrame) -> plt.Figure:
    if "executor" not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No executor data", ha="center", va="center")
        return fig

    levels = [c for c in ("interface_ok", "mock_ok", "live_ok") if c in df.columns]
    mat = df.set_index("executor")[levels]
    # Map each cell: true → 1.0, false → 0.0, anything else (None/NaN) → 0.5.
    # CSV round-trip means values can be bool, int, or the strings
    # "True"/"False". ``applymap`` was removed in pandas 2.2 so use ``map``.
    def _score(v):
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("true", "1"):  return 1.0
            if s in ("false", "0"): return 0.0
            return 0.5
        if v is True or v == 1:  return 1.0
        if v is False or v == 0: return 0.0
        return 0.5
    mat_num = mat.map(_score) if hasattr(mat, "map") else mat.applymap(_score)

    fig, ax = plt.subplots(figsize=(6, max(2, 0.5 * len(mat))))
    im = ax.imshow(mat_num.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels)
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index)
    for i in range(mat_num.shape[0]):
        for j in range(mat_num.shape[1]):
            score = mat_num.iloc[i, j]
            label = "✓" if score == 1.0 else ("✗" if score == 0.0 else "—")
            ax.text(j, i, label, ha="center", va="center",
                    color="white" if score != 0.5 else "black")
    ax.set_title("Benchmark D — Executor coverage")
    fig.tight_layout()
    return fig


# ── CLI: regenerate all figures ──────────────────────────────────

def _latest(run_dir: Path) -> Optional[Path]:
    if not run_dir.exists():
        return None
    subs = [p for p in run_dir.iterdir() if p.is_dir()]
    return max(subs, key=lambda p: p.stat().st_mtime) if subs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    root = Path(args.results)
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for bench_name, fn in (
        ("planning",   planning_figure),
        ("recovery",   recovery_figure),
        ("generation", generation_figure),
        ("executors",  executor_matrix_figure),
    ):
        latest = _latest(root / bench_name)
        if latest is None:
            print(f"[skip] no results for {bench_name}")
            continue
        csv = latest / "metrics.csv"
        if not csv.exists() or csv.stat().st_size == 0:
            print(f"[skip] empty metrics for {bench_name}")
            continue
        df = pd.read_csv(csv)
        if df.empty:
            print(f"[skip] empty dataframe for {bench_name}")
            continue
        fig = fn(df)
        fig.savefig(fig_dir / f"{bench_name}.pdf")
        fig.savefig(fig_dir / f"{bench_name}.png", dpi=150)
        plt.close(fig)
        print(f"[ok]   {bench_name} → {fig_dir/bench_name}.pdf")


if __name__ == "__main__":
    main()
