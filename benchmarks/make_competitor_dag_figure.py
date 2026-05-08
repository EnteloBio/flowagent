"""Figure + paired statistics for the competitor DAG-awareness ablation
(Benchmark J).

Reads ``paired_metrics.csv`` produced by
``bench_competitor_dag_ablation.py`` and emits, **per competitor**:

* ``figure_competitor_dag__<comp>.pdf`` / ``.png`` -- one panel per
  metric showing arm means with bootstrap 95% CIs.
* ``figure_competitor_dag__<comp>__stats.tsv`` -- per-metric
  paired-prompt comparison (Wilcoxon signed-rank for continuous,
  McNemar for ``overall_pass``). Rows are matched on
  ``(model, input_id, replicate)`` -- same prompt graded by both arms.

This is a competitor-aware sibling of ``make_ablation_figure.py``: it
keeps the same metric set, the same statistical tests, and the same
visual layout so reviewers can compare FlowAgent's H ablation
side-by-side with the competitor J ablation. Differences:

* Ingests the cross-competitor ``paired_metrics.csv`` (rows tagged
  with a ``competitor`` column) and renders one PDF per competitor.
  Pass ``--competitor=<slug>`` to render only one.
* The arm labels on the x-axis read ``"DAG-aware prompt"`` and
  ``"DAG-blind prompt"`` -- explicit about what's being toggled (a
  prompt template, not a schema flag like FlowAgent's H).

Usage::

    # Render every competitor in the CSV:
    python make_competitor_dag_figure.py \\
        --paired results/competitor_dag_ablation/<ts>/paired_metrics.csv \\
        --out figure_competitor_dag

    # Single competitor:
    python make_competitor_dag_figure.py \\
        --paired results/competitor_dag_ablation/<ts>/claude_code/paired_metrics.csv \\
        --competitor=claude_code \\
        --out figure_competitor_dag__claude_code
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Same metric set as Benchmark H so the two figures are directly
# comparable. (key, label, axis kind).
_PLOT_METRICS: List[Tuple[str, str, str]] = [
    # Row 1 -- gating + tool quality
    ("overall_pass",           "Overall pass rate",        "fraction"),
    ("tools_present_fraction", "Tools present fraction",   "fraction"),
    ("hallucination_rate",     "Hallucination rate",       "fraction"),
    # Row 2 -- preset fidelity (subset metrics, flagged in red by _plot)
    ("preset_command_f1",      "Preset command F1",        "fraction"),
    ("preset_name_jaccard",    "Preset name Jaccard",      "fraction"),
    ("num_steps",              "Step count",               "count"),
    # Row 3 -- structural shape (sanity for the ablation)
    ("dag_edge_density",       "DAG edge density",         "open"),
    ("parallel_width",         "Parallel width",           "count"),
    ("stage_efficiency",       "Stage efficiency",         "open"),
]

_STATS_METRICS = [m for m, _, _ in _PLOT_METRICS if m != "overall_pass"]
_BINARY_METRICS = ["overall_pass"]

# Pair join key. Note: ``model`` is the competitor's driver model
# (e.g. ``claude-haiku-4-5``), constant within a single sweep, so the
# joined CSV typically has a single value here. Including it in the
# key keeps the join compatible with multi-model sweeps.
_PAIR_KEY = ("model", "input_id", "replicate")


# ── Paired stats (mirror of make_ablation_figure) ────────────────────

def _bootstrap_ci(values: np.ndarray, *, n_boot: int = 2000,
                  alpha: float = 0.05, seed: int = 42) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    if values.size == 1:
        v = float(values[0])
        return v, v, v
    rng = np.random.default_rng(seed)
    boots = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)


def _wilcoxon(diffs: np.ndarray) -> Dict[str, float]:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    n = int(diffs.size)
    if n == 0:
        return {"n": 0, "mean_diff": float("nan"),
                "median_diff": float("nan"), "p_value": float("nan"),
                "test": "wilcoxon_skipped"}
    mean_diff = float(diffs.mean())
    median_diff = float(np.median(diffs))
    nonzero = np.count_nonzero(diffs)
    if nonzero == 0:
        return {"n": n, "mean_diff": mean_diff, "median_diff": median_diff,
                "p_value": 1.0, "test": "all-zero"}
    try:
        from scipy.stats import wilcoxon as _w  # type: ignore
        stat = _w(diffs, zero_method="wilcox", alternative="two-sided")
        return {"n": n, "mean_diff": mean_diff, "median_diff": median_diff,
                "p_value": float(stat.pvalue), "test": "wilcoxon"}
    except Exception:
        return {"n": n, "mean_diff": mean_diff, "median_diff": median_diff,
                "p_value": float("nan"), "test": "wilcoxon_unavailable"}


def _mcnemar_binary(arm_a: np.ndarray, arm_b: np.ndarray) -> Dict[str, float]:
    a = np.asarray(arm_a).astype(bool)
    b = np.asarray(arm_b).astype(bool)
    n = int(min(a.size, b.size))
    if n == 0:
        return {"n": 0, "b_only": 0, "c_only": 0,
                "p_value": float("nan"), "test": "mcnemar_skipped"}
    b_only = int(np.sum(~a & b))   # B improves, A fails
    c_only = int(np.sum(a & ~b))   # A improves, B fails
    if b_only + c_only == 0:
        return {"n": n, "b_only": b_only, "c_only": c_only,
                "p_value": 1.0, "test": "all-concordant"}
    try:
        from math import comb
        n_disc = b_only + c_only
        k = max(b_only, c_only)
        tail = sum(comb(n_disc, i) for i in range(k, n_disc + 1)) / (2 ** n_disc)
        return {"n": n, "b_only": b_only, "c_only": c_only,
                "p_value": float(min(1.0, 2 * tail)), "test": "mcnemar"}
    except Exception:
        return {"n": n, "b_only": b_only, "c_only": c_only,
                "p_value": float("nan"), "test": "mcnemar_unavailable"}


# ── Pair-up + plot ───────────────────────────────────────────────────

def _pair_arms(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "arm" not in df.columns:
        raise SystemExit("paired_metrics.csv missing 'arm' column")
    aware = df[df["arm"] == "dag_aware"].copy()
    blind = df[df["arm"] == "dag_blind"].copy()
    if aware.empty or blind.empty:
        raise SystemExit(
            f"missing arm rows: aware={len(aware)}, blind={len(blind)}")
    aware = aware.set_index(list(_PAIR_KEY))
    blind = blind.set_index(list(_PAIR_KEY))
    common = aware.index.intersection(blind.index)
    return aware.loc[common], blind.loc[common]


def _plot(df: pd.DataFrame, *, out_base: Path, title_suffix: str) -> Path:
    aware = df[df["arm"] == "dag_aware"]
    blind = df[df["arm"] == "dag_blind"]

    n_metrics = len(_PLOT_METRICS)
    ncols = 3
    nrows = int(np.ceil(n_metrics / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.4 * nrows))
    axes = np.atleast_2d(axes).flatten()

    # Pre-compute per-panel n; flag subset panels (preset_*) in red.
    panel_ns: List[Tuple[int, int]] = []
    for key, _label, _kind in _PLOT_METRICS:
        if key not in df.columns:
            panel_ns.append((0, 0))
            continue
        a_vals = pd.to_numeric(aware.get(key), errors="coerce").to_numpy()
        b_vals = pd.to_numeric(blind.get(key), errors="coerce").to_numpy()
        n_a = int(np.sum(~np.isnan(a_vals)))
        n_b = int(np.sum(~np.isnan(b_vals)))
        panel_ns.append((n_a, n_b))
    valid_ns = [max(a, b) for a, b in panel_ns if max(a, b) > 0]
    modal_n = int(max(valid_ns)) if valid_ns else 0
    subset_threshold = max(1, int(modal_n * 0.7)) if modal_n else 0

    for ax, ((key, label, kind), (n_a, n_b)) in zip(
        axes, zip(_PLOT_METRICS, panel_ns),
    ):
        if key not in df.columns:
            ax.set_visible(False)
            continue

        a_vals = pd.to_numeric(aware.get(key), errors="coerce").to_numpy()
        b_vals = pd.to_numeric(blind.get(key), errors="coerce").to_numpy()

        a_mean, a_lo, a_hi = _bootstrap_ci(a_vals)
        b_mean, b_lo, b_hi = _bootstrap_ci(b_vals)

        x = np.array([0, 1])
        means = np.array([a_mean, b_mean])
        lo = np.array([a_lo, b_lo])
        hi = np.array([a_hi, b_hi])
        yerr = np.vstack([means - lo, hi - means])

        ax.bar(x, means, yerr=yerr, capsize=4,
               color=["#1e40af", "#94a3b8"],   # indigo (DAG-aware) / slate (blind)
               edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        # Explicit labels make it obvious what's being toggled: a
        # prompt template, not a schema (unlike FlowAgent's H).
        ax.set_xticklabels(["DAG-aware\nprompt", "DAG-blind\nprompt"],
                           fontsize=9)

        is_subset = (
            modal_n > 0
            and 0 < max(n_a, n_b) < subset_threshold
        )
        n_disp = n_a if n_a == n_b else f"{n_a}/{n_b}"
        title_n_suffix = (
            f"\n(subset: n={n_disp} of {modal_n})" if is_subset
            else f"  (n={n_disp})"
        )
        ax.set_title(
            label + title_n_suffix,
            fontsize=10,
            color="#b22222" if is_subset else "black",
            fontweight="bold" if is_subset else "normal",
        )
        if kind == "fraction":
            ax.set_ylim(0.0, 1.0 if max(means) <= 1 else max(means) * 1.1)
        elif kind == "count":
            ax.set_ylim(0, max(1.0, max(hi) * 1.15))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

        annotation_lines = [f"n_aware = {n_a}", f"n_blind = {n_b}"]
        if is_subset:
            annotation_lines.append(
                f"({100 * max(n_a, n_b) / max(modal_n, 1):.0f}% of corpus)")
        ax.text(
            0.04, 0.96,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=8.5,
            color="#b22222" if is_subset else "#333333",
            fontweight="bold" if is_subset else "normal",
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="white", alpha=0.85,
                      edgecolor="#b22222" if is_subset else "#cccccc",
                      linewidth=0.8),
        )

    for k in range(len(_PLOT_METRICS), len(axes)):
        axes[k].set_visible(False)

    fig.suptitle(
        f"Competitor DAG-prompt ablation -- {title_suffix}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    pdf_path = out_base.with_suffix(".pdf")
    png_path = out_base.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return pdf_path


def _build_stats_table(aware: pd.DataFrame, blind: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for key in _STATS_METRICS:
        if key not in aware.columns or key not in blind.columns:
            continue
        a_vals = pd.to_numeric(aware[key], errors="coerce").to_numpy()
        b_vals = pd.to_numeric(blind[key], errors="coerce").to_numpy()
        diffs = a_vals - b_vals
        stats = _wilcoxon(diffs)
        rows.append({
            "metric": key,
            "kind": "continuous",
            "mean_dag_aware": float(np.nanmean(a_vals)) if a_vals.size else float("nan"),
            "mean_dag_blind": float(np.nanmean(b_vals)) if b_vals.size else float("nan"),
            **stats,
        })
    for key in _BINARY_METRICS:
        if key not in aware.columns or key not in blind.columns:
            continue
        a_vals = aware[key].to_numpy()
        b_vals = blind[key].to_numpy()
        # Cast booleans to float so subtraction in stats helpers is well-typed.
        a_float = a_vals.astype(bool).astype(float)
        b_float = b_vals.astype(bool).astype(float)
        stats = _mcnemar_binary(a_vals, b_vals)
        rows.append({
            "metric": key,
            "kind": "binary",
            "mean_dag_aware": float(np.mean(a_float)) if a_float.size else float("nan"),
            "mean_dag_blind": float(np.mean(b_float)) if b_float.size else float("nan"),
            "mean_diff": float("nan"),
            "median_diff": float("nan"),
            **stats,
        })
    return pd.DataFrame(rows)


def _render_for_competitor(df_comp: pd.DataFrame, *, out_base: Path,
                           competitor: str) -> None:
    aware, blind = _pair_arms(df_comp)
    if aware.empty:
        print(f"[skip] {competitor}: no paired rows after joining arms.")
        return

    pdf_path = _plot(df_comp, out_base=out_base, title_suffix=competitor)
    stats = _build_stats_table(aware, blind)
    stats_path = Path(str(out_base) + "__stats.tsv")
    stats.to_csv(stats_path, sep="\t", index=False, float_format="%.6g")

    print(f"\n=== {competitor} ===")
    print(f"[ok] figure -> {pdf_path}")
    print(f"[ok] stats  -> {stats_path}")
    print()
    print(stats.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--paired", required=True,
        help="Path to paired_metrics.csv (cross-competitor or per-comp).",
    )
    ap.add_argument(
        "--competitor", default=None,
        help="Render only this competitor (default: render every competitor "
             "found in the CSV).",
    )
    ap.add_argument(
        "--out", default="figure_competitor_dag",
        help="Output basename. Per-competitor outputs are written as "
             "``<out>__<comp>.pdf`` / ``.png`` and "
             "``<out>__<comp>__stats.tsv``.",
    )
    args = ap.parse_args()

    paired_path = Path(args.paired)
    if not paired_path.exists():
        raise SystemExit(f"paired_metrics.csv not found: {paired_path}")

    df = pd.read_csv(paired_path)
    # Per-competitor CSVs from bench_competitor_dag_ablation don't include
    # a ``competitor`` column. The cross-competitor CSV does. Synthesise
    # one when missing so the per-competitor branch below works either
    # way.
    if "competitor" not in df.columns:
        if args.competitor:
            df["competitor"] = args.competitor
        else:
            df["competitor"] = "claude_code"

    competitors = (
        [args.competitor]
        if args.competitor
        else sorted(df["competitor"].dropna().unique())
    )
    if not competitors:
        raise SystemExit("No competitor rows in paired CSV.")

    for comp in competitors:
        sub = df[df["competitor"] == comp]
        if sub.empty:
            print(f"[skip] no rows for competitor={comp!r}")
            continue
        out_base = Path(f"{args.out}__{comp}")
        _render_for_competitor(sub, out_base=out_base, competitor=comp)


if __name__ == "__main__":
    main()
