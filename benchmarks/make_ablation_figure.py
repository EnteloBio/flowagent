"""Figure + paired statistics for the DAG-awareness ablation (Benchmark H).

Reads ``paired_metrics.csv`` produced by ``bench_ablation.py`` and emits:

* ``figure_ablation.pdf`` / ``.png`` -- one panel per metric showing
  arm means with bootstrap 95% CIs.
* ``stats_ablation.tsv`` -- per-metric paired-prompt comparison
  (Wilcoxon signed-rank for continuous metrics, McNemar for binary
  ``overall_pass``). Rows are matched on
  ``(model, input_id, replicate)`` so the same prompt is graded by
  the DAG-aware and DAG-blind planner under identical seeds.

Metrics covered (all written to ``stats_ablation.tsv`` regardless of
whether they are plotted):

  - tools_present_fraction      -- tool-coverage rate
  - hallucination_rate          -- fraction of plan tools we don't recognise
  - preset_command_f1           -- token-F1 vs the preset's gold command
  - preset_name_jaccard         -- step-name Jaccard vs the preset
  - dag_edge_density            -- edges / (steps - 1); ablation sanity
  - parallel_width              -- max width of a topological layer
  - num_steps                   -- raw step count
  - overall_pass                -- gating boolean (McNemar)

Usage::

    python make_ablation_figure.py \\
        --paired results/ablation/<ts>/paired_metrics.csv \\
        --out figure_ablation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Metrics shown on the figure. (key, display label, axis kind).
# ``axis`` tells the plotter how to format the y-axis: "fraction"
# means 0..1, "count" means integer counts, "open" means free.
_PLOT_METRICS: List[Tuple[str, str, str]] = [
    ("tools_present_fraction", "Tools present fraction",  "fraction"),
    ("hallucination_rate",     "Hallucination rate",       "fraction"),
    ("preset_command_f1",      "Preset command F1",        "fraction"),
    ("dag_edge_density",       "DAG edge density",         "open"),
    ("parallel_width",         "Parallel width",           "count"),
    ("num_steps",              "Step count",               "count"),
]

# Metrics also exported to stats_ablation.tsv.
_STATS_METRICS = [m for m, _, _ in _PLOT_METRICS] + [
    "preset_name_jaccard",
]
_BINARY_METRICS = ["overall_pass"]


def _bootstrap_ci(values: np.ndarray, *, n_boot: int = 2000,
                  alpha: float = 0.05, seed: int = 42) -> Tuple[float, float, float]:
    """Return (mean, lo, hi) using percentile bootstrap; gracefully
    handles all-NaN, all-equal, and empty arrays."""
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


# ── Paired stats ───────────────────────────────────────────────────

def _wilcoxon(diffs: np.ndarray) -> Dict[str, float]:
    """Wilcoxon signed-rank on paired differences. Falls back to
    NaN p-value if scipy is missing or all diffs are zero."""
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    n = int(diffs.size)
    if n == 0:
        return {"n": 0, "mean_diff": float("nan"),
                "median_diff": float("nan"), "p_value": float("nan"),
                "test": "wilcoxon_skipped"}
    mean_diff = float(diffs.mean())
    median_diff = float(np.median(diffs))
    p_value: float
    test = "wilcoxon"
    nonzero = np.count_nonzero(diffs)
    if nonzero == 0:
        # All paired diffs zero -- no signal. Reporting p=1 is the
        # convention.
        p_value = 1.0
        test = "all-zero"
    else:
        try:
            from scipy.stats import wilcoxon as _w  # type: ignore
            stat = _w(diffs, zero_method="wilcox", alternative="two-sided")
            p_value = float(stat.pvalue)
        except Exception:
            p_value = float("nan")
            test = "wilcoxon_unavailable"
    return {"n": n, "mean_diff": mean_diff, "median_diff": median_diff,
            "p_value": p_value, "test": test}


def _mcnemar_binary(arm_a: np.ndarray, arm_b: np.ndarray) -> Dict[str, float]:
    """McNemar's test on paired binary outcomes (arm_a vs arm_b).

    Reports the discordant counts ``b`` (a=False, b=True) and ``c``
    (a=True, b=False) so the direction of any effect is readable.
    """
    a = np.asarray(arm_a).astype(bool)
    b = np.asarray(arm_b).astype(bool)
    n = int(min(a.size, b.size))
    if n == 0:
        return {"n": 0, "b_only": 0, "c_only": 0,
                "p_value": float("nan"), "test": "mcnemar_skipped"}
    b_only = int(np.sum(~a & b))   # B improves, A fails
    c_only = int(np.sum(a & ~b))   # A improves, B fails
    test = "mcnemar"
    p_value: float
    if b_only + c_only == 0:
        p_value = 1.0
        test = "all-concordant"
    else:
        try:
            # Exact binomial: P(X >= max(b,c) | n=b+c, p=0.5) * 2.
            from math import comb
            n_disc = b_only + c_only
            k = max(b_only, c_only)
            tail = sum(comb(n_disc, i) for i in range(k, n_disc + 1)) / (2 ** n_disc)
            p_value = float(min(1.0, 2 * tail))
        except Exception:
            p_value = float("nan")
            test = "mcnemar_unavailable"
    return {"n": n, "b_only": b_only, "c_only": c_only,
            "p_value": p_value, "test": test}


# ── Pair-up by (model, input_id, replicate) ────────────────────────

_PAIR_KEY = ("model", "input_id", "replicate")


def _pair_arms(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "arm" not in df.columns:
        raise SystemExit("paired_metrics.csv missing 'arm' column")
    aware = df[df["arm"] == "dag_aware"].copy()
    blind = df[df["arm"] == "dag_blind"].copy()
    if aware.empty:
        raise SystemExit("no dag_aware rows in paired_metrics.csv")
    if blind.empty:
        raise SystemExit("no dag_blind rows in paired_metrics.csv")
    # Inner join on the pair key.
    aware = aware.set_index(list(_PAIR_KEY))
    blind = blind.set_index(list(_PAIR_KEY))
    common = aware.index.intersection(blind.index)
    return aware.loc[common], blind.loc[common]


# ── Plot ───────────────────────────────────────────────────────────

def _plot(df: pd.DataFrame, *, out_base: Path) -> Path:
    aware = df[df["arm"] == "dag_aware"]
    blind = df[df["arm"] == "dag_blind"]

    n_metrics = len(_PLOT_METRICS)
    ncols = 3
    nrows = int(np.ceil(n_metrics / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.4 * nrows))
    axes = np.atleast_2d(axes).flatten()

    for ax, (key, label, kind) in zip(axes, _PLOT_METRICS):
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

        bars = ax.bar(x, means, yerr=yerr, capsize=4,
                      color=["#1f77b4", "#d62728"],
                      edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(["DAG-aware", "DAG-blind"])
        ax.set_title(label, fontsize=10)
        if kind == "fraction":
            ax.set_ylim(0.0, 1.0 if max(means) <= 1 else max(means) * 1.1)
        elif kind == "count":
            ax.set_ylim(0, max(1.0, max(hi) * 1.15))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

        # Annotate with a small sample size note.
        n_a = int(np.sum(~np.isnan(a_vals)))
        n_b = int(np.sum(~np.isnan(b_vals)))
        ax.text(
            0.98, 0.96,
            f"n_aware={n_a}\nn_blind={n_b}",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color="grey",
        )

    # Hide unused axes
    for k in range(len(_PLOT_METRICS), len(axes)):
        axes[k].set_visible(False)

    fig.suptitle("DAG awareness ablation -- planning-only", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    pdf_path = out_base.with_suffix(".pdf")
    png_path = out_base.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return pdf_path


# ── TSV stats ──────────────────────────────────────────────────────

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
        stats = _mcnemar_binary(a_vals, b_vals)
        rows.append({
            "metric": key,
            "kind": "binary",
            "mean_dag_aware": float(np.mean(a_vals.astype(bool))) if a_vals.size else float("nan"),
            "mean_dag_blind": float(np.mean(b_vals.astype(bool))) if b_vals.size else float("nan"),
            "mean_diff": float("nan"),
            "median_diff": float("nan"),
            **stats,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--paired", required=True,
        help="Path to paired_metrics.csv produced by bench_ablation.py",
    )
    ap.add_argument("--out", default="figure_ablation",
                    help="Output basename (writes .pdf and .png)")
    ap.add_argument("--stats-out", default=None,
                    help="Path for stats_ablation.tsv "
                         "(default: <out>__stats.tsv)")
    args = ap.parse_args()

    paired_path = Path(args.paired)
    if not paired_path.exists():
        raise SystemExit(f"paired_metrics.csv not found: {paired_path}")

    df = pd.read_csv(paired_path)
    aware, blind = _pair_arms(df)

    if aware.empty:
        raise SystemExit("No paired rows after joining dag_aware vs dag_blind.")

    out_base = Path(args.out)
    pdf_path = _plot(df, out_base=out_base)

    stats_path = (Path(args.stats_out)
                  if args.stats_out
                  else Path(str(out_base) + "__stats.tsv"))
    stats = _build_stats_table(aware, blind)
    stats.to_csv(stats_path, sep="\t", index=False, float_format="%.6g")

    print(f"[ok] figure -> {pdf_path}")
    print(f"[ok] stats  -> {stats_path}")
    print()
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
