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
#
# ``overall_pass`` is plotted as a fraction (= per-arm pass rate)
# alongside the continuous metrics so the headline gating outcome is
# visible on the figure, not buried in the stats TSV. The McNemar
# discordance counts (b_only / c_only) and p-value still come from
# ``_build_stats_table`` -- the bar panel just shows the marginal
# means with bootstrap CIs, matching every other panel's layout.
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

# Metrics also exported to stats_ablation.tsv.
# ``overall_pass`` is plotted (above) but uses a paired McNemar test, so
# it's listed in _BINARY_METRICS for stats; everything else is Wilcoxon.
_STATS_METRICS = [m for m, _, _ in _PLOT_METRICS if m != "overall_pass"]
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

    # ── Pre-compute per-metric paired n, and the modal n across panels.
    # When a metric is only defined on a subset of prompts (e.g.
    # ``preset_command_f1`` only on prompts with ``gold_preset``), its n
    # will be much smaller than the modal n. We highlight those panels
    # with a red title and an explicit "subset" tag so the reader
    # cannot misread a 5-paired-prompt subset panel as a 66-prompt
    # whole-corpus panel just because the bars look similar.
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
    # A panel is a "subset" if its n is at least 30% smaller than the
    # full-corpus n. The 30% threshold catches small-tier metrics
    # (preset_*, fidelity-only, etc.) without flagging panels that are
    # one or two cells shy of full because a couple of cells errored.
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

        bars = ax.bar(x, means, yerr=yerr, capsize=4,
                      color=["#1f77b4", "#d62728"],
                      edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(["DAG-aware", "DAG-blind"])

        # Title carries n inline so it's adjacent to the metric name
        # and impossible to miss. Red for subset panels.
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

        # In-panel annotation: explicit n + subset warning. Larger,
        # darker, anchored top-left of the plot area to avoid the bars.
        annotation_lines = [f"n_aware = {n_a}", f"n_blind = {n_b}"]
        if is_subset:
            annotation_lines.append(f"({100 * max(n_a, n_b) / max(modal_n, 1):.0f}% of corpus)")
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
