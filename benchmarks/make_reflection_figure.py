"""Figure + paired statistics for the completeness-reflection ablation.

Companion to ``make_ablation_figure.py``. Reads the
``paired_metrics.csv`` produced by ``bench_reflection_ablation.py``
and writes:

* ``figure_reflection.pdf`` / ``.png`` -- one panel per metric showing
  per-arm means with bootstrap 95% CIs. Headline metrics are
  ``completeness_pass`` and ``stage_efficiency`` (the two new metrics
  introduced alongside the reflection loop), followed by the existing
  ``overall_pass``, ``dag_edge_density``, ``parallel_width`` for
  reference.

* ``stats_reflection.tsv`` -- per-metric paired comparison
  (Wilcoxon signed-rank for continuous metrics; McNemar for binary
  pass-rates). Rows are matched on ``(model, input_id, replicate)``
  so ``reflect_on`` and ``reflect_off`` see the same prompt under the
  same seed.

Usage::

    python make_reflection_figure.py \\
        --paired results/reflection/<ts>/paired_metrics.csv \\
        --out figure_reflection
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# (key, display label, axis kind). ``axis``: "fraction" 0..1, "count"
# integer, "open" free-scale.
_PLOT_METRICS: List[Tuple[str, str, str]] = [
    ("completeness_pass",          "Completeness pass rate",   "fraction"),
    ("stage_efficiency",           "Stage efficiency",          "open"),
    ("overall_pass",               "Overall pass rate",         "fraction"),
    ("dag_edge_density",           "DAG edge density",          "open"),
    ("parallel_width",             "Parallel width",            "count"),
    ("num_completeness_failures",  "Completeness failures",     "count"),
    ("hallucination_rate",         "Hallucination rate",        "fraction"),
    ("tools_present_fraction",     "Tools present fraction",    "fraction"),
    ("num_steps",                  "Step count",                "count"),
]

_STATS_CONTINUOUS = [
    "completeness_pass",
    "stage_efficiency",
    "stage_efficiency_raw",
    "dag_edge_density",
    "parallel_width",
    "num_completeness_failures",
    "hallucination_rate",
    "tools_present_fraction",
    "num_steps",
    "completeness_attempts",
]

_STATS_BINARY = ["overall_pass", "completeness_pass"]


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
        p_value = float(stat.pvalue)
        return {"n": n, "mean_diff": mean_diff, "median_diff": median_diff,
                "p_value": p_value, "test": "wilcoxon"}
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
        p_value = float(min(1.0, 2 * tail))
        return {"n": n, "b_only": b_only, "c_only": c_only,
                "p_value": p_value, "test": "mcnemar"}
    except Exception:
        return {"n": n, "b_only": b_only, "c_only": c_only,
                "p_value": float("nan"), "test": "mcnemar_unavailable"}


_PAIR_KEY = ("model", "input_id", "replicate")


def _pair_arms(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "arm" not in df.columns:
        raise SystemExit("paired_metrics.csv missing 'arm' column")
    on = df[df["arm"] == "reflect_on"].copy()
    off = df[df["arm"] == "reflect_off"].copy()
    if on.empty:
        raise SystemExit("no reflect_on rows in paired_metrics.csv")
    if off.empty:
        raise SystemExit("no reflect_off rows in paired_metrics.csv")
    on = on.set_index(list(_PAIR_KEY))
    off = off.set_index(list(_PAIR_KEY))
    common = on.index.intersection(off.index)
    return on.loc[common], off.loc[common]


def _plot(df: pd.DataFrame, *, out_base: Path) -> Path:
    on = df[df["arm"] == "reflect_on"]
    off = df[df["arm"] == "reflect_off"]

    n_metrics = len(_PLOT_METRICS)
    ncols = 3
    nrows = int(np.ceil(n_metrics / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.4 * nrows))
    axes = np.atleast_2d(axes).flatten()

    # Pre-compute per-metric paired n + modal n; flag subset panels in
    # red so the reader can't misread a small-subset panel's bars as
    # full-corpus. Mirrors make_ablation_figure.py's _plot.
    panel_ns: List[Tuple[int, int]] = []
    for key, _label, _kind in _PLOT_METRICS:
        if key not in df.columns:
            panel_ns.append((0, 0))
            continue
        a_vals = pd.to_numeric(on.get(key), errors="coerce").to_numpy()
        b_vals = pd.to_numeric(off.get(key), errors="coerce").to_numpy()
        panel_ns.append((
            int(np.sum(~np.isnan(a_vals))),
            int(np.sum(~np.isnan(b_vals))),
        ))
    valid_ns = [max(a, b) for a, b in panel_ns if max(a, b) > 0]
    modal_n = int(max(valid_ns)) if valid_ns else 0
    subset_threshold = max(1, int(modal_n * 0.7)) if modal_n else 0

    for ax, ((key, label, kind), (n_a, n_b)) in zip(
        axes, zip(_PLOT_METRICS, panel_ns),
    ):
        if key not in df.columns:
            ax.set_visible(False)
            continue

        a_vals = pd.to_numeric(on.get(key), errors="coerce").to_numpy()
        b_vals = pd.to_numeric(off.get(key), errors="coerce").to_numpy()

        a_mean, a_lo, a_hi = _bootstrap_ci(a_vals)
        b_mean, b_lo, b_hi = _bootstrap_ci(b_vals)

        x = np.array([0, 1])
        means = np.array([a_mean, b_mean])
        lo = np.array([a_lo, b_lo])
        hi = np.array([a_hi, b_hi])
        # Replace NaNs with means for the err arrays so matplotlib doesn't choke.
        yerr = np.vstack([
            np.nan_to_num(means - lo, nan=0.0),
            np.nan_to_num(hi - means, nan=0.0),
        ])

        ax.bar(
            x, np.nan_to_num(means, nan=0.0),
            yerr=yerr, capsize=4,
            color=["#2ca02c", "#7f7f7f"],
            edgecolor="black", linewidth=0.6,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(["reflect_on", "reflect_off"])

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
            ax.set_ylim(0.0, 1.0 if (np.nanmax(means) <= 1) else float(np.nanmax(means)) * 1.1)
        elif kind == "count":
            ymax = float(np.nanmax(hi)) if np.isfinite(hi).any() else 1.0
            ax.set_ylim(0, max(1.0, ymax * 1.15))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

        annotation_lines = [f"n_on  = {n_a}", f"n_off = {n_b}"]
        if is_subset:
            annotation_lines.append(
                f"({100 * max(n_a, n_b) / max(modal_n, 1):.0f}% of corpus)"
            )
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
        "Completeness-reflection ablation -- planning-only "
        "(reflect_on = DAG-Plan-style structural rules + LLM retry)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    pdf_path = out_base.with_suffix(".pdf")
    png_path = out_base.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return pdf_path


def _build_stats_table(on: pd.DataFrame, off: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for key in _STATS_CONTINUOUS:
        if key not in on.columns or key not in off.columns:
            continue
        # Cast to float explicitly so booleans (e.g. completeness_pass)
        # support subtraction without raising.
        a_vals = pd.to_numeric(on[key], errors="coerce").to_numpy(dtype=float)
        b_vals = pd.to_numeric(off[key], errors="coerce").to_numpy(dtype=float)
        diffs = a_vals - b_vals
        stats = _wilcoxon(diffs)
        rows.append({
            "metric": key,
            "kind": "continuous",
            "mean_reflect_on":  float(np.nanmean(a_vals)) if a_vals.size else float("nan"),
            "mean_reflect_off": float(np.nanmean(b_vals)) if b_vals.size else float("nan"),
            **stats,
        })
    for key in _STATS_BINARY:
        if key not in on.columns or key not in off.columns:
            continue
        a_vals = on[key].to_numpy()
        b_vals = off[key].to_numpy()
        # Coerce string-y truthiness to bool.
        def _b(arr):
            return np.array([
                str(v).strip().lower() in ("true", "1", "1.0", "yes")
                if not isinstance(v, (int, float, bool, np.bool_))
                else bool(v)
                for v in arr
            ])
        stats = _mcnemar_binary(_b(a_vals), _b(b_vals))
        rows.append({
            "metric": key,
            "kind": "binary",
            "mean_reflect_on":  float(np.mean(_b(a_vals))) if a_vals.size else float("nan"),
            "mean_reflect_off": float(np.mean(_b(b_vals))) if b_vals.size else float("nan"),
            **stats,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired", required=True,
                    help="Path to paired_metrics.csv produced by bench_reflection_ablation.py")
    ap.add_argument("--out", default="figure_reflection",
                    help="Output basename (no extension)")
    args = ap.parse_args()

    csv_path = Path(args.paired)
    if not csv_path.is_file():
        raise SystemExit(f"paired csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    out_base = csv_path.parent / args.out

    pdf = _plot(df, out_base=out_base)
    print(f"[fig] figure -> {pdf}")
    print(f"[fig]        -> {pdf.with_suffix('.png')}")

    on, off = _pair_arms(df)
    stats = _build_stats_table(on, off)
    tsv = csv_path.parent / "stats_reflection.tsv"
    stats.to_csv(tsv, sep="\t", index=False, float_format="%.4f")
    print(f"[stats] paired stats -> {tsv}")
    if not stats.empty:
        # Pretty-print a brief summary on stdout for quick readout.
        cols = ["metric", "mean_reflect_on", "mean_reflect_off",
                "mean_diff" if "mean_diff" in stats.columns else None,
                "p_value", "test"]
        cols = [c for c in cols if c and c in stats.columns]
        print()
        print(stats[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
