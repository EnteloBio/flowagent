"""Figure + paired statistics for the workflow tool-hint ablation (Benchmark M).

Reads ``paired_metrics.csv`` from ``bench_tool_hint_ablation.py`` and writes
``figure_tool_hint.pdf`` / ``.png`` plus ``stats_tool_hint.tsv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from make_reflection_figure import (
    _bootstrap_ci,
    _mcnemar_binary,
    _wilcoxon,
)

_ARM_ON = "hint_on"
_ARM_OFF = "hint_off"
_PAIR_KEY = ("model", "input_id", "replicate")

_PLOT_METRICS: List[Tuple[str, str, str]] = [
    ("overall_pass",           "Overall pass rate",        "fraction"),
    ("completeness_pass",      "Completeness pass rate",   "fraction"),
    ("tools_present_fraction", "Tools present fraction",   "fraction"),
    ("hallucination_rate",     "Hallucination rate",       "fraction"),
    ("no_forbidden_tools",     "No forbidden tools",     "fraction"),
    ("num_hallucinated_tools", "Hallucinated tool count",  "count"),
    ("type_correct",           "Workflow type correct",    "fraction"),
    ("num_steps",              "Step count",               "count"),
]

_STATS_CONTINUOUS = [
    "tools_present_fraction",
    "hallucination_rate",
    "num_hallucinated_tools",
    "num_steps",
    "num_completeness_failures",
]

_STATS_BINARY = [
    "overall_pass",
    "completeness_pass",
    "no_forbidden_tools",
    "type_correct",
]


def _pair_arms(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "arm" not in df.columns:
        raise SystemExit("paired_metrics.csv missing 'arm' column")
    on = df[df["arm"] == _ARM_ON].copy()
    off = df[df["arm"] == _ARM_OFF].copy()
    if on.empty:
        raise SystemExit(f"no {_ARM_ON} rows in paired_metrics.csv")
    if off.empty:
        raise SystemExit(f"no {_ARM_OFF} rows in paired_metrics.csv")
    on = on.set_index(list(_PAIR_KEY))
    off = off.set_index(list(_PAIR_KEY))
    common = on.index.intersection(off.index)
    return on.loc[common], off.loc[common]


def _plot(df: pd.DataFrame, *, out_base: Path) -> Path:
    on = df[df["arm"] == _ARM_ON]
    off = df[df["arm"] == _ARM_OFF]

    ncols = 3
    nrows = int(np.ceil(len(_PLOT_METRICS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.4 * nrows))
    axes = np.atleast_2d(axes).flatten()

    for ax, (key, label, kind) in zip(axes, _PLOT_METRICS):
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
        yerr = np.vstack([means - lo, hi - means])
        ax.bar(x, means, color=["#1f77b4", "#c44e52"], width=0.55,
               edgecolor="#333333", linewidth=0.6)
        ax.errorbar(x, means, yerr=yerr, fmt="none", color="#333333",
                    capsize=4, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(["hint on", "hint off"], fontsize=9)
        ax.set_title(label, fontsize=10)
        if kind == "fraction":
            ax.set_ylim(0, 1.12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    for k in range(len(_PLOT_METRICS), len(axes)):
        axes[k].set_visible(False)

    fig.suptitle(
        "Workflow tool-hint ablation: per-workflow allowlist in planner prompt",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf_path = out_base.with_suffix(".pdf")
    fig.savefig(pdf_path)
    fig.savefig(out_base.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return pdf_path


def _as_bool(arr) -> np.ndarray:
    return np.array([
        str(v).strip().lower() in ("true", "1", "1.0", "yes")
        if not isinstance(v, (int, float, bool, np.bool_))
        else bool(v)
        for v in arr
    ])


def _build_stats_table(on: pd.DataFrame, off: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for key in _STATS_CONTINUOUS:
        if key not in on.columns or key not in off.columns:
            continue
        a_vals = pd.to_numeric(on[key], errors="coerce").to_numpy(dtype=float)
        b_vals = pd.to_numeric(off[key], errors="coerce").to_numpy(dtype=float)
        stats = _wilcoxon(a_vals - b_vals)
        rows.append({
            "metric": key,
            "kind": "continuous",
            "mean_hint_on":  float(np.nanmean(a_vals)) if a_vals.size else float("nan"),
            "mean_hint_off": float(np.nanmean(b_vals)) if b_vals.size else float("nan"),
            **stats,
        })
    for key in _STATS_BINARY:
        if key not in on.columns or key not in off.columns:
            continue
        a_vals = on[key].to_numpy()
        b_vals = off[key].to_numpy()
        stats = _mcnemar_binary(_as_bool(a_vals), _as_bool(b_vals))
        rows.append({
            "metric": key,
            "kind": "binary",
            "mean_hint_on":  float(np.mean(_as_bool(a_vals))) if a_vals.size else float("nan"),
            "mean_hint_off": float(np.mean(_as_bool(b_vals))) if b_vals.size else float("nan"),
            **stats,
        })
    return pd.DataFrame(rows)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired", required=True)
    ap.add_argument("--out", default="figure_tool_hint")
    args = ap.parse_args(argv)

    csv_path = Path(args.paired)
    if not csv_path.is_file():
        raise SystemExit(f"paired csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    out_base = csv_path.parent / args.out
    pdf = _plot(df, out_base=out_base)
    print(f"[fig] figure -> {pdf}")

    on, off = _pair_arms(df)
    stats = _build_stats_table(on, off)
    tsv = csv_path.parent / "stats_tool_hint.tsv"
    stats.to_csv(tsv, sep="\t", index=False, float_format="%.4f")
    print(f"[stats] paired stats -> {tsv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
