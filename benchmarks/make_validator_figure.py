"""Figure + paired statistics for the validator-on-vs-off ablation (todo T0).

Reads ``paired_metrics.csv`` produced by ``bench_validator_ablation.py`` and
emits:

* ``figure_validator.pdf`` / ``.png`` -- one panel per metric showing
  arm means with bootstrap 95% CIs.
* ``stats_validator.tsv`` -- per-metric paired-prompt comparison
  (Wilcoxon signed-rank for continuous metrics, McNemar for binary
  ``overall_pass``). Rows are matched on
  ``(model, input_id, replicate)`` so the same prompt is graded by
  the validator-on and validator-off planner under identical seeds.

Reuses :mod:`make_ablation_figure`'s bootstrap CI / Wilcoxon / McNemar
helpers verbatim — the only difference vs the DAG-awareness ablation is
arm naming (``validator_on`` / ``validator_off`` instead of
``dag_aware`` / ``dag_blind``) and the figure title.

Metrics covered (all written to ``stats_validator.tsv`` regardless of
whether they are plotted):

  - tools_present_fraction      -- tool-coverage rate
  - hallucination_rate          -- fraction of plan tools we don't recognise
  - preset_command_f1           -- token-F1 vs the preset's gold command
  - preset_name_jaccard         -- step-name Jaccard vs the preset
  - dag_edge_density            -- edges / (steps - 1); structural sanity
  - parallel_width              -- max width of a topological layer
  - num_steps                   -- raw step count
  - overall_pass                -- gating boolean (McNemar)

Usage::

    python make_validator_figure.py \\
        --paired results/validator_ablation/<ts>/paired_metrics.csv \\
        --out figure_validator
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse the existing ablation figure's bootstrap / Wilcoxon / McNemar /
# subset-flagging logic so any improvement to the DAG-awareness figure
# automatically lifts this one too.
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from make_ablation_figure import (  # noqa: E402
    _PLOT_METRICS,
    _STATS_METRICS,
    _BINARY_METRICS,
    _bootstrap_ci,
    _mcnemar_binary,
    _wilcoxon,
)


_ARM_ON = "validator_on"
_ARM_OFF = "validator_off"
_PAIR_KEY = ("model", "input_id", "replicate")


# ── Pair-up by (model, input_id, replicate) ────────────────────────

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


# ── Plot ───────────────────────────────────────────────────────────

def _plot(df: pd.DataFrame, *, out_base: Path) -> Path:
    on = df[df["arm"] == _ARM_ON]
    off = df[df["arm"] == _ARM_OFF]

    n_metrics = len(_PLOT_METRICS)
    ncols = 3
    nrows = int(np.ceil(n_metrics / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.4 * nrows))
    axes = np.atleast_2d(axes).flatten()

    panel_ns: List[Tuple[int, int]] = []
    for key, _label, _kind in _PLOT_METRICS:
        if key not in df.columns:
            panel_ns.append((0, 0))
            continue
        a_vals = pd.to_numeric(on.get(key), errors="coerce").to_numpy()
        b_vals = pd.to_numeric(off.get(key), errors="coerce").to_numpy()
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

        a_vals = pd.to_numeric(on.get(key), errors="coerce").to_numpy()
        b_vals = pd.to_numeric(off.get(key), errors="coerce").to_numpy()

        a_mean, a_lo, a_hi = _bootstrap_ci(a_vals)
        b_mean, b_lo, b_hi = _bootstrap_ci(b_vals)

        x = np.array([0, 1])
        means = np.array([a_mean, b_mean])
        lo = np.array([a_lo, b_lo])
        hi = np.array([a_hi, b_hi])
        yerr = np.vstack([means - lo, hi - means])

        ax.bar(x, means, yerr=yerr, capsize=4,
               color=["#1f77b4", "#d62728"],
               edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(["Validator on", "Validator off"])

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

        annotation_lines = [f"n_on  = {n_a}", f"n_off = {n_b}"]
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

    for k in range(len(_PLOT_METRICS), len(axes)):
        axes[k].set_visible(False)

    fig.suptitle("Validator on/off ablation -- planning-only", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    pdf_path = out_base.with_suffix(".pdf")
    png_path = out_base.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return pdf_path


# ── TSV stats ──────────────────────────────────────────────────────

def _build_stats_table(on: pd.DataFrame, off: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for key in _STATS_METRICS:
        if key not in on.columns or key not in off.columns:
            continue
        a_vals = pd.to_numeric(on[key], errors="coerce").to_numpy()
        b_vals = pd.to_numeric(off[key], errors="coerce").to_numpy()
        diffs = a_vals - b_vals
        stats = _wilcoxon(diffs)
        rows.append({
            "metric": key,
            "kind": "continuous",
            "mean_validator_on": float(np.nanmean(a_vals)) if a_vals.size else float("nan"),
            "mean_validator_off": float(np.nanmean(b_vals)) if b_vals.size else float("nan"),
            **stats,
        })
    for key in _BINARY_METRICS:
        if key not in on.columns or key not in off.columns:
            continue
        a_vals = on[key].to_numpy()
        b_vals = off[key].to_numpy()
        stats = _mcnemar_binary(a_vals, b_vals)
        rows.append({
            "metric": key,
            "kind": "binary",
            "mean_validator_on": float(np.mean(a_vals.astype(bool))) if a_vals.size else float("nan"),
            "mean_validator_off": float(np.mean(b_vals.astype(bool))) if b_vals.size else float("nan"),
            "mean_diff": float("nan"),
            "median_diff": float("nan"),
            **stats,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--paired", required=True,
        help="Path to paired_metrics.csv produced by bench_validator_ablation.py",
    )
    ap.add_argument("--out", default="figure_validator",
                    help="Output basename (writes .pdf and .png)")
    ap.add_argument("--stats-out", default=None,
                    help="Path for stats_validator.tsv "
                         "(default: <out>__stats.tsv)")
    args = ap.parse_args()

    paired_path = Path(args.paired)
    if not paired_path.exists():
        raise SystemExit(f"paired_metrics.csv not found: {paired_path}")

    df = pd.read_csv(paired_path)
    on, off = _pair_arms(df)

    if on.empty:
        raise SystemExit(
            f"No paired rows after joining {_ARM_ON} vs {_ARM_OFF}."
        )

    out_base = Path(args.out)
    pdf_path = _plot(df, out_base=out_base)

    stats_path = (Path(args.stats_out)
                  if args.stats_out
                  else Path(str(out_base) + "__stats.tsv"))
    stats = _build_stats_table(on, off)
    stats.to_csv(stats_path, sep="\t", index=False, float_format="%.6g")

    print(f"[ok] figure -> {pdf_path}")
    print(f"[ok] stats  -> {stats_path}")
    print()
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
