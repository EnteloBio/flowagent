"""Figure + paired statistics for the CoVe verifier on/off ablation (todo T4).

Reads ``paired_metrics.csv`` produced by ``bench_cove_ablation.py`` and
emits:

* ``figure_cove.pdf`` / ``.png`` -- one panel per metric showing arm
  means with bootstrap 95% CIs.
* ``stats_cove.tsv`` -- per-metric paired-prompt comparison (Wilcoxon
  signed-rank for continuous, McNemar for binary ``overall_pass``). Rows
  matched on ``(model, input_id, replicate)`` so the same prompt is
  graded by the verifier-on and verifier-off planner under identical
  seeds.
* ``stats_cove_signal.tsv`` -- *additional* table specific to T4: how
  often the verifier flagged plans on the verifier-on arm, broken down
  by overall_pass outcome. Lets the reviewer assess the verifier's
  precision / recall as a failure predictor without re-running the
  contingency-table snippet from the bench-script README.

Reuses :mod:`make_ablation_figure`'s bootstrap CI / Wilcoxon / McNemar
helpers verbatim (same as :mod:`make_validator_figure`) so any
improvement to the DAG-awareness figure automatically lifts this one.

Usage::

    python make_cove_figure.py \\
        --paired results/cove_ablation/<ts>/paired_metrics.csv \\
        --out figure_cove
"""

from __future__ import annotations

import argparse
import json
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


_ARM_ON = "verifier_on"
_ARM_OFF = "verifier_off"
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
        ax.set_xticklabels(["Verifier on", "Verifier off"])

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

    fig.suptitle(
        "CoVe verifier on/off ablation -- planning-only "
        "(annotation mode, no abstention)",
        fontsize=12,
    )
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
            "mean_verifier_on": float(np.nanmean(a_vals)) if a_vals.size else float("nan"),
            "mean_verifier_off": float(np.nanmean(b_vals)) if b_vals.size else float("nan"),
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
            "mean_verifier_on": float(np.mean(a_vals.astype(bool))) if a_vals.size else float("nan"),
            "mean_verifier_off": float(np.mean(b_vals.astype(bool))) if b_vals.size else float("nan"),
            "mean_diff": float("nan"),
            "median_diff": float("nan"),
            **stats,
        })
    return pd.DataFrame(rows)


# ── Verifier-signal contingency table ──────────────────────────────
#
# T4-specific output: how often did the verifier flag plans on the
# verifier-on arm, and did flagged plans actually fail more often than
# unflagged ones? Loaded straight from the JSONL (the paired CSV
# doesn't carry the per-cell ``_verifier`` envelope by default — only
# scalar columns make it through the harness's auto-flattening).

def _build_signal_table(jsonl_path: Path) -> pd.DataFrame:
    """Read a verifier_on results.jsonl and build the contingency table.

    Returns rows for ``(flagged, overall_pass)`` and summary metrics
    (precision, recall, abstention rate). Empty DataFrame if the
    JSONL is missing or no cells carry a ``_verifier`` envelope.
    """
    if not jsonl_path.exists():
        return pd.DataFrame()

    cells: List[Dict[str, object]] = []
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        env = (r.get("plan") or {}).get("_verifier") or {}
        if not env:
            continue
        weight = int(env.get("weighted_concern_count") or 0)
        threshold = int(env.get("abstention_threshold") or 2)
        flagged = weight >= threshold
        cells.append({
            "input_id": r.get("input_id"),
            "replicate": r.get("replicate"),
            "weighted_concern_count": weight,
            "threshold": threshold,
            "flagged": flagged,
            "overall_pass": bool(r.get("overall_pass")),
            "n_concerns": len(env.get("concerns") or []),
            "n_flagged_concerns": sum(
                1 for c in (env.get("concerns") or []) if c.get("concern")
            ),
        })

    if not cells:
        return pd.DataFrame()

    df = pd.DataFrame(cells)
    n = len(df)
    tp = int(((df["flagged"]) & (~df["overall_pass"])).sum())   # flagged & failed (correct)
    fp = int(((df["flagged"]) & (df["overall_pass"])).sum())    # flagged & passed (false alarm)
    fn = int(((~df["flagged"]) & (~df["overall_pass"])).sum())  # unflagged & failed (miss)
    tn = int(((~df["flagged"]) & (df["overall_pass"])).sum())   # unflagged & passed

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) else float("nan")
    abstain_rate = (tp + fp) / n if n else float("nan")
    failure_rate_overall = (tp + fn) / n if n else float("nan")

    rows = [
        {"row": "flagged & failed     (TP)",   "n": tp,
         "share_of_total": tp / n},
        {"row": "flagged & passed     (FP)",   "n": fp,
         "share_of_total": fp / n},
        {"row": "unflagged & failed   (FN)",   "n": fn,
         "share_of_total": fn / n},
        {"row": "unflagged & passed   (TN)",   "n": tn,
         "share_of_total": tn / n},
        {"row": "TOTAL",                       "n": n,
         "share_of_total": 1.0},
        {"row": "abstention rate (flagged share)", "n": tp + fp,
         "share_of_total": abstain_rate},
        {"row": "failure rate (overall_pass=False share)", "n": tp + fn,
         "share_of_total": failure_rate_overall},
        {"row": "verifier precision (TP / (TP+FP))",
         "n": float("nan"), "share_of_total": precision},
        {"row": "verifier recall    (TP / (TP+FN))",
         "n": float("nan"), "share_of_total": recall},
    ]
    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--paired", required=True,
        help="Path to paired_metrics.csv produced by bench_cove_ablation.py",
    )
    ap.add_argument("--out", default="figure_cove",
                    help="Output basename (writes .pdf and .png)")
    ap.add_argument("--stats-out", default=None,
                    help="Path for stats_cove.tsv "
                         "(default: <out>__stats.tsv)")
    ap.add_argument("--signal-out", default=None,
                    help="Path for stats_cove_signal.tsv "
                         "(default: <out>__signal.tsv)")
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

    # Verifier-signal table from the verifier_on JSONL.
    run_dir = paired_path.parent
    signal_path = (Path(args.signal_out)
                   if args.signal_out
                   else Path(str(out_base) + "__signal.tsv"))
    signal = _build_signal_table(run_dir / _ARM_ON / "results.jsonl")
    if not signal.empty:
        signal.to_csv(signal_path, sep="\t", index=False, float_format="%.6g")

    print(f"[ok] figure -> {pdf_path}")
    print(f"[ok] stats  -> {stats_path}")
    if not signal.empty:
        print(f"[ok] signal -> {signal_path}")
    else:
        print("[warn] no _verifier envelopes in verifier_on results — "
              "signal table not written")
    print()
    print(stats.to_string(index=False))
    if not signal.empty:
        print()
        print("--- Verifier signal (verifier_on arm) ---")
        print(signal.to_string(index=False))


if __name__ == "__main__":
    main()
