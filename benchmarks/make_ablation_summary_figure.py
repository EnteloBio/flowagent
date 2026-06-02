"""Combined FlowAgent planning-stack ablation summary (Benchmarks H, I, K, L, M).

Reads the newest ``paired_metrics.csv`` from each ablation run tree and
emits a single two-panel figure comparing **component ON vs OFF** pass
rates across the four toggles that isolate FlowAgent architecture layers
(DAG schema, completeness reflection, command validator, CoVe verifier,
tool-hint allowlist).

Benchmark J (competitor DAG-prompt ablation) is intentionally excluded —
that tests an external framework, not a FlowAgent component.

Outputs:

* ``figure_ablation_summary.pdf`` / ``.png``
* ``figure_ablation_summary__stats.tsv`` — per-component marginal pass
  rates, paired McNemar p-values, and discordant counts.

Usage::

    python make_ablation_summary_figure.py \\
        --results-base results \\
        --out figure_ablation_summary
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from make_ablation_figure import _bootstrap_ci, _mcnemar_binary

_PAIR_KEY = ("model", "input_id", "replicate")

# (results subdir, ON arm, OFF arm, short x-axis label)
_COMPONENTS: List[Tuple[str, str, str, str]] = [
    ("ablation", "dag_aware", "dag_blind", "DAG schema\n(H)"),
    ("reflection", "reflect_on", "reflect_off", "Completeness\nreflection (I)"),
    ("validator_ablation", "validator_on", "validator_off", "Command\nvalidator (K)"),
    ("cove_ablation", "verifier_on", "verifier_off", "CoVe verifier\n(L)"),
    ("tool_hint_ablation", "hint_on", "hint_off", "Tool hint\n(M)"),
]

_METRICS: List[Tuple[str, str]] = [
    ("overall_pass", "Overall pass rate"),
    ("completeness_pass", "Completeness pass rate"),
]

_ON_COLOR = "#1f77b4"
_OFF_COLOR = "#c44e52"


@dataclass(frozen=True)
class ComponentStats:
    subdir: str
    label: str
    on_arm: str
    off_arm: str
    n_pairs: int
    metric: str
    mean_on: float
    lo_on: float
    hi_on: float
    mean_off: float
    lo_off: float
    hi_off: float
    p_value: float
    b_only: int
    c_only: int
    paired_csv: Optional[Path] = None


def _latest_paired_csv(results_root: Path, bench_subdir: str) -> Optional[Path]:
    run_dir = results_root / bench_subdir
    if not run_dir.is_dir():
        return None
    subs = [
        p for p in run_dir.iterdir()
        if p.is_dir()
        and not p.name.startswith("_")
        and (p / "paired_metrics.csv").is_file()
        and (p / "paired_metrics.csv").stat().st_size > 0
    ]
    if not subs:
        return None
    latest = max(subs, key=lambda p: p.stat().st_mtime)
    return latest / "paired_metrics.csv"


def _pair_arms(
    df: pd.DataFrame, on_arm: str, off_arm: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "arm" not in df.columns:
        raise ValueError("paired_metrics.csv missing 'arm' column")
    on = df[df["arm"] == on_arm].copy()
    off = df[df["arm"] == off_arm].copy()
    if on.empty or off.empty:
        raise ValueError(f"missing rows for arms {on_arm!r} / {off_arm!r}")
    on = on.set_index(list(_PAIR_KEY))
    off = off.set_index(list(_PAIR_KEY))
    common = on.index.intersection(off.index)
    return on.loc[common], off.loc[common]


def _as_bool_series(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.array([], dtype=bool)
    return df[col].map(
        lambda v: bool(v) if pd.notna(v) else False
    ).to_numpy(dtype=bool)


def _summarise_component(
    *,
    subdir: str,
    label: str,
    on_arm: str,
    off_arm: str,
    metric: str,
    paired_csv: Path,
) -> Optional[ComponentStats]:
    df = pd.read_csv(paired_csv)
    if metric not in df.columns:
        return None
    try:
        on_df, off_df = _pair_arms(df, on_arm, off_arm)
    except ValueError:
        return None
    if on_df.empty:
        return None

    on_vals = _as_bool_series(on_df, metric).astype(float)
    off_vals = _as_bool_series(off_df, metric).astype(float)
    mcn = _mcnemar_binary(on_vals.astype(bool), off_vals.astype(bool))
    mean_on, lo_on, hi_on = _bootstrap_ci(on_vals)
    mean_off, lo_off, hi_off = _bootstrap_ci(off_vals)

    return ComponentStats(
        subdir=subdir,
        label=label,
        on_arm=on_arm,
        off_arm=off_arm,
        n_pairs=int(on_df.shape[0]),
        metric=metric,
        mean_on=mean_on,
        lo_on=lo_on,
        hi_on=hi_on,
        mean_off=mean_off,
        lo_off=lo_off,
        hi_off=hi_off,
        p_value=float(mcn["p_value"]),
        b_only=int(mcn["b_only"]),
        c_only=int(mcn["c_only"]),
        paired_csv=paired_csv,
    )


def collect_stats(results_root: Path) -> List[ComponentStats]:
    rows: List[ComponentStats] = []
    for subdir, on_arm, off_arm, label in _COMPONENTS:
        paired = _latest_paired_csv(results_root, subdir)
        if paired is None:
            print(f"[warn] no paired_metrics.csv for {subdir}", file=sys.stderr)
            continue
        for metric, _title in _METRICS:
            stat = _summarise_component(
                subdir=subdir,
                label=label,
                on_arm=on_arm,
                off_arm=off_arm,
                metric=metric,
                paired_csv=paired,
            )
            if stat is not None:
                rows.append(stat)
            else:
                print(
                    f"[warn] {subdir}: could not summarise {metric}",
                    file=sys.stderr,
                )
    return rows


def _format_p(p: float) -> str:
    if np.isnan(p):
        return "p=—"
    if p < 0.001:
        return "p<0.001"
    if p < 0.01:
        return f"p={p:.3f}"
    return f"p={p:.2f}"


def _plot(stats: Sequence[ComponentStats], *, out_base: Path) -> Path:
    by_metric: Dict[str, List[ComponentStats]] = {}
    for row in stats:
        by_metric.setdefault(row.metric, []).append(row)

    metrics_to_plot = [
        (key, title) for key, title in _METRICS if key in by_metric
    ]
    if not metrics_to_plot:
        raise SystemExit("no component stats to plot")

    fig, axes = plt.subplots(
        1, len(metrics_to_plot),
        figsize=(3.8 * len(metrics_to_plot), 4.6),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for ax, (metric_key, panel_title) in zip(axes_flat, metrics_to_plot):
        rows = by_metric[metric_key]
        n_comp = len(rows)
        x_centers = np.arange(n_comp)
        width = 0.34

        for i, row in enumerate(rows):
            xc = x_centers[i]
            on_mean = row.mean_on
            off_mean = row.mean_off
            on_yerr = np.array([[on_mean - row.lo_on], [row.hi_on - on_mean]])
            off_yerr = np.array([[off_mean - row.lo_off], [row.hi_off - off_mean]])

            ax.bar(
                xc - width / 2, on_mean, width,
                yerr=on_yerr, capsize=3,
                color=_ON_COLOR, edgecolor="black", linewidth=0.5,
                label="ON (default)" if i == 0 else "_nolegend_",
            )
            ax.bar(
                xc + width / 2, off_mean, width,
                yerr=off_yerr, capsize=3,
                color=_OFF_COLOR, edgecolor="black", linewidth=0.5,
                alpha=0.85,
                label="OFF (ablated)" if i == 0 else "_nolegend_",
            )

            delta_pp = (on_mean - off_mean) * 100
            sign = "+" if delta_pp >= 0 else ""
            y_top = max(row.hi_on, row.hi_off, on_mean, off_mean)
            ax.text(
                xc, min(1.02, y_top + 0.06),
                f"{sign}{delta_pp:.1f} pp\n{_format_p(row.p_value)}",
                ha="center", va="bottom", fontsize=8,
            )

        ax.set_xticks(x_centers)
        ax.set_xticklabels([r.label for r in rows], fontsize=9)
        ax.set_ylim(0.0, 1.08)
        ax.set_ylabel("Pass rate")
        letter = "a" if metric_key == metrics_to_plot[0][0] else "b"
        ax.set_title(f"{letter}  {panel_title}", loc="left", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        if metric_key == metrics_to_plot[0][0]:
            ax.legend(loc="upper right", frameon=True, fontsize=8)

    fig.suptitle(
        "FlowAgent planning-stack ablation summary\n"
        "(paired ON vs OFF per component; Benchmarks H / I / K / L)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    pdf_path = out_base.with_suffix(".pdf")
    png_path = out_base.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return pdf_path


def _write_stats_tsv(stats: Sequence[ComponentStats], path: Path) -> None:
    rows = []
    for s in stats:
        rows.append({
            "component": s.subdir,
            "label": s.label.replace("\n", " "),
            "metric": s.metric,
            "on_arm": s.on_arm,
            "off_arm": s.off_arm,
            "n_pairs": s.n_pairs,
            "mean_on": s.mean_on,
            "mean_off": s.mean_off,
            "delta_pp": (s.mean_on - s.mean_off) * 100,
            "p_value": s.p_value,
            "b_only_off_wins": s.b_only,
            "c_only_on_wins": s.c_only,
            "paired_csv": str(s.paired_csv) if s.paired_csv else "",
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, float_format="%.6g")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-base", default="results",
                    help="Benchmark results root (default: results)")
    ap.add_argument("--out", default="figure_ablation_summary",
                    help="Output path stem (default: figure_ablation_summary)")
    args = ap.parse_args(argv)

    here = Path(__file__).resolve().parent
    results_root = Path(args.results_base)
    if not results_root.is_absolute():
        results_root = here / results_root

    out_base = Path(args.out)
    if not out_base.is_absolute():
        out_base = here / out_base
    out_base.parent.mkdir(parents=True, exist_ok=True)

    stats = collect_stats(results_root)
    if not stats:
        raise SystemExit(
            f"no ablation paired results under {results_root} "
            f"(expected ablation/, reflection/, validator_ablation/, cove_ablation/)"
        )

    pdf_path = _plot(stats, out_base=out_base)
    stats_path = Path(str(out_base) + "__stats.tsv")
    _write_stats_tsv(stats, stats_path)
    print(f"[ok]   ablation_summary → {pdf_path}")
    print(f"[ok]   ablation_summary stats → {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
