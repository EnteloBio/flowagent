"""Combined FlowAgent planning-stack ablation summary (Benchmarks H, I, K, L, M).

Reads the newest ``paired_metrics.csv`` from each ablation run tree and
emits:

* ``figure_ablation_summary.pdf`` / ``.png`` — paired ON vs OFF pass rates
  (overall + structural completeness) across architecture toggles.
* ``figure_ablation_summary_secondary.pdf`` / ``.png`` — paired ON vs OFF
  for tool coverage, hallucination rate, DAG edge density, and stage
  efficiency (publication supplement panels).
* ``figure_ablation_summary__stats.tsv`` — marginal pass rates + McNemar
  (backward-compatible headline table).
* ``figure_ablation_summary__metrics.tsv`` — long-format paired statistics
  for every metric present in the run (Wilcoxon / McNemar, bootstrap CIs).
* ``figure_ablation_summary__metrics_wide.tsv`` — one row per component,
  columns ``<metric>_on_mean`` / ``_off_mean`` / ``_delta`` / ``_p`` for
  direct paste into manuscript tables.

Benchmark J (competitor DAG-prompt ablation) is intentionally excluded —
that tests an external framework, not a FlowAgent component.

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

from make_ablation_figure import _bootstrap_ci, _mcnemar_binary, _wilcoxon

_PAIR_KEY = ("model", "input_id", "replicate")

# (results subdir, ON arm, OFF arm, short x-axis label)
_COMPONENTS: List[Tuple[str, str, str, str]] = [
    ("ablation", "dag_aware", "dag_blind", "DAG schema\n(H)"),
    ("reflection", "reflect_on", "reflect_off", "Completeness\nreflection (I)"),
    ("validator_ablation", "validator_on", "validator_off", "Command\nvalidator (K)"),
    ("cove_ablation", "verifier_on", "verifier_off", "CoVe verifier\n(L)"),
    ("tool_hint_ablation", "hint_on", "hint_off", "Tool hint\n(M)"),
]

# Headline pass-rate panels on the primary figure.
_HEADLINE_METRICS: List[Tuple[str, str]] = [
    ("overall_pass", "Overall pass rate"),
    ("completeness_pass", "Completeness pass rate"),
]

# Secondary figure panels (continuous / rate metrics comparable across arms).
_SECONDARY_FIGURE_METRICS: List[Tuple[str, str]] = [
    ("tools_present_fraction", "Mean tool coverage"),
    ("hallucination_rate", "Hallucination rate"),
    ("dag_edge_density", "DAG edge density"),
    ("stage_efficiency", "Stage efficiency"),
]

# kind: "binary" | "continuous"; direction guides signed delta interpretation.
_ALL_METRICS: List[Tuple[str, str, str, str]] = [
    # Pass / rubric gates
    ("overall_pass", "Overall pass rate", "binary", "higher_better"),
    ("completeness_pass", "Completeness pass rate", "binary", "higher_better"),
    ("type_correct", "Workflow type correct", "binary", "higher_better"),
    ("no_forbidden_tools", "No forbidden tools", "binary", "higher_better"),
    ("step_count_ok", "Step count OK", "binary", "higher_better"),
    ("plan_valid", "Plan structurally valid", "binary", "higher_better"),
    ("dag_valid", "DAG valid", "binary", "higher_better"),
    # Tool / command quality
    ("tools_present_fraction", "Tools present fraction", "continuous", "higher_better"),
    ("hallucination_rate", "Hallucination rate", "continuous", "lower_better"),
    ("num_hallucinated_tools", "Hallucinated tool count", "continuous", "lower_better"),
    ("commands_well_formed_fraction", "Commands well-formed fraction", "continuous", "higher_better"),
    # Structural shape
    ("dag_edge_density", "DAG edge density", "continuous", "higher_better"),
    ("parallel_width", "Parallel width", "continuous", "neutral"),
    ("stage_efficiency", "Stage efficiency", "continuous", "higher_better"),
    ("stage_efficiency_raw", "Stage efficiency (raw)", "continuous", "higher_better"),
    ("num_steps", "Step count", "continuous", "neutral"),
    ("num_completeness_failures", "Completeness failure count", "continuous", "lower_better"),
    ("completeness_attempts", "Completeness reflection attempts", "continuous", "lower_better"),
    # Preset concordance (subset of prompts with gold_preset)
    ("preset_command_f1", "Preset command F1", "continuous", "higher_better"),
    ("preset_name_jaccard", "Preset name Jaccard", "continuous", "higher_better"),
    # Cost / latency (when recorded)
    ("cost_usd", "Cost (USD)", "continuous", "lower_better"),
    ("wall_seconds", "Wall time (s)", "continuous", "lower_better"),
    ("prompt_tokens", "Prompt tokens", "continuous", "lower_better"),
    ("completion_tokens", "Completion tokens", "continuous", "lower_better"),
    ("llm_calls", "LLM calls", "continuous", "lower_better"),
]

_ON_COLOR = "#1f77b4"
_OFF_COLOR = "#c44e52"


@dataclass(frozen=True)
class MetricSummary:
    subdir: str
    label: str
    on_arm: str
    off_arm: str
    metric: str
    metric_label: str
    kind: str
    direction: str
    n_pairs: int
    mean_on: float
    lo_on: float
    hi_on: float
    mean_off: float
    lo_off: float
    hi_off: float
    delta: float
    median_diff: float
    p_value: float
    test: str
    b_only: int
    c_only: int
    paired_csv: Optional[Path] = None


# Backward-compatible alias used by tests and paired_figure_report.
ComponentStats = MetricSummary


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


def _as_float_series(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.array([], dtype=float)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _signed_delta(
    mean_on: float, mean_off: float, *, direction: str,
) -> float:
    """ON-minus-OFF, flipped for lower-is-better metrics."""
    raw = mean_on - mean_off
    if direction == "lower_better":
        return -raw
    return raw


def _summarise_metric(
    *,
    subdir: str,
    label: str,
    on_arm: str,
    off_arm: str,
    metric: str,
    metric_label: str,
    kind: str,
    direction: str,
    paired_csv: Path,
) -> Optional[MetricSummary]:
    df = pd.read_csv(paired_csv)
    if metric not in df.columns:
        return None
    try:
        on_df, off_df = _pair_arms(df, on_arm, off_arm)
    except ValueError:
        return None
    if on_df.empty:
        return None

    if kind == "binary":
        on_vals = _as_bool_series(on_df, metric).astype(float)
        off_vals = _as_bool_series(off_df, metric).astype(float)
        if on_vals.size == 0:
            return None
        mcn = _mcnemar_binary(on_vals.astype(bool), off_vals.astype(bool))
        mean_on, lo_on, hi_on = _bootstrap_ci(on_vals)
        mean_off, lo_off, hi_off = _bootstrap_ci(off_vals)
        return MetricSummary(
            subdir=subdir,
            label=label,
            on_arm=on_arm,
            off_arm=off_arm,
            metric=metric,
            metric_label=metric_label,
            kind=kind,
            direction=direction,
            n_pairs=int(on_df.shape[0]),
            mean_on=mean_on,
            lo_on=lo_on,
            hi_on=hi_on,
            mean_off=mean_off,
            lo_off=lo_off,
            hi_off=hi_off,
            delta=_signed_delta(mean_on, mean_off, direction=direction),
            median_diff=float("nan"),
            p_value=float(mcn["p_value"]),
            test=str(mcn.get("test", "mcnemar")),
            b_only=int(mcn["b_only"]),
            c_only=int(mcn["c_only"]),
            paired_csv=paired_csv,
        )

    on_vals = _as_float_series(on_df, metric)
    off_vals = _as_float_series(off_df, metric)
    mask = ~(np.isnan(on_vals) | np.isnan(off_vals))
    on_vals = on_vals[mask]
    off_vals = off_vals[mask]
    if on_vals.size == 0:
        return None
    diffs = on_vals - off_vals
    wil = _wilcoxon(diffs)
    mean_on, lo_on, hi_on = _bootstrap_ci(on_vals)
    mean_off, lo_off, hi_off = _bootstrap_ci(off_vals)
    return MetricSummary(
        subdir=subdir,
        label=label,
        on_arm=on_arm,
        off_arm=off_arm,
        metric=metric,
        metric_label=metric_label,
        kind=kind,
        direction=direction,
        n_pairs=int(on_vals.size),
        mean_on=mean_on,
        lo_on=lo_on,
        hi_on=hi_on,
        mean_off=mean_off,
        lo_off=lo_off,
        hi_off=hi_off,
        delta=_signed_delta(mean_on, mean_off, direction=direction),
        median_diff=float(wil.get("median_diff", float("nan"))),
        p_value=float(wil["p_value"]),
        test=str(wil.get("test", "wilcoxon")),
        b_only=0,
        c_only=0,
        paired_csv=paired_csv,
    )


def collect_all_summaries(results_root: Path) -> List[MetricSummary]:
    rows: List[MetricSummary] = []
    for subdir, on_arm, off_arm, label in _COMPONENTS:
        paired = _latest_paired_csv(results_root, subdir)
        if paired is None:
            print(f"[warn] no paired_metrics.csv for {subdir}", file=sys.stderr)
            continue
        for key, mlabel, kind, direction in _ALL_METRICS:
            stat = _summarise_metric(
                subdir=subdir,
                label=label,
                on_arm=on_arm,
                off_arm=off_arm,
                metric=key,
                metric_label=mlabel,
                kind=kind,
                direction=direction,
                paired_csv=paired,
            )
            if stat is not None:
                rows.append(stat)
    return rows


def collect_stats(results_root: Path) -> List[MetricSummary]:
    """Headline pass-rate rows only (backward compatible)."""
    headline = {k for k, _ in _HEADLINE_METRICS}
    return [r for r in collect_all_summaries(results_root) if r.metric in headline]


def _summarise_component(
    *,
    subdir: str,
    label: str,
    on_arm: str,
    off_arm: str,
    metric: str,
    paired_csv: Path,
) -> Optional[MetricSummary]:
    meta = next((m for m in _ALL_METRICS if m[0] == metric), None)
    if meta is None:
        return None
    key, mlabel, kind, direction = meta
    return _summarise_metric(
        subdir=subdir,
        label=label,
        on_arm=on_arm,
        off_arm=off_arm,
        metric=key,
        metric_label=mlabel,
        kind=kind,
        direction=direction,
        paired_csv=paired_csv,
    )


def _format_p(p: float) -> str:
    if np.isnan(p):
        return "p=—"
    if p < 0.001:
        return "p<0.001"
    if p < 0.01:
        return f"p={p:.3f}"
    return f"p={p:.2f}"


def _plot_grouped_bars(
    stats: Sequence[MetricSummary],
    *,
    metric_key: str,
    panel_title: str,
    ax: plt.Axes,
    letter: str,
    show_legend: bool,
    ylabel: str = "Mean",
) -> None:
    rows = [r for r in stats if r.metric == metric_key]
    if not rows:
        ax.set_visible(False)
        return

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
            label="ON (default)" if i == 0 and show_legend else "_nolegend_",
        )
        ax.bar(
            xc + width / 2, off_mean, width,
            yerr=off_yerr, capsize=3,
            color=_OFF_COLOR, edgecolor="black", linewidth=0.5,
            alpha=0.85,
            label="OFF (ablated)" if i == 0 and show_legend else "_nolegend_",
        )

        if row.kind == "binary":
            delta_pp = (row.mean_on - row.mean_off) * 100
            delta_txt = f"{delta_pp:+.1f} pp"
        else:
            delta_txt = f"Δ={row.mean_on - row.mean_off:+.3g}"
        y_top = max(row.hi_on, row.hi_off, on_mean, off_mean)
        ax.text(
            xc, y_top + max(0.02, 0.04 * y_top),
            f"{delta_txt}\n{_format_p(row.p_value)}",
            ha="center", va="bottom", fontsize=7.5,
        )

    ax.set_xticks(x_centers)
    ax.set_xticklabels([r.label for r in rows], fontsize=9)
    ymax = max(
        (max(r.hi_on, r.hi_off, r.mean_on, r.mean_off) for r in rows),
        default=1.0,
    )
    if metric_key in {"overall_pass", "completeness_pass", "tools_present_fraction",
                      "hallucination_rate", "type_correct", "no_forbidden_tools"}:
        ax.set_ylim(0.0, min(1.12, ymax * 1.15 + 0.08))
    else:
        ax.set_ylim(0.0, ymax * 1.2 + 0.05)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{letter}  {panel_title}", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    if show_legend:
        ax.legend(loc="upper right", frameon=True, fontsize=8)


def _plot(stats: Sequence[MetricSummary], *, out_base: Path) -> Path:
    headline = [r for r in stats if r.metric in {k for k, _ in _HEADLINE_METRICS}]
    metrics_to_plot = [
        (key, title) for key, title in _HEADLINE_METRICS
        if any(r.metric == key for r in headline)
    ]
    if not metrics_to_plot:
        raise SystemExit("no component stats to plot")

    fig, axes = plt.subplots(
        1, len(metrics_to_plot),
        figsize=(3.8 * len(metrics_to_plot), 4.6),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for i, (metric_key, panel_title) in enumerate(metrics_to_plot):
        _plot_grouped_bars(
            headline,
            metric_key=metric_key,
            panel_title=panel_title,
            ax=axes_flat[i],
            letter=chr(ord("a") + i),
            show_legend=(i == 0),
            ylabel="Pass rate",
        )

    fig.suptitle(
        "FlowAgent planning-stack ablation summary\n"
        "(paired ON vs OFF per component; Benchmarks H / I / K / L / M)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    pdf_path = out_base.with_suffix(".pdf")
    png_path = out_base.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return pdf_path


def _plot_secondary(stats: Sequence[MetricSummary], *, out_base: Path) -> Optional[Path]:
    secondary = [r for r in stats if r.metric in {k for k, _ in _SECONDARY_FIGURE_METRICS}]
    panels = [
        (key, title) for key, title in _SECONDARY_FIGURE_METRICS
        if any(r.metric == key for r in secondary)
    ]
    if not panels:
        return None

    ncols = 2
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.2 * ncols, 3.8 * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for i, (metric_key, panel_title) in enumerate(panels):
        _plot_grouped_bars(
            secondary,
            metric_key=metric_key,
            panel_title=panel_title,
            ax=axes_flat[i],
            letter=chr(ord("a") + i),
            show_legend=(i == 0),
        )

    for k in range(len(panels), len(axes_flat)):
        axes_flat[k].set_visible(False)

    fig.suptitle(
        "FlowAgent ablation — secondary planning metrics\n"
        "(paired ON vs OFF; bootstrap 95% CIs)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    stem = Path(str(out_base) + "_secondary")
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return pdf_path


def _write_stats_tsv(stats: Sequence[MetricSummary], path: Path) -> None:
    rows = []
    for s in stats:
        delta_pp = (s.mean_on - s.mean_off) * 100 if s.kind == "binary" else float("nan")
        rows.append({
            "component": s.subdir,
            "label": s.label.replace("\n", " "),
            "metric": s.metric,
            "on_arm": s.on_arm,
            "off_arm": s.off_arm,
            "n_pairs": s.n_pairs,
            "mean_on": s.mean_on,
            "ci_lo_on": s.lo_on,
            "ci_hi_on": s.hi_on,
            "mean_off": s.mean_off,
            "ci_lo_off": s.lo_off,
            "ci_hi_off": s.hi_off,
            "delta_pp": delta_pp,
            "p_value": s.p_value,
            "b_only_off_wins": s.b_only,
            "c_only_on_wins": s.c_only,
            "paired_csv": str(s.paired_csv) if s.paired_csv else "",
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, float_format="%.6g")


def _write_metrics_tsv(stats: Sequence[MetricSummary], path: Path) -> None:
    rows = []
    for s in stats:
        rows.append({
            "component": s.subdir,
            "label": s.label.replace("\n", " "),
            "benchmark_arm_on": s.on_arm,
            "benchmark_arm_off": s.off_arm,
            "metric": s.metric,
            "metric_label": s.metric_label,
            "kind": s.kind,
            "direction": s.direction,
            "n_pairs": s.n_pairs,
            "mean_on": s.mean_on,
            "ci_lo_on": s.lo_on,
            "ci_hi_on": s.hi_on,
            "mean_off": s.mean_off,
            "ci_lo_off": s.lo_off,
            "ci_hi_off": s.hi_off,
            "delta_on_minus_off": s.mean_on - s.mean_off,
            "signed_delta": s.delta,
            "median_paired_diff": s.median_diff,
            "p_value": s.p_value,
            "test": s.test,
            "b_only_off_wins": s.b_only,
            "c_only_on_wins": s.c_only,
            "paired_csv": str(s.paired_csv) if s.paired_csv else "",
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, float_format="%.6g")


def _write_metrics_wide_tsv(stats: Sequence[MetricSummary], path: Path) -> None:
    """One row per component; metric stats as prefixed columns."""
    by_comp: Dict[str, Dict[str, object]] = {}
    for s in stats:
        comp = s.subdir
        if comp not in by_comp:
            by_comp[comp] = {
                "component": comp,
                "label": s.label.replace("\n", " "),
                "on_arm": s.on_arm,
                "off_arm": s.off_arm,
                "n_pairs_modal": s.n_pairs,
            }
        prefix = s.metric
        by_comp[comp][f"{prefix}_on_mean"] = s.mean_on
        by_comp[comp][f"{prefix}_off_mean"] = s.mean_off
        by_comp[comp][f"{prefix}_delta"] = s.mean_on - s.mean_off
        by_comp[comp][f"{prefix}_signed_delta"] = s.delta
        by_comp[comp][f"{prefix}_p"] = s.p_value
        by_comp[comp][f"{prefix}_n"] = s.n_pairs
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(by_comp.values())).to_csv(
        path, sep="\t", index=False, float_format="%.6g",
    )


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

    all_stats = collect_all_summaries(results_root)
    headline_stats = [s for s in all_stats if s.metric in {k for k, _ in _HEADLINE_METRICS}]
    if not headline_stats:
        raise SystemExit(
            f"no ablation paired results under {results_root} "
            f"(expected ablation/, reflection/, validator_ablation/, "
            f"cove_ablation/, tool_hint_ablation/)"
        )

    pdf_path = _plot(headline_stats, out_base=out_base)
    sec_path = _plot_secondary(all_stats, out_base=out_base)

    stats_path = Path(str(out_base) + "__stats.tsv")
    metrics_path = Path(str(out_base) + "__metrics.tsv")
    wide_path = Path(str(out_base) + "__metrics_wide.tsv")
    _write_stats_tsv(headline_stats, stats_path)
    _write_metrics_tsv(all_stats, metrics_path)
    _write_metrics_wide_tsv(all_stats, wide_path)

    print(f"[ok]   ablation_summary → {pdf_path}")
    if sec_path is not None:
        print(f"[ok]   ablation_summary_secondary → {sec_path}")
    print(f"[ok]   ablation_summary stats → {stats_path}")
    print(f"[ok]   ablation_summary metrics → {metrics_path}")
    print(f"[ok]   ablation_summary metrics (wide) → {wide_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
