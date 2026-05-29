"""Render paired ablation figures into ``results/figures/`` for ``make report``."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

_BENCH_DIR = Path(__file__).resolve().parent.parent


def _ensure_bench_on_path() -> None:
    root = str(_BENCH_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)


def _latest_paired_csv(results_root: Path, bench_subdir: str) -> Optional[Path]:
    """Return the newest ``paired_metrics.csv`` under ``results/<subdir>/``."""
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


def _write_stats(stats: pd.DataFrame, stats_path: Path) -> None:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(stats_path, sep="\t", index=False, float_format="%.6g")


def _render_ablation(paired_csv: Path, out_base: Path) -> None:
    _ensure_bench_on_path()
    from make_ablation_figure import _build_stats_table, _pair_arms, _plot

    df = pd.read_csv(paired_csv)
    aware, blind = _pair_arms(df)
    if aware.empty:
        print("[skip] ablation: no paired rows after joining arms")
        return
    pdf_path = _plot(df, out_base=out_base)
    _write_stats(_build_stats_table(aware, blind),
                 Path(str(out_base) + "__stats.tsv"))
    print(f"[ok]   ablation → {pdf_path}")


def _render_reflection(paired_csv: Path, out_base: Path) -> None:
    _ensure_bench_on_path()
    from make_reflection_figure import _build_stats_table, _pair_arms, _plot

    df = pd.read_csv(paired_csv)
    on, off = _pair_arms(df)
    if on.empty:
        print("[skip] reflection: no paired rows after joining arms")
        return
    pdf_path = _plot(df, out_base=out_base)
    _write_stats(_build_stats_table(on, off),
                 Path(str(out_base) + "__stats.tsv"))
    print(f"[ok]   reflection → {pdf_path}")


def _render_validator(paired_csv: Path, out_base: Path) -> None:
    _ensure_bench_on_path()
    from make_validator_figure import _build_stats_table, _pair_arms, _plot

    df = pd.read_csv(paired_csv)
    on, off = _pair_arms(df)
    if on.empty:
        print("[skip] validator_ablation: no paired rows after joining arms")
        return
    pdf_path = _plot(df, out_base=out_base)
    _write_stats(_build_stats_table(on, off),
                 Path(str(out_base) + "__stats.tsv"))
    print(f"[ok]   validator_ablation → {pdf_path}")


def _render_cove(paired_csv: Path, out_base: Path) -> None:
    _ensure_bench_on_path()
    from make_cove_figure import (
        _ARM_ON,
        _build_signal_table,
        _build_stats_table,
        _pair_arms,
        _plot,
    )

    df = pd.read_csv(paired_csv)
    on, off = _pair_arms(df)
    if on.empty:
        print("[skip] cove_ablation: no paired rows after joining arms")
        return
    pdf_path = _plot(df, out_base=out_base)
    _write_stats(_build_stats_table(on, off),
                 Path(str(out_base) + "__stats.tsv"))

    run_dir = paired_csv.parent
    signal = _build_signal_table(run_dir / _ARM_ON / "results.jsonl")
    if not signal.empty:
        signal_path = Path(str(out_base) + "__signal.tsv")
        signal.to_csv(signal_path, sep="\t", index=False, float_format="%.6g")
        print(f"[ok]   cove_ablation_signal → {signal_path}")
    print(f"[ok]   cove_ablation → {pdf_path}")


def _render_competitor_dag(paired_csv: Path, fig_dir: Path) -> None:
    _ensure_bench_on_path()
    from make_competitor_dag_figure import _render_for_competitor

    df = pd.read_csv(paired_csv)
    if "competitor" not in df.columns:
        df = df.copy()
        df["competitor"] = "claude_code"

    rendered = False
    for comp in sorted(df["competitor"].dropna().unique()):
        sub = df[df["competitor"] == comp]
        if sub.empty:
            continue
        out_base = fig_dir / f"competitor_dag__{comp}"
        _render_for_competitor(sub, out_base=out_base, competitor=comp)
        rendered = True

    if rendered:
        print(f"[ok]   competitor_dag → {fig_dir}/competitor_dag__*.pdf")


def render_paired_ablation_figures(results_root: Path, fig_dir: Path) -> None:
    """Find latest paired ablation runs and write figures under ``fig_dir``."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    simple: list[tuple[str, str, str, Callable[[Path, Path], None]]] = [
        ("ablation", "ablation", "ablation", _render_ablation),
        ("reflection", "reflection", "reflection", _render_reflection),
        ("validator_ablation", "validator_ablation", "validator_ablation",
         _render_validator),
        ("cove_ablation", "cove_ablation", "cove_ablation", _render_cove),
    ]
    for label, subdir, out_name, render_fn in simple:
        paired = _latest_paired_csv(results_root, subdir)
        if paired is None:
            print(f"[skip] no paired results for {label}")
            continue
        try:
            render_fn(paired, fig_dir / out_name)
        except SystemExit as exc:
            print(f"[skip] {label}: {exc}")
        except Exception as exc:
            print(f"[skip] {label}: {exc}")

    paired = _latest_paired_csv(results_root, "competitor_dag_ablation")
    if paired is None:
        print("[skip] no paired results for competitor_dag_ablation")
        return
    try:
        _render_competitor_dag(paired, fig_dir)
    except SystemExit as exc:
        print(f"[skip] competitor_dag_ablation: {exc}")
    except Exception as exc:
        print(f"[skip] competitor_dag_ablation: {exc}")
