"""Tests for the combined ablation summary figure (H/I/K/L/M)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from make_ablation_summary_figure import (
    _pair_arms,
    _summarise_component,
    collect_all_summaries,
    collect_stats,
    main,
)


def _write_paired(
    run_dir: Path,
    *,
    on_arm: str,
    off_arm: str,
    on_pass: list[bool],
    off_pass: list[bool],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (a, b) in enumerate(zip(on_pass, off_pass)):
        base = {"model": "m", "input_id": f"p{i}", "replicate": 0}
        rows.append({
            **base, "arm": on_arm,
            "overall_pass": a, "completeness_pass": a,
            "tools_present_fraction": 0.9 if a else 0.5,
            "hallucination_rate": 0.0 if a else 0.2,
            "dag_edge_density": 1.0 if a else 0.0,
            "stage_efficiency": 1.0 if a else 0.8,
            "cost_usd": 0.01,
        })
        rows.append({
            **base, "arm": off_arm,
            "overall_pass": b, "completeness_pass": b,
            "tools_present_fraction": 0.7 if b else 0.4,
            "hallucination_rate": 0.1 if b else 0.3,
            "dag_edge_density": 0.0,
            "stage_efficiency": 0.9 if b else 0.7,
            "cost_usd": 0.012,
        })
    pd.DataFrame(rows).to_csv(run_dir / "paired_metrics.csv", index=False)


def test_pair_arms_inner_join() -> None:
    df = pd.DataFrame([
        {"model": "m", "input_id": "a", "replicate": 0, "arm": "dag_aware",
         "overall_pass": True},
        {"model": "m", "input_id": "a", "replicate": 0, "arm": "dag_blind",
         "overall_pass": False},
        {"model": "m", "input_id": "b", "replicate": 0, "arm": "dag_aware",
         "overall_pass": True},
    ])
    on, off = _pair_arms(df, "dag_aware", "dag_blind")
    assert len(on) == 1
    assert bool(on.iloc[0]["overall_pass"]) is True
    assert bool(off.iloc[0]["overall_pass"]) is False


def test_summarise_component_computes_delta() -> None:
    tmp = Path(pytest.importorskip("tempfile").mkdtemp())
    run = tmp / "ablation" / "2026-01-01T00-00-00"
    _write_paired(
        run,
        on_arm="dag_aware",
        off_arm="dag_blind",
        on_pass=[True, True, False],
        off_pass=[False, True, False],
    )
    stat = _summarise_component(
        subdir="ablation",
        label="DAG",
        on_arm="dag_aware",
        off_arm="dag_blind",
        metric="overall_pass",
        paired_csv=run / "paired_metrics.csv",
    )
    assert stat is not None
    assert stat.n_pairs == 3
    assert stat.mean_on == pytest.approx(2 / 3)
    assert stat.mean_off == pytest.approx(1 / 3)
    assert stat.c_only == 1


def test_collect_all_summaries_includes_continuous_metrics(tmp_path: Path) -> None:
    run = tmp_path / "ablation" / "2026-01-01T00-00-00"
    _write_paired(
        run,
        on_arm="dag_aware",
        off_arm="dag_blind",
        on_pass=[True, True],
        off_pass=[True, False],
    )
    rows = collect_all_summaries(tmp_path)
    metrics = {r.metric for r in rows if r.subdir == "ablation"}
    assert "tools_present_fraction" in metrics
    assert "hallucination_rate" in metrics
    assert "cost_usd" in metrics


def test_collect_and_main(tmp_path: Path) -> None:
    for subdir, on_arm, off_arm in [
        ("ablation", "dag_aware", "dag_blind"),
        ("reflection", "reflect_on", "reflect_off"),
        ("validator_ablation", "validator_on", "validator_off"),
        ("cove_ablation", "verifier_on", "verifier_off"),
        ("tool_hint_ablation", "hint_on", "hint_off"),
    ]:
        run = tmp_path / subdir / "2026-01-01T00-00-00"
        _write_paired(
            run,
            on_arm=on_arm,
            off_arm=off_arm,
            on_pass=[True, True],
            off_pass=[True, False],
        )

    stats = collect_stats(tmp_path)
    assert len(stats) == 10  # 5 components × 2 metrics

    out = tmp_path / "fig" / "ablation_summary"
    assert main(["--results-base", str(tmp_path), "--out", str(out)]) == 0
    assert out.with_suffix(".pdf").is_file()
    assert Path(str(out) + "_secondary.pdf").is_file()
    assert Path(str(out) + "__stats.tsv").is_file()
    assert Path(str(out) + "__metrics.tsv").is_file()
    assert Path(str(out) + "__metrics_wide.tsv").is_file()
