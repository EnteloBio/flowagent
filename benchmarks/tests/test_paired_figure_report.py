"""Tests for paired ablation figure discovery."""

from pathlib import Path

from harness.paired_figure_report import _latest_paired_csv


def test_latest_paired_csv_picks_newest_run(tmp_path: Path) -> None:
    root = tmp_path / "results"
    old = root / "ablation" / "2026-01-01T00-00-00"
    new = root / "ablation" / "2026-02-01T00-00-00"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    (old / "paired_metrics.csv").write_text("arm\n")
    (new / "paired_metrics.csv").write_text("arm\n")

    assert _latest_paired_csv(root, "ablation") == new / "paired_metrics.csv"


def test_latest_paired_csv_skips_empty_and_missing(tmp_path: Path) -> None:
    root = tmp_path / "results"
    empty = root / "ablation" / "empty-run"
    empty.mkdir(parents=True)
    (empty / "paired_metrics.csv").write_text("")

    assert _latest_paired_csv(root, "ablation") is None
    assert _latest_paired_csv(root, "missing_bench") is None
