"""Smoke test for Benchmark M (tool-hint ablation)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent
PY = sys.executable


def test_tool_hint_ablation_mock(tmp_path):
    proc = subprocess.run(
        [
            PY, str(BENCH_DIR / "bench_tool_hint_ablation.py"),
            "--mock",
            "--models=gpt-4.1",
            "--replicates=1",
            "--limit=2",
            "--out", str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(BENCH_DIR),
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout

    run_dirs = list((tmp_path / "tool_hint_ablation").iterdir())
    assert run_dirs
    paired = run_dirs[0] / "paired_metrics.csv"
    assert paired.is_file()

    rows = paired.read_text().strip().splitlines()
    assert len(rows) >= 3  # header + 2 arms × 2 prompts

    for arm in ("hint_on", "hint_off"):
        metrics = run_dirs[0] / arm / "metrics.csv"
        assert metrics.is_file()
        results = json.loads((run_dirs[0] / arm / "results.json").read_text())
        assert results
        arms = {r["arm"] for r in results}
        assert arm in arms
