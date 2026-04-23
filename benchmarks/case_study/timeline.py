"""Panel 5a — execution-trace Gantt for a single FlowAgent run.

Reads the workflow manifest + checkpoint files produced by the run,
renders a per-step Gantt coloured by outcome:
    completed  — green
    recovered  — amber  (Level-1 LLM recovery fired + succeeded)
    skipped    — grey   (smart-resume cached)
    failed     — red    (recovery failed or unrecoverable refusal)

Usage::

    python benchmarks/case_study/timeline.py \\
        --run-dir benchmarks/results/realworld_GSE52778 \\
        --out figures/fig5a_timeline

Produces ``fig5a_timeline.pdf`` + ``fig5a_timeline.svg`` next to each other.

Fields the script looks for in each step record (flexible — falls back to
sensible defaults when anything is missing):
    name, status, start_time, end_time, recovery_attempt, command
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch

_STATUS_COLOURS = {
    "completed": "#15803d",
    "success":   "#15803d",
    "recovered": "#E69F00",
    "skipped":   "#9ca3af",
    "cached":    "#9ca3af",
    "failed":    "#b91c1c",
    "error":     "#b91c1c",
    "rejected":  "#b91c1c",
}
_STATUS_LABELS = {
    "completed": "Completed",
    "recovered": "Recovered by LLM",
    "skipped":   "Skipped (smart-resume)",
    "failed":    "Failed / refused",
}


def _parse_ts(x: Any) -> Optional[float]:
    """Parse a timestamp that might be ISO-8601, epoch-seconds, or already a float."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return datetime.fromisoformat(str(x).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


def _normalise_status(raw: str, record: Dict[str, Any]) -> str:
    s = (raw or "").lower()
    if record.get("recovery_attempt"):
        return "recovered"
    if s in ("completed", "success"):
        return "completed"
    if s in ("skipped", "cached"):
        return "skipped"
    if s in ("failed", "error", "rejected"):
        return "failed"
    return "completed" if not s else s


def _load_steps(run_dir: Path) -> List[Dict[str, Any]]:
    """Pull step records from whichever files exist in the run directory."""
    candidates = [
        run_dir / "flowagent_output" / "Unnamed_Workflow" / "workflow_manifest.json",
        run_dir / "workflow_manifest.json",
        *run_dir.glob("flowagent_output/**/workflow_manifest.json"),
    ]
    for path in candidates:
        if path.is_file():
            with path.open() as f:
                blob = json.load(f)
            steps = (
                blob.get("steps")
                or blob.get("results")
                or blob.get("workflow", {}).get("steps")
            )
            if steps:
                return steps
    # Fallback: scan the checkpoint dir
    ckpt_dir = run_dir / "workflow_state"
    if ckpt_dir.is_dir():
        checkpoints = sorted(ckpt_dir.glob("*.json"))
        if checkpoints:
            with checkpoints[-1].open() as f:
                blob = json.load(f)
            return blob.get("steps") or blob.get("results") or []
    raise SystemExit(f"No workflow manifest or checkpoint found under {run_dir}")


def render(steps: List[Dict[str, Any]], out_prefix: Path, title: str) -> None:
    # Collect (name, status, start, end). For records missing times,
    # lay them out by declared order with unit-width bars so the figure
    # still communicates the status mix.
    bars: List[Dict[str, Any]] = []
    t0 = None
    for i, rec in enumerate(steps):
        start = _parse_ts(rec.get("start_time") or rec.get("started_at"))
        end = _parse_ts(rec.get("end_time") or rec.get("finished_at"))
        if start is not None and t0 is None:
            t0 = start
        bars.append({
            "name": rec.get("name") or rec.get("step_name") or f"step_{i+1}",
            "status": _normalise_status(str(rec.get("status", "")), rec),
            "start": start,
            "end": end,
            "index": i,
        })

    # Normalise times: if any are present, rebase to seconds-from-start and
    # fall back to unit-width for missing ones. If none present, use
    # unit-width bars stacked left-to-right.
    has_times = any(b["start"] is not None and b["end"] is not None for b in bars)
    if has_times:
        t0 = min(b["start"] for b in bars if b["start"] is not None)
        for b in bars:
            s = (b["start"] or t0) - t0
            e = (b["end"] or (s + 1)) - t0
            b["t_start"], b["t_width"] = s, max(e - s, 0.5)
    else:
        for i, b in enumerate(bars):
            b["t_start"], b["t_width"] = float(i), 0.9

    fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(bars) + 1.8))
    for b in bars:
        ax.barh(
            b["name"],
            b["t_width"],
            left=b["t_start"],
            color=_STATUS_COLOURS.get(b["status"], "#6b7280"),
            edgecolor="#111827",
            linewidth=0.5,
            alpha=0.95,
        )

    ax.set_xlabel("Wall time (s)" if has_times else "Execution order")
    ax.invert_yaxis()
    ax.set_title(title, fontsize=11, loc="left", pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", which="both", length=0)
    ax.grid(axis="x", linestyle=":", color="#d1d5db", alpha=0.6)
    ax.set_axisbelow(True)

    legend_handles = [
        Patch(facecolor=_STATUS_COLOURS[k], edgecolor="#111827", label=v)
        for k, v in _STATUS_LABELS.items()
    ]
    ax.legend(
        handles=legend_handles, loc="lower right",
        frameon=False, fontsize=9, ncol=2,
    )

    plt.tight_layout()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_prefix}.pdf")
    fig.savefig(f"{out_prefix}.svg")
    print(f"Wrote {out_prefix}.pdf and {out_prefix}.svg ({len(bars)} steps)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path,
                   help="Working directory of the flowagent run (contains workflow_state/, flowagent_output/)")
    p.add_argument("--out", required=True, type=Path,
                   help="Output prefix (without extension); .pdf and .svg written")
    p.add_argument("--title", default="GSE52778 case study — per-step execution trace")
    args = p.parse_args()

    steps = _load_steps(args.run_dir.resolve())
    render(steps, args.out, args.title)


if __name__ == "__main__":
    main()
