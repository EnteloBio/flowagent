"""Re-score an existing planning benchmark run without re-calling the LLM.

Useful when scoring logic changes (e.g. tightened/loosened metrics or
new prompt synonyms) and you don't want to spend another ~$2 + 20 min to
regenerate the underlying plans.

Reads:  results/planning/<TS>/results.json
        corpus/prompts.yaml (current expectations)

Writes: results/planning/<TS>/rescored_<NEW_TS>/{results.json, metrics.csv}
        Original files are NOT modified.

Usage:
    # Rescore the latest planning run
    python rescore_planning.py

    # Rescore a specific run
    python rescore_planning.py --run results/planning/2026-04-13T11-08-06
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE_DIR = Path(__file__).parent
sys.path.insert(0, str(_HERE_DIR))
sys.path.insert(0, str(_HERE_DIR.parent))

from harness.metrics import score_plan          # noqa: E402
from harness.runner import _write_csv, load_yaml  # noqa: E402


def _latest_planning_run(base: Path) -> Optional[Path]:
    pdir = base / "planning"
    if not pdir.exists():
        return None
    runs = [p for p in pdir.iterdir() if p.is_dir() and not p.name.startswith("rescored")]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="Path to a planning run directory; "
                                   "defaults to the most recent under results/planning/")
    ap.add_argument("--prompts", default=str(_HERE_DIR / "corpus" / "prompts.yaml"))
    ap.add_argument("--results-base", default="results")
    args = ap.parse_args()

    run_dir = Path(args.run) if args.run else _latest_planning_run(Path(args.results_base))
    if run_dir is None or not run_dir.exists():
        raise SystemExit(f"No planning run found at {args.run or args.results_base}")

    src = run_dir / "results.json"
    if not src.exists():
        raise SystemExit(f"{src} does not exist")

    rows = json.loads(src.read_text())
    prompts_by_id = {p["id"]: p for p in load_yaml(Path(args.prompts))["prompts"]}

    rescored: List[Dict[str, Any]] = []
    for row in rows:
        plan = row.get("plan")
        prompt_id = row.get("input_id") or row.get("prompt_id")
        expected = prompts_by_id.get(prompt_id)
        if not (plan and expected):
            rescored.append(row)  # pass-through if we can't rescore
            continue
        # Re-compute every metric, preserve identifiers + plan + timing
        new_metrics = score_plan(plan, expected)
        rescored.append({
            **{k: row[k] for k in
               ("model", "provider", "input_id", "prompt", "replicate",
                "wall_seconds", "plan") if k in row},
            **new_metrics,
        })

    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out = run_dir / f"rescored_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(rescored, indent=2, default=str))
    _write_csv(out / "metrics.csv", rescored)

    # Summary stats
    n = len(rescored)
    n_valid = sum(1 for r in rescored if r.get("plan_valid"))
    n_type  = sum(1 for r in rescored if r.get("type_correct"))
    n_dag   = sum(1 for r in rescored if r.get("dag_valid"))
    n_pass  = sum(1 for r in rescored if r.get("overall_pass"))
    print(f"[ok] rescored {n} rows → {out}")
    print(f"     plan_valid:    {n_valid}/{n}  ({100*n_valid/n:.0f}%)")
    print(f"     type_correct:  {n_type}/{n}  ({100*n_type/n:.0f}%)")
    print(f"     dag_valid:     {n_dag}/{n}  ({100*n_dag/n:.0f}%)")
    print(f"     overall_pass:  {n_pass}/{n}  ({100*n_pass/n:.0f}%)")


if __name__ == "__main__":
    main()
