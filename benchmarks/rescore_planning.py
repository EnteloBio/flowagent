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

    # Rescore every run under results/planning/ (needed before merge)
    python rescore_planning.py --all

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

from harness.metrics import score_plan, cost_usd    # noqa: E402
from harness.runner import _write_csv, load_yaml    # noqa: E402


def _planning_run_dirs(base: Path) -> List[Path]:
    pdir = base / "planning"
    if not pdir.exists():
        return []
    return sorted(
        (p for p in pdir.iterdir()
         if p.is_dir()
         and not p.name.startswith("rescored")
         and not p.name.startswith("_")
         and (p / "results.json").exists()),
        key=lambda p: p.stat().st_mtime,
    )


def _latest_planning_run(base: Path) -> Optional[Path]:
    runs = _planning_run_dirs(base)
    return runs[-1] if runs else None


def rescore_run(
    run_dir: Path,
    *,
    prompts_path: Path,
    models_cfg: Dict[str, Any],
) -> Path:
    """Re-score one planning run; return the ``rescored_<ts>/`` output dir."""
    src = run_dir / "results.json"
    if not src.exists():
        raise FileNotFoundError(f"{src} does not exist")

    rows = json.loads(src.read_text())
    prompts_by_id = {p["id"]: p for p in load_yaml(prompts_path)["prompts"]}
    pricing_by_id = {m["id"]: m for m in models_cfg.get("models", [])}

    rescored: List[Dict[str, Any]] = []
    for row in rows:
        plan = row.get("plan")
        prompt_id = row.get("input_id") or row.get("prompt_id")
        expected = prompts_by_id.get(prompt_id)
        if not (plan and expected):
            rescored.append(row)  # pass-through if we can't rescore
            continue
        new_metrics = score_plan(plan, expected)

        prompt_tok     = int(row.get("prompt_tokens", 0) or 0)
        completion_tok = int(row.get("completion_tokens", 0) or 0)
        cost = cost_usd(prompt_tok, completion_tok,
                        pricing_by_id.get(row.get("model"), {}))

        rescored.append({
            **{k: row[k] for k in
               ("model", "provider", "input_id", "prompt", "replicate",
                "wall_seconds", "plan") if k in row},
            "prompt_tokens":     prompt_tok,
            "completion_tokens": completion_tok,
            "llm_calls":         int(row.get("llm_calls", 0) or 0),
            "cost_usd":          cost,
            **new_metrics,
        })

    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out = run_dir / f"rescored_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(rescored, indent=2, default=str))
    _write_csv(out / "metrics.csv", rescored)

    n = len(rescored)
    n_pass = sum(1 for r in rescored if r.get("overall_pass"))
    n_hall = sum(1 for r in rescored if int(r.get("num_hallucinated_tools") or 0) > 0)
    print(f"[ok] rescored {n} rows → {out}")
    print(f"     overall_pass:  {n_pass}/{n}  ({100*n_pass/n:.0f}%)")
    print(f"     any_hallucination (typo+unknown): {n_hall}/{n}  ({100*n_hall/n:.0f}%)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="Path to a planning run directory; "
                                   "defaults to the most recent under results/planning/")
    ap.add_argument("--all", action="store_true",
                    help="Rescore every run under results/planning/ "
                         "(use before merge when scoring logic changes)")
    ap.add_argument("--prompts", default=str(_HERE_DIR / "corpus" / "prompts.yaml"))
    ap.add_argument("--results-base", default="results")
    args = ap.parse_args()

    base = Path(args.results_base)
    prompts_path = Path(args.prompts)
    models_cfg = load_yaml(_HERE_DIR / "config" / "models.yaml")

    if args.all:
        runs = _planning_run_dirs(base)
        if not runs:
            raise SystemExit(f"No planning runs found under {base / 'planning'}")
        for run_dir in runs:
            print(f"[rescore] {run_dir.name}")
            rescore_run(run_dir, prompts_path=prompts_path, models_cfg=models_cfg)
        return

    run_dir = Path(args.run) if args.run else _latest_planning_run(base)
    if run_dir is None or not run_dir.exists():
        raise SystemExit(f"No planning run found at {args.run or args.results_base}")

    rescore_run(run_dir, prompts_path=prompts_path, models_cfg=models_cfg)


if __name__ == "__main__":
    main()
