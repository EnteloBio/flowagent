"""Benchmark J -- DAG-awareness ablation for COMPETITOR frameworks.

Runs the same prompt corpus through a competitor's planner under two
prompt-level conditions and produces paired-prompt results so the
question "does telling a competitor's framework about the dependency
DAG improve plan quality?" can be answered with a Wilcoxon signed-rank
test on identical inputs.

Why this benchmark exists
-------------------------
Benchmark H (``bench_ablation.py``) tests FlowAgent's *own* DAG
awareness by toggling ``LLM_DAG_AWARE``, which flips both the planner
prompt AND the structured-output schema (``WorkflowPlanSchemaNoDAG``
loses the ``dependencies`` field entirely). That ablation conflates
"prompt-level DAG instruction" with "schema-level DAG enforcement".

For competitor frameworks we don't control the schema. The cleanest
question we *can* ask is: "Does *prompt-level* DAG instruction
improve the framework's output?" Specifically:

* Are the resulting plans actually DAGs (``dag_edge_density > 0``)?
* Do they have meaningful parallelism (``parallel_width > 1``)?
* Does plan quality change on the gating metrics
  (``overall_pass``, ``tools_present_fraction``, etc.)?

If a competitor improves only marginally with DAG instructions, that's
a signal that schema-level enforcement (FlowAgent's contribution) is
the necessary intervention -- not just any mention of the word
"dependency" in a prompt.

Currently supported competitors
-------------------------------
* ``claude_code`` -- Anthropic's Claude Code CLI, driven via
  :mod:`harness.claude_code_shim`. The shim ships with two prompt
  templates (DAG-aware vs DAG-blind); the default is DAG-blind for
  fair head-to-head comparisons in Benchmark E.
  ``ClaudeCodeCompetitor(with_dag=True)`` opts-in to the DAG-aware
  template via ``--with-dag-instruction`` -- this is what Benchmark
  J's ``dag_aware`` arm uses.

Other competitors (Biomni, Edison) need their own shim-level prompt
toggle before they can be added here. The framework is set up to take
them with minimal additional code.

Arms
----
* ``dag_aware`` -- competitor receives the DAG-aware prompt template
  (``dependencies`` field + topological-order rule).
* ``dag_blind`` -- competitor receives the DAG-blind prompt template
  (no ``dependencies`` field, no DAG / topological rules).

Output
------
``results/competitor_dag_ablation/<timestamp>/<competitor>/`` per arm:

  * ``dag_aware/results.jsonl`` + ``results.json`` + ``metrics.csv`` + ``manifest.json``
  * ``dag_blind/results.jsonl`` + ...
  * ``<competitor>__paired_metrics.csv`` -- per (input, replicate, arm)
                                            row, joined for the figure.

A top-level ``paired_metrics.csv`` consolidates every competitor's
paired rows (with a ``competitor`` column) so a single figure script
can render multi-competitor side-by-side panels.

Usage::

    # Pilot (3 prompts, 1 replicate, 2 arms) -- ~$0.50 on Claude Haiku 4.5
    python bench_competitor_dag_ablation.py \\
        --competitors=claude_code \\
        --model=claude-haiku-4-5 \\
        --replicates=1 --limit=3

    # Full sweep
    python bench_competitor_dag_ablation.py \\
        --competitors=claude_code \\
        --model=claude-sonnet-4-5 \\
        --replicates=3
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from harness.competitors import (                              # noqa: E402
    ClaudeCodeCompetitor, Competitor,
)
from harness.runner import (                                   # noqa: E402
    load_yaml, set_provider, sweep, timestamped_dir, write_manifest,
)
from bench_competitors import _run_cell                        # noqa: E402

LOG = logging.getLogger("bench_competitor_dag_ablation")


# ── Competitor factory: builds the two ablation arms for a given id ─

def _build_arm_competitors(
    competitor_id: str, *, model: Optional[str],
) -> Dict[str, Competitor]:
    """Return ``{"dag_aware": <comp>, "dag_blind": <comp>}`` for a slug.

    Adding a new competitor here is the only place the rest of the
    benchmark needs to know about. Each competitor must already
    implement a ``with_dag`` kwarg + a shim-level prompt-template
    toggle (default DAG-blind for fair head-to-head; ``with_dag=True``
    opts-in to the DAG-aware template). The competitor's slug differs
    across arms (e.g. ``claude_code`` vs ``claude_code_dag_aware``),
    handled transparently by ``__init__``.
    """
    competitor_id = competitor_id.lower()
    if competitor_id == "claude_code":
        return {
            "dag_aware": ClaudeCodeCompetitor(model=model, with_dag=True),
            "dag_blind": ClaudeCodeCompetitor(model=model, with_dag=False),
        }
    raise SystemExit(
        f"Unknown competitor for Benchmark J: {competitor_id!r}. "
        f"Currently supported: claude_code. "
        f"Adding Biomni / Edison requires the same prompt-template "
        f"toggle (and ``with_dag`` kwarg) the Claude Code shim has. "
        f"Edison already exposes ``with_dag`` -- enabling the J arm "
        f"for it just needs an entry here."
    )


# ── Per-cell runner that delegates to bench_competitors._run_cell ───

def _make_runner(competitor: Competitor, *, mock: bool, timeout: float,
                 model_cfg: Dict[str, Any], arm: str,
                 competitor_logical_id: str):
    """Wrap ``_run_cell`` to match ``sweep``'s ``(model_cfg, entry, rep)``
    signature so we get resume + concurrency + flat results.jsonl for free.

    The returned coroutine normalises ``row['competitor']`` to the
    *logical* competitor slug (``claude_code``) -- not the arm-specific
    one (``claude_code_dag_aware``). ``_run_cell`` populates
    ``competitor`` with ``competitor.id`` which differs across arms so
    the two ablation arms can co-exist in the same registry; that's
    fine for Benchmark E but breaks the paired-CSV joiner here, which
    pivots ``arm`` against a single logical competitor. We preserve
    the per-arm slug under ``competitor_arm`` so a figure / debugger
    can still verify each row came from the right shim invocation.
    """
    async def _runner(model_cfg_arg: Dict[str, Any],
                      entry: Dict[str, Any], rep: int) -> Dict[str, Any]:
        row = await _run_cell(
            competitor, entry, rep,
            mock=mock, timeout=timeout, model_cfg=model_cfg_arg,
        )
        # Tag with arm + (per-arm slug) and rewrite ``competitor`` to
        # the logical id so per-arm rows join under the same key.
        row["arm"] = arm
        row["competitor_arm"] = competitor.id
        row["competitor"] = competitor_logical_id
        return row
    return _runner


# ── Paired CSV consolidator (per-competitor + cross-competitor) ─────

def _write_paired_csv(out_dir: Path,
                      arms: Dict[str, List[Dict[str, Any]]],
                      *, basename: str = "paired_metrics") -> Path:
    """Write a single CSV joining both arms by (model, input_id, replicate).

    Mirrors ``bench_ablation._write_paired_csv``: takes the union of
    scalar columns across the two arms, then writes one row per
    completed cell. The downstream figure script joins by
    ``(competitor, model, input_id, replicate)``.
    """
    target = out_dir / f"{basename}.csv"
    keys: List[str] = ["arm"]
    seen = {"arm"}
    for rows in arms.values():
        for r in rows:
            for k, v in r.items():
                if k in seen:
                    continue
                if isinstance(v, (str, int, float, bool)) or v is None:
                    keys.append(k); seen.add(k)
    with target.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for arm, rows in arms.items():
            for r in rows:
                w.writerow({k: r.get(k) for k in keys})
    return target


def _load_prompts(path: Path, ids: Optional[List[str]],
                  *, limit: Optional[int]) -> List[Dict[str, Any]]:
    """Load prompts.yaml and optionally restrict to a subset / limit."""
    data = load_yaml(path)["prompts"]
    if ids:
        wanted = set(ids)
        keep = [p for p in data if p["id"] in wanted]
        missing = wanted - {p["id"] for p in keep}
        if missing:
            raise SystemExit(f"Unknown prompt ids: {sorted(missing)}")
        data = keep
    if limit:
        data = data[:limit]
    return data


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Competitor DAG-awareness ablation (Benchmark J).")
    ap.add_argument("--competitors", default="claude_code",
                    help="Comma-separated competitor ids to ablate "
                         "(default: claude_code). Currently the only "
                         "wired-up competitor is Claude Code.")
    ap.add_argument("--model", default="claude-haiku-4-5",
                    help="Driver model id passed to the competitor's "
                         "shim (default: claude-haiku-4-5).")
    ap.add_argument("--config", default=str(_HERE / "config" / "models.yaml"),
                    help="models.yaml for cost / provider lookup.")
    ap.add_argument("--prompts", default=None,
                    help="Comma-separated prompt ids (default: full corpus).")
    ap.add_argument("--prompts-file",
                    default=str(_HERE / "corpus" / "prompts.yaml"))
    ap.add_argument("--replicates", type=int, default=3,
                    help="Replicates per (competitor, prompt, arm) cell.")
    ap.add_argument("--limit", type=int, default=None,
                    help="If set, only the first N prompts are run.")
    ap.add_argument("--timeout", type=float, default=300,
                    help="Per-cell timeout in seconds (default 300).")
    ap.add_argument("--concurrency", type=int, default=2,
                    help="Concurrent cells (default 2 -- conservative because "
                         "Claude Code spawns Node per call).")
    ap.add_argument("--mock", action="store_true",
                    help="Use canned mock plans (no real CLI calls).")
    ap.add_argument("--arms", default="dag_aware,dag_blind",
                    help="Which arms to run (default: both, comma-separated).")
    ap.add_argument("--out", default="results")
    ap.add_argument("--resume", default=None,
                    help="Path to an existing results/competitor_dag_ablation/"
                         "<ts>/ dir; skips cells already in each arm's "
                         "results.jsonl.")
    args = ap.parse_args()

    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO"),
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Resolve model_cfg (used for cost lookup / provider routing). The
    # model id may not exist in models.yaml (e.g. Claude Code accepts
    # internal aliases that aren't priced in our YAML); fall back to a
    # bare cfg in that case so the harness still runs.
    cfg = load_yaml(Path(args.config))
    model_cfg = next(
        (m for m in cfg["models"] if m["id"] == args.model), None,
    )
    if model_cfg is None:
        LOG.warning(
            "Model %r not in %s -- using a stub cfg with zero pricing. "
            "Cost figures will come from each competitor's own usage tally.",
            args.model, args.config,
        )
        model_cfg = {"id": args.model, "provider": "anthropic",
                     "pricing": {"input_per_1k": 0.0, "output_per_1k": 0.0}}
    if not args.mock:
        try:
            set_provider(model_cfg)
        except Exception as e:
            # Claude Code drives its own auth; not having the matching
            # FlowAgent provider env var shouldn't block the run.
            LOG.warning("set_provider failed (non-fatal for Claude Code): %s", e)

    competitors_to_run = [c.strip() for c in args.competitors.split(",")
                          if c.strip()]
    if not competitors_to_run:
        raise SystemExit("--competitors must list at least one slug")

    arms_to_run = [a.strip() for a in args.arms.split(",") if a.strip()]
    valid_arms = {"dag_aware", "dag_blind"}
    bad = [a for a in arms_to_run if a not in valid_arms]
    if bad:
        raise SystemExit(f"Unknown arm(s): {bad}; valid: {sorted(valid_arms)}")

    prompt_ids = args.prompts.split(",") if args.prompts else None
    prompts = _load_prompts(Path(args.prompts_file), prompt_ids,
                            limit=args.limit)

    if args.resume:
        out_dir = Path(args.resume)
        if not out_dir.is_dir():
            raise SystemExit(f"--resume dir does not exist: {out_dir}")
        print(f"[resume] reusing {out_dir}")
    else:
        out_dir = timestamped_dir(Path(args.out), "competitor_dag_ablation")

    # Synthetic single-element ``models`` list so ``sweep`` (which loops
    # over ``models × inputs × replicates``) sees one logical "model"
    # and the row keys are clean.
    models_for_sweep = [model_cfg]

    # Cross-competitor combined paired CSV: one row per (competitor,
    # arm, model, input_id, replicate). Built as we go so a Ctrl-C
    # mid-sweep still leaves a useful partial CSV.
    cross_paired_rows: List[Dict[str, Any]] = []

    for comp_id in competitors_to_run:
        comp_dir = out_dir / comp_id
        comp_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n========== Competitor: {comp_id} ==========")

        try:
            arm_competitors = _build_arm_competitors(comp_id, model=args.model)
        except SystemExit as e:
            LOG.error(str(e))
            continue

        # Up-front availability check so we fail fast (and loudly) when
        # the CLI / API key is not configured -- much friendlier than
        # 198 cells all returning ``not-available``.
        for arm, comp in arm_competitors.items():
            ok, why = comp.available()
            print(f"  arm={arm}: {comp.name} -- "
                  f"{'available' if ok else 'NOT AVAILABLE: ' + why.splitlines()[0]}")

        arm_results: Dict[str, List[Dict[str, Any]]] = {}

        async def _run_competitor_arms() -> Dict[str, List[Dict[str, Any]]]:
            results: Dict[str, List[Dict[str, Any]]] = {}
            for arm in arms_to_run:
                arm_dir = comp_dir / arm
                arm_dir.mkdir(parents=True, exist_ok=True)
                comp = arm_competitors[arm]
                print(f"\n=== Arm: {comp_id}/{arm} (with_dag={arm == 'dag_aware'}) ===")

                runner = _make_runner(
                    comp, mock=args.mock, timeout=args.timeout,
                    model_cfg=model_cfg, arm=arm,
                    competitor_logical_id=comp_id,
                )
                sweep_result = await sweep(
                    runner,
                    models=models_for_sweep,
                    inputs=prompts,
                    replicates=args.replicates,
                    out_dir=arm_dir,
                    benchmark_name=f"competitor_dag_ablation:{comp_id}/{arm}",
                    concurrency=args.concurrency,
                )
                results[arm] = sweep_result.results
                print(f"[ok] arm={arm} wrote {len(sweep_result.results)} rows -> {arm_dir}")
            return results

        arm_results = asyncio.run(_run_competitor_arms())

        # Per-competitor paired CSV for the figure.
        if len(arm_results) >= 2:
            paired = _write_paired_csv(
                comp_dir, arm_results, basename="paired_metrics",
            )
            print(f"\n[paired:{comp_id}] -> {paired}")

        # Tag rows with the competitor before merging into the cross-
        # competitor CSV so the figure script can pivot on it.
        for arm, rows in arm_results.items():
            for r in rows:
                r.setdefault("competitor", comp_id)
                cross_paired_rows.append(r)

    # Combined cross-competitor paired CSV at the run root.
    if cross_paired_rows:
        keys: List[str] = ["competitor", "arm"]
        seen = set(keys)
        for r in cross_paired_rows:
            for k, v in r.items():
                if k in seen:
                    continue
                if isinstance(v, (str, int, float, bool)) or v is None:
                    keys.append(k); seen.add(k)
        cross_path = out_dir / "paired_metrics.csv"
        with cross_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in cross_paired_rows:
                w.writerow({k: r.get(k) for k in keys})
        print(f"\n[paired:cross] {len(cross_paired_rows)} rows -> {cross_path}")

    # Top-level manifest summarising the run.
    write_manifest(out_dir, benchmark="competitor_dag_ablation",
                   models=[model_cfg], extra={
                       "competitors": competitors_to_run,
                       "arms": arms_to_run,
                       "num_prompts": len(prompts),
                       "replicates": args.replicates,
                       "concurrency": args.concurrency,
                       "mock": args.mock,
                       "resumed": bool(args.resume),
                   })


if __name__ == "__main__":
    main()
