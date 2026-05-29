"""Benchmark E — Head-to-head against other agentic bioinformatics systems.

Each competitor (FlowAgent baseline, BioMaster, AutoBA, Biomni, …)
implements the ``Competitor`` interface in ``harness/competitors.py`` and
must produce a FlowAgent-compatible plan dict. This module drives the
evaluation loop, scoring every competitor with the same ``score_plan``
metrics so the comparison is apples-to-apples.

Mock mode (``--mock``): uses canned responses derived from the prompt's
``gold_preset`` (where available) or the ``expected_tools`` list, so the
harness can be exercised without API keys. A warning is logged per
competitor whose real adapter is not installed.

Usage
-----
Full sweep::

    python bench_competitors.py --replicates=3

Subset of competitors / prompts::

    python bench_competitors.py --competitors=flowagent,biomaster \\
        --prompts=rnaseq_kallisto_basic,hard_full_germline_pipeline \\
        --replicates=2

Opt-in competitors
------------------
Some lanes are excluded from the default sweep because they are slow,
expensive, or not directly comparable (see :data:`_OPT_IN_COMPETITORS`).
To include them, name them explicitly via ``--competitors``. As of
this writing only ``edison`` is opt-in; routine ``make competitors``
runs therefore skip Edison Analysis. The full four-way comparison
remains available via ``make competitors-all`` (which lists ``edison``
explicitly) or ``--competitors=edison`` for an Edison-only sweep.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE_DIR = Path(__file__).parent
sys.path.insert(0, str(_HERE_DIR))
sys.path.insert(0, str(_HERE_DIR.parent))

from harness.competitors import (                              # noqa: E402
    Competitor, CompetitorResult, RawLLMCompetitor, build_registry, _empty_plan,
)
from harness.metrics import score_plan, cost_usd               # noqa: E402
from harness.mock_plans import mock_plan_from_prompt             # noqa: E402
from harness.runner import (                                    # noqa: E402
    load_yaml, set_provider, timestamped_dir, write_manifest, _write_csv,
)

LOG = logging.getLogger("bench_competitors")
HERE = Path(__file__).parent


# ── Opt-in competitors (excluded from the default sweep) ─────────
#
# Some competitors are too slow / expensive / unreliable to include in
# routine ``make competitors`` runs but are still wanted for the
# manuscript-grade comparison or for explicit opt-in via
# ``--competitors=<name>``.
#
# ``edison`` belongs here because:
#   * Its analysis API runs the workflow end-to-end (3-15 min/task) while
#     every other competitor only *plans* it — they are not directly
#     comparable on wall-clock or pass-rate.
#   * Each cell consumes real Edison credits even when the harness times
#     out, which makes accidental inclusion costly.
#   * The harness's default ``--timeout=180`` triggers a kill before
#     Edison's polling loop typically finishes (its own
#     ``EDISON_TIMEOUT`` defaults to 1800), so default-sweep cells fail
#     systematically.
#
# Edison stays in :func:`harness.competitors.build_default_competitor_registry`
# (so the DAG-toggle invariant tests and Benchmark J still see it), and
# it remains opt-in via ``--competitors=edison`` on the CLI or via
# ``make competitors-all`` which names it explicitly.
_OPT_IN_COMPETITORS: frozenset[str] = frozenset({"edison"})


def _filter_to_run_set(
    registry: Dict[str, Competitor],
    requested: Optional[str],
) -> Dict[str, Competitor]:
    """Apply the default-vs-explicit-opt-in selection for a sweep.

    * If ``requested`` is set (a comma-separated list of competitor ids,
      from ``--competitors``), return exactly that subset. Opt-in
      competitors are included when named explicitly.
    * Otherwise return the registry minus :data:`_OPT_IN_COMPETITORS`,
      so a routine ``make competitors`` invocation does not pull in
      slow / expensive lanes by accident.

    Raises :class:`SystemExit` if ``requested`` names no known competitor.
    """
    if requested:
        wanted = {s.strip() for s in requested.split(",") if s.strip()}
        out = {k: v for k, v in registry.items() if k in wanted}
        if not out:
            raise SystemExit(f"No known competitors in {requested!r}")
        return out
    return {k: v for k, v in registry.items() if k not in _OPT_IN_COMPETITORS}


# ── Mock fallback ────────────────────────────────────────────────

def _mock_plan(prompt_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic plan derived from ``expected_tools`` / ``gold_preset``.

    Same heuristic as ``bench_planning._mock_plan`` — one step per expected
    tool, padded to ``expected_min_steps``. Not a real agent invocation;
    used only when ``--mock`` is passed so the harness runs offline.
    """
    return mock_plan_from_prompt(prompt_entry, step_name=lambda i: f"s{i}")


# ── Per-cell runner ──────────────────────────────────────────────

async def _run_cell(competitor: Competitor, prompt_entry: Dict[str, Any],
                    replicate: int, *, mock: bool, timeout: float,
                    model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    row_model = model_cfg.get("id", "")
    if isinstance(competitor, RawLLMCompetitor):
        row_model = competitor.model_id
    row_base = {
        "competitor":  competitor.id,
        "competitor_name": competitor.name,
        "input_id":    prompt_entry["id"],
        "prompt":      prompt_entry["prompt"],
        "replicate":   replicate,
        "model":       row_model,
    }

    if mock:
        plan = _mock_plan(prompt_entry)
        metrics = score_plan(plan, prompt_entry)
        return {
            **row_base,
            "wall_seconds": 0.0,
            "prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0,
            "cost_usd": 0.0, "error": None,
            "plan": plan,
            **metrics,
        }

    ok, why = competitor.available()
    if not ok:
        LOG.warning("Skipping %s: %s", competitor.id, why.splitlines()[0])
        return {
            **row_base,
            "wall_seconds": 0.0,
            "prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0,
            "cost_usd": 0.0,
            "error": f"not-available: {why.splitlines()[0]}",
            "plan": _empty_plan(),
            **score_plan(_empty_plan(), prompt_entry),
        }

    try:
        t0 = time.perf_counter()
        cres: CompetitorResult = await asyncio.wait_for(
            competitor.plan(prompt_entry["prompt"]),
            timeout=timeout,
        )
        wall = time.perf_counter() - t0
    except asyncio.TimeoutError:
        return {
            **row_base,
            "wall_seconds": timeout,
            "prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0,
            "cost_usd": 0.0,
            "error": f"timeout after {timeout:.0f}s",
            "plan": _empty_plan(),
            **score_plan(_empty_plan(), prompt_entry),
        }
    except Exception as exc:
        return {
            **row_base,
            "wall_seconds": 0.0,
            "prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0,
            "cost_usd": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
            "plan": _empty_plan(),
            **score_plan(_empty_plan(), prompt_entry),
        }

    metrics = score_plan(cres.plan, prompt_entry)
    # Prefer the competitor's measured cost; else compute from tokens.
    final_cost = cres.cost_usd if cres.cost_usd > 0 else cost_usd(
        cres.prompt_tokens, cres.completion_tokens, model_cfg,
    )
    return {
        **row_base,
        "wall_seconds":      cres.wall_seconds or wall,
        "prompt_tokens":     cres.prompt_tokens,
        "completion_tokens": cres.completion_tokens,
        "llm_calls":         cres.llm_calls,
        "cost_usd":          final_cost,
        "error":             cres.error,
        "plan":              cres.plan,
        **metrics,
    }


# ── Driver ───────────────────────────────────────────────────────

async def _drive(competitors: Dict[str, Competitor],
                 prompts: List[Dict[str, Any]],
                 replicates: int, *, mock: bool, timeout: float,
                 model_cfg: Dict[str, Any],
                 out_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    total = len(competitors) * len(prompts) * replicates
    done = 0

    for competitor in competitors.values():
        ok, why = competitor.available()
        marker = "" if ok else f"  [not available: {why.splitlines()[0]}]"
        print(f"== {competitor.name}{marker}", flush=True)

        for entry in prompts:
            for rep in range(replicates):
                done += 1
                print(f"[{done}/{total}] {competitor.id} × "
                      f"{entry['id']} × rep{rep} ... ", end="", flush=True)
                t0 = time.perf_counter()
                row = await _run_cell(competitor, entry, rep,
                                      mock=mock, timeout=timeout,
                                      model_cfg=model_cfg)
                elapsed = time.perf_counter() - t0
                status = ("pass" if row.get("overall_pass")
                          else (row.get("error") or "fail"))
                # Show a short line in the progress stream; long errors (e.g. shim
                # tracebacks) would be unreadable. Print the full error on a
                # second line when it is long or non-trivial.
                line1 = str(status) if len(str(status)) <= 72 else str(status)[:69] + "…"
                print(f"{line1} ({elapsed:.1f}s)", flush=True)
                if row.get("error") and len(str(row["error"])) > 72:
                    print(f"    {row['error']}", flush=True)
                rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(rows, indent=2, default=str))
    _write_csv(out_dir / "metrics.csv", rows)
    write_manifest(
        out_dir, benchmark="competitors",
        models=[model_cfg] if model_cfg else [],
        extra={"competitors": list(competitors.keys()),
               "num_prompts": len(prompts),
               "replicates": replicates, "mock": mock},
    )

    # Per-competitor rollup: pass / fail / crash, plus mean cost and wall.
    # "crash" = an error row where we got no plan to score (distinct from
    # a scored plan that failed the scoring gates). Written alongside as
    # summary.tsv so it's easy to paste into a manuscript table.
    summary = _summarise(rows)
    _print_summary(summary)
    (out_dir / "summary.tsv").write_text(_format_summary_tsv(summary))
    return rows


def _summarise(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group rows by competitor and compute pass/fail/crash counts + means.

    Reports two co-primary outcomes per competitor:

    * ``pass_rate`` — strict ``overall_pass`` rate. Every rubric gate
      (workflow type, expected tools, forbidden tools, min step count,
      schema, DAG) must hold.
    * ``tool_recovery`` — mean ``tools_present_fraction`` over scored
      cells (crashes excluded). Partial-credit view, so a 5-of-6 plan
      contributes 0.83 instead of 0.

    pass = ``overall_pass`` is True
    crash = ``error`` is set AND the plan has zero scored steps (no plan
            produced — distinct from a plan that failed scoring)
    fail = everything else (scored but didn't pass)
    """
    by_comp: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_comp.setdefault(r.get("competitor", "?"), []).append(r)

    def _is_crash(r: Dict[str, Any]) -> bool:
        return bool(r.get("error")) and not (r.get("plan") or {}).get("steps")

    out: List[Dict[str, Any]] = []
    for comp, cells in by_comp.items():
        total = len(cells)
        n_pass = sum(1 for r in cells if r.get("overall_pass"))
        n_crash = sum(1 for r in cells if _is_crash(r))
        n_fail = total - n_pass - n_crash
        scored = [r for r in cells if not _is_crash(r)]
        if scored:
            tool_recovery = sum(
                float(r.get("tools_present_fraction") or 0.0) for r in scored
            ) / len(scored)
        else:
            tool_recovery = 0.0
        def _mean(key: str) -> float:
            vals = [float(r.get(key) or 0.0) for r in cells]
            return sum(vals) / len(vals) if vals else 0.0
        out.append({
            "competitor": comp,
            "name": cells[0].get("competitor_name", comp),
            "total": total,
            "pass": n_pass,
            "fail": n_fail,
            "crash": n_crash,
            "pass_rate": n_pass / total if total else 0.0,
            "tool_recovery": tool_recovery,
            "n_scored": len(scored),
            "mean_cost_usd": _mean("cost_usd"),
            "mean_wall_s": _mean("wall_seconds"),
        })
    # Sort descending by pass rate so the leader is on top.
    out.sort(key=lambda s: (-s["pass_rate"], s["competitor"]))
    return out


def _print_summary(summary: List[Dict[str, Any]]) -> None:
    print("\nHead-to-head rollup (two co-primary metrics + cost):")
    print("  Pass% = strict overall_pass rate.  "
          "Tools% = mean expected-tool fraction (partial credit, "
          "crashes excluded).")
    header = (f"  {'Competitor':<14} {'Pass':>8} {'Fail':>6} "
              f"{'Crash':>6} {'Pass%':>7} {'Tools%':>7} "
              f"{'$/cell':>9} {'Wall':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for s in summary:
        pr = f"{s['pass_rate'] * 100:.1f}%"
        tr = f"{s['tool_recovery'] * 100:.1f}%"
        cost = f"${s['mean_cost_usd']:.4f}"
        wall = f"{s['mean_wall_s']:.1f}s"
        row = (f"  {s['name']:<14} "
               f"{s['pass']:>3}/{s['total']:<4} "
               f"{s['fail']:>6} {s['crash']:>6} "
               f"{pr:>7} {tr:>7} {cost:>9} {wall:>7}")
        print(row)


def _format_summary_tsv(summary: List[Dict[str, Any]]) -> str:
    keys = ["competitor", "name", "total", "pass", "fail", "crash",
            "pass_rate", "tool_recovery", "n_scored",
            "mean_cost_usd", "mean_wall_s"]
    lines = ["\t".join(keys)]
    for s in summary:
        lines.append("\t".join(str(s[k]) for k in keys))
    return "\n".join(lines) + "\n"


def _load_prompts(path: Path, ids: Optional[List[str]]) -> List[Dict[str, Any]]:
    data = load_yaml(path)["prompts"]
    if not ids:
        return data
    wanted = set(ids)
    keep = [p for p in data if p["id"] in wanted]
    missing = wanted - {p["id"] for p in keep}
    if missing:
        raise SystemExit(f"Unknown prompt ids: {sorted(missing)}")
    return keep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--competitors",
                    help="Comma-separated competitor ids. Default: all "
                         "registered EXCEPT opt-in lanes "
                         f"({', '.join(sorted(_OPT_IN_COMPETITORS))}). "
                         "Name them explicitly to include them, e.g. "
                         "``--competitors=edison`` for an Edison-only sweep, "
                         "or use ``make competitors-all``.")
    ap.add_argument("--raw-models",
                    default="gpt-5.4,claude-opus-4-7,gemini-2.5-pro",
                    help="Comma-separated model IDs to run as zero-shot "
                         "raw-LLM baselines (one provider call, no "
                         "scaffolding). Each becomes a ``raw_<model_id>`` "
                         "competitor. Set to empty string to disable.")
    ap.add_argument("--prompts",
                    help="Comma-separated prompt ids "
                         "(default: 10 representative prompts)")
    ap.add_argument("--prompts-file",
                    default=str(HERE / "corpus" / "prompts.yaml"))
    ap.add_argument("--model", default="gpt-4.1",
                    help="Model ID — passed through to every competitor")
    ap.add_argument("--config", default=str(HERE / "config" / "models.yaml"))
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=180,
                    help="Per-cell timeout in seconds")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument(
        "--edison-budget-credits", type=float, default=None,
        help="Hard cap on cumulative Edison Analysis credits across this "
             "process (writes a shared budget file the shim reads). Once "
             "exceeded, further Edison cells short-circuit with an error "
             "envelope instead of submitting tasks. Has no effect if the "
             "Edison adapter is not registered or unavailable.",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO"),
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Model config — used for FlowAgent + cost lookup
    cfg = load_yaml(Path(args.config))
    model_cfg = next(
        (m for m in cfg["models"] if m["id"] == args.model), None,
    )
    if model_cfg is None:
        raise SystemExit(
            f"Model {args.model!r} not found in {args.config}. "
            f"Available: {[m['id'] for m in cfg['models']]}"
        )
    if not args.mock:
        set_provider(model_cfg)

    if args.edison_budget_credits is not None:
        # The Edison shim reads ``EDISON_BUDGET_CREDITS`` (and tracks
        # cumulative usage in ``EDISON_BUDGET_FILE``). Setting it here
        # lets a long sweep enforce the cap even though each task runs
        # in a fresh subprocess.
        os.environ["EDISON_BUDGET_CREDITS"] = str(args.edison_budget_credits)
        # Reset the budget tally for this run so a stale file from an
        # earlier sweep doesn't make every cell short-circuit.
        from tempfile import gettempdir
        budget_file = (
            os.environ.get("EDISON_BUDGET_FILE")
            or str(Path(gettempdir()) / "edison_budget.json")
        )
        try:
            Path(budget_file).write_text('{"credits_used": 0.0}')
        except Exception:
            pass

    # Competitors — includes zero-shot raw-LLM baselines alongside the
    # scaffolded agentic systems (FlowAgent / BioMaster / AutoBA /
    # Biomni / ClaudeCode / Edison).
    raw_models = [m.strip() for m in (args.raw_models or "").split(",") if m.strip()]
    registry = build_registry(
        model_cfg=model_cfg,
        raw_models=raw_models,
        models_yaml_cfg=cfg,
    )
    registry = _filter_to_run_set(registry, args.competitors)

    # Prompts — default to a compact balanced subset if none specified
    default_subset = [
        "rnaseq_kallisto_basic", "rnaseq_hisat2_htseq", "chipseq_macs2",
        "atacseq_basic", "variant_bwa_gatk", "scrna_kb",
        "hard_full_germline_pipeline", "hard_methylation_bismark_wgbs",
        "hard_rnaseq_full_de_pipeline", "hard_metagenomics_kraken",
    ]
    prompt_ids = (args.prompts.split(",") if args.prompts else default_subset)
    prompts = _load_prompts(Path(args.prompts_file), prompt_ids)

    out_dir = timestamped_dir(Path(args.out), "competitors")
    rows = asyncio.run(_drive(
        registry, prompts, args.replicates,
        mock=args.mock, timeout=args.timeout,
        model_cfg=model_cfg, out_dir=out_dir,
    ))
    n_pass = sum(1 for r in rows if r.get("overall_pass"))
    print(f"[ok] {n_pass}/{len(rows)} passed → {out_dir}")


if __name__ == "__main__":
    main()
