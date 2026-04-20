"""Benchmark E — Head-to-head against other agentic bioinformatics systems.

Each competitor (FlowAgent baseline, BioMaster, and future additions)
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
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE_DIR = Path(__file__).parent
sys.path.insert(0, str(_HERE_DIR))
sys.path.insert(0, str(_HERE_DIR.parent))

from harness.competitors import (                              # noqa: E402
    Competitor, CompetitorResult, build_registry, _empty_plan,
)
from harness.metrics import score_plan, cost_usd               # noqa: E402
from harness.runner import (                                    # noqa: E402
    load_yaml, set_provider, timestamped_dir, write_manifest, _write_csv,
)

LOG = logging.getLogger("bench_competitors")
HERE = Path(__file__).parent


# ── Mock fallback ────────────────────────────────────────────────

def _mock_plan(prompt_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic plan derived from ``expected_tools`` / ``gold_preset``.

    Same heuristic as ``bench_planning._mock_plan`` — one step per expected
    tool, padded to ``expected_min_steps``. Not a real agent invocation;
    used only when ``--mock`` is passed so the harness runs offline.
    """
    gold = prompt_entry.get("gold_preset")
    if gold:
        try:
            from flowagent.presets.catalog import get_preset
            preset = get_preset(gold)
            if preset:
                return {"workflow_type": preset["workflow_type"],
                        "steps": copy.deepcopy(preset["steps"])}
        except Exception:
            pass
    tools = prompt_entry.get("expected_tools") or ["fastqc"]
    steps = [
        {"name": f"s{i}", "command": f"{t} input.fastq.gz",
         "dependencies": [f"s{i-1}"] if i else [],
         "outputs": [], "description": ""}
        for i, t in enumerate(tools)
    ]
    wf = prompt_entry.get("expected_workflow_type", "custom")
    if isinstance(wf, list):
        wf = wf[0]
    return {"workflow_type": wf, "steps": steps}


# ── Per-cell runner ──────────────────────────────────────────────

async def _run_cell(competitor: Competitor, prompt_entry: Dict[str, Any],
                    replicate: int, *, mock: bool, timeout: float,
                    model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    row_base = {
        "competitor":  competitor.id,
        "competitor_name": competitor.name,
        "input_id":    prompt_entry["id"],
        "prompt":      prompt_entry["prompt"],
        "replicate":   replicate,
        "model":       model_cfg.get("id", ""),
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
                print(f"{status[:40]} ({elapsed:.1f}s)", flush=True)
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

    pass = ``overall_pass`` is True
    crash = ``error`` is set AND the plan has zero scored steps (no plan
            produced — distinct from a plan that failed scoring)
    fail = everything else (scored but didn't pass)
    """
    by_comp: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_comp.setdefault(r.get("competitor", "?"), []).append(r)

    out: List[Dict[str, Any]] = []
    for comp, cells in by_comp.items():
        total = len(cells)
        n_pass = sum(1 for r in cells if r.get("overall_pass"))
        n_crash = sum(
            1 for r in cells
            if r.get("error") and not (r.get("plan") or {}).get("steps")
        )
        n_fail = total - n_pass - n_crash
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
            "mean_cost_usd": _mean("cost_usd"),
            "mean_wall_s": _mean("wall_seconds"),
        })
    # Sort descending by pass rate so the leader is on top.
    out.sort(key=lambda s: (-s["pass_rate"], s["competitor"]))
    return out


def _print_summary(summary: List[Dict[str, Any]]) -> None:
    print("\nHead-to-head rollup (pass / fail / crash per competitor):")
    header = (f"  {'Competitor':<14} {'Pass':>8} {'Fail':>6} "
              f"{'Crash':>6} {'Pass%':>7} {'$/cell':>9} {'Wall':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for s in summary:
        pr = f"{s['pass_rate'] * 100:.1f}%"
        cost = f"${s['mean_cost_usd']:.4f}"
        wall = f"{s['mean_wall_s']:.1f}s"
        row = (f"  {s['name']:<14} "
               f"{s['pass']:>3}/{s['total']:<4} "
               f"{s['fail']:>6} {s['crash']:>6} "
               f"{pr:>7} {cost:>9} {wall:>7}")
        print(row)


def _format_summary_tsv(summary: List[Dict[str, Any]]) -> str:
    keys = ["competitor", "name", "total", "pass", "fail", "crash",
            "pass_rate", "mean_cost_usd", "mean_wall_s"]
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
                    help="Comma-separated competitor ids "
                         "(default: all registered)")
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

    # Competitors
    registry = build_registry(model_cfg=model_cfg)
    if args.competitors:
        wanted = set(args.competitors.split(","))
        registry = {k: v for k, v in registry.items() if k in wanted}
        if not registry:
            raise SystemExit(f"No known competitors in {args.competitors!r}")

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
    import os
    main()
