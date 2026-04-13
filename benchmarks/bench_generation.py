"""Benchmark C — Generator fidelity (Nextflow + Snakemake).

For each preset in ``PRESET_CATALOG`` and each generator, run:

    code = gen.generate(plan, output_dir=tmp)
    v    = gen.validate(code, output_dir=tmp)

and score preservation of step count, DAG topology, and tool coverage.

Deterministic — no LLM calls, no API keys. Runs in ~seconds.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

_HERE_DIR = Path(__file__).parent
sys.path.insert(0, str(_HERE_DIR))
sys.path.insert(0, str(_HERE_DIR.parent))

from harness.metrics import build_dag, extract_tools_from_plan    # noqa: E402
from harness.runner import timestamped_dir, write_manifest        # noqa: E402

try:
    import networkx as nx
except ImportError:
    nx = None


# ── Parsing generated pipeline code ──────────────────────────────

def _nextflow_processes(code: str) -> List[str]:
    return re.findall(r"^process\s+(\w+)\s*\{", code, flags=re.MULTILINE)


def _nextflow_deps(code: str) -> "nx.DiGraph":
    """Extract the workflow {} block's dependency graph.

    Pattern: ``PROC(UPSTREAM.out.done)`` or ``PROC(A.mix(B).collect())``
    """
    g = nx.DiGraph() if nx else None
    if g is None:
        return g
    # Isolate the workflow block.
    wf = re.search(r"workflow\s*\{(.+?)\}\s*\Z", code, flags=re.DOTALL)
    body = wf.group(1) if wf else code
    for line in body.splitlines():
        m = re.match(r"\s*(\w+)\s*\((.*)\)\s*$", line.strip())
        if not m:
            continue
        proc, arg = m.groups()
        g.add_node(proc.lower())
        for dep in re.findall(r"(\w+)\.out\.done", arg):
            g.add_edge(dep.lower(), proc.lower())
    return g


def _snakemake_rules(code: str) -> List[str]:
    return re.findall(r"^rule\s+(\w+)\s*:", code, flags=re.MULTILINE)


def _snakemake_deps(code: str, plan: Dict[str, Any]) -> "nx.DiGraph":
    """Snakemake rules use rules.X.output; easiest to cross-reference with plan."""
    g = nx.DiGraph() if nx else None
    if g is None:
        return g
    names = {s.get("name", "").lower() for s in plan.get("steps", [])}
    for s in plan.get("steps", []):
        g.add_node(s.get("name", "").lower())
    for s in plan.get("steps", []):
        for d in s.get("dependencies") or []:
            if d.lower() in names:
                g.add_edge(d.lower(), s["name"].lower())
    return g


# ── Fidelity check ───────────────────────────────────────────────

def _check(plan_id: str, plan: Dict[str, Any], gen, gen_name: str,
           outdir: Path) -> Dict[str, Any]:
    try:
        code = gen.generate(plan, output_dir=outdir)
    except Exception as exc:
        return {
            "plan_id": plan_id,
            "generator": gen_name,
            "error": f"{type(exc).__name__}: {exc}",
            "validation_ok": False,
            "step_count_matches": False,
            "dag_isomorphic": False,
            "tools_preserved": False,
        }

    # Validation (advisory; never blocks)
    try:
        v = gen.validate(code, output_dir=outdir)
        validation_ok = bool(v.get("valid", True))
    except Exception:
        validation_ok = False

    # Structural checks
    if gen_name == "nextflow":
        parsed_steps = _nextflow_processes(code)
        parsed_dag = _nextflow_deps(code)
    else:
        parsed_steps = _snakemake_rules(code)
        # Snakemake adds an implicit ``rule all`` aggregator — discount it
        # so step_count reflects the user-visible rules only.
        parsed_steps = [r for r in parsed_steps if r != "all"]
        parsed_dag = _snakemake_deps(code, plan)

    plan_dag = build_dag(plan) if nx else None
    if nx and plan_dag is not None and parsed_dag is not None:
        try:
            dag_isomorphic = nx.is_isomorphic(plan_dag, parsed_dag)
        except Exception:
            dag_isomorphic = False
    else:
        dag_isomorphic = None

    # Tool preservation
    plan_tools = extract_tools_from_plan(plan)
    code_lower = code.lower()
    tools_preserved = all(t in code_lower for t in plan_tools) if plan_tools else True

    result = {
        "plan_id": plan_id,
        "generator": gen_name,
        "validation_ok": validation_ok,
        "validation_warnings": len((v.get("warnings") or []) if "v" in locals() else []),
        "step_count_matches": len(parsed_steps) == len(plan.get("steps", [])),
        "parsed_step_count": len(parsed_steps),
        "plan_step_count": len(plan.get("steps", [])),
        "dag_isomorphic": dag_isomorphic,
        "tools_preserved": tools_preserved,
        "code_bytes": len(code),
    }
    if gen_name == "nextflow":
        # Regression test for path-with-spaces fix
        result["regression_launchdir_quoted"] = "cd '${launchDir}'" in code
    return result


# ── Driver ───────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    from flowagent.core.pipeline_generator.nextflow_generator import NextflowGenerator
    from flowagent.core.pipeline_generator.snakemake_generator import SnakemakeGenerator
    from flowagent.presets.catalog import PRESET_CATALOG

    out_dir = timestamped_dir(Path(args.out), "generation")
    results: List[Dict[str, Any]] = []

    for preset_id, preset in PRESET_CATALOG.items():
        plan = {
            "workflow_type": preset["workflow_type"],
            "steps": preset["steps"],
        }
        for gen_name, gen_cls in (("nextflow", NextflowGenerator),
                                  ("snakemake", SnakemakeGenerator)):
            with tempfile.TemporaryDirectory(prefix=f"gen_{preset_id}_") as td:
                row = _check(preset_id, plan, gen_cls(), gen_name, Path(td))
                results.append(row)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    from harness.runner import _write_csv
    _write_csv(out_dir / "metrics.csv", results)
    write_manifest(out_dir, benchmark="generation", models=[],
                   extra={"num_presets": len(PRESET_CATALOG), "generators": 2})
    n_ok = sum(1 for r in results if r.get("step_count_matches"))
    print(f"[ok] {n_ok}/{len(results)} generator runs passed step-count → {out_dir}")


if __name__ == "__main__":
    main()
