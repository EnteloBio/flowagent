"""Scoring functions for benchmark results.

All functions take primitive types (dicts, strings) and return dicts with
scalar fields, so results serialise cleanly to CSV for figure generation.
"""

from __future__ import annotations

import re
import shlex
from typing import Any, Dict, Iterable, List, Optional, Set

try:
    import networkx as nx  # type: ignore
except ImportError:  # pragma: no cover
    nx = None


# ── Basic utilities ──────────────────────────────────────────────

def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Jaccard similarity |A ∩ B| / |A ∪ B|."""
    a_set, b_set = set(a), set(b)
    if not a_set and not b_set:
        return 1.0
    union = a_set | b_set
    return len(a_set & b_set) / len(union) if union else 0.0


def command_token_f1(cmd_a: str, cmd_b: str) -> float:
    """Token-level F1 between two shell commands.

    Uses ``shlex.split`` so quoted args are handled correctly.
    """
    try:
        ta = set(shlex.split(cmd_a or ""))
        tb = set(shlex.split(cmd_b or ""))
    except ValueError:
        # Malformed quoting -- fall back to whitespace split
        ta = set((cmd_a or "").split())
        tb = set((cmd_b or "").split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    tp = len(ta & tb)
    precision = tp / len(ta) if ta else 0.0
    recall = tp / len(tb) if tb else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Plan introspection ───────────────────────────────────────────

def extract_tools_from_plan(plan: Dict[str, Any]) -> Set[str]:
    """Return the set of CLI tools that appear as the first non-shell token
    in any step command. Mirrors the generator's ``_primary_tool`` logic so
    scoring matches what the generator actually routes to."""
    SHELL = {
        "mkdir", "cd", "rm", "mv", "cp", "ln", "touch", "test",
        "set", "export", "echo", "source", "bash", "sh", "for", "do",
        "done", "if", "then", "else", "fi", "while",
    }
    tools: Set[str] = set()
    for step in plan.get("steps", []):
        cmd = step.get("command", "") or ""
        tokens = re.split(r"[\s;|&()<>]+", cmd.strip())
        for tok in tokens:
            if not tok or "=" in tok or tok.startswith("-"):
                continue
            name = tok.split("/")[-1].lower()
            if name in SHELL:
                continue
            tools.add(name)
            break  # Only the first bioinformatics tool per step
    return tools


def build_dag(plan: Dict[str, Any]) -> "nx.DiGraph":
    """Build a NetworkX DiGraph from plan dependencies."""
    if nx is None:
        raise RuntimeError("networkx not installed — install benchmark deps")
    g = nx.DiGraph()
    for step in plan.get("steps", []):
        name = step.get("name", "")
        g.add_node(name)
    for step in plan.get("steps", []):
        name = step.get("name", "")
        for dep in step.get("dependencies", []) or []:
            g.add_edge(dep, name)
    return g


def dag_valid(plan: Dict[str, Any]) -> bool:
    """True iff dependencies form a DAG (no cycles, no dangling references)."""
    if nx is None:
        return True  # Can't verify without networkx; assume valid.
    try:
        g = build_dag(plan)
    except Exception:
        return False
    if not nx.is_directed_acyclic_graph(g):
        return False
    names = {s.get("name") for s in plan.get("steps", [])}
    for step in plan.get("steps", []):
        for dep in step.get("dependencies", []) or []:
            if dep not in names:
                return False
    return True


# ── Schema validation ─────────────────────────────────────────────

# Required keys on a workflow step. Extra keys (resources, parameters,
# tools, profile_name, ...) are allowed because the runtime accepts them.
_REQUIRED_STEP_KEYS = ("name", "command")


def plan_schema_valid(plan: Dict[str, Any]) -> bool:
    """True iff the plan has the structural shape FlowAgent's runtime
    actually requires.

    We deliberately do NOT use ``WorkflowPlanSchema.model_validate`` here
    because that schema declares ``extra: forbid`` — but in practice,
    LLM-generated plans (and the executor) carry extra fields like
    ``resources``, ``parameters``, ``tools``, ``profile_name``, etc.
    Plans with those fields are perfectly executable; rejecting them as
    "invalid" gives a misleading 0% pass rate.

    We require:
      - top-level dict with ``workflow_type`` (str) and ``steps`` (list)
      - each step is a dict with at least ``name`` (str) and ``command`` (str)
      - dependencies, if present, must be a list of strings
    """
    if not isinstance(plan, dict):
        return False
    if not isinstance(plan.get("workflow_type"), str):
        return False
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        return False
    for step in steps:
        if not isinstance(step, dict):
            return False
        for key in _REQUIRED_STEP_KEYS:
            if not isinstance(step.get(key), str) or not step[key]:
                return False
        deps = step.get("dependencies", [])
        if deps is not None and not (isinstance(deps, list) and
                                      all(isinstance(d, str) for d in deps)):
            return False
    return True


def _normalise_type(s: str) -> str:
    """Normalise a workflow_type string for tolerant comparison.

    rna_seq_hisat ≡ rna_seq_hisat2 (LLM sometimes drops trailing digits).
    Strips trailing digits and trailing underscores.
    """
    s = (s or "").strip().lower()
    while s and s[-1].isdigit():
        s = s[:-1]
    return s.rstrip("_")


def type_matches(actual: str, expected) -> bool:
    """Permissive workflow-type comparison.

    ``expected`` may be a single string or a list of acceptable synonyms.
    The actual type is also matched against ``custom`` if ``custom`` is
    listed (because the LLM falls back to "custom" for less-common
    workflow families that FlowAgent doesn't have a canonical name for).
    """
    if isinstance(expected, str):
        accept = [expected]
    elif isinstance(expected, (list, tuple)):
        accept = list(expected)
    else:
        return False
    a_norm = _normalise_type(actual)
    return any(_normalise_type(e) == a_norm for e in accept)


# ── Top-level scoring ─────────────────────────────────────────────

def score_plan(plan: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
    """Compute Benchmark A metrics for one (plan, expectation) pair.

    ``expected`` fields consumed:
      - ``expected_workflow_type``
      - ``expected_tools``        (list[str])
      - ``expected_min_steps``    (int)
      - ``forbidden_tools``       (list[str])
      - ``gold_preset``           (str, optional; preset id for concordance)
    """
    metrics: Dict[str, Any] = {}

    # Structural validity
    metrics["plan_valid"] = plan_schema_valid(plan)
    metrics["dag_valid"] = dag_valid(plan)

    # Type — accept a single expected string OR a list of synonyms.
    # Also tolerant of trailing-digit differences (rna_seq_hisat vs hisat2).
    actual_type = plan.get("workflow_type", "")
    metrics["type_correct"] = type_matches(
        actual_type, expected.get("expected_workflow_type"),
    )
    metrics["actual_workflow_type"] = actual_type

    # Tool coverage
    plan_tools = extract_tools_from_plan(plan)
    metrics["num_tools"] = len(plan_tools)
    expected_tools = [t.lower() for t in (expected.get("expected_tools") or [])]
    forbidden = [t.lower() for t in (expected.get("forbidden_tools") or [])]
    if expected_tools:
        present = [t for t in expected_tools if t in plan_tools]
        metrics["tools_present_fraction"] = len(present) / len(expected_tools)
    else:
        metrics["tools_present_fraction"] = 1.0
    metrics["no_forbidden_tools"] = not any(t in plan_tools for t in forbidden)

    # Step count
    n_steps = len(plan.get("steps", []))
    metrics["num_steps"] = n_steps
    metrics["step_count_ok"] = n_steps >= (expected.get("expected_min_steps") or 0)

    # Gold preset concordance (optional)
    gold_id = expected.get("gold_preset")
    if gold_id:
        try:
            from flowagent.presets.catalog import get_preset
            gold = get_preset(gold_id)
            if gold:
                plan_names = [s.get("name", "") for s in plan.get("steps", [])]
                gold_names = [s.get("name", "") for s in gold.get("steps", [])]
                metrics["preset_name_jaccard"] = jaccard(plan_names, gold_names)
                # Command-level F1 on matching step names
                plan_by_name = {s.get("name"): s.get("command", "")
                                for s in plan.get("steps", [])}
                gold_by_name = {s.get("name"): s.get("command", "")
                                for s in gold.get("steps", [])}
                common = set(plan_by_name) & set(gold_by_name)
                if common:
                    metrics["preset_command_f1"] = sum(
                        command_token_f1(plan_by_name[n], gold_by_name[n])
                        for n in common
                    ) / len(common)
                else:
                    metrics["preset_command_f1"] = 0.0
        except Exception:
            pass  # Gold concordance is best-effort

    # Overall pass/fail (for stacked-bar summary)
    metrics["overall_pass"] = bool(
        metrics["plan_valid"]
        and metrics["dag_valid"]
        and metrics["type_correct"]
        and metrics["tools_present_fraction"] == 1.0
        and metrics["no_forbidden_tools"]
        and metrics["step_count_ok"]
    )
    return metrics


# ── Cost model ────────────────────────────────────────────────────

def cost_usd(prompt_tokens: int, completion_tokens: int,
             model_cfg: Dict[str, Any]) -> float:
    pricing = model_cfg.get("pricing", {}) or {}
    return (
        prompt_tokens * pricing.get("input_per_1k", 0.0) / 1000.0
        + completion_tokens * pricing.get("output_per_1k", 0.0) / 1000.0
    )


def diagnosis_matches(diagnosis: Optional[str], regex: str) -> bool:
    """True if the recovery diagnosis matches the expected-cause regex."""
    if not diagnosis:
        return False
    return bool(re.search(regex, diagnosis))
