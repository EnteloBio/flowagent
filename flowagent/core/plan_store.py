"""Helpers for saving and loading plan artifacts to/from disk.

A plan artifact is a JSON file with a well-known schema that can be
reviewed, version-controlled, and re-executed without re-prompting the
LLM. Alongside the raw plan a human-readable Markdown summary is written
so a bioinformatician can ``cat workflow.md`` to see what will run.
"""

from __future__ import annotations

import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


PLAN_FILENAME = "workflow.json"
PLAN_MD_FILENAME = "workflow.md"


def save_plan(
    plan: Dict[str, Any],
    output_dir: Path,
    *,
    prompt: Optional[str] = None,
    model: Optional[str] = None,
) -> Path:
    """Write *plan* as ``workflow.json`` + ``workflow.md`` inside *output_dir*.

    Returns the path to the JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / PLAN_FILENAME
    md_path = output_dir / PLAN_MD_FILENAME

    # Annotate with provenance
    annotated = dict(plan)
    annotated.setdefault("_meta", {})
    annotated["_meta"]["saved_at"] = datetime.now(timezone.utc).isoformat()
    if prompt:
        annotated["_meta"]["prompt"] = prompt
    if model:
        annotated["_meta"]["model"] = model

    plan_path.write_text(json.dumps(annotated, indent=2))
    md_path.write_text(_plan_to_markdown(annotated))
    return plan_path


def load_plan(path: Path) -> Dict[str, Any]:
    """Load a plan from *path* (directory or explicit .json file)."""
    if path.is_dir():
        path = path / PLAN_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Plan not found: {path}")
    return json.loads(path.read_text())


def diff_plans(a: Dict[str, Any], b: Dict[str, Any]) -> str:
    """Return a human-readable diff of two plan dicts."""
    lines: List[str] = []
    steps_a = {s["name"]: s for s in a.get("steps", [])}
    steps_b = {s["name"]: s for s in b.get("steps", [])}
    all_names = list(steps_a) + [n for n in steps_b if n not in steps_a]

    for name in all_names:
        sa = steps_a.get(name)
        sb = steps_b.get(name)
        if sa is None:
            lines.append(f"+ [new]    {name}")
            lines.append(f"           cmd: {sb.get('command','')}")
        elif sb is None:
            lines.append(f"- [removed] {name}")
        else:
            cmd_a = sa.get("command", "")
            cmd_b = sb.get("command", "")
            deps_a = set(sa.get("dependencies", []))
            deps_b = set(sb.get("dependencies", []))
            if cmd_a != cmd_b:
                lines.append(f"~ [changed] {name}")
                lines.append(f"  - {_truncate(cmd_a)}")
                lines.append(f"  + {_truncate(cmd_b)}")
            elif deps_a != deps_b:
                added_deps = deps_b - deps_a
                removed_deps = deps_a - deps_b
                lines.append(f"~ [deps]    {name}")
                for d in sorted(added_deps):
                    lines.append(f"  + dep: {d}")
                for d in sorted(removed_deps):
                    lines.append(f"  - dep: {d}")
            else:
                lines.append(f"  [same]    {name}")

    if not lines:
        return "(plans are identical)"
    return "\n".join(lines)


# ── Internal ──────────────────────────────────────────────────

def _plan_to_markdown(plan: Dict[str, Any]) -> str:
    """Convert a plan dict to a human-readable Markdown summary."""
    meta = plan.get("_meta", {})
    name = plan.get("name", plan.get("workflow_type", "Workflow"))
    desc = plan.get("description", "")
    steps = plan.get("steps", [])
    prompt = meta.get("prompt", "")
    model = meta.get("model", "")
    saved_at = meta.get("saved_at", "")

    parts: List[str] = [f"# {name}"]
    if desc:
        parts.append(f"\n{desc}")
    parts.append("\n---\n")
    if prompt:
        parts.append(f"**Prompt:** {prompt}\n")
    if model:
        parts.append(f"**Model:** {model}\n")
    if saved_at:
        parts.append(f"**Generated:** {saved_at}\n")
    parts.append(f"\n## Steps ({len(steps)} total)\n")

    for i, step in enumerate(steps, 1):
        name_s = step.get("name", f"step_{i}")
        cmd = step.get("command", "")
        deps = step.get("dependencies", [])
        desc_s = step.get("description", "")
        parts.append(f"### {i}. `{name_s}`")
        if desc_s:
            parts.append(f"\n{desc_s}\n")
        if deps:
            parts.append(f"\n**Depends on:** {', '.join(deps)}\n")
        if cmd:
            # Wrap long commands
            wrapped = textwrap.fill(cmd, width=100, subsequent_indent="  ")
            parts.append(f"\n```bash\n{wrapped}\n```\n")

    return "\n".join(parts)


def _truncate(s: str, width: int = 80) -> str:
    return s if len(s) <= width else s[:width - 3] + "..."
