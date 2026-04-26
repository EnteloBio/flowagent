"""LLM-driven narrative for FlowAgent notebooks.

Renders prose that's tailored to the *specific* pipeline that ran. The
deterministic exporter in :mod:`flowagent.utils.export_notebook` produces
the same scaffold (plan table, code cells, status badges) for every run;
this module fills in the assay-aware content — what a reader of an RNA-seq
notebook would expect to see (transcript quantification, DE caveats,
glucocorticoid pathway interpretation), distinct from what a reader of a
ChIP-seq, ATAC-seq, or variant-calling notebook would expect.

Single LLM call per notebook. Returns a structured dict that the exporter
weaves into the cell stream. Failures fall back gracefully — a notebook
with the deterministic skeleton still ships.

Output schema:

    {
        "intro":               str,           # 2-3 markdown paragraphs
        "step_narrative":      {step: str},   # 1-2 sentences per step
        "results_interpretation": str,        # may be empty if no results
        "followup":            str,           # markdown bullet list
    }
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Cap context size — for a 30-step pipeline with verbose stderr, the raw
# results blob can blow past the model's context. We summarise.
_MAX_STDOUT_CHARS_PER_STEP = 600
_MAX_STDERR_CHARS_PER_STEP = 600

_SYSTEM_PROMPT = (
    "You are a careful computational-biology research assistant annotating "
    "a bioinformatics workflow notebook. Your prose will be embedded into "
    "a Jupyter notebook that captures the pipeline a colleague just ran. "
    "Tailor every paragraph to the specific assay (e.g. RNA-seq vs ChIP-seq "
    "vs methylation vs variant calling), reference genome, and dataset that "
    "the workflow plan reveals — generic boilerplate is not useful here. "
    "Do not invent results that are not in the supplied evidence; if a "
    "results block is empty, explain what the reader should look for once "
    "the run completes rather than fabricating numbers."
)


def _summarise_results(results: List[Dict[str, Any]]) -> str:
    """Compact textual digest of step results for the prompt context."""
    if not results:
        return "(no run results yet — pipeline has not executed)"
    lines: List[str] = []
    for r in results:
        name   = r.get("step_name") or r.get("name") or "?"
        status = r.get("status", "?")
        rc     = r.get("returncode")
        wall   = r.get("wall_seconds") or r.get("duration_seconds")
        wall_s = f"{wall:.1f}s" if isinstance(wall, (int, float)) else "?"
        head   = f"- {name} [{status}] rc={rc} wall={wall_s}"
        out    = (r.get("stdout") or "").strip()
        err    = (r.get("stderr") or "").strip()
        if out:
            head += f"\n    stdout: {out[:_MAX_STDOUT_CHARS_PER_STEP]}"
        if err:
            head += f"\n    stderr: {err[:_MAX_STDERR_CHARS_PER_STEP]}"
        lines.append(head)
    return "\n".join(lines)


def _summarise_plan(workflow: Dict[str, Any]) -> str:
    """Render the plan as numbered, dependency-annotated bullets."""
    lines = [f"Workflow name: {workflow.get('name', '?')}"]
    for i, s in enumerate(workflow.get("steps", []), 1):
        deps = ", ".join(s.get("dependencies") or []) or "—"
        lines.append(
            f"{i}. {s.get('name','?')}  (deps: {deps})\n"
            f"   description: {s.get('description','')}\n"
            f"   command: {(s.get('command') or '')[:300]}"
        )
    return "\n".join(lines)


def _build_user_prompt(prompt: Optional[str], workflow: Dict[str, Any],
                       results: List[Dict[str, Any]]) -> str:
    return (
        "Original natural-language request:\n"
        f"  \"\"\"\n  {prompt or '(none provided)'}\n  \"\"\"\n\n"
        "Workflow plan that was executed:\n"
        f"{_summarise_plan(workflow)}\n\n"
        "Results summary (per step, truncated):\n"
        f"{_summarise_results(results)}\n\n"
        "Return JSON with EXACTLY these four keys, and no other keys:\n\n"
        "  \"intro\": A 2-3 paragraph scientific overview. Open by naming "
        "the assay and any GEO/SRA/ENCODE accession present in the prompt. "
        "Mention the reference genome and the analytical question. End "
        "with a one-sentence pointer to the per-step cells below.\n\n"
        "  \"step_narrative\": A JSON object mapping each step name "
        "(verbatim) to a 1-2 sentence rationale explaining WHY this step "
        "exists in *this* pipeline. Do NOT rephrase the description; "
        "explain the technical purpose (\"kallisto pseudo-alignment was "
        "chosen over STAR because…\", \"FastQC is run before quantification "
        "to flag adapter contamination that would inflate misalignment "
        "rates…\").\n\n"
        "  \"results_interpretation\": If the results block contains real "
        "numbers, write a 2-3 paragraph interpretation citing specific "
        "step outputs (filenames, counts, exit codes) and flagging "
        "anything that warrants follow-up (failed steps, suspicious QC). "
        "If results are empty, write a 1-2 paragraph \"what to look for\" "
        "guide instead.\n\n"
        "  \"followup\": A markdown bullet list of 2-3 plausible "
        "follow-up analyses appropriate to *this* assay and dataset. "
        "Constrain suggestions to analyses that build on the data already "
        "in hand; mark anything requiring new data as such.\n\n"
        "Return ONLY the JSON object, with no markdown fences, no preamble, "
        "and no trailing commentary."
    )


_JSON_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


def _parse_response(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from an LLM reply, defensively."""
    if not text:
        return None
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    intro       = str(obj.get("intro") or "").strip()
    interp      = str(obj.get("results_interpretation") or "").strip()
    followup    = str(obj.get("followup") or "").strip()
    sn          = obj.get("step_narrative") or {}
    step_narrative: Dict[str, str] = {}
    if isinstance(sn, dict):
        for k, v in sn.items():
            if isinstance(k, str) and isinstance(v, str):
                step_narrative[k] = v.strip()
    if not (intro or interp or followup or step_narrative):
        return None
    return {
        "intro":                  intro,
        "step_narrative":         step_narrative,
        "results_interpretation": interp,
        "followup":               followup,
    }


async def generate_notebook_narrative(
    workflow: Dict[str, Any],
    *,
    prompt: Optional[str],
    results: Optional[List[Dict[str, Any]]] = None,
    llm: Any = None,
) -> Optional[Dict[str, Any]]:
    """Ask the configured LLM to write narrative for this notebook.

    Returns the parsed dict on success, ``None`` on any failure (no LLM,
    network error, malformed response). Callers should treat ``None`` as
    "skip narrative; ship the deterministic notebook unchanged".

    Args:
        workflow: Parsed ``workflow.json`` content.
        prompt:   Original user prompt.
        results:  Per-step result dicts (may be partial / empty).
        llm:      Optional pre-constructed ``LLMInterface``. Constructed
                  on demand if not supplied — uses whichever provider /
                  model the calling process has set in the environment.
    """
    try:
        if llm is None:
            from flowagent.core.llm import LLMInterface
            llm = LLMInterface()
        user = _build_user_prompt(prompt, workflow, results or [])
        # ``LLMInterface._call_openai`` is the provider-agnostic chat call
        # (the name is legacy — it routes through whichever provider is
        # configured). It takes a messages list and returns the assistant
        # text. We build a system + user pair so the model adheres to the
        # JSON-only contract baked into ``_SYSTEM_PROMPT``.
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user},
        ]
        reply = await llm._call_openai(messages)
        parsed = _parse_response(reply or "")
        if parsed is None:
            logger.warning("Notebook narrative: LLM reply did not parse as "
                           "valid JSON; falling back to deterministic notebook.")
        return parsed
    except Exception as exc:
        logger.warning("Notebook narrative generation failed: %s", exc)
        return None
