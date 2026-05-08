"""Subprocess shim that drives Anthropic's Claude Code CLI for Benchmark E.

Reads a prompt + optional input-file list on the command line, invokes
``claude --print --output-format json --permission-mode plan -p <prompt>``
in a scratch working directory, parses the CLI's JSON output, extracts
the embedded workflow plan, and prints a single JSON envelope on stdout
in the same shape used by ``biomni_shim.py`` / ``biomaster_shim.py`` /
``autoba_shim.py``.

Why a shim and not a direct subprocess call from competitors.py:

* Process isolation: Claude Code spawns a Node runtime with its own
  signal handling. Running it in a subprocess (the harness already
  invokes shims this way) gives us a real kill-on-timeout and prevents
  CC's own colour/TTY heuristics from corrupting the parent stdout.
* JSON envelope: the shim translates Claude Code's CLI output into the
  shared shim envelope so ``ClaudeCodeCompetitor._invoke_shim`` can
  reuse ``_parse_shim_stdout_json``.

CLI envelope (printed verbatim to stdout, single line / object):

    {
      "plan":              { "workflow_type": ..., "steps": [...] },
      "prompt_tokens":     <int>,
      "completion_tokens": <int>,
      "llm_calls":         <int>,
      "cost_usd":          <float>,
      "wall_seconds":      <float>,
      "dag_aware":         <bool>,
      "error":             <str | null>
    }

The plan shape matches FlowAgent's ``WorkflowPlanSchema`` so
``harness.metrics.score_plan`` can grade it on the same rubric.

DAG-awareness convention:

By default the shim uses a **DAG-blind** prompt template -- no
``dependencies`` field in the schema example, no topological-order
rule. This is the fair head-to-head baseline for Benchmark E:
FlowAgent's contribution is its DAG-aware planner, so giving Claude
Code a free DAG instruction in its prompt would be a confound.
Pass ``--with-dag-instruction`` to opt-in to the DAG-aware template
(the prompt-level equivalent of FlowAgent's ``LLM_DAG_AWARE=true``)
-- this is what Benchmark J's ``dag_aware`` arm uses to test whether
prompt-level DAG instruction alone changes Claude Code's plan
quality.

The toggle is symmetric -- the only difference between the two
templates is the ``dependencies`` field + topological-order rule.
Every other rule (tool-first command, no side effects, no markdown
fences, etc.) is byte-identical, pinned by the unit test
``TestSelectTemplate.test_non_dag_rules_unchanged_between_arms``.

Configuration knobs (env vars):

* ``CLAUDE_CODE_BIN``     : explicit path to the ``claude`` binary
                            (default: resolved via ``shutil.which``).
* ``CLAUDE_CODE_TIMEOUT`` : per-prompt wall-clock cap, seconds (default 300).
* ``CLAUDE_CODE_MODEL``   : model id passed via ``--model`` (default: CLI default).

Run as ``python claude_code_shim.py --prompt <text> [--files JSON]
[--model <id>] [--with-dag-instruction]``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Plan extraction (same JSON envelope as the planner) ─────────────
#
# Two templates. The DEFAULT (``_PLAN_INSTRUCTION_TEMPLATE_NO_DAG``) is
# the DAG-blind prompt -- it does NOT ask for a ``dependencies`` field
# and has no topological-order rule. This is the fair head-to-head
# baseline for Benchmark E: FlowAgent's differentiator is its
# DAG-aware planner, so Claude Code shouldn't get a free DAG
# instruction in its prompt.
#
# ``_PLAN_INSTRUCTION_TEMPLATE`` (DAG-aware) is opt-in via
# ``--with-dag-instruction``; this is the prompt-level equivalent of
# FlowAgent's ``LLM_DAG_AWARE=true`` and is used by Benchmark J's
# ``dag_aware`` arm. Toggling between the two is what Benchmark J
# actually measures.

_PLAN_INSTRUCTION_TEMPLATE = """You are a bioinformatics pipeline planner.

Task: {prompt}

Available input files (treat as already on disk): {files}

Return EXACTLY ONE JSON object describing the workflow plan, with this schema:

{{
  "workflow_type": "<rna_seq_kallisto | rna_seq_star | rna_seq_hisat | chip_seq | atac_seq | variant_calling | single_cell_10x | single_cell_kb | qc_only | custom>",
  "steps": [
    {{
      "name": "<unique_snake_case_id>",
      "command": "<runnable shell pipeline using real bioinformatics tools>",
      "dependencies": ["<prior step name>"],
      "outputs": ["<expected output paths>"]
    }}
  ]
}}

Rules:
- Steps must be in topological order.
- ``dependencies`` must reference names of prior steps exactly.
- The first token of each ``command`` MUST be the bioinformatics tool
  itself (``fastqc``, ``kallisto``, ``samtools``, etc.). Do not prepend
  ``mkdir -p`` or other shell prefixes -- emit a separate step for setup.
- Cover every step from raw input to the requested final output.
- Do NOT actually create files, edit code, or call any tools. Just emit
  the JSON plan.

Return ONLY the JSON object, no markdown fences, no commentary, no
explanation, no preamble.
"""


# DAG-blind variant for Benchmark J. Same shell of a prompt, but:
#   * the schema example has no ``dependencies`` field,
#   * the topological-order rule is removed,
#   * the dependency-reference rule is removed.
# Everything else (workflow_type vocabulary, command rules, no-side-
# effects rule, no-fences rule) is identical to the DAG-aware
# template, so the only experimental variable is the DAG instruction
# itself. Symmetric to FlowAgent's ``WorkflowPlanSchemaNoDAG``.
_PLAN_INSTRUCTION_TEMPLATE_NO_DAG = """You are a bioinformatics pipeline planner.

Task: {prompt}

Available input files (treat as already on disk): {files}

Return EXACTLY ONE JSON object describing the workflow plan, with this schema:

{{
  "workflow_type": "<rna_seq_kallisto | rna_seq_star | rna_seq_hisat | chip_seq | atac_seq | variant_calling | single_cell_10x | single_cell_kb | qc_only | custom>",
  "steps": [
    {{
      "name": "<unique_snake_case_id>",
      "command": "<runnable shell pipeline using real bioinformatics tools>",
      "outputs": ["<expected output paths>"]
    }}
  ]
}}

Rules:
- The first token of each ``command`` MUST be the bioinformatics tool
  itself (``fastqc``, ``kallisto``, ``samtools``, etc.). Do not prepend
  ``mkdir -p`` or other shell prefixes -- emit a separate step for setup.
- Cover every step from raw input to the requested final output.
- Do NOT actually create files, edit code, or call any tools. Just emit
  the JSON plan.

Return ONLY the JSON object, no markdown fences, no commentary, no
explanation, no preamble.
"""


def _select_template(*, dag_aware: bool) -> str:
    """Return the prompt template for the requested ablation arm.

    Exposed as a module-level helper so tests can pin the two arms
    diverge only in the DAG-related sentences -- never via accidental
    drift in unrelated rules.
    """
    return (
        _PLAN_INSTRUCTION_TEMPLATE
        if dag_aware
        else _PLAN_INSTRUCTION_TEMPLATE_NO_DAG
    )


# ── JSON extraction from free-form Claude output ────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first valid top-level JSON object out of free-form text.

    Tries, in order: (1) the whole text, (2) anything inside ```json …```
    fences, (3) a brace-balance scan for the longest top-level
    ``{ … }``. Returns ``None`` if nothing parses.
    """
    text = (text or "").strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for m in _FENCE_RE.finditer(text):
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            continue

    # Brace-balance scan. Walk until braces close at the same depth they
    # opened; try to parse each candidate. Stops at first parse-success.
    starts = [i for i, c in enumerate(text) if c == "{"]
    for start in starts:
        depth = 0
        for end in range(start, len(text)):
            c = text[end]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:end + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    return None


# ── Claude CLI invocation ──────────────────────────────────────────

def _resolve_claude_bin() -> Optional[str]:
    """Find the ``claude`` binary, honouring ``CLAUDE_CODE_BIN`` first."""
    explicit = os.environ.get("CLAUDE_CODE_BIN")
    if explicit:
        if Path(explicit).exists() and os.access(explicit, os.X_OK):
            return explicit
        return None
    found = shutil.which("claude")
    return found


def _invoke_claude(
    prompt_text: str,
    *,
    binary: str,
    model: Optional[str],
    timeout: float,
    cwd: Path,
) -> Tuple[Dict[str, Any], str, str, int]:
    """Run Claude Code in print-mode and return (parsed CLI JSON, stdout, stderr, returncode)."""
    argv: List[str] = [
        binary,
        "--print",
        "--output-format", "json",
        "--permission-mode", "plan",
    ]
    if model:
        argv += ["--model", model]
    argv += ["-p", prompt_text]

    proc = subprocess.run(
        argv,
        capture_output=True,
        text=False,
        cwd=str(cwd),
        timeout=timeout,
        check=False,
    )
    out = proc.stdout.decode(errors="replace")
    err = proc.stderr.decode(errors="replace")
    parsed: Dict[str, Any] = {}
    if out.strip():
        # Claude Code emits a JSON object on stdout in --output-format json.
        # Some builds also stream NDJSON; tolerate both.
        try:
            parsed = json.loads(out)
        except json.JSONDecodeError:
            for line in reversed(out.splitlines()):
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        parsed = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
    return parsed, out, err, proc.returncode


# ── Envelope ───────────────────────────────────────────────────────

def _envelope(
    plan: Dict[str, Any],
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    llm_calls: int = 0,
    cost_usd: float = 0.0,
    wall_seconds: float = 0.0,
    dag_aware: bool = True,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "plan":              plan,
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "llm_calls":         llm_calls,
        "cost_usd":          cost_usd,
        "wall_seconds":      wall_seconds,
        "dag_aware":         dag_aware,
        "error":             error,
    }


def _empty_plan() -> Dict[str, Any]:
    return {"workflow_type": "custom", "steps": []}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Claude Code competitor shim")
    ap.add_argument("--prompt", required=True, help="Natural-language task")
    ap.add_argument("--files", default="[]",
                    help="JSON-encoded list of 'name: description' strings")
    ap.add_argument("--model", default=None,
                    help="Override model id (CLAUDE_CODE_MODEL also honoured)")
    # Ablation arm switch for Benchmark J. The default is DAG-blind so
    # head-to-head Benchmark E doesn't quietly hand Claude Code the same
    # DAG instruction FlowAgent's planner uses; pass --with-dag-instruction
    # to opt-in to the DAG-aware template (Benchmark J's `dag_aware` arm).
    ap.add_argument(
        "--with-dag-instruction", dest="with_dag_instruction",
        action="store_true",
        help="Use the DAG-aware prompt template (asks for `dependencies` "
             "field + topological-order rule). Default is DAG-blind so "
             "head-to-head benchmarks don't confound Claude Code with a "
             "DAG instruction FlowAgent reserves for its own planner.",
    )
    args = ap.parse_args(argv)

    dag_aware = args.with_dag_instruction

    binary = _resolve_claude_bin()
    if not binary:
        print(json.dumps(_envelope(
            _empty_plan(),
            dag_aware=dag_aware,
            error=("claude binary not found on PATH. Install Claude Code "
                   "and authenticate, or set CLAUDE_CODE_BIN to its path."),
        )))
        return 0  # Soft-skip: harness logs error per-cell.

    try:
        files = json.loads(args.files) if args.files else []
        if not isinstance(files, list):
            files = []
    except json.JSONDecodeError:
        files = []

    model = args.model or os.environ.get("CLAUDE_CODE_MODEL")
    try:
        timeout = float(os.environ.get("CLAUDE_CODE_TIMEOUT", "300"))
    except ValueError:
        timeout = 300.0

    structured_prompt = _select_template(dag_aware=dag_aware).format(
        prompt=args.prompt,
        files=", ".join(files) if files else "(none provided)",
    )

    # Run inside a scratch CWD so any side-effects from --permission-mode plan
    # (logs, settings probes, etc.) don't pollute the harness output dir.
    scratch = Path(os.environ.get("CLAUDE_CODE_SCRATCH", ".")).resolve()
    scratch.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    try:
        cli_json, stdout, stderr, rc = _invoke_claude(
            structured_prompt,
            binary=binary,
            model=model,
            timeout=timeout,
            cwd=scratch,
        )
    except subprocess.TimeoutExpired:
        wall = time.perf_counter() - t0
        print(json.dumps(_envelope(
            _empty_plan(), wall_seconds=wall,
            dag_aware=dag_aware,
            error=f"timeout after {timeout:.0f}s",
        )))
        return 0
    except FileNotFoundError as fnf:
        print(json.dumps(_envelope(
            _empty_plan(),
            dag_aware=dag_aware,
            error=f"FileNotFoundError: {fnf}",
        )))
        return 0

    wall = time.perf_counter() - t0

    # Pull the model's reply text out of the CLI JSON. Claude Code uses
    # ``result`` for the final assistant message in ``--output-format json``.
    reply_text = ""
    if isinstance(cli_json, dict):
        reply_text = (
            cli_json.get("result")
            or cli_json.get("message")
            or cli_json.get("text")
            or ""
        )
        if isinstance(reply_text, list):
            # Some CLI versions return a list of content blocks
            parts: List[str] = []
            for block in reply_text:
                if isinstance(block, dict):
                    parts.append(str(block.get("text") or block.get("content") or ""))
                else:
                    parts.append(str(block))
            reply_text = "\n".join(parts)
    if not reply_text:
        # Last resort: search the raw stdout for a JSON object.
        reply_text = stdout

    plan_obj = _extract_json_object(reply_text) or {}
    if not isinstance(plan_obj.get("steps"), list):
        # Sometimes the model wraps the plan inside another key.
        for key in ("plan", "workflow", "pipeline"):
            inner = plan_obj.get(key) if isinstance(plan_obj, dict) else None
            if isinstance(inner, dict) and isinstance(inner.get("steps"), list):
                plan_obj = inner
                break

    # Normalise step shape to the FlowAgent schema. Keep this lenient --
    # competitors.py runs ``_normalise_plan`` again before scoring.
    normalised_steps: List[Dict[str, Any]] = []
    for step in (plan_obj.get("steps") or []):
        if not isinstance(step, dict):
            continue
        normalised_steps.append({
            "name":         str(step.get("name") or step.get("id") or ""),
            "command":      str(step.get("command") or step.get("cmd") or ""),
            "dependencies": list(step.get("dependencies") or step.get("deps") or []),
            "outputs":      list(step.get("outputs") or []),
            "description":  str(step.get("description") or ""),
        })

    plan = {
        "workflow_type": str(plan_obj.get("workflow_type")
                              or plan_obj.get("pipeline_type")
                              or "custom"),
        "steps": normalised_steps,
    }

    usage = (cli_json.get("usage") if isinstance(cli_json, dict) else None) or {}
    in_tok = int(usage.get("input_tokens", 0) or 0)
    out_tok = int(usage.get("output_tokens", 0) or 0)
    cache_creation = int(usage.get("cache_creation_input_tokens", 0) or 0)
    cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
    cost = float(cli_json.get("total_cost_usd", 0.0) or 0.0) if isinstance(cli_json, dict) else 0.0
    turns = int(cli_json.get("num_turns", 1) or 1) if isinstance(cli_json, dict) else 1

    err_msg: Optional[str] = None
    if rc != 0:
        err_msg = f"claude exited rc={rc}"
        if stderr.strip():
            err_msg += f": {stderr.strip()[:500]}"
    elif isinstance(cli_json, dict) and cli_json.get("is_error"):
        err_msg = str(cli_json.get("error") or cli_json.get("message") or "Claude Code reported is_error=true")
    elif not normalised_steps:
        err_msg = "Claude Code returned no parseable workflow plan"

    print(json.dumps(_envelope(
        plan,
        prompt_tokens=in_tok + cache_creation + cache_read,
        completion_tokens=out_tok,
        llm_calls=turns,
        cost_usd=cost,
        wall_seconds=wall,
        dag_aware=dag_aware,
        error=err_msg,
    )))
    return 0


if __name__ == "__main__":
    sys.exit(main())
