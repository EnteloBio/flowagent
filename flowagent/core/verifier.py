"""Chain-of-Verification (CoVe) plan verifier with abstention semantics.

Closes the same-LLM self-correction trap (todo T4 in the FlowAgent
architecture review). Built on the Dhuliawala et al. 2024 *factored*
CoVe pattern (`arXiv:2309.11495`): the verifier sees only the user task
and the generated plan, never the generator's chain-of-thought, so it
can't anchor on the same hallucinations the generator produced.

Empirical motivation lives in two places:

  1. The recovery-taxonomy benchmark (Benchmark B). Across 9 reasoning
     models on unrecoverable-tier faults, ~37% correctly refused, ~57%
     produced a confabulated "fix" (silent failure 23% + unsafe repair
     34%). LLMs are reliably bad at correction-on-feedback when in the
     same context as the original generation.

  2. T0 (validator on/off ablation, claude-haiku-4-5). The validator's
     retry pressure produced 4× more retries with no net gain in
     overall_pass and a regression in completeness convergence — the
     same self-correction failure mode the recovery taxonomy measures,
     surfacing on the planning side.

This module's *prevention* answer to that finding: an independent
verifier that doesn't try to fix what it finds. It produces a list of
concerns; the planner decides whether to ship, annotate, or abstain
based on a count threshold (D4-flavored conformal abstention rather
than D1's retry-feedback loop).

Gating
------
* ``FLOWAGENT_COVE_VERIFY``        (default false) — feature toggle.
* ``FLOWAGENT_COVE_ABSTAIN``       (default false) — when true, the
                                   planner raises on threshold breach
                                   instead of just annotating.
* ``FLOWAGENT_COVE_THRESHOLD``     (default 2) — minimum number of
                                   concerns to trigger abstention.
                                   Severities ``high`` count double so
                                   a single ship-blocking concern is
                                   enough on its own.

Cost
----
One structured LLM call per plan attempt. ~$0.01-0.05 per plan
depending on model, ~$2-10 over a 198-cell benchmark.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from .schemas import VerificationResult, to_json_schema

logger = logging.getLogger(__name__)


# ── Env-flag gating ────────────────────────────────────────────

def is_enabled() -> bool:
    """Read FLOWAGENT_COVE_VERIFY fresh on every call.

    Off by default so existing benchmarks see today's behaviour
    unchanged. Per-call freshness mirrors the validator-enabled and
    DAG-aware flags so the harness can flip per-cell.
    """
    return os.environ.get("FLOWAGENT_COVE_VERIFY", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }


def is_abstention_enabled() -> bool:
    return os.environ.get("FLOWAGENT_COVE_ABSTAIN", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }


def get_abstention_threshold() -> int:
    """Concern-count threshold above which the planner abstains.

    Default 2: a single concern is treated as a warning (annotated on
    the plan envelope, plan still ships); two or more concerns is
    treated as enough signal to refuse. Severity ``high`` concerns
    count double via :func:`_weighted_concern_count` so one
    ship-blocking issue is enough on its own.
    """
    raw = os.environ.get("FLOWAGENT_COVE_THRESHOLD", "2")
    try:
        v = int(raw)
        return max(1, v)
    except (ValueError, TypeError):
        return 2


# ── Question template ──────────────────────────────────────────
#
# Five targeted questions that map onto the failure modes catalogued
# in §1.3 of the architecture review. Each is phrased so a "no real
# problem" answer is short and unambiguous, and a "yes, problem" answer
# names the specific tool / step / token at fault. The verifier model
# returns one ``VerificationConcern`` per question with concern=True iff
# it identified a real issue.

_VERIFIER_QUESTIONS: List[str] = [
    (
        "Are all tool names invoked by step commands real bioinformatics "
        "tools that exist on Bioconda or Bioconductor? List any token "
        "you do not recognise as real software (a typo, a fictional name, "
        "or a wrapper script that no step writes). If all are real, "
        "answer 'all real'."
    ),
    (
        "Given the experimental design implied by the user's task, are "
        "the chosen tools appropriate for that assay type? List any tool "
        "that is being applied to data it cannot consume (e.g. a ChIP-seq "
        "peak caller on RNA-seq input, a single-cell tool on bulk data). "
        "If everything is appropriate, answer 'appropriate'."
    ),
    (
        "Does every analysis chain (alignment / quantification / peak "
        "calling / differential expression / variant calling) end in a "
        "report aggregator (multiqc, deseq2 results CSV, narrowPeak "
        "summary, ...) or terminate as a final user-facing artefact? "
        "List any analysis step whose output is dangling. If all are "
        "wired correctly, answer 'all wired'."
    ),
    (
        "Looking at step ordering and dependencies, does any step "
        "reference an input path that no upstream step produces? List "
        "the step name and the missing input. If every step's inputs "
        "are produced upstream or supplied externally, answer 'consistent'."
    ),
    (
        "Does the plan match the assay type the user asked for? If the "
        "user asked for ChIP-seq and the plan does RNA-seq (or vice "
        "versa), name the mismatch. If it matches, answer 'matches'."
    ),
]


_VERIFIER_SYSTEM_PROMPT = (
    "You are an INDEPENDENT bioinformatics workflow reviewer. You did not "
    "write the plan you are about to review and you must not assume any "
    "of its choices are correct.\n\n"
    "You will be shown:\n"
    "  - the user's original task description\n"
    "  - a workflow plan as JSON (steps with names, commands, "
    "dependencies, outputs)\n\n"
    "You will NOT be shown the prompt the plan-generation model used or "
    "any reasoning trace from it. Your job is to answer a fixed list of "
    "review questions about the plan strictly on its own merits. Do not "
    "speculate, do not interpret ambiguous prompts charitably, and do "
    "not invent fixes — your role is to flag concerns, not propose "
    "repairs.\n\n"
    "For each question, set ``concern=true`` only if you identified a "
    "real problem; otherwise ``concern=false``. Severity:\n"
    "  - high   : ship-blocking (fictional tool, wrong workflow class, "
    "missing inputs that prevent execution)\n"
    "  - medium : should fix but plan may still run (suboptimal tool, "
    "missing report aggregator, redundant steps)\n"
    "  - low    : nit / preference (cosmetic naming, ordering nitpicks)\n\n"
    "Be terse. Cite specific step names / tool tokens / paths in your "
    "answer; do not write paragraphs."
)


# ── Plan rendering ─────────────────────────────────────────────

def _plan_for_verifier(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Strip generator-internal envelope keys before showing the verifier.

    The plan dict may carry ``_completeness``, ``_validator``, etc. —
    those are generator-side instrumentation and would tip the verifier
    off to which checks have already passed. Drop them so the verifier
    judges the plan blind.
    """
    return {
        k: v for k, v in plan.items()
        if not (isinstance(k, str) and k.startswith("_"))
    }


# ── Severity weighting ─────────────────────────────────────────

def _weighted_concern_count(concerns: List[Dict[str, Any]]) -> int:
    """Count concerns weighted by severity.

    ``high`` counts as 2 (a single ship-blocking concern hits the
    default threshold of 2 on its own). ``medium`` counts as 1.
    ``low`` counts as 0 — informational only.

    This lets the threshold env var stay a simple int while still
    distinguishing nits from real problems.
    """
    weights = {"high": 2, "medium": 1, "low": 0}
    total = 0
    for c in concerns:
        if not c.get("concern"):
            continue
        sev = (c.get("severity") or "medium").lower()
        total += weights.get(sev, 1)
    return total


# ── Main entry point ───────────────────────────────────────────

async def verify_plan_independently(
    plan: Dict[str, Any],
    *,
    user_prompt: str,
    provider: Any,
    questions: Optional[List[str]] = None,
    input_files: Optional[List[str]] = None,
) -> Tuple[bool, List[Dict[str, Any]], int]:
    """Run a CoVe-style independent verifier on *plan*.

    Returns ``(ok, concerns, weighted_count)`` where:

    * ``ok`` — True iff the weighted concern count is below the
      abstention threshold. False means the plan should be flagged
      (and, if ``FLOWAGENT_COVE_ABSTAIN=true``, refused).
    * ``concerns`` — list of ``{question, answer, concern, severity}``
      dicts, one per question. Stored on the plan envelope so a
      reviewer can audit *why* the verifier flagged the plan, not just
      that it did.
    * ``weighted_count`` — the severity-weighted total used for the
      threshold comparison. Surfaced for benchmark instrumentation
      (so we can plot a "verifier signal density" histogram).

    On any error path (LLM call fails, schema validation fails,
    provider lacks ``chat_structured``), returns
    ``(True, [], 0)`` — the verifier defers rather than blocks.
    Robustness here matters because this runs after the planner has
    already converged; a verifier crash should never lose a good plan.
    """
    if not is_enabled():
        return (True, [], 0)

    if questions is None:
        questions = _VERIFIER_QUESTIONS

    plan_view = _plan_for_verifier(plan)
    questions_block = "\n".join(
        f"  Q{i+1}. {q}" for i, q in enumerate(questions)
    )

    # Telling the verifier which files are *externally supplied* removes
    # the most common false-positive on the dependency-consistency
    # question: it stops flagging FASTQ inputs as "missing upstream
    # producer" when they're user-staged data that no step needs to
    # produce. Empty / None means "no externally-supplied inputs known"
    # and the verifier should treat any unproduced path as suspect.
    if input_files:
        externals_block = (
            "EXTERNALLY SUPPLIED INPUT FILES (these are user-staged "
            "and do not need an upstream producer):\n"
            + "\n".join(f"  - {f}" for f in input_files)
            + "\n\n"
        )
    else:
        externals_block = ""

    user_msg = (
        f"USER TASK:\n{user_prompt}\n\n"
        f"{externals_block}"
        f"PLAN UNDER REVIEW:\n```json\n{json.dumps(plan_view, indent=2)}\n```"
        f"\n\nReview questions:\n{questions_block}\n\n"
        f"Return one ``VerificationConcern`` per question, in order. "
        f"Use ``concern=true`` only when you identified a real problem."
    )

    messages = [
        {"role": "system", "content": _VERIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    try:
        schema = to_json_schema(VerificationResult)
        resp = await provider.chat_structured(messages, schema)
        raw = (
            json.loads(resp.content)
            if isinstance(resp.content, str)
            else resp.content
        )
        result = VerificationResult.model_validate(raw)
        concerns_list = [c.model_dump() for c in result.concerns]
    except Exception as e:
        logger.debug("CoVe verifier call failed (%s); deferring", e)
        return (True, [], 0)

    weighted = _weighted_concern_count(concerns_list)
    threshold = get_abstention_threshold()
    ok = weighted < threshold

    if not ok:
        # Log the specific concerns at INFO so a benchmark sweep's log
        # is auditable without extra plumbing. The plan envelope also
        # carries them, but the log is easier to grep across cells.
        flagged = [c for c in concerns_list if c.get("concern")]
        logger.info(
            "CoVe verifier flagged plan (weighted=%d ≥ threshold=%d): %s",
            weighted, threshold,
            "; ".join(
                f"[{c.get('severity', '?')}] {c.get('answer', '')[:120]}"
                for c in flagged
            ),
        )

    return (ok, concerns_list, weighted)


__all__ = [
    "get_abstention_threshold",
    "is_abstention_enabled",
    "is_enabled",
    "verify_plan_independently",
]
