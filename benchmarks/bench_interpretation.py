"""Benchmark G — biological-interpretation evaluation.

Stresses the FlowAgent *reporting* module: given the outputs of a
prior FlowAgent run (DE table, QC summary, etc.), can the model under
test answer concrete biological questions about them — and abstain when
the supplied evidence is genuinely insufficient?

Two question modes:

    mcq         single-letter answer; scored by exact match against the
                YAML's ``answer`` field. ``refusal_acceptable: true``
                questions accept the refusal option as correct, which
                tests calibration of "I don't know".

    open_ended  free-text response; scored by an LLM-as-judge that
                consumes the rubric, the reference answer, and the
                model's response, and emits a 0–100 score plus a
                justification.

Mock mode emits deterministic fake answers (always picks "A" for MCQ,
returns a stub for open-ended) so the harness is exercisable without
API keys; useful for CI smoke tests of the scoring path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from harness.runner import (  # noqa: E402
    _write_csv, load_yaml, set_provider, timestamped_dir, write_manifest,
)


# ── Prompt assembly ──────────────────────────────────────────────

_SYSTEM_DATA = (
    "You are a careful computational-biology research assistant. "
    "Answer using only the supplied analysis outputs and any dataset "
    "metadata block prepended to them. If those outputs do not contain "
    "the evidence needed to answer, pick the refusal option or say so "
    "explicitly."
)

_SYSTEM_PRIORS = (
    "You are a careful computational-biology research assistant. "
    "You may use established biological knowledge together with any "
    "supplied analysis outputs. Answer the question directly; do not "
    "refuse merely because the supplied files lack detail when textbook "
    "biology suffices."
)

_SYSTEM_REFUSAL = (
    "You are a careful computational-biology research assistant. "
    "Answer using only the supplied analysis outputs. If the supplied "
    "data genuinely cannot answer the question, pick the refusal option."
)


def _system_prompt(evidence_class: str) -> str:
    if evidence_class == "internal_knowledge":
        return _SYSTEM_PRIORS
    if evidence_class == "calibrated_refusal":
        return _SYSTEM_REFUSAL
    return _SYSTEM_DATA

# ── Judge calibration ────────────────────────────────────────────
#
# The reviewer flagged that the previous judge prompt told the model
# nothing about the score scale or the pass mark. Replies were therefore
# implicitly anchored on whatever calibration the judge's pre-training
# happened to bake in. The new prompt:
#
#   - States the pass mark (≥60) explicitly.
#   - Defines five anchored score bands with concrete examples of what
#     belongs in each.
#   - Requires structured JSON output with rubric ``hits`` / ``misses``
#     / ``fabrications`` lists plus a ``grounding_quote`` taken
#     verbatim from the candidate answer, so each judgment is auditable.
#   - Forbids credit for fabricated facts.
#
# A separate calibration harness (see ``benchmarks/judge_calibration.py``)
# is provided to re-score a sample of N=30 responses with a second
# judge model and report inter-judge κ.

_JUDGE_SCORE_ANCHORS = (
    "  0-20  : Fabricated, contradicted by the supplied evidence, or "
    "answers a different question. Confidently wrong direction.\n"
    "  21-40 : Generic / textbook prose with no grounding in the "
    "supplied inputs; or hits at most one rubric item.\n"
    "  41-59 : Partially correct. Hits some rubric items but misses "
    "most, or hits them at the wrong granularity.\n"
    "  60-79 : Covers the majority of rubric items, with minor gaps "
    "or one factual slip. PASSING quality.\n"
    "  80-100: Hits all rubric items, properly grounded in the "
    "candidate's own quoted evidence, no fabrications."
)

_SYSTEM_JUDGE = (
    "You are a strict scientific-writing grader. You will receive "
    "(i) a question, (ii) a grading rubric, (iii) a reference answer, "
    "and (iv) a candidate answer to grade. The pass mark is 60. "
    "Use these anchored score bands:\n\n"
    f"{_JUDGE_SCORE_ANCHORS}\n\n"
    "Do not award credit for fabricated facts, even if fluently "
    "written; mark them in ``fabrications``. Score on the rubric "
    "items listed, NOT on writing style. Return strictly the JSON "
    "schema requested — no commentary outside the JSON."
)


def _read_text_path(path: Path) -> str:
    """Read a text or gzip-compressed text file."""
    if path.name.endswith(".gz") or path.suffix == ".gz":
        import gzip
        with gzip.open(path, "rt", errors="replace") as fh:
            return fh.read()
    return path.read_text(errors="replace")


def _summarize_bed_like(text: str) -> str:
    """Return a one-line summary for BED / narrowPeak content."""
    rows = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    if not rows:
        return "peak rows: 0"
    widths = []
    chroms: Dict[str, int] = {}
    for ln in rows:
        parts = ln.split("\t")
        if len(parts) < 3:
            continue
        chrom = parts[0]
        chroms[chrom] = chroms.get(chrom, 0) + 1
        try:
            widths.append(int(parts[2]) - int(parts[1]))
        except ValueError:
            pass
    med_w = sorted(widths)[len(widths) // 2] if widths else 0
    top_chr = max(chroms, key=chroms.get) if chroms else "?"
    return (
        f"peak rows: {len(rows)}; median width: {med_w} bp; "
        f"most peaks on: {top_chr} ({chroms.get(top_chr, 0)} rows)"
    )


def _summarize_vcf(text: str) -> str:
    """Return a one-line summary for VCF variant records."""
    n = indel = 0
    chroms: set = set()
    for ln in text.splitlines():
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split("\t")
        if len(parts) < 5:
            continue
        n += 1
        chroms.add(parts[0])
        ref, alt = parts[3], parts[4].split(",")[0]
        if len(ref) > 1 or len(alt) > 1:
            indel += 1
    if n == 0:
        return "variant records: 0"
    frac = indel / n
    return (
        f"variant records: {n}; chromosomes present: {sorted(chroms)}; "
        f"indel fraction (|REF|>1 or |ALT|>1): {frac:.1%}"
    )


def _bundle_inputs(
    input_paths: Dict[str, Path],
    char_budget: int = 24_000,
    *,
    dataset_context: str = "",
) -> str:
    """Concatenate the named input files into a single context block.

    Each input is preceded by a header line ``=== <name> (path=<rel>) ===``
    and is truncated to fit a per-input share of ``char_budget``.  Gzip
    files are decompressed.  Large BED/VCF inputs get a computed summary
    line so row-count / composition questions remain answerable even when
    the excerpt is truncated.
    """
    chunks: List[str] = []
    if dataset_context.strip():
        chunks.append(f"=== dataset metadata ===\n{dataset_context.strip()}\n")
    if not input_paths:
        return "".join(chunks)
    share = max(2_000, char_budget // max(1, len(input_paths)))
    for name, path in input_paths.items():
        header = f"\n=== {name} ({path}) ===\n"
        try:
            text = _read_text_path(path)
        except FileNotFoundError:
            chunks.append(header + f"[file not found: {path}]\n")
            continue
        summary = ""
        low = name.lower()
        if low.endswith("vcf") or path.name.endswith(".vcf.gz"):
            summary = _summarize_vcf(text) + "\n"
        elif "peak" in low or path.suffix.lower() in (".bed", ".narrowpeak"):
            summary = _summarize_bed_like(text) + "\n"
        body = summary + text
        if len(body) > share:
            body = body[:share] + f"\n... [truncated; {len(text)-share} chars omitted]\n"
        chunks.append(header + body)
    return "".join(chunks)


def _mcq_prompt(question: Dict[str, Any], context: str) -> str:
    choices = "\n".join(f"  {k}) {v}" for k, v in question["choices"].items())
    valid = "/".join(question["choices"].keys())
    system = _system_prompt(question.get("evidence_class", ""))
    return (
        f"{system}\n\n"
        f"Analysis outputs:\n{context}\n\n"
        f"Question: {question['question']}\n\n"
        f"Choices:\n{choices}\n\n"
        f"Respond in EXACTLY this format, using the literal tags shown:\n\n"
        f"  <answer>X</answer>\n"
        f"  <explain>One short sentence of justification.</explain>\n\n"
        f"where X is one of {valid}. The ``<answer>`` tag must contain "
        f"only a single capital letter and nothing else. Do not put any "
        f"prose outside the two tags."
    )


def _open_prompt(question: Dict[str, Any], context: str) -> str:
    system = _system_prompt(question.get("evidence_class", "data_required"))
    return (
        f"{system}\n\n"
        f"Analysis outputs:\n{context}\n\n"
        f"Question: {question['question']}\n\n"
        f"Answer concisely and stay grounded in the supplied evidence."
    )


_JUDGE_JSON_SCHEMA = (
    '{\n'
    '  "score": <integer 0-100, anchored to the bands above>,\n'
    '  "hits": ["<rubric item the candidate hit>", ...],\n'
    '  "misses": ["<rubric item the candidate missed>", ...],\n'
    '  "fabrications": ["<claim the candidate made that is unsupported '
    'by the supplied inputs>", ...],\n'
    '  "grounding_quote": "<short verbatim quote from the candidate '
    'answer showing it grounded in the supplied evidence; empty '
    'string if there is no such grounding>",\n'
    '  "justification": "<one short paragraph explaining the score>"\n'
    '}'
)


def _judge_prompt(question: Dict[str, Any], candidate: str) -> str:
    return (
        f"{_SYSTEM_JUDGE}\n\n"
        f"Question:\n{question['question']}\n\n"
        f"Rubric:\n{question.get('rubric','(no rubric provided)')}\n\n"
        f"Reference answer:\n{question.get('reference_answer','(none)')}\n\n"
        f"Candidate answer:\n{candidate}\n\n"
        f"PASS MARK: a score of 60 or higher counts as a passing answer; "
        f"60 means the candidate covered the majority of rubric items with "
        f"only minor gaps. Do NOT pad scores into the 60-79 band out of "
        f"politeness — apply the anchors strictly.\n\n"
        f"Return EXACTLY this JSON schema and nothing else:\n\n"
        f"{_JUDGE_JSON_SCHEMA}"
    )


# ── LLM call helpers ─────────────────────────────────────────────

async def _call_llm(prompt: str, *, model_cfg: Dict[str, Any]) -> str:
    """Single-shot text completion via FlowAgent's LLMInterface.

    ``LLMInterface._call_openai`` is the provider-agnostic chat method
    (the legacy name; it routes through whichever provider is configured).
    """
    set_provider(model_cfg)
    from flowagent.core.llm import LLMInterface  # late import: env must be set first
    llm = LLMInterface()
    messages = [{"role": "user", "content": prompt}]
    resp = await llm._call_openai(messages)
    return (resp or "").strip()


# ── MCQ letter extraction ────────────────────────────────────────
#
# The reviewer of FlowAgent's interpretation benchmark observed that the
# previous extractor returned ``I`` for replies like "I believe the
# answer is B" because it pulled the FIRST standalone capital letter
# from the first line. The new parser is tag-aware (the prompt asks for
# ``<answer>X</answer>``) with a tiered regex fallback for models that
# return prose instead. ``valid_choices`` constrains the letters that
# count — important for questions with non-A-D choice keys.

_TAG_RE = re.compile(
    r"<\s*answer\s*>\s*([A-Z])\s*<\s*/\s*answer\s*>",
    flags=re.IGNORECASE,
)

_TIERED_PATTERNS = (
    # Tier 1: the entire reply (or first line) is just one letter.
    re.compile(r"^\s*([A-Z])\s*[\.\)\]:]?\s*$",
               flags=re.MULTILINE),
    # Tier 2: explicit "Answer: X" / "Answer - X" header.
    re.compile(r"^\s*answer\s*[:\-]\s*\(?([A-Z])\)?",
               flags=re.IGNORECASE | re.MULTILINE),
    # Tier 3: "the answer is X" / "answer is (X)" / "option X" / "choice X".
    re.compile(
        r"\b(?:answer|choice|option|select|pick)\s*"
        r"(?:would\s+be|is|=|:)?\s*\(?([A-Z])\)?",
        flags=re.IGNORECASE,
    ),
    # Tier 4: "(X)" anywhere — common when the LLM brackets the letter.
    re.compile(r"\(\s*([A-Z])\s*\)"),
)


def _extract_letter(
    reply: str,
    valid_choices: Optional[List[str]] = None,
) -> Optional[str]:
    """Extract the chosen letter from an MCQ reply.

    Strategy:
      1. Look for a ``<answer>X</answer>`` tag (the format the prompt
         asks for explicitly).
      2. Fall back through a tiered regex sequence in priority order.
      3. As a last resort, take the LAST standalone capital letter in
         the reply that is in ``valid_choices`` — last not first, so
         "I believe the answer is B" returns ``B`` instead of ``I``.

    ``valid_choices`` is a list of allowed letters (e.g.
    ``["A", "B", "C", "D"]``); when provided, candidates not in the
    set are rejected. The fallback path (step 3) requires this set so
    it cannot "discover" a stray ``I`` or ``T`` in narrative text.
    """
    text = reply or ""
    valid = {c.upper() for c in (valid_choices or [])}

    def _ok(letter: str) -> Optional[str]:
        letter = (letter or "").upper()
        if not letter or len(letter) != 1 or not letter.isalpha():
            return None
        if valid and letter not in valid:
            return None
        return letter

    # 1. Structured tag.
    m = _TAG_RE.search(text)
    cand = _ok(m.group(1)) if m else None
    if cand:
        return cand

    # 2. Tiered regex fallbacks.
    for pat in _TIERED_PATTERNS:
        for m in pat.finditer(text):
            cand = _ok(m.group(1))
            if cand:
                return cand

    # 3. Last-resort: take the LAST in-set capital letter.
    if valid:
        last = None
        for m in re.finditer(r"\b([A-Z])\b", text):
            if m.group(1) in valid:
                last = m.group(1)
        if last:
            return last
    return None


def _parse_judge_json(reply: str) -> Dict[str, Any]:
    """Robust JSON extraction from a possibly-noisy judge reply.

    Returns a dict with keys ``score`` (float|None), ``justification``
    (str), ``hits`` (list[str]), ``misses`` (list[str]),
    ``fabrications`` (list[str]) and ``grounding_quote`` (str). Missing
    or malformed fields collapse to safe defaults so the upstream row
    schema is always populated.
    """
    out: Dict[str, Any] = {
        "score":            None,
        "justification":    "",
        "hits":             [],
        "misses":           [],
        "fabrications":     [],
        "grounding_quote":  "",
    }
    if not reply:
        return out
    # Greedy match — the judge sometimes wraps the JSON in prose despite
    # the instructions; we want the largest balanced-looking block.
    m = re.search(r"\{.*\}", reply, flags=re.DOTALL)
    if not m:
        out["justification"] = reply.strip()[:600]
        return out
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        out["justification"] = reply.strip()[:600]
        return out

    raw = obj.get("score")
    try:
        score = float(raw)
        score = max(0.0, min(100.0, score))
    except (TypeError, ValueError):
        score = None
    out["score"] = score
    out["justification"] = str(obj.get("justification", ""))[:600]

    def _as_str_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(x)[:300] for x in value if str(x).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()[:300]]
        return []

    out["hits"]            = _as_str_list(obj.get("hits"))
    out["misses"]          = _as_str_list(obj.get("misses"))
    out["fabrications"]    = _as_str_list(obj.get("fabrications"))
    out["grounding_quote"] = str(obj.get("grounding_quote", ""))[:300]
    return out


# ── Per-question scoring ─────────────────────────────────────────

async def _score_mcq(question: Dict[str, Any], context: str,
                     model_cfg: Dict[str, Any], *, mock: bool) -> Dict[str, Any]:
    valid_choices = list(question["choices"].keys())
    if mock:
        reply = "<answer>A</answer><explain>mock</explain>"
        letter = "A"
    else:
        reply = await _call_llm(_mcq_prompt(question, context), model_cfg=model_cfg)
        letter = _extract_letter(reply, valid_choices=valid_choices)
    truth = question["answer"]
    refusal_letters = {k for k, v in question["choices"].items()
                       if "cannot determine" in str(v).lower()}
    is_refusal = letter in refusal_letters
    correct = (
        letter == truth
        or (question.get("refusal_acceptable") and is_refusal)
    )
    return {
        "answer_given":    letter or "",
        "answer_truth":    truth,
        "is_refusal":      bool(is_refusal),
        "correct":         bool(correct),
        "evidence_class":  question.get("evidence_class", ""),
        "raw_response":    reply[:400],
    }


_PASS_MARK = 60.0


def _flatten_list(items: List[str]) -> str:
    """Newline-join a short list for storage in a CSV cell."""
    return "\n".join(items)[:1000]


async def _score_open(question: Dict[str, Any], context: str,
                      model_cfg: Dict[str, Any],
                      judge_cfg: Dict[str, Any], *, mock: bool) -> Dict[str, Any]:
    if mock:
        candidate = "(mock candidate answer — no LLM call made)"
        parsed: Dict[str, Any] = {
            "score":           50.0,
            "justification":   "(mock judge)",
            "hits":            [],
            "misses":          ["(mock — full rubric)"],
            "fabrications":    [],
            "grounding_quote": "",
        }
    else:
        candidate = await _call_llm(_open_prompt(question, context),
                                    model_cfg=model_cfg)
        judge_reply = await _call_llm(_judge_prompt(question, candidate),
                                      model_cfg=judge_cfg)
        parsed = _parse_judge_json(judge_reply)
    score = parsed["score"]
    return {
        "candidate_answer":     candidate[:1000],
        "judge_score":          score,
        "judge_justification":  parsed["justification"],
        "judge_hits":           _flatten_list(parsed["hits"]),
        "judge_misses":         _flatten_list(parsed["misses"]),
        "judge_fabrications":   _flatten_list(parsed["fabrications"]),
        "judge_grounding_quote": parsed["grounding_quote"],
        "correct":              (score is not None and score >= _PASS_MARK),
        "evidence_class":       question.get("evidence_class", ""),
    }


# ── Sweep ────────────────────────────────────────────────────────

async def _run_sweep(datasets: List[Dict[str, Any]],
                     model_cfg: Dict[str, Any],
                     judge_cfg: Dict[str, Any],
                     inputs_base: Path, *, mock: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ds in datasets:
        input_paths = {k: inputs_base / v
                       for k, v in (ds.get("inputs") or {}).items()}
        context = _bundle_inputs(
            input_paths,
            dataset_context=ds.get("analysis_context", ""),
        )
        for q in ds["questions"]:
            t0 = time.perf_counter()
            try:
                if q["type"] == "mcq":
                    detail = await _score_mcq(q, context, model_cfg, mock=mock)
                elif q["type"] == "open_ended":
                    detail = await _score_open(q, context, model_cfg,
                                               judge_cfg, mock=mock)
                else:
                    detail = {"error": f"unknown type {q['type']!r}"}
            except Exception as exc:  # pragma: no cover - defensive
                detail = {"error": f"{type(exc).__name__}: {exc}"}
            wall = round(time.perf_counter() - t0, 3)

            # Schema-stable row: always include ``correct`` so downstream
            # plotting code can group/aggregate even when every cell errored
            # out (e.g. invalid API key killed the whole sweep).
            row: Dict[str, Any] = {
                "dataset":       ds["id"],
                "accession":     ds.get("accession", ""),
                "question_id":   q["id"],
                "question_type": q["type"],
                "evidence_class": q.get("evidence_class", ""),
                "model":         model_cfg["id"],
                "provider":      model_cfg["provider"],
                "judge_model":   judge_cfg["id"] if q["type"] == "open_ended" else "",
                "wall_seconds":  wall,
                "correct":       False,
                "answer_given":  "",
                "answer_truth":  q.get("answer", ""),
                "is_refusal":    False,
                "judge_score":   None,
                "raw_response":  "",
                "candidate_answer":     "",
                "judge_justification":  "",
                "judge_hits":           "",
                "judge_misses":         "",
                "judge_fabrications":   "",
                "judge_grounding_quote": "",
            }
            row.update(detail)
            rows.append(row)
            print(f"  [{q['type']:9}]  {q['id']}  "
                  f"correct={detail.get('correct')}  ({wall:.1f}s)")
    return rows


# ── CLI ──────────────────────────────────────────────────────────

def _resolve_model(models_cfg: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    for m in models_cfg.get("models", []):
        if m["id"] == model_id:
            return m
    raise SystemExit(f"model '{model_id}' not in models.yaml")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions",   default="config/interpretation_questions.yaml")
    ap.add_argument("--models-yaml", default="config/models.yaml")
    ap.add_argument("--inputs-base", default=".")
    ap.add_argument("--out",         default="results")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--model",      help="Single model ID under test")
    grp.add_argument("--models",     help="Comma-separated model IDs to sweep")
    grp.add_argument("--all-models", action="store_true",
                     help="Sweep every model in models.yaml")
    ap.add_argument("--judge",       default="gpt-5.4",
                    help="Judge model for open-ended scoring (default: gpt-5.4)")
    ap.add_argument("--datasets",    default="",
                    help="Comma-separated dataset IDs (default: all)")
    ap.add_argument("--mock", action="store_true",
                    help="Skip LLM calls; emit deterministic fake answers")
    args = ap.parse_args()

    qpath = _HERE / args.questions if not Path(args.questions).is_absolute() \
        else Path(args.questions)
    mpath = _HERE / args.models_yaml if not Path(args.models_yaml).is_absolute() \
        else Path(args.models_yaml)

    qcfg = load_yaml(qpath)
    mcfg = load_yaml(mpath)
    datasets = qcfg.get("datasets", [])
    if args.datasets:
        wanted = {d.strip() for d in args.datasets.split(",") if d.strip()}
        datasets = [d for d in datasets if d["id"] in wanted]
    if not datasets:
        sys.exit("no datasets selected")

    # Resolve the model list under test. ``--model`` runs one; ``--models``
    # takes a comma-separated list; ``--all-models`` sweeps the registry.
    if args.all_models:
        model_ids = [m["id"] for m in mcfg.get("models", [])]
    elif args.models:
        model_ids = [s.strip() for s in args.models.split(",") if s.strip()]
    else:
        model_ids = [args.model]
    model_cfgs = [_resolve_model(mcfg, mid) for mid in model_ids]
    judge_cfg  = _resolve_model(mcfg, args.judge)

    # Run each model in turn under a single event loop. Using one
    # ``asyncio.run`` per model would close the loop between iterations,
    # which leaves httpx's async cleanup tasks orphaned and prints
    # "Event loop is closed" warnings — harmless but noisy. One loop
    # for the whole sweep avoids that.
    async def _run_all():
        rows: List[Dict[str, Any]] = []
        for i, mc in enumerate(model_cfgs, 1):
            print(f"\n[{i}/{len(model_cfgs)}] sweeping model={mc['id']}")
            rows.extend(await _run_sweep(
                datasets, mc, judge_cfg,
                inputs_base=Path(args.inputs_base), mock=args.mock,
            ))
        return rows

    rows = asyncio.run(_run_all())

    # Roll up per-(model, dataset, type) summary for the manifest.
    by: Dict[tuple, Dict[str, Any]] = {}
    for r in rows:
        key = (r["model"], r["dataset"], r["question_type"])
        d = by.setdefault(key, {"n": 0, "correct": 0, "judge_sum": 0.0,
                                 "judge_n": 0})
        d["n"] += 1
        d["correct"] += int(bool(r.get("correct")))
        if r["question_type"] == "open_ended" and r.get("judge_score") is not None:
            d["judge_sum"] += r["judge_score"]
            d["judge_n"]   += 1
    summary = []
    for (model, ds, qt), d in by.items():
        row = {"model": model, "dataset": ds, "type": qt, "n": d["n"],
               "accuracy": d["correct"] / d["n"] if d["n"] else None}
        if qt == "open_ended" and d["judge_n"]:
            row["mean_judge_score"] = d["judge_sum"] / d["judge_n"]
        summary.append(row)

    out_dir = timestamped_dir(Path(args.out), "interpretation")
    _write_csv(out_dir / "metrics.csv", rows)
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2, default=str))
    write_manifest(
        out_dir, benchmark="interpretation",
        models=model_cfgs + [judge_cfg],
        extra={
            "models_under_test": [m["id"] for m in model_cfgs],
            "judge_model":       judge_cfg["id"],
            "datasets":          [d["id"] for d in datasets],
            "summary":           summary,
            "mock":              args.mock,
        },
    )
    print(f"\n[ok] wrote {len(rows)} rows → {out_dir}/metrics.csv")
    # Print a compact per-model overview so the user can sanity-check
    # before running the report stage.
    by_model: Dict[str, Dict[str, Any]] = {}
    for s in summary:
        m = by_model.setdefault(s["model"], {"mcq_n": 0, "mcq_c": 0,
                                              "open_n": 0, "open_sum": 0.0})
        if s["type"] == "mcq":
            m["mcq_n"] += s["n"]
            m["mcq_c"] += int(round(s["accuracy"] * s["n"]))
        else:
            if s.get("mean_judge_score") is not None:
                m["open_n"]   += s["n"]
                m["open_sum"] += s["mean_judge_score"] * s["n"]
    print("\n[summary] per-model rollup:")
    for model, d in by_model.items():
        mcq_acc = d["mcq_c"] / d["mcq_n"] if d["mcq_n"] else 0.0
        open_mean = d["open_sum"] / d["open_n"] if d["open_n"] else float("nan")
        print(f"   {model:35s}  mcq={mcq_acc:.0%} ({d['mcq_c']}/{d['mcq_n']})"
              f"  open_judge_mean={open_mean:.1f}")


if __name__ == "__main__":
    main()
