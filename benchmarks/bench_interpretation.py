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

_SYSTEM_RESPONDENT = (
    "You are a careful computational-biology research assistant. "
    "Answer using only the supplied analysis outputs; if those outputs "
    "do not contain the evidence needed to answer, say so explicitly."
)

_SYSTEM_JUDGE = (
    "You are a strict scientific-writing grader. You will receive "
    "(i) a question, (ii) a grading rubric, (iii) a reference answer, "
    "and (iv) a candidate answer to grade. Return JSON with keys "
    "``score`` (integer 0-100) and ``justification`` (one short paragraph). "
    "Do not award credit for fabricated facts, even if fluently written."
)


def _bundle_inputs(input_paths: Dict[str, Path], char_budget: int = 24_000) -> str:
    """Concatenate the named input files into a single context block.

    Each input is preceded by a header line ``=== <name> (path=<rel>) ===``
    and is truncated to fit a per-input share of ``char_budget``.
    """
    if not input_paths:
        return ""
    share = max(2_000, char_budget // max(1, len(input_paths)))
    chunks: List[str] = []
    for name, path in input_paths.items():
        header = f"\n=== {name} ({path}) ===\n"
        try:
            text = path.read_text(errors="replace")
        except FileNotFoundError:
            chunks.append(header + f"[file not found: {path}]\n")
            continue
        if len(text) > share:
            text = text[:share] + f"\n... [truncated; {len(text)-share} chars omitted]\n"
        chunks.append(header + text)
    return "".join(chunks)


def _mcq_prompt(question: Dict[str, Any], context: str) -> str:
    choices = "\n".join(f"  {k}) {v}" for k, v in question["choices"].items())
    return (
        f"{_SYSTEM_RESPONDENT}\n\n"
        f"Analysis outputs:\n{context}\n\n"
        f"Question: {question['question']}\n\n"
        f"Choices:\n{choices}\n\n"
        f"Reply with a single capital letter (A, B, C, ...) on its "
        f"own line, optionally followed by a one-sentence justification."
    )


def _open_prompt(question: Dict[str, Any], context: str) -> str:
    return (
        f"{_SYSTEM_RESPONDENT}\n\n"
        f"Analysis outputs:\n{context}\n\n"
        f"Question: {question['question']}\n\n"
        f"Answer concisely and stay grounded in the supplied evidence."
    )


def _judge_prompt(question: Dict[str, Any], candidate: str) -> str:
    return (
        f"{_SYSTEM_JUDGE}\n\n"
        f"Question:\n{question['question']}\n\n"
        f"Rubric:\n{question.get('rubric','(no rubric provided)')}\n\n"
        f"Reference answer:\n{question.get('reference_answer','(none)')}\n\n"
        f"Candidate answer:\n{candidate}\n\n"
        f"Return JSON: {{\"score\": <0-100>, \"justification\": \"...\"}}"
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


_LETTER_RE = re.compile(r"\b([A-Z])\b")


def _extract_letter(reply: str) -> Optional[str]:
    """Pull the first standalone capital letter from a reply."""
    m = _LETTER_RE.search(reply.strip().split("\n", 1)[0])
    return m.group(1) if m else None


def _parse_judge_json(reply: str) -> Tuple[Optional[float], str]:
    """Robust JSON extraction from a possibly-noisy judge reply."""
    m = re.search(r"\{.*\}", reply, flags=re.DOTALL)
    if not m:
        return None, reply.strip()[:200]
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, reply.strip()[:200]
    raw = obj.get("score")
    try:
        score = float(raw)
        score = max(0.0, min(100.0, score))
    except (TypeError, ValueError):
        score = None
    return score, str(obj.get("justification", ""))[:600]


# ── Per-question scoring ─────────────────────────────────────────

async def _score_mcq(question: Dict[str, Any], context: str,
                     model_cfg: Dict[str, Any], *, mock: bool) -> Dict[str, Any]:
    if mock:
        reply, letter = "A (mock)", "A"
    else:
        reply = await _call_llm(_mcq_prompt(question, context), model_cfg=model_cfg)
        letter = _extract_letter(reply)
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
        "raw_response":    reply[:400],
    }


async def _score_open(question: Dict[str, Any], context: str,
                      model_cfg: Dict[str, Any],
                      judge_cfg: Dict[str, Any], *, mock: bool) -> Dict[str, Any]:
    if mock:
        candidate = "(mock candidate answer — no LLM call made)"
        score, just = 50.0, "(mock judge)"
    else:
        candidate = await _call_llm(_open_prompt(question, context),
                                    model_cfg=model_cfg)
        judge_reply = await _call_llm(_judge_prompt(question, candidate),
                                      model_cfg=judge_cfg)
        score, just = _parse_judge_json(judge_reply)
    return {
        "candidate_answer": candidate[:1000],
        "judge_score":      score,
        "judge_justification": just,
        "correct":          (score is not None and score >= 60.0),
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
        context = _bundle_inputs(input_paths)
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
