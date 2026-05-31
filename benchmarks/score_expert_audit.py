"""Score a completed expert audit against the LLM judge (and across experts).

Reads a filled ``expert_audit_scoring.tsv`` (optionally merged with
``expert_audit_key.tsv`` for LLM columns) and reports:

* expert vs LLM agreement (Pearson r, Cohen's κ on pass/fail)
* expert A vs expert B agreement on ``overlap_item=Y`` rows
* per-item disagreement table for supplement examples

Usage::

    python score_expert_audit.py \\
        --scoring results/expert_audit/<ts>/expert_audit_scoring.tsv \\
        --key results/expert_audit/<ts>/expert_audit_key.tsv

If experts filled ``expert_audit_full.tsv`` (with LLM columns present),
``--key`` is optional.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import bench_interpretation as bi  # noqa: E402
from harness.expert_audit_stats import summarise_paired_scores  # noqa: E402


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _try_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_pass(value: Any, *, score: Optional[float], pass_mark: float) -> Optional[bool]:
    if value is None:
        if score is not None:
            return score >= pass_mark
        return None
    s = str(value).strip().lower()
    if not s:
        if score is not None:
            return score >= pass_mark
        return None
    if s in ("yes", "y", "true", "1", "pass", "p"):
        return True
    if s in ("no", "n", "false", "0", "fail", "f"):
        return False
    return score >= pass_mark if score is not None else None


def _merge_key(
    scoring: List[Dict[str, str]],
    key_rows: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    key_by_id = {r["audit_id"]: r for r in key_rows if r.get("audit_id")}
    merged: List[Dict[str, str]] = []
    for row in scoring:
        out = dict(row)
        extra = key_by_id.get(row.get("audit_id", ""), {})
        for k, v in extra.items():
            if k not in out or not str(out.get(k, "")).strip():
                out[k] = v
        merged.append(out)
    return merged


def _expert_entries(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Expand rows with multiple expert_id values into scored entries."""
    entries: List[Dict[str, Any]] = []
    for row in rows:
        score = _try_float(row.get("expert_score"))
        if score is None:
            continue
        expert_id = (row.get("expert_id") or "expert").strip() or "expert"
        passed = _parse_pass(row.get("expert_pass"), score=score, pass_mark=bi._PASS_MARK)
        entries.append({
            "audit_id": row.get("audit_id", ""),
            "question_id": row.get("question_id", ""),
            "expert_id": expert_id,
            "expert_score": score,
            "expert_pass": passed,
            "llm_judge_score": _try_float(row.get("llm_judge_score")),
            "overlap_item": (row.get("overlap_item") or "").strip().upper() == "Y",
            "model": row.get("model", ""),
        })
    return entries


def _human_human_summary(
    entries: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Pair experts on overlap items when two distinct expert_ids exist."""
    by_audit: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for e in entries:
        if not e["overlap_item"]:
            continue
        by_audit.setdefault(e["audit_id"], {})[e["expert_id"]] = e

    pairs: List[Tuple[float, float]] = []
    pass_pairs: List[Tuple[bool, bool]] = []
    for audit_id, experts in by_audit.items():
        if len(experts) < 2:
            continue
        ids = sorted(experts)
        a, b = experts[ids[0]], experts[ids[1]]
        pairs.append((a["expert_score"], b["expert_score"]))
        if a["expert_pass"] is not None and b["expert_pass"] is not None:
            pass_pairs.append((a["expert_pass"], b["expert_pass"]))

    if not pairs:
        return None

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    summary = summarise_paired_scores(xs, ys, pass_mark=bi._PASS_MARK)
    summary["comparison"] = "expert_vs_expert"
    summary["expert_a"] = sorted({e["expert_id"] for e in entries if e["overlap_item"]})[:1]
    summary["n_overlap_pairs"] = len(pairs)
    if len(pass_pairs) < len(pairs):
        summary["note"] = (
            "Some overlap rows lack expert_pass; κ uses rows with both passes set."
        )
    return summary


def _expert_vs_llm_summary(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One score per audit_id: mean expert score vs LLM judge."""
    by_audit: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        aid = e["audit_id"]
        if aid not in by_audit:
            by_audit[aid] = {"scores": [], "llm": e["llm_judge_score"]}
        by_audit[aid]["scores"].append(e["expert_score"])
        if e["llm_judge_score"] is not None:
            by_audit[aid]["llm"] = e["llm_judge_score"]

    xs: List[float] = []
    ys: List[float] = []
    for rec in by_audit.values():
        if not rec["scores"] or rec["llm"] is None:
            continue
        xs.append(sum(rec["scores"]) / len(rec["scores"]))
        ys.append(rec["llm"])

    summary = summarise_paired_scores(xs, ys, pass_mark=bi._PASS_MARK)
    summary["comparison"] = "expert_vs_llm"
    return summary


def _disagreements(
    entries: List[Dict[str, Any]],
    *,
    pass_mark: float,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Largest |expert − LLM| gaps for supplement examples."""
    by_audit: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        aid = e["audit_id"]
        if aid not in by_audit:
            by_audit[aid] = dict(e)
            by_audit[aid]["expert_scores"] = []
        by_audit[aid]["expert_scores"].append(e["expert_score"])

    rows: List[Dict[str, Any]] = []
    for aid, rec in by_audit.items():
        llm = rec.get("llm_judge_score")
        if llm is None:
            continue
        expert_mean = sum(rec["expert_scores"]) / len(rec["expert_scores"])
        delta = expert_mean - llm
        a_pass = expert_mean >= pass_mark
        l_pass = llm >= pass_mark
        if abs(delta) < 10 and a_pass == l_pass:
            continue
        rows.append({
            "audit_id": aid,
            "question_id": rec.get("question_id", ""),
            "model": rec.get("model", ""),
            "expert_score_mean": round(expert_mean, 1),
            "llm_judge_score": llm,
            "score_delta": round(delta, 1),
            "expert_pass": a_pass,
            "llm_judge_pass": l_pass,
            "pass_flip": a_pass != l_pass,
        })
    rows.sort(key=lambda r: abs(r["score_delta"]), reverse=True)
    return rows[:limit]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--scoring", required=True,
                    help="Filled expert_audit_scoring.tsv (or directory)")
    ap.add_argument("--scoring-extra", action="append", default=[],
                    help="Additional completed scoring sheets (second annotator)")
    ap.add_argument("--key", default="",
                    help="expert_audit_key.tsv for LLM columns (optional if merged)")
    ap.add_argument("--out", default="",
                    help="Output directory (default: same dir as scoring file)")
    args = ap.parse_args()

    def _resolve_scoring(path_str: str) -> Path:
        p = Path(path_str)
        if not p.is_absolute():
            p = _HERE / p
        if p.is_dir():
            p = p / "expert_audit_scoring.tsv"
        if not p.exists():
            raise SystemExit(f"scoring file not found: {p}")
        return p

    scoring_path = _resolve_scoring(args.scoring)
    scoring_rows: List[Dict[str, str]] = []
    for path_str in [args.scoring, *args.scoring_extra]:
        scoring_rows.extend(_read_tsv(_resolve_scoring(path_str)))

    if args.key:
        key_path = Path(args.key)
        if not key_path.is_absolute():
            key_path = _HERE / key_path
        if key_path.is_dir():
            key_path = key_path / "expert_audit_key.tsv"
        if not key_path.exists():
            raise SystemExit(f"key file not found: {key_path}")
        scoring_rows = _merge_key(scoring_rows, _read_tsv(key_path))

    entries = _expert_entries(scoring_rows)
    if not entries:
        raise SystemExit(
            "no rows with expert_score filled — annotators must complete the scoring sheet"
        )

    expert_llm = _expert_vs_llm_summary(entries)
    expert_expert = _human_human_summary(entries)
    disagreements = _disagreements(entries, pass_mark=bi._PASS_MARK)

    summary: Dict[str, Any] = {
        "pass_mark": bi._PASS_MARK,
        "n_expert_scores": len(entries),
        "n_unique_audit_ids": len({e["audit_id"] for e in entries}),
        "expert_vs_llm": expert_llm,
        "expert_vs_expert_overlap": expert_expert,
    }

    out_dir = Path(args.out) if args.out else scoring_path.parent
    if not out_dir.is_absolute():
        out_dir = _HERE / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "expert_audit_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    if disagreements:
        fields = list(disagreements[0].keys())
        with (out_dir / "expert_audit_disagreements.tsv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(disagreements)

    print("=== expert audit summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir / 'expert_audit_summary.json'}")
    if disagreements:
        print(f"wrote {out_dir / 'expert_audit_disagreements.tsv'}")


if __name__ == "__main__":
    main()
