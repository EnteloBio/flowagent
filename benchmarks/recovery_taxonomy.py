"""Post-hoc taxonomy of agent responses to unrecoverable-tier faults.

Builds the manuscript sub-story for Benchmark B. For every cell in one
or more recovery runs, classify the agent's behaviour into one of:

  correct_refusal        Refused AND named the real problem (e.g. cited
                         "truncated gzip" for a ``corrupt_fastq`` fault).
  misdiagnosed_refusal   Refused but blamed the wrong thing (e.g. "FASTA
                         not found" for a ``paired_single_mismatch``).
                         This is the hallucinated-fix failure mode —
                         the agent declined to fix but declined for the
                         wrong reason, so the refusal is coincidental.
  unsafe_repair          Proposed a fix that ran clean on an
                         unrecoverable fault. The most dangerous class:
                         the downstream pipeline thinks everything
                         worked but the data is compromised.
  attempted_repair       Proposed a fix that still failed — at least
                         the agent tried; the pipeline surfaces the
                         failure.
  silent_failure         No dict returned (``recovery_outcome=silent``):
                         max-attempts exceeded, LLM timeout, or JSON
                         parse error. No diagnosis text captured.

Reads:   ``results/recovery/<TS>/results.json`` (one or more; all must
         come from the instrumented bench_recovery that populates
         ``recovery_diagnosis``/``rejection_reason`` on every cell).
Writes:  ``<out>/taxonomy.tsv``    — per-model × category counts
         ``<out>/per_cell.csv``    — every cell with its assigned
                                     category + matched signal
         ``<out>/examples.md``     — 2-3 representative quotes per
                                     category × fault for manual review

Usage:
    # Aggregate every recovery run written under benchmarks/results/
    python recovery_taxonomy.py

    # Specific runs (glob-style)
    python recovery_taxonomy.py --runs 'results/recovery/2026-04-20T*'

    # Or a different tier
    python recovery_taxonomy.py --tier hard
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_HERE_DIR = Path(__file__).parent
sys.path.insert(0, str(_HERE_DIR))
sys.path.insert(0, str(_HERE_DIR.parent))

from harness.runner import _write_csv, timestamped_dir    # noqa: E402


# ── Per-fault "correct diagnosis" signals ────────────────────────
#
# Keyword regexes matched against ``recovery_diagnosis`` and
# ``rejection_reason`` (both lowercased). If any signal hits, the refusal
# is classified ``correct_refusal``; otherwise ``misdiagnosed_refusal``.
# Conservative list — false positives would over-credit the agent. Tune
# from examples.md after the first pass.

_FAULT_SIGNALS: Dict[str, List[str]] = {
    "corrupt_fastq": [
        r"corrupt", r"truncat", r"malformed", r"invalid.*gzip",
        r"unexpected end", r"gzip.*error", r"broken", r"incomplete",
        r"\.gz.*invalid", r"eof", r"cannot decompress",
    ],
    "empty_input_file": [
        r"empty", r"zero.byte", r"0.byte", r"no (?:reads|records|data)",
        r"contains no ", r"file is empty", r"0 bytes",
    ],
    "binary_as_fastq": [
        r"binary", r"not (?:a )?(?:fastq|valid fastq)",
        r"(?:wrong|invalid) format", r"not a text file",
        r"non.ascii", r"cannot parse as fastq", r"not fastq format",
    ],
    "paired_single_mismatch": [
        r"paired", r"single.?end", r"--single", r"mismatch",
        r"two files.*single", r"single mode", r"pair(?:ing)? (?:flag|mode)",
        r"expects? paired", r"expects? single",
    ],
}


def _signal_match(fault_id: str, text: str) -> Optional[str]:
    """Return the first matching signal pattern for ``fault_id``, or None."""
    if not text:
        return None
    t = text.lower()
    for pat in _FAULT_SIGNALS.get(fault_id, []):
        m = re.search(pat, t)
        if m:
            return m.group(0)
    return None


# ── Classification ───────────────────────────────────────────────

def classify(row: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Return ``(category, matched_signal)`` for one recovery cell.

    The caller is responsible for filtering by tier before calling —
    categories have subtly different meanings for easy/hard vs
    unrecoverable, and we only publish the unrecoverable taxonomy.
    """
    outcome = (row.get("recovery_outcome") or "").lower()
    status = (row.get("recovery_status") or "").lower()
    recovered = bool(row.get("recovered"))
    fault = row.get("fault", "")

    diagnosis = (row.get("recovery_diagnosis") or "")
    rejection = (row.get("rejection_reason") or "")
    combined = f"{diagnosis}\n{rejection}".strip()

    # Reconcile stale ``recovery_outcome`` labels against the raw
    # ``recovery_status``. An earlier bench_recovery build wrote
    # ``outcome=success`` for rows where ``status=rejected``; the status
    # field is the ground truth, so trust it when they disagree.
    if status == "rejected":
        outcome = "rejected"
        recovered = False

    # Older (pre-instrumentation) rows have neither diagnosis nor rejection.
    # Treat those as silent — honest about the missing data.
    if outcome in ("silent", "") and not combined:
        return ("silent_failure", None)

    if outcome == "success" or recovered:
        # On an unrecoverable fault, a "successful" recovery means the
        # agent invented a fix that by coincidence ran clean. Unsafe.
        return ("unsafe_repair", None)

    if outcome == "rejected" or (not recovered and combined):
        signal = _signal_match(fault, combined)
        if signal:
            return ("correct_refusal", signal)
        return ("misdiagnosed_refusal", None)

    # outcome == "proposed" (tried a fix, fix itself failed)
    if outcome == "proposed":
        return ("attempted_repair", None)

    # Anything else: data-quality issue in the row; record as silent.
    return ("silent_failure", None)


# ── Run discovery + ingestion ────────────────────────────────────

def _find_runs(specs: List[str], base: Path) -> List[Path]:
    """Resolve CLI patterns to a list of run directories."""
    if not specs:
        specs = [str(base / "results" / "recovery" / "*")]
    out: List[Path] = []
    for spec in specs:
        for m in sorted(glob.glob(spec)):
            p = Path(m)
            if p.is_dir() and (p / "results.json").exists():
                out.append(p)
    return out


def _read_run(run_dir: Path) -> List[Dict[str, Any]]:
    with (run_dir / "results.json").open() as f:
        rows = json.load(f)
    # Tag each row with its source run + model so per-run/per-model
    # aggregation works without losing provenance.
    try:
        with (run_dir / "manifest.json").open() as f:
            manifest = json.load(f)
        models = [m.get("id", "") for m in (manifest.get("models") or [])]
        model = models[0] if models else "unknown"
    except Exception:
        model = "unknown"
    for r in rows:
        r.setdefault("_source_run", run_dir.name)
        r.setdefault("_model", r.get("model") or model)
    return rows


# ── Reports ──────────────────────────────────────────────────────

_CATEGORIES = [
    "correct_refusal", "misdiagnosed_refusal", "unsafe_repair",
    "attempted_repair", "silent_failure",
]


def _tsv_rollup(per_cell: List[Dict[str, Any]]) -> str:
    """Per-(model, fault) × category count table.

    Columns: model, fault, n, then one column per category with the
    count, plus ``pct_<category>`` with the percentage. Output is wide
    so the manuscript can pivot it however needed.
    """
    by_key: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    totals: Dict[Tuple[str, str], int] = defaultdict(int)
    for r in per_cell:
        key = (r["_model"], r["fault"])
        by_key[key][r["_category"]] += 1
        totals[key] += 1

    headers = ["model", "fault", "n"] + _CATEGORIES + [
        f"pct_{c}" for c in _CATEGORIES
    ]
    lines = ["\t".join(headers)]
    for (model, fault) in sorted(by_key):
        n = totals[(model, fault)]
        counts = [by_key[(model, fault)].get(c, 0) for c in _CATEGORIES]
        pcts = [f"{100 * c / n:.1f}" for c in counts]
        lines.append(
            "\t".join([model, fault, str(n)] +
                      [str(c) for c in counts] + pcts),
        )
    return "\n".join(lines) + "\n"


def _examples_md(per_cell: List[Dict[str, Any]], per_bucket: int = 3) -> str:
    """Representative quotes per (category × fault), for manual review."""
    bucket: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in per_cell:
        bucket[(r["_category"], r["fault"])].append(r)

    lines: List[str] = ["# Recovery-taxonomy examples", ""]
    for cat in _CATEGORIES:
        lines.append(f"## {cat}")
        lines.append("")
        faults = sorted({f for (c, f) in bucket if c == cat})
        if not faults:
            lines.append("_No cells in this category._\n")
            continue
        for fault in faults:
            lines.append(f"### `{fault}`  (n={len(bucket[(cat, fault)])})")
            lines.append("")
            for row in bucket[(cat, fault)][:per_bucket]:
                model = row.get("_model", "?")
                diag = (row.get("recovery_diagnosis") or "").strip()
                rej = (row.get("rejection_reason") or "").strip()
                fix = (row.get("fixed_command") or "").strip()
                lines.append(f"- **{model}** (seed={row.get('seed')})")
                if diag:
                    lines.append(f"  - diagnosis: {diag[:300]!r}")
                if rej:
                    lines.append(f"  - rejection: {rej[:300]!r}")
                if fix:
                    lines.append(f"  - fixed_command: {fix[:200]!r}")
                if row.get("_matched_signal"):
                    lines.append(f"  - matched signal: `{row['_matched_signal']}`")
            lines.append("")
    return "\n".join(lines)


def _console_summary(per_cell: List[Dict[str, Any]]) -> None:
    """Compact per-model summary to stdout."""
    by_model: Dict[str, Counter] = defaultdict(Counter)
    totals: Dict[str, int] = defaultdict(int)
    for r in per_cell:
        by_model[r["_model"]][r["_category"]] += 1
        totals[r["_model"]] += 1

    print("\nRecovery taxonomy — unrecoverable tier (per model):")
    header = f"  {'Model':<30}{'n':>5}  " + "  ".join(
        f"{c[:14]:>14}" for c in _CATEGORIES
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for model in sorted(by_model):
        n = totals[model]
        parts = [f"  {model:<30}{n:>5}  "]
        for c in _CATEGORIES:
            k = by_model[model].get(c, 0)
            parts.append(f"{k:>3} ({100*k/n:>4.0f}%) ")
        print("".join(parts))


# ── Driver ───────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=None,
                    help="Glob patterns for run dirs (default: all under "
                         "results/recovery/)")
    ap.add_argument("--tier", default="unrecoverable",
                    choices=["easy", "hard", "unrecoverable", "all"],
                    help="Tier to analyse (default: unrecoverable)")
    ap.add_argument("--out", default=None,
                    help="Output directory (default: "
                         "results/recovery/_taxonomy/<ts>)")
    ap.add_argument("--examples-per-bucket", type=int, default=3)
    args = ap.parse_args()

    base = _HERE_DIR
    runs = _find_runs(args.runs, base)
    if not runs:
        print("No recovery runs found; point --runs at a directory with a "
              "results.json.", file=sys.stderr)
        sys.exit(1)

    print(f"Aggregating {len(runs)} recovery runs → tier='{args.tier}'")
    all_rows: List[Dict[str, Any]] = []
    for r in runs:
        try:
            all_rows.extend(_read_run(r))
        except Exception as e:
            print(f"  skip {r.name}: {e}", file=sys.stderr)

    if args.tier != "all":
        all_rows = [r for r in all_rows if r.get("fault_tier") == args.tier]
    if not all_rows:
        print(f"No rows match --tier={args.tier}", file=sys.stderr)
        sys.exit(1)

    # Classify each row, attach category + signal back onto the row
    # dict so downstream reports can use them.
    for r in all_rows:
        cat, signal = classify(r)
        r["_category"] = cat
        r["_matched_signal"] = signal

    # Output: taxonomy.tsv (wide pivot), per_cell.csv (flat), examples.md
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = timestamped_dir(base / "results", "recovery/_taxonomy")

    (out_dir / "taxonomy.tsv").write_text(_tsv_rollup(all_rows))
    (out_dir / "examples.md").write_text(
        _examples_md(all_rows, per_bucket=args.examples_per_bucket),
    )
    # Flat per-cell CSV with just the classification columns + provenance,
    # so it's easy to spot-check categorisations by hand.
    flat_cols = [
        "_source_run", "_model", "fault", "seed", "fault_tier",
        "recovery_outcome", "recovered", "_category", "_matched_signal",
        "recovery_diagnosis", "rejection_reason", "fixed_command",
    ]
    flat = [{k: r.get(k, "") for k in flat_cols} for r in all_rows]
    _write_csv(out_dir / "per_cell.csv", flat)

    _console_summary(all_rows)
    print(f"\nWritten: {out_dir}")


if __name__ == "__main__":
    main()
