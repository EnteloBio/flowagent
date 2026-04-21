"""Build sample_conditions.tsv from a GEO SOFT family file and SRA runinfo.csv.

Standalone CLI so it can be invoked directly from a workflow step without
pulling in the planner / LLM layers. Usage::

    python -m flowagent.utils.build_sample_sheet \\
        --soft raw_data/metadata/GSE186412_family.soft \\
        --runinfo raw_data/GSE186412_runinfo.csv \\
        --labels Blank,Pamc3 \\
        --out sample_conditions.tsv

When every sample in the SOFT file matches one of the labels, the output is
a two-column TSV ready for DESeq2 (``sample_id\\tcondition``). When labels
are missing or some samples don't match, a four-column template is written
(``sample_id\\ttitle\\tcharacteristics\\tcondition``) so the user can fill
the ``condition`` column by hand before re-running the downstream steps.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, Iterator, List


def parse_soft(path: str) -> Iterator[Dict[str, str]]:
    """Yield one dict per ``^SAMPLE`` block in a GEO SOFT family file."""
    current: Dict[str, str] | None = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("^SAMPLE"):
                if current and current.get("gsm"):
                    yield current
                _, _, gsm = line.partition("=")
                current = {
                    "gsm": gsm.strip(),
                    "title": "",
                    "characteristics": "",
                    "srx": "",
                }
            elif current is None:
                continue
            elif line.startswith("!Sample_title"):
                _, _, val = line.partition("=")
                current["title"] = val.strip()
            elif line.startswith("!Sample_characteristics_ch1"):
                _, _, val = line.partition("=")
                existing = current["characteristics"]
                current["characteristics"] = (
                    f"{existing}; {val.strip()}" if existing else val.strip()
                )
            elif line.startswith("!Sample_relation") and "SRA" in line:
                match = re.search(r"(SRX\d+)", line)
                if match:
                    current["srx"] = match.group(1)
        if current and current.get("gsm"):
            yield current


def parse_runinfo(path: str) -> Dict[str, List[str]]:
    """Return ``SRX -> [SRR, ...]`` from a GEO ``runinfo.csv``."""
    srx_to_srr: Dict[str, List[str]] = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            srr = (row.get("Run") or "").strip()
            srx = (row.get("Experiment") or "").strip()
            if srr and srx:
                srx_to_srr.setdefault(srx, []).append(srr)
    return srx_to_srr


def match_label(sample: Dict[str, str], labels: List[str]) -> str:
    """Return the first label whose whole-word appears in title/characteristics."""
    haystack = f"{sample.get('title', '')} {sample.get('characteristics', '')}".lower()
    for label in labels:
        if re.search(rf"\b{re.escape(label.lower())}\b", haystack):
            return label
    return ""


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--soft", required=True, help="Decompressed GEO SOFT family file")
    parser.add_argument("--runinfo", required=True, help="SRA runinfo.csv")
    parser.add_argument(
        "--labels",
        default="",
        help="Comma-separated condition labels (e.g. 'Blank,Pamc3'). "
             "Empty string emits a fill-in template.",
    )
    parser.add_argument("--out", required=True, help="Output TSV path")
    args = parser.parse_args(argv)

    labels = [lab.strip() for lab in args.labels.split(",") if lab.strip()]
    if len(set(l.lower() for l in labels)) < len(labels):
        print(
            "WARNING: duplicate labels ignored",
            file=sys.stderr,
        )
    srx_to_srr = parse_runinfo(args.runinfo)

    rows: List[Dict[str, str]] = []
    unmatched_gsms: List[str] = []
    for sample in parse_soft(args.soft):
        condition = match_label(sample, labels) if labels else ""
        srrs = srx_to_srr.get(sample["srx"], [])
        if not srrs:
            # Runinfo didn't map this sample — fall back to GSM as the id
            # so the user can still match it manually.
            srrs = [sample["gsm"]]
        for srr in srrs:
            rows.append({
                "sample_id": srr,
                "title": sample["title"],
                "characteristics": sample["characteristics"],
                "condition": condition,
            })
        if labels and not condition:
            unmatched_gsms.append(sample["gsm"])

    if not rows:
        print(f"ERROR: no samples parsed from {args.soft}", file=sys.stderr)
        return 1

    all_matched = bool(labels) and not unmatched_gsms
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        if all_matched:
            f.write("sample_id\tcondition\n")
            for r in rows:
                f.write(f"{r['sample_id']}\t{r['condition']}\n")
        else:
            f.write("sample_id\ttitle\tcharacteristics\tcondition\n")
            for r in rows:
                f.write(
                    f"{r['sample_id']}\t{r['title']}\t"
                    f"{r['characteristics']}\t{r['condition']}\n"
                )

    print(f"Wrote {len(rows)} rows to {out_path}")
    if unmatched_gsms:
        print(
            f"WARNING: {len(set(unmatched_gsms))} sample(s) had no condition matched; "
            f"edit the 'condition' column of {out_path} before running DESeq2.",
            file=sys.stderr,
        )
    elif not labels:
        print(
            f"No labels supplied — template written. Fill the 'condition' column "
            f"of {out_path} before running DESeq2.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
