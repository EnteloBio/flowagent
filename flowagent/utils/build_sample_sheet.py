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


# Stable NCBI SRA runinfo CSV schema — used when the downloaded file is
# missing its header row (which happens when the ``download_geo_metadata``
# bash loop's first iteration returns header-only output and subsequent
# iterations strip their own headers via ``tail -n +2``).
_SRA_RUNINFO_COLUMNS = [
    "Run", "ReleaseDate", "LoadDate", "spots", "bases", "spots_with_mates",
    "avgLength", "size_MB", "AssemblyName", "download_path",
    "Experiment", "LibraryName", "LibraryStrategy", "LibrarySelection",
    "LibrarySource", "LibraryLayout", "InsertSize", "InsertDev", "Platform",
    "Model", "SRAStudy", "BioProject", "ProjectID", "Sample", "BioSample",
    "SampleType", "TaxID", "ScientificName", "SampleName", "g1k_pop_code",
    "source", "g1k_analysis_group", "Subject_ID", "Sex", "Disease", "Tumor",
    "Affection_Status", "Analyte_Type", "Histological_Type", "Body_Site",
    "CenterName", "Submission", "dbgap_study_accession", "Consent",
    "RunHash", "ReadHash",
]


def parse_runinfo(path: str) -> Dict[str, List[str]]:
    """Return ``SRX -> [SRR, ...]`` from a GEO ``runinfo.csv``.

    Tolerant of missing header rows — the ``esearch | efetch -format runinfo``
    loop sometimes drops the initial header depending on how the caller
    concatenates per-GSM results. If the first line's first field looks like
    an SRR accession rather than the literal ``Run``, we fall back to the
    stable SRA runinfo column schema.
    """
    srx_to_srr: Dict[str, List[str]] = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
        f.seek(0)

        first_field = first_line.split(",", 1)[0].strip() if first_line else ""
        has_header = first_field == "Run"

        if has_header:
            reader = csv.DictReader(f)
        else:
            # Headerless file — apply the canonical SRA schema. Only as many
            # columns as we actually have in the data are populated; extra
            # schema columns are ignored safely by ``row.get()`` below.
            reader = csv.DictReader(f, fieldnames=_SRA_RUNINFO_COLUMNS)

        for row in reader:
            srr = (row.get("Run") or "").strip()
            srx = (row.get("Experiment") or "").strip()
            if srr and srx and srr.startswith(("SRR", "ERR", "DRR")):
                srx_to_srr.setdefault(srx, []).append(srr)
    return srx_to_srr


def match_label(sample: Dict[str, str], labels: List[str]) -> str:
    """Return the first label that is a case-insensitive prefix of a token
    in the sample's title or characteristics.

    A ``token`` is an ``\\w+`` run — alphanumerics plus underscore. Prefix
    matching is deliberate so short labels like ``Dex`` match treatments
    like ``Dexamethasone`` (the standard GEO pattern of abbreviating a
    drug name in the sample ID while spelling it out in ``treatment:``).

    Labels that contain a non-word character (typically a hyphen, e.g.
    ``Dex-treated`` extracted from a prompt like ``between Dex-treated
    and untreated``) are also tried by their first ``\\w+`` sub-token.
    That way descriptor suffixes like ``-treated`` / ``-stimulated`` /
    ``-group`` don't have to literally appear in the SOFT characteristics
    to match — the core drug/condition name suffices.

    Combo treatments like ``Albuterol_Dex`` are a single token, and ``Dex``
    is *not* a prefix of ``Albuterol_Dex``, so short labels don't
    accidentally capture combos — the user would have to supply
    ``Albuterol_Dex`` (or similar) explicitly.

    Labels are checked in the order the user provided them, so put the
    more specific label first when there's potential overlap.
    """
    haystack = f"{sample.get('title', '')} {sample.get('characteristics', '')}"
    tokens = re.findall(r"\w+", haystack)
    for label in labels:
        candidates = [label]
        first_sub = next(iter(re.findall(r"\w+", label)), "")
        if first_sub and first_sub != label:
            candidates.append(first_sub)
        for cand in candidates:
            lab = cand.lower()
            if not lab:
                continue
            for tok in tokens:
                if tok.lower().startswith(lab):
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

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if labels:
        # Labels given → write a DE-ready 2-column TSV with ONLY the matched
        # rows. Unmatched samples are out-of-scope for this comparison and
        # would otherwise contaminate downstream DESeq2 (empty ``condition``
        # becomes a spurious factor level).
        matched_rows = [r for r in rows if r["condition"]]
        if not matched_rows:
            print(
                f"ERROR: no samples matched any of the labels {labels!r}. "
                f"Check label spelling against {args.soft}.",
                file=sys.stderr,
            )
            return 1

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("sample_id\tcondition\n")
            for r in matched_rows:
                f.write(f"{r['sample_id']}\t{r['condition']}\n")

        # Per-label tally for quick sanity check.
        from collections import Counter
        counts = Counter(r["condition"] for r in matched_rows)
        tally = ", ".join(f"{c} {lab}" for lab, c in counts.most_common())
        print(f"Wrote {len(matched_rows)} rows to {out_path} ({tally})")
        if unmatched_gsms:
            skipped = len(set(unmatched_gsms))
            print(
                f"Excluded {skipped} out-of-scope sample(s) not matching "
                f"any of {labels!r} (Albuterol combos, etc).",
                file=sys.stderr,
            )
        return 0

    # No labels → write the 4-column template so the user can fill in
    # conditions by hand. Keep every row so they can see what's available.
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("sample_id\ttitle\tcharacteristics\tcondition\n")
        for r in rows:
            f.write(
                f"{r['sample_id']}\t{r['title']}\t"
                f"{r['characteristics']}\t{r['condition']}\n"
            )
    print(f"Wrote {len(rows)} rows to {out_path}")
    print(
        f"No labels supplied — template written. Fill the 'condition' column "
        f"of {out_path} before running DESeq2.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
