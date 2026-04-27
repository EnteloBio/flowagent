"""Convert a GEO-deposited MACS ``*_peaks.txt`` file to 3-column BED.

The MACS legacy text format used by GSE32222's GSM-level deposits has
a few comment-prefixed header lines, then a tab-separated table whose
first three columns are ``chr``, ``start``, ``end``. We strip the
comments, keep those three columns, sort by chrom/start, and write
the BED next to the input. The post-process step is invoked by the
download orchestrator with one positional argument: the path to the
gunzipped MACS .txt produced by the preceding direct-URL fetch.

The output BED replaces ``_gsm798425_macs.txt`` with the case's
declared ``reference`` filename — the orchestrator hands us the
already-gunzipped intermediate, so we only need to write the final
BED at the path declared in fidelity_cases.yaml.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: macs_txt_to_bed.py <gunzipped_macs.txt>")
    src = Path(sys.argv[1])
    if not src.exists():
        sys.exit(f"input not found: {src}")

    # Output filename is hard-coded to the YAML's ``reference`` value
    # for this case. We resolve relative to the script's directory.
    out = src.parent / "gse32222_er_canonical.bed"

    rows: list[tuple[str, int, int]] = []
    with src.open() as fh:
        header_seen = False
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if not header_seen:
                # The first non-comment row in MACS .txt is the column header
                # (``chr  start  end  length  ...``); skip it.
                header_seen = True
                continue
            if len(cols) < 3:
                continue
            try:
                rows.append((cols[0], int(cols[1]), int(cols[2])))
            except ValueError:
                # Malformed numeric — skip silently; MACS .txt is well-behaved
                # in practice but defensive parsing keeps the build resilient.
                continue

    rows.sort(key=lambda r: (r[0], r[1]))
    with out.open("w") as fh:
        for chrom, start, end in rows:
            fh.write(f"{chrom}\t{start}\t{end}\n")
    src.unlink(missing_ok=True)  # remove the intermediate
    print(f"[ok] wrote {len(rows):,} peaks → {out}")


if __name__ == "__main__":
    main()
