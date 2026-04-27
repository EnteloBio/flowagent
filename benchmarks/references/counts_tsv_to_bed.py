"""Extract a 3-column BED from the GSE74912 deposited counts matrix.

NCBI's GSE74912 supplementary file ``GSE74912_ATACseq_All_Counts.txt.gz``
is a tab-separated table where the first three columns are
``Chr  Start  End`` (the consensus peak coordinates) and every
subsequent column is per-sample read counts at that peak. The
Benchmark F runner only needs the peak coordinates, so we extract
columns 1-3 and discard the rest.

Invoked by the download orchestrator with one positional argument:
the path to the gunzipped intermediate TSV.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: counts_tsv_to_bed.py <gunzipped_counts.tsv>")
    src = Path(sys.argv[1])
    if not src.exists():
        sys.exit(f"input not found: {src}")

    out = src.parent / "gse74912_corces_consensus.bed"

    rows: list[tuple[str, int, int]] = []
    with src.open() as fh:
        header = next(fh, "").rstrip("\n").split("\t")
        # Sanity-check the column header; abort if the format diverged.
        if [h.lower() for h in header[:3]] != ["chr", "start", "end"]:
            sys.exit(f"unexpected header in {src}: {header[:3]}")
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 3:
                continue
            try:
                rows.append((cols[0], int(cols[1]), int(cols[2])))
            except ValueError:
                continue

    rows.sort(key=lambda r: (r[0], r[1]))
    with out.open("w") as fh:
        for chrom, start, end in rows:
            fh.write(f"{chrom}\t{start}\t{end}\n")
    src.unlink(missing_ok=True)
    print(f"[ok] wrote {len(rows):,} peaks → {out}")


if __name__ == "__main__":
    main()
