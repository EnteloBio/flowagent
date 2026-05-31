#!/usr/bin/env python3
"""Download GIAB HG002 v4.2.1 benchmark VCF and subset to chr20.

Materialises ``references/giab_na12878_chr20_truth.vcf.gz`` for Benchmark G.
"""

from __future__ import annotations

import gzip
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

_HERE = Path(__file__).parent
_OUT = _HERE / "giab_na12878_chr20_truth.vcf.gz"
_URL = (
    "https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/"
    "AshkenazimTrio/HG002_NA24385_son/NISTv4.2.1/GRCh38/"
    "HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"
)


def _http_get(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (FlowAgent benchmark fetch)",
    })
    with urllib.request.urlopen(req, timeout=600) as resp, dest.open("wb") as fh:
        shutil.copyfileobj(resp, fh)


def _subset_python(src: Path, dest: Path) -> int:
    n = 0
    with gzip.open(src, "rt") as fi, gzip.open(dest, "wt") as fo:
        for line in fi:
            if line.startswith("#"):
                fo.write(line)
                continue
            chrom = line.split("\t", 1)[0]
            if chrom not in ("chr20", "20"):
                continue
            fo.write(line)
            n += 1
    return n


def main() -> None:
    if _OUT.exists() and _OUT.stat().st_size > 1000:
        print(f"[skip] {_OUT} already exists")
        return
    with tempfile.TemporaryDirectory() as td:
        full = Path(td) / "full.vcf.gz"
        print(f"[fetch] {_URL}")
        _http_get(_URL, full)
        # Prefer bcftools when available (faster on huge files).
        try:
            subprocess.run(
                ["bcftools", "view", "-r", "chr20", "-Oz", "-o", str(_OUT), str(full)],
                check=True, capture_output=True,
            )
            with gzip.open(_OUT, "rt") as f:
                n = sum(1 for ln in f if not ln.startswith("#"))
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("[info] bcftools unavailable — Python chr20 subset")
            n = _subset_python(full, _OUT)
    print(f"[ok] wrote {n} chr20 variants → {_OUT}")


if __name__ == "__main__":
    main()
