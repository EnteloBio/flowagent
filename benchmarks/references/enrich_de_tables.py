#!/usr/bin/env python3
"""Add ``gene_symbol`` to Benchmark F/G DE reference tables in-place.

The interpretation MCQs name genes by symbol (CRISPLD2, Krt14, …) but the
frozen DESeq2/edgeR recipes originally emitted Ensembl / Entrez IDs only.
This script maps IDs → symbols via MyGene.info and rewrites each TSV with
columns ``gene_id, gene_symbol, log2FoldChange, padj``.

Usage (from ``benchmarks/``)::

    python references/enrich_de_tables.py
    python references/enrich_de_tables.py --only gse52778_himes_DE.tsv
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List

import pandas as pd

_HERE = Path(__file__).parent

_TABLES = {
    "gse52778_himes_DE.tsv":      ("human", "ensembl.gene"),
    "gse60450_fu_DE.tsv":         ("mouse", "entrezgene"),
    "gse152418_covid_blood_DE.tsv": ("human", "ensembl.gene"),
}


_ENSEMBL = {
    "human": "https://rest.ensembl.org/lookup/id",
    "mouse": "https://rest.ensembl.org/lookup/id",
}


def _map_symbols(ids: List[str], species: str, scope: str) -> Dict[str, str]:
    """Batch map Ensembl gene IDs → gene symbol via Ensembl REST."""
    del scope  # Ensembl lookup is ID-native
    clean = list(dict.fromkeys(str(i).split(".")[0] for i in ids))
    out: Dict[str, str] = {}
    chunk = 900
    for i in range(0, len(clean), chunk):
        batch = clean[i:i + chunk]
        payload = json.dumps({"ids": batch}).encode()
        req = urllib.request.Request(
            _ENSEMBL[species],
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            hits = json.loads(resp.read())
        for ens_id, meta in hits.items():
            if not isinstance(meta, dict):
                continue
            sym = meta.get("display_name") or meta.get("external_name") or ""
            if sym:
                out[str(ens_id).split(".")[0]] = str(sym)
    return out


def enrich(path: Path, species: str, scope: str) -> None:
    df = pd.read_csv(path, sep="\t")
    if "gene_symbol" in df.columns and df["gene_symbol"].notna().sum() > 100:
        print(f"[skip] {path.name}: already has gene_symbol")
        return
    gid = "gene_id" if "gene_id" in df.columns else df.columns[0]
    sym_map = _map_symbols(df[gid].astype(str).tolist(), species, scope)
    df["gene_symbol"] = (
        df[gid].astype(str).str.replace(r"\.\d+$", "", regex=True).map(sym_map).fillna("")
    )
    cols = [gid, "gene_symbol", "log2FoldChange", "padj"]
    cols = [c for c in cols if c in df.columns]
    extra = [c for c in df.columns if c not in cols]
    df = df[cols + extra]
    df.to_csv(path, sep="\t", index=False)
    n_sym = (df["gene_symbol"] != "").sum()
    print(f"[ok] {path.name}: {n_sym}/{len(df)} rows with gene_symbol")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="Single filename under references/")
    args = ap.parse_args()
    targets = _TABLES
    if args.only:
        if args.only not in _TABLES:
            sys.exit(f"unknown table {args.only!r}; known: {list(_TABLES)}")
        targets = {args.only: _TABLES[args.only]}
    for fname, (species, scope) in targets.items():
        path = _HERE / fname
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        enrich(path, species, scope)


if __name__ == "__main__":
    main()
