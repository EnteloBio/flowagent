"""Comparison metrics for Benchmark F — output fidelity.

Each function takes a candidate file path, a reference file path, and a
``params`` dict from ``fidelity_cases.yaml``, and returns a flat dict of
numeric metrics that the runner serialises into ``metrics.csv``.
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import pandas as pd
from scipy import stats


# ── DE-table comparison ──────────────────────────────────────────

def compare_de_table(candidate: Path, reference: Path,
                     params: Dict[str, Any]) -> Dict[str, Any]:
    """Score a DE-results table against a reference.

    Returns:
        spearman_lfc      — Spearman ρ on log2FC across the gene
                            intersection (NaN if intersection < 10).
        n_overlap         — number of genes present in both tables.
        jaccard_top_n     — Jaccard on top-N significant genes
                            (padj < alpha & |log2FC| > min_lfc, ranked
                            by |log2FC|).
        n_sig_candidate, n_sig_reference — sig-gene counts before
                            top-N truncation.
    """
    gid  = params.get("gene_id_column", "gene_id")
    lfc  = params.get("log2fc_column",  "log2FoldChange")
    padj = params.get("padj_column",    "padj")
    alpha   = float(params.get("alpha", 0.05))
    min_lfc = float(params.get("min_lfc", 1.0))
    top_n   = int(params.get("top_n", 200))

    cand = _read_table(candidate)
    refr = _read_table(reference)
    for need in (gid, lfc):
        if need not in cand.columns:
            return {"error": f"candidate missing column: {need}"}
        if need not in refr.columns:
            return {"error": f"reference missing column: {need}"}

    cand = cand[[gid, lfc] + ([padj] if padj in cand.columns else [])].copy()
    refr = refr[[gid, lfc] + ([padj] if padj in refr.columns else [])].copy()
    cand[gid] = cand[gid].astype(str).str.replace(r"\.\d+$", "", regex=True)
    refr[gid] = refr[gid].astype(str).str.replace(r"\.\d+$", "", regex=True)

    merged = cand.merge(refr, on=gid, suffixes=("_cand", "_ref"))
    merged = merged.dropna(subset=[f"{lfc}_cand", f"{lfc}_ref"])

    out: Dict[str, Any] = {
        "n_candidate": len(cand),
        "n_reference": len(refr),
        "n_overlap":   len(merged),
    }
    if len(merged) >= 10:
        rho, p = stats.spearmanr(merged[f"{lfc}_cand"], merged[f"{lfc}_ref"])
        out["spearman_lfc"] = float(rho)
        out["spearman_p"]   = float(p)
    else:
        out["spearman_lfc"] = float("nan")
        out["spearman_p"]   = float("nan")

    sig_cand = _top_n_sig(cand, gid, lfc, padj, alpha, min_lfc, top_n)
    sig_ref  = _top_n_sig(refr, gid, lfc, padj, alpha, min_lfc, top_n)
    out["n_sig_candidate"] = len(sig_cand)
    out["n_sig_reference"] = len(sig_ref)
    out["jaccard_top_n"]   = _jaccard(sig_cand, sig_ref)
    out["top_n_used"]      = top_n
    return out


def _top_n_sig(df: pd.DataFrame, gid: str, lfc: str, padj: str,
               alpha: float, min_lfc: float, top_n: int) -> Set[str]:
    sig = df.copy()
    if padj in sig.columns:
        sig = sig[pd.to_numeric(sig[padj], errors="coerce") < alpha]
    sig = sig[pd.to_numeric(sig[lfc], errors="coerce").abs() > min_lfc]
    sig = sig.assign(_abslfc=sig[lfc].abs()).sort_values("_abslfc",
                                                          ascending=False)
    return set(sig[gid].head(top_n).tolist())


# ── BED / peak comparison ────────────────────────────────────────

def compare_peak_bed(candidate: Path, reference: Path,
                     params: Dict[str, Any]) -> Dict[str, Any]:
    """Jaccard on peak intervals with reciprocal-overlap criterion.

    A candidate peak matches a reference peak if both cover ≥``min_overlap``
    of each other (default 0.5). The Jaccard index is
    |matched_pairs| / (|cand| + |ref| - |matched_pairs|).
    """
    min_overlap = float(params.get("min_overlap", 0.5))
    cand = _read_bed(candidate)
    refr = _read_bed(reference)
    matched = _bed_match_pairs(cand, refr, min_overlap)
    union = len(cand) + len(refr) - matched
    return {
        "n_candidate":  len(cand),
        "n_reference":  len(refr),
        "n_matched":    matched,
        "jaccard_peak": matched / union if union else float("nan"),
    }


def _read_bed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None,
                     names=["chrom", "start", "end"], usecols=[0, 1, 2],
                     comment="#", dtype={"chrom": str, "start": int, "end": int})
    return df.sort_values(["chrom", "start"]).reset_index(drop=True)


def _bed_match_pairs(a: pd.DataFrame, b: pd.DataFrame,
                     min_overlap: float) -> int:
    """Count pairs of intervals satisfying reciprocal overlap.

    O(N log N) per chromosome via a sort+two-pointer sweep — acceptable
    for typical peak counts (<1M); a true bedtools join would be faster
    but adds a binary dependency.
    """
    matched = 0
    for chrom, ga in a.groupby("chrom"):
        gb = b[b["chrom"] == chrom].sort_values("start").reset_index(drop=True)
        if gb.empty:
            continue
        for _, ra in ga.iterrows():
            sa, ea = ra["start"], ra["end"]
            la = ea - sa
            j = gb["start"].searchsorted(ea)
            i = max(0, gb["end"].searchsorted(sa) - 1)
            for k in range(i, min(j + 1, len(gb))):
                sb, eb = int(gb.at[k, "start"]), int(gb.at[k, "end"])
                ov = max(0, min(ea, eb) - max(sa, sb))
                lb = eb - sb
                if la > 0 and lb > 0 and ov / la >= min_overlap and ov / lb >= min_overlap:
                    matched += 1
                    break
    return matched


# ── VCF comparison ───────────────────────────────────────────────

def compare_vcf(candidate: Path, reference: Path,
                params: Dict[str, Any]) -> Dict[str, Any]:
    """F1 on (chrom, pos, ref, alt) variant tuples."""
    region = params.get("region")  # "chr22:16000000-50000000"
    cset = _read_vcf_keys(candidate, region=region)
    rset = _read_vcf_keys(reference, region=region)
    tp = len(cset & rset)
    fp = len(cset - rset)
    fn = len(rset - cset)
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) else float("nan")
    f1   = 2 * prec * rec / (prec + rec) if prec and rec else float("nan")
    return {"n_candidate": len(cset), "n_reference": len(rset),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1}


def _read_vcf_keys(path: Path, region: Optional[str] = None) -> Set[Tuple]:
    keys: Set[Tuple] = set()
    chrom_filter, lo, hi = None, None, None
    if region:
        c, _, span = region.partition(":")
        chrom_filter = c
        if span:
            a, _, b = span.partition("-")
            lo, hi = int(a), int(b)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 5:
                continue
            chrom, pos, _, ref, alt = cols[:5]
            if chrom_filter and chrom != chrom_filter:
                continue
            try:
                p = int(pos)
            except ValueError:
                continue
            if lo is not None and (p < lo or p > hi):
                continue
            for a in alt.split(","):
                keys.add((chrom, p, ref, a))
    return keys


# ── Helpers ──────────────────────────────────────────────────────

def _read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if str(path).endswith((".tsv", ".tsv.gz", ".txt")) else ","
    return pd.read_csv(path, sep=sep)


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


COMPARATORS = {
    "de_table": compare_de_table,
    "peak_bed": compare_peak_bed,
    "vcf":      compare_vcf,
}
