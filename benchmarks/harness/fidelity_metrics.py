"""Comparison metrics for Benchmark F — output fidelity.

Each function takes a candidate file path, a reference file path, and a
``params`` dict from ``fidelity_cases.yaml``, and returns a flat dict of
numeric metrics that the runner serialises into ``metrics.csv``.
"""

from __future__ import annotations

import gzip
import io
import json
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ── DE-table comparison ──────────────────────────────────────────

# DESeq2's ``write.csv(res, ...)`` puts the rownames in an unnamed first
# column, which pandas reads as ``Unnamed: 0``. tximport-summarised
# tables sometimes use ``Gene`` / ``gene`` / ``ensembl_id``. We try the
# canonical name first, then accept any of these aliases and rename to
# the canonical name so downstream code can assume one schema.
_GENE_ID_ALIASES = (
    "gene_id", "Gene", "gene", "feature_id",
    "ensembl_id", "ensembl_gene_id", "GeneID",
    "Unnamed: 0", "",
)


def _resolve_gene_col(df: pd.DataFrame, want: str) -> Optional[str]:
    """Return the column name that should be treated as the gene ID."""
    if want in df.columns:
        return want
    for alias in _GENE_ID_ALIASES:
        if alias in df.columns:
            return alias
    return None


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

    # Resolve the gene-ID column on each side. Both tables get renamed
    # to the canonical ``gid`` so the rest of the function only needs
    # one column name regardless of which alias the candidate used.
    cand_gid = _resolve_gene_col(cand, gid)
    refr_gid = _resolve_gene_col(refr, gid)
    if cand_gid is None:
        return {"error": f"candidate has no recognised gene-ID column "
                         f"(tried {gid!r} and aliases); columns are "
                         f"{list(cand.columns)[:8]}"}
    if refr_gid is None:
        return {"error": f"reference has no recognised gene-ID column "
                         f"(tried {gid!r} and aliases); columns are "
                         f"{list(refr.columns)[:8]}"}
    if cand_gid != gid:
        cand = cand.rename(columns={cand_gid: gid})
    if refr_gid != gid:
        refr = refr.rename(columns={refr_gid: gid})

    if lfc not in cand.columns:
        return {"error": f"candidate missing log2FC column: {lfc}"}
    if lfc not in refr.columns:
        return {"error": f"reference missing log2FC column: {lfc}"}

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


# ── Methylation table (Benchmark F case 8 — WGBS) ────────────────

def compare_methylation_table(candidate: Path, reference: Path,
                              params: Dict[str, Any]) -> Dict[str, Any]:
    """Per-CpG β-value comparison.

    Both candidate and reference are TSVs with at least
    ``(chrom, pos, beta)`` columns. We join on (chrom, pos), compute
    Spearman ρ on β-values across the intersection, plus a binary
    concordance: fraction of CpGs where both calls agree on the
    hypo (β < ``hypo_threshold``) / mid / hyper (β > ``hyper_threshold``)
    classification. Concordance is more robust than Spearman to the
    inevitable ~10% disagreement at borderline-methylated sites.
    """
    chrom_col = params.get("chrom_column", "chrom")
    pos_col   = params.get("pos_column",   "pos")
    beta_col  = params.get("beta_column",  "beta_value")
    hypo  = float(params.get("hypo_threshold",  0.25))
    hyper = float(params.get("hyper_threshold", 0.75))
    min_n = int(params.get("min_intersection", 1000))

    cand = _read_table(candidate)
    refr = _read_table(reference)
    for need in (chrom_col, pos_col, beta_col):
        if need not in cand.columns:
            return {"error": f"candidate missing column: {need}"}
        if need not in refr.columns:
            return {"error": f"reference missing column: {need}"}

    merged = cand[[chrom_col, pos_col, beta_col]].merge(
        refr[[chrom_col, pos_col, beta_col]],
        on=[chrom_col, pos_col], suffixes=("_cand", "_ref"),
    )
    n_overlap = len(merged)
    out: Dict[str, Any] = {
        "n_candidate":  len(cand),
        "n_reference":  len(refr),
        "n_overlap":    n_overlap,
    }
    if n_overlap < min_n:
        out["error"] = f"intersection too small: {n_overlap} < {min_n}"
        out["spearman_beta"] = float("nan")
        out["concordance"]   = float("nan")
        return out

    cand_b = pd.to_numeric(merged[f"{beta_col}_cand"], errors="coerce")
    refr_b = pd.to_numeric(merged[f"{beta_col}_ref"],  errors="coerce")
    keep = cand_b.notna() & refr_b.notna()
    cand_b, refr_b = cand_b[keep], refr_b[keep]
    rho, p = stats.spearmanr(cand_b, refr_b)
    out["spearman_beta"] = float(rho)
    out["spearman_p"]    = float(p)

    def _bin(s):
        return pd.cut(s, bins=[-0.001, hypo, hyper, 1.001],
                      labels=["hypo", "mid", "hyper"])
    out["concordance"] = float((_bin(cand_b) == _bin(refr_b)).mean())
    return out


# ── Single-cell cluster markers (Benchmark F case 9) ─────────────

def compare_cluster_markers(candidate: Path, reference: Path,
                            params: Dict[str, Any]) -> Dict[str, Any]:
    """Per-cluster top-N marker-gene Jaccard with name harmonisation.

    Both files have a ``cluster_label`` column and a ``gene`` column;
    rows are ranked by ``rank_column`` (default ``log2FC``). We compute
    Jaccard on the top-N markers per cluster after harmonising cluster
    names via ``params["cluster_alias_map"]`` (so a candidate calling a
    cluster "CD14 Mono" matches the reference's "Monocyte"). Returns
    mean Jaccard across recovered clusters and a recovery count
    (clusters with Jaccard ≥ ``min_cluster_recovery``).
    """
    cluster_col = params.get("cluster_column", "cluster_label")
    gene_col    = params.get("gene_column",    "gene")
    rank_col    = params.get("rank_column",    "log2FC")
    top_n       = int(params.get("top_n_per_cluster", 20))
    min_recov   = float(params.get("min_cluster_recovery", 0.5))
    aliases     = params.get("cluster_alias_map") or {}

    # Build alias lookup: {alternative_name: canonical_name}
    canon: Dict[str, str] = {}
    for canonical, alts in aliases.items():
        canon[canonical.lower().strip()] = canonical
        for a in alts:
            canon[a.lower().strip()] = canonical

    def _canonicalise(s: str) -> str:
        return canon.get(str(s).lower().strip(), str(s).strip())

    cand = _read_table(candidate)
    refr = _read_table(reference)
    for need in (cluster_col, gene_col):
        if need not in cand.columns:
            return {"error": f"candidate missing column: {need}"}
        if need not in refr.columns:
            return {"error": f"reference missing column: {need}"}

    cand[cluster_col] = cand[cluster_col].map(_canonicalise)
    refr[cluster_col] = refr[cluster_col].map(_canonicalise)

    def _top_n(df):
        out: Dict[str, Set[str]] = {}
        if rank_col in df.columns:
            df = df.assign(_abs=pd.to_numeric(df[rank_col], errors="coerce").abs())
            df = df.sort_values("_abs", ascending=False)
        for cluster, g in df.groupby(cluster_col):
            out[str(cluster)] = set(g[gene_col].astype(str).head(top_n).tolist())
        return out

    cand_markers = _top_n(cand)
    refr_markers = _top_n(refr)
    shared = set(cand_markers) & set(refr_markers)
    if not shared:
        return {"error": "no overlapping cluster names after harmonisation",
                "n_candidate_clusters": len(cand_markers),
                "n_reference_clusters": len(refr_markers)}

    per_cluster_jaccard = {
        c: _jaccard(cand_markers[c], refr_markers[c]) for c in shared
    }
    n_recovered = sum(1 for j in per_cluster_jaccard.values() if j >= min_recov)
    return {
        "n_candidate_clusters": len(cand_markers),
        "n_reference_clusters": len(refr_markers),
        "n_shared_clusters":    len(shared),
        "n_recovered_clusters": n_recovered,
        "mean_jaccard":         float(np.mean(list(per_cluster_jaccard.values()))),
        "median_jaccard":       float(np.median(list(per_cluster_jaccard.values()))),
        "per_cluster_jaccard":  json.dumps(per_cluster_jaccard),
        "top_n_used":           top_n,
    }


# ── Taxonomic profile (Benchmark F case 10 — metagenomics) ───────

def compare_taxonomic_profile(candidate: Path, reference: Path,
                              params: Dict[str, Any]) -> Dict[str, Any]:
    """Relative-abundance vector comparison at a fixed taxonomic rank.

    Both files have ``(taxon, rel_abundance)`` columns. We compute
    Spearman ρ on the log10 abundance vector across the union of taxa
    (zero-fill missing taxa with a small pseudocount), plus
    Bray-Curtis dissimilarity (0 = identical, 1 = totally disjoint).
    """
    taxon_col = params.get("taxon_column",     "taxon")
    abund_col = params.get("abundance_column", "rel_abundance")
    min_n     = int(params.get("min_intersection", 5))
    pseudo    = 1e-6

    cand = _read_table(candidate)
    refr = _read_table(reference)
    for need in (taxon_col, abund_col):
        if need not in cand.columns:
            return {"error": f"candidate missing column: {need}"}
        if need not in refr.columns:
            return {"error": f"reference missing column: {need}"}

    cand_v = cand.set_index(taxon_col)[abund_col].astype(float)
    refr_v = refr.set_index(taxon_col)[abund_col].astype(float)
    union  = cand_v.index.union(refr_v.index)
    cand_v = cand_v.reindex(union, fill_value=0.0)
    refr_v = refr_v.reindex(union, fill_value=0.0)
    intersection = (cand.index.intersection(refr.index)
                    if hasattr(cand, "index") else set())
    n_intersection = (cand_v > 0).astype(int).add((refr_v > 0).astype(int)).eq(2).sum()

    out: Dict[str, Any] = {
        "n_candidate_taxa": int((cand_v > 0).sum()),
        "n_reference_taxa": int((refr_v > 0).sum()),
        "n_intersection":   int(n_intersection),
        "n_union":          int(len(union)),
    }
    if n_intersection < min_n:
        out["error"] = f"intersection too small: {n_intersection} < {min_n}"
        out["spearman_log_abundance"] = float("nan")
        out["bray_curtis"] = float("nan")
        return out

    rho, p = stats.spearmanr(np.log10(cand_v + pseudo),
                              np.log10(refr_v + pseudo))
    out["spearman_log_abundance"] = float(rho)
    out["spearman_p"]             = float(p)
    # Bray-Curtis on raw relative abundances (sum to 1)
    cand_norm = cand_v / cand_v.sum() if cand_v.sum() > 0 else cand_v
    refr_norm = refr_v / refr_v.sum() if refr_v.sum() > 0 else refr_v
    num = np.abs(cand_norm - refr_norm).sum()
    den = (cand_norm + refr_norm).sum()
    out["bray_curtis"] = float(num / den) if den > 0 else float("nan")
    return out


# ── Plan-constraint check (Benchmark F case 11 — adversarial) ────

def compare_plan_constraint(candidate: Path, reference: Path,
                            params: Dict[str, Any]) -> Dict[str, Any]:
    """Inspect a candidate ``workflow.json`` for forbidden / required tools.

    The "candidate" is FlowAgent's emitted workflow.json. We scan all
    step commands + names for the tools listed in ``forbidden_in_plan``
    (must NOT appear) and ``required_anywhere`` (at least one MUST
    appear). Score: 1.0 if both conditions met, 0.5 if one met,
    0.0 if neither. The reference path here is a stub — used only to
    confirm the constraint definition file exists; the real check is
    against ``params``.
    """
    forbidden = {t.lower() for t in params.get("forbidden_in_plan") or []}
    required  = {t.lower() for t in params.get("required_anywhere") or []}
    min_score = float(params.get("min_correction_score", 1.0))

    if not candidate.exists():
        return {"error": f"workflow.json not found: {candidate}"}
    try:
        wf = json.loads(candidate.read_text())
    except json.JSONDecodeError as exc:
        return {"error": f"workflow.json parse failed: {exc}"}
    steps = wf.get("steps", [])

    haystack = " ".join(
        f"{s.get('name','')} {s.get('command','')}" for s in steps
    ).lower()
    forbidden_hits = {t for t in forbidden if t in haystack}
    required_hits  = {t for t in required  if t in haystack}

    forbidden_ok = not forbidden_hits
    required_ok  = bool(required_hits) if required else True
    score = (0.5 * int(forbidden_ok)) + (0.5 * int(required_ok))

    return {
        "n_steps":            len(steps),
        "forbidden_hits":     ",".join(sorted(forbidden_hits)) or "",
        "required_hits":      ",".join(sorted(required_hits)) or "",
        "forbidden_ok":       bool(forbidden_ok),
        "required_ok":        bool(required_ok),
        "constraint_score":   float(score),
        "passed":             bool(score >= min_score),
    }


COMPARATORS = {
    "de_table":           compare_de_table,
    "peak_bed":           compare_peak_bed,
    "vcf":                compare_vcf,
    "methylation_table":  compare_methylation_table,
    "cluster_markers":    compare_cluster_markers,
    "taxonomic_profile":  compare_taxonomic_profile,
    "plan_constraint":    compare_plan_constraint,
}
