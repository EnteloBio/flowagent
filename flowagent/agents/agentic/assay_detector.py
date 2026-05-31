"""Assay-agnostic output detector for FlowAgent reporting.

``detect_assay`` inspects a results directory and returns a dict
describing what kind of outputs are present, which is used by
``AgenticAnalysisSystem`` to select the appropriate report template.

Supported assay categories
---------------------------
- ``rna_seq``      — kallisto / STAR / featureCounts abundance files
- ``chip_atac``    — BED/narrowPeak/broadPeak files from MACS2 / HMMRATAC
- ``variant``      — VCF/BCF files (GATK, DeepVariant, freebayes …)
- ``generic``      — any results directory not matching the above

Each category also includes a ``metrics`` sub-dict with the key numbers
a user would care about:
- rna_seq:    n_samples, n_transcripts_mean, mapping_rate_mean
- chip_atac:  n_peak_files, n_peaks_total, frip_proxy (peak/total_reads if available)
- variant:    n_vcfs, n_variants_total, ti_tv (if computable)
- generic:    file_inventory (extension → count)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── File-extension signatures ─────────────────────────────────

_RNA_SEQ_GLOBS = ["abundance.tsv", "abundance.h5", "*.genes.results", "*.isoforms.results",
                  "counts.txt", "featurecounts.txt", "*ReadsPerGene.out.tab"]
_PEAK_EXTENSIONS = {".narrowPeak", ".broadPeak", ".bed", ".bdg"}
_VCF_EXTENSIONS = {".vcf", ".vcf.gz", ".bcf"}


def detect_assay(results_dir: Path) -> Dict[str, Any]:
    """Inspect *results_dir* and return an assay summary dict."""
    results_dir = Path(results_dir)
    info: Dict[str, Any] = {
        "assay": "generic",
        "results_dir": str(results_dir),
        "metrics": {},
        "files": {},
    }

    # Inventory all files
    all_files = list(results_dir.rglob("*"))
    ext_counts: Dict[str, int] = {}
    for f in all_files:
        if f.is_file():
            ext = "".join(f.suffixes).lower() or f.suffix.lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
    info["files"] = ext_counts

    # ── RNA-seq ───────────────────────────────────────────────
    abundance_files = list(results_dir.rglob("abundance.tsv")) + list(results_dir.rglob("abundance.h5"))
    counts_files = (
        list(results_dir.rglob("counts.txt"))
        + list(results_dir.rglob("featurecounts.txt"))
        + list(results_dir.rglob("*ReadsPerGene.out.tab"))
        + list(results_dir.rglob("*.genes.results"))
    )
    if abundance_files or counts_files:
        info["assay"] = "rna_seq"
        info["metrics"] = _rna_seq_metrics(abundance_files, counts_files)
        return info

    # ── ChIP-seq / ATAC-seq (peaks) ───────────────────────────
    peak_files = [
        f for f in results_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in {".narrowpeak", ".broadpeak"}
        or (f.suffix.lower() == ".bed" and "peak" in f.stem.lower())
    ]
    # Also accept any .bed in a peaks/ subdirectory
    for bed_file in results_dir.rglob("*.bed"):
        if "peak" in str(bed_file).lower() and bed_file not in peak_files:
            peak_files.append(bed_file)
    if peak_files:
        info["assay"] = "chip_atac"
        info["metrics"] = _peak_metrics(peak_files, results_dir)
        return info

    # ── Variant calling ───────────────────────────────────────
    vcf_files = [
        f for f in results_dir.rglob("*")
        if f.is_file() and any(str(f).endswith(ext) for ext in (".vcf", ".vcf.gz", ".bcf"))
        and "truth" not in f.name.lower()  # exclude reference VCFs
    ]
    if vcf_files:
        info["assay"] = "variant"
        info["metrics"] = _vcf_metrics(vcf_files)
        return info

    # ── Generic fallback ──────────────────────────────────────
    info["metrics"] = {"file_inventory": ext_counts, "total_files": len(all_files)}
    return info


def assay_report_summary(info: Dict[str, Any]) -> str:
    """Return a short human-readable summary for inclusion in a report."""
    assay = info.get("assay", "generic")
    m = info.get("metrics", {})

    if assay == "rna_seq":
        n = m.get("n_samples", 0)
        mr = m.get("mapping_rate_mean_pct")
        mr_str = f", mean mapping rate {mr:.1f}%" if mr is not None else ""
        return f"RNA-seq: {n} sample(s){mr_str}"

    if assay == "chip_atac":
        n_files = m.get("n_peak_files", 0)
        n_peaks = m.get("n_peaks_total", 0)
        return f"ChIP/ATAC: {n_files} peak file(s), {n_peaks:,} peaks total"

    if assay == "variant":
        n_vcfs = m.get("n_vcfs", 0)
        n_var = m.get("n_variants_total", 0)
        titv = m.get("ti_tv")
        titv_str = f", Ti/Tv {titv:.2f}" if titv else ""
        return f"Variant calling: {n_vcfs} VCF(s), {n_var:,} variants{titv_str}"

    total = m.get("total_files", 0)
    inv = m.get("file_inventory", {})
    ext_str = ", ".join(f"{v}×{k}" for k, v in sorted(inv.items(), key=lambda x: -x[1])[:5])
    return f"Generic results: {total} files ({ext_str})"


# ── Assay-specific metrics helpers ────────────────────────────

def _rna_seq_metrics(abundance_files: List[Path], counts_files: List[Path]) -> Dict[str, Any]:
    n_samples = len(abundance_files) or len(counts_files)
    mapping_rates: List[float] = []
    n_transcripts: List[int] = []

    for af in abundance_files[:20]:  # cap at 20 to keep it fast
        try:
            run_info = af.parent / "run_info.json"
            if run_info.exists():
                with run_info.open() as f:
                    ri = json.load(f)
                    n_proc = ri.get("n_processed", 0)
                    n_pseudo = ri.get("n_pseudoaligned", 0)
                    if n_proc > 0:
                        mapping_rates.append(n_pseudo / n_proc * 100)
        except Exception:
            pass

        try:
            lines = af.read_text().splitlines()
            n_transcripts.append(max(0, len(lines) - 1))
        except Exception:
            pass

    return {
        "n_samples": n_samples,
        "n_transcripts_mean": int(sum(n_transcripts) / len(n_transcripts)) if n_transcripts else None,
        "mapping_rate_mean_pct": (sum(mapping_rates) / len(mapping_rates)) if mapping_rates else None,
    }


def _peak_metrics(peak_files: List[Path], results_dir: Path) -> Dict[str, Any]:
    n_peaks_total = 0
    for pf in peak_files:
        try:
            lines = [l for l in pf.read_text().splitlines() if l and not l.startswith("#")]
            n_peaks_total += len(lines)
        except Exception:
            pass

    # FRiP proxy: look for flagstat outputs
    frip: Optional[float] = None
    for flagstat in results_dir.rglob("*.flagstat"):
        try:
            text = flagstat.read_text()
            for line in text.splitlines():
                if "mapped" in line and "%" in line:
                    pct_str = line.split("(")[1].split(")")[0].strip().rstrip("%")
                    frip = float(pct_str)
                    break
        except Exception:
            pass
        if frip is not None:
            break

    return {
        "n_peak_files": len(peak_files),
        "n_peaks_total": n_peaks_total,
        "frip_proxy": frip,
    }


def _vcf_metrics(vcf_files: List[Path]) -> Dict[str, Any]:
    n_variants = 0
    transitions = 0
    transversions = 0

    _TRANSITIONS = {frozenset("AG"), frozenset("CT")}

    for vf in vcf_files[:5]:  # limit to avoid reading huge VCFs
        try:
            opener = None
            if str(vf).endswith(".gz"):
                import gzip
                opener = gzip.open(vf, "rt", errors="replace")
            else:
                opener = vf.open(errors="replace")
            with opener as fh:
                for line in fh:
                    if line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) < 5:
                        continue
                    ref, alt = parts[3].upper(), parts[4].upper().split(",")[0]
                    if len(ref) == 1 and len(alt) == 1:
                        n_variants += 1
                        pair = frozenset([ref, alt])
                        if pair in _TRANSITIONS:
                            transitions += 1
                        else:
                            transversions += 1
                    else:
                        n_variants += 1  # indel / MNP
        except Exception:
            pass

    ti_tv = (transitions / transversions) if transversions > 0 else None

    return {
        "n_vcfs": len(vcf_files),
        "n_variants_total": n_variants,
        "ti_tv": ti_tv,
        "n_transitions": transitions,
        "n_transversions": transversions,
    }
