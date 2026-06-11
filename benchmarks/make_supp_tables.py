"""Generate manuscript supplementary tables 1 (prompt corpus) and 2 (models).

Supplementary Table 1 — the Benchmark A *explicit-tool* prompt corpus
(``tier: transcription``): one row per prompt with its difficulty tier
(standard vs hard), biological domain, scored workflow type, the rubric
tools, the minimum-step floor, and the forbidden-tool list.

Supplementary Table 2 — the provider-model registry actually evaluated in
the planning sweep, joined to the empirical per-plan token / call / cost /
pass-rate observed on the explicit-tool tier of the latest merged run.

Both tables are emitted as Markdown (manuscript paste) and TSV
(machine-readable) under ``results/figures/``.

Usage::

    python make_supp_tables.py
    python make_supp_tables.py --run results/planning/_merged/<TS>
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_HERE = Path(__file__).parent

# ── Domain labels for the explicit-tool corpus (Table 1) ──────────────
# Keyed by prompt id; keeps the supplementary table readable without
# re-deriving the domain from tool lists.
_DOMAIN: Dict[str, str] = {
    # Standard tier
    "rnaseq_kallisto_basic": "RNA-seq",
    "rnaseq_kallisto_explicit_ref": "RNA-seq",
    "rnaseq_star_featurecounts": "RNA-seq",
    "rnaseq_salmon": "RNA-seq",
    "rnaseq_hisat2_htseq": "RNA-seq",
    "rnaseq_with_trim": "RNA-seq",
    "rnaseq_mouse": "RNA-seq",
    "rnaseq_geo": "RNA-seq",
    "chipseq_macs2": "ChIP-seq",
    "chipseq_single_end": "ChIP-seq",
    "chipseq_with_qc": "ChIP-seq",
    "atacseq_basic": "ATAC-seq",
    "atacseq_paired": "ATAC-seq",
    "variant_bwa_gatk": "Variant calling",
    "variant_bwa_bcftools": "Variant calling",
    "variant_whole_genome": "Variant calling",
    "scrna_kb": "scRNA-seq",
    "scrna_starsolo": "scRNA-seq",
    "ref_download_qc": "Reference / QC",
    "fastqc_only": "QC",
    "adv_single_end_kallisto": "RNA-seq (adversarial)",
    "adv_cross_species": "RNA-seq (adversarial)",
    "adv_multi_sample": "RNA-seq (adversarial)",
    # Hard tier
    "hard_full_germline_pipeline": "Variant calling",
    "hard_methylation_bismark_wgbs": "Methylation (WGBS)",
    "hard_longread_nanopore_variants": "Variant calling (long-read)",
    "hard_strand_specific_rnaseq": "RNA-seq",
    "hard_sv_calling_manta": "Structural variants",
    "hard_cutandrun_full": "CUT&RUN",
    "hard_rnaseq_full_de_pipeline": "RNA-seq",
    "hard_metagenomics_kraken": "Metagenomics",
    "hard_amplicon_dada2": "Amplicon (16S)",
    "hard_atac_full_nucleosome": "ATAC-seq",
    "hard_rnaseq_salmon_full": "RNA-seq",
    "hard_somatic_mutect2": "Variant calling (somatic)",
    "hard_longread_pacbio_assembly": "Assembly (long-read)",
    "hard_chip_diffbind": "ChIP-seq",
    "hard_smallrna_mirdeep2": "Small-RNA / miRNA",
    "hard_hic_cooler": "Hi-C",
    "hard_shortread_assembly_spades": "Assembly (short-read)",
    "hard_cfdna_lowpass_cnv": "CNV / cfDNA",
}

_PROVIDER_BY_PREFIX = [
    ("gpt", "openai"), ("o1", "openai"), ("o3", "openai"), ("o4", "openai"),
    ("claude", "anthropic"),
    ("gemini", "google"),
]


def _provider_for(model: str) -> str:
    for pref, prov in _PROVIDER_BY_PREFIX:
        if model.startswith(pref):
            return prov
    return "other"


def _as_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    return [str(v)]


def _primary_type(v: Any) -> str:
    lst = _as_list(v)
    return lst[0] if lst else ""


def _clean_prompt(p: str) -> str:
    return " ".join(str(p).split())


# ── Table 1: prompt corpus ────────────────────────────────────────────

def build_table1(prompts_yaml: Path) -> List[Dict[str, Any]]:
    cfg = yaml.safe_load(prompts_yaml.read_text())
    out: List[Dict[str, Any]] = []
    for p in cfg.get("prompts", []):
        # Only the explicit-tool (transcription) tier belongs in Table 1.
        if p.get("tier", "transcription") == "inference":
            continue
        pid = p["id"]
        tier = "Hard" if pid.startswith("hard_") else "Standard"
        out.append({
            "id": pid,
            "tier": tier,
            "domain": _DOMAIN.get(pid, ""),
            "workflow_type": _primary_type(p.get("expected_workflow_type")),
            "expected_tools": ", ".join(_as_list(p.get("expected_tools"))),
            "min_steps": p.get("expected_min_steps", ""),
            "forbidden_tools": ", ".join(_as_list(p.get("forbidden_tools"))),
            "prompt": _clean_prompt(p.get("prompt", "")),
        })
    # Standard first, then Hard, preserving corpus order within each tier.
    out.sort(key=lambda r: 0 if r["tier"] == "Standard" else 1)
    return out


def render_table1_md(rows: List[Dict[str, Any]]) -> str:
    n_std = sum(r["tier"] == "Standard" for r in rows)
    n_hard = sum(r["tier"] == "Hard" for r in rows)
    head = (
        f"**Supplementary Table 1. Benchmark A prompt corpus "
        f"({len(rows)} explicit-tool prompts: {n_std} standard-tier, "
        f"{n_hard} hard-tier).** Each prompt is scored against a rubric of "
        f"the workflow type, the required (expected) tools, a minimum step "
        f"count, and a forbidden-tool list.\n"
    )
    cols = ["#", "Prompt ID", "Tier", "Domain", "Workflow type",
            "Expected tools", "Min steps", "Forbidden tools"]
    lines = [head, "| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for i, r in enumerate(rows, 1):
        lines.append("| " + " | ".join([
            str(i), f"`{r['id']}`", r["tier"], r["domain"],
            f"`{r['workflow_type']}`" if r["workflow_type"] else "—",
            r["expected_tools"] or "—", str(r["min_steps"]),
            r["forbidden_tools"] or "—",
        ]) + " |")
    return "\n".join(lines)


# ── Table 2: model registry × empirical ───────────────────────────────

def _latest_merged(base: Path) -> Optional[Path]:
    root = base / "planning" / "_merged"
    if not root.exists():
        return None
    runs = sorted([p for p in root.iterdir() if p.is_dir()],
                  key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def _f(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def build_table2(models_yaml: Path, metrics_csv: Path,
                 tier: str = "transcription") -> List[Dict[str, Any]]:
    reg = {m["id"]: m for m in yaml.safe_load(models_yaml.read_text())["models"]}

    by_model: Dict[str, Dict[str, List[float]]] = {}
    n_pass: Dict[str, List[int]] = {}
    with metrics_csv.open() as f:
        for r in csv.DictReader(f):
            if tier and r.get("tier") != tier:
                continue
            m = r.get("model")
            if not m:
                continue
            d = by_model.setdefault(m, {k: [] for k in
                ("prompt_tokens", "completion_tokens", "cost_usd",
                 "llm_calls", "wall_seconds")})
            for k in d:
                val = _f(r.get(k))
                if val is not None:
                    d[k].append(val)
            op = str(r.get("overall_pass", "")).strip().lower()
            if op in ("true", "false", "1", "0"):
                n_pass.setdefault(m, []).append(1 if op in ("true", "1") else 0)

    def _mean(xs: List[float]) -> Optional[float]:
        return statistics.mean(xs) if xs else None

    def _median(xs: List[float]) -> Optional[float]:
        return statistics.median(xs) if xs else None

    rows: List[Dict[str, Any]] = []
    for m in sorted(by_model):
        meta = reg.get(m, {})
        passes = n_pass.get(m, [])
        d = by_model[m]
        rows.append({
            "model": m,
            "provider": meta.get("provider") or _provider_for(m),
            "family": meta.get("family", ""),
            "tier": meta.get("tier", ""),
            "context_k": meta.get("context_k", ""),
            "reasoning": meta.get("reasoning", ""),
            "input_per_1k": (meta.get("pricing") or {}).get("input_per_1k", ""),
            "output_per_1k": (meta.get("pricing") or {}).get("output_per_1k", ""),
            "n_cells": len(passes),
            "pass_rate": (statistics.mean(passes) if passes else None),
            "mean_in": _mean(d["prompt_tokens"]),
            "mean_out": _mean(d["completion_tokens"]),
            "mean_calls": _mean(d["llm_calls"]),
            "median_wall": _median(d["wall_seconds"]),
            "mean_cost": _mean(d["cost_usd"]),
        })
    order = {"openai": 0, "anthropic": 1, "google": 2}
    rows.sort(key=lambda r: (order.get(r["provider"], 9), r["model"]))
    return rows


def render_table2_md(rows: List[Dict[str, Any]]) -> str:
    head = (
        f"**Supplementary Table 2. Provider-model combinations evaluated "
        f"in the planning sweep ({len(rows)} models).** Registry metadata "
        f"(provider, family, support tier, context window, unit pricing) is "
        f"joined to the empirical per-plan tokens, LLM calls, wall time, "
        f"cost, and overall pass rate observed on the explicit-tool tier of "
        f"the merged sweep.\n"
    )
    cols = ["Model ID", "Provider", "Family", "Tier", "Ctx (k)", "Reasoning",
            "In $/1k", "Out $/1k", "N", "Pass", "Mean in", "Mean out",
            "Mean calls", "Median wall (s)", "Mean cost ($)"]
    lines = [head, "| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]

    def s(v, fmt=None):
        if v is None or v == "":
            return "—"
        if fmt:
            return fmt(v)
        return str(v)

    for r in rows:
        lines.append("| " + " | ".join([
            f"`{r['model']}`",
            s(r["provider"]), s(r["family"]), s(r["tier"]),
            s(r["context_k"], lambda v: f"{int(v)}"),
            "yes" if r["reasoning"] is True else ("no" if r["reasoning"] is False else "—"),
            s(r["input_per_1k"], lambda v: f"{float(v):.5f}"),
            s(r["output_per_1k"], lambda v: f"{float(v):.5f}"),
            s(r["n_cells"]),
            s(r["pass_rate"], lambda v: f"{v:.1%}"),
            s(r["mean_in"], lambda v: f"{int(round(v)):,}"),
            s(r["mean_out"], lambda v: f"{int(round(v)):,}"),
            s(r["mean_calls"], lambda v: f"{v:.1f}"),
            s(r["median_wall"], lambda v: f"{v:.1f}"),
            s(r["mean_cost"], lambda v: f"{v:.5f}"),
        ]) + " |")
    return "\n".join(lines)


def _write_tsv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-base", default="results")
    ap.add_argument("--run", default=None,
                    help="Merged planning run dir (default: latest)")
    ap.add_argument("--prompts", default="corpus/prompts.yaml")
    ap.add_argument("--models-yaml", default="config/models.yaml")
    ap.add_argument("--out-dir", default="results/figures")
    ap.add_argument("--tier", default="transcription",
                    help="Scoring tier for Table 2 empirical join")
    args = ap.parse_args()

    base = (_HERE / args.results_base) if not Path(args.results_base).is_absolute() \
        else Path(args.results_base)
    out_dir = (_HERE / args.out_dir) if not Path(args.out_dir).is_absolute() \
        else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Table 1 ──
    t1 = build_table1(_HERE / args.prompts)
    (out_dir / "supp_table1_prompts.md").write_text(render_table1_md(t1) + "\n")
    _write_tsv(out_dir / "supp_table1_prompts.tsv", t1)
    n_std = sum(r["tier"] == "Standard" for r in t1)
    n_hard = sum(r["tier"] == "Hard" for r in t1)
    print(f"[ok] Table 1 → {out_dir/'supp_table1_prompts.md'} "
          f"({len(t1)} prompts: {n_std} standard, {n_hard} hard)")

    # ── Table 2 ──
    run_dir = Path(args.run) if args.run else _latest_merged(base)
    if run_dir is None or not (run_dir / "metrics.csv").exists():
        print("[warn] no merged planning run with metrics.csv; skipping Table 2")
        return
    t2 = build_table2(_HERE / args.models_yaml, run_dir / "metrics.csv",
                      tier=args.tier)
    (out_dir / "supp_table2_models.md").write_text(render_table2_md(t2) + "\n")
    _write_tsv(out_dir / "supp_table2_models.tsv", t2)
    n_plans = len(t1) * len(t2) * 3
    print(f"[ok] Table 2 → {out_dir/'supp_table2_models.md'} "
          f"({len(t2)} models; source {run_dir.name})")
    print(f"[info] corpus×models×3 = {len(t1)} × {len(t2)} × 3 = {n_plans:,} plans")


if __name__ == "__main__":
    main()
