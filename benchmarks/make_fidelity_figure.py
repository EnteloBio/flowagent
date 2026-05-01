"""Concordance figure for Benchmark F (output fidelity).

Headline figure for the paper: one hex-bin scatter per RNA-seq DE case
showing FlowAgent's log2FoldChange against the published reference's,
annotated with Spearman rho, Jaccard@200, and the gene-overlap count.

Reads candidate + reference tables on demand from
``--bulk-dir/<case>__<model>__rep<N>/<output_relpath>`` and
``--reference-base/<reference>`` (i.e. the same paths
``bench_fidelity.py`` uses), so the figure script is self-contained
and does not require a prior ``metrics.csv``.

Usage::

    python make_fidelity_figure.py \\
        --cases   config/fidelity_cases.yaml \\
        --bulk-dir results/fidelity_runs \\
        --reference-base . \\
        --model gpt-4.1 --replicate 0 \\
        --out figure_fidelity_concordance

Outputs ``<out>.pdf`` + ``.png``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from harness.fidelity_metrics import compare_de_table  # noqa: E402
from harness.runner import load_yaml  # noqa: E402


# Display labels per case-id. Falls back to the raw case-id if a
# mapping is missing — keeps the figure usable when new cases are added.
_TITLE_MAP = {
    "gse52778_dex_de":          "GSE52778 — Dex vs untreated",
    "gse60450_mammary_de":      "GSE60450 — basal vs luminal",
    "gse152418_covid_blood_de": "GSE152418 — COVID vs healthy",
}
_SUBTITLE_MAP = {
    "gse52778_dex_de":          "Himes 2014 · DESeq2",
    "gse60450_mammary_de":      "Fu 2015 · edgeR / limma-voom",
    "gse152418_covid_blood_de": "Arunachalam 2020 · DESeq2",
}

# Mirror the alias list inside ``compare_de_table`` so this script
# can resolve gene-ID columns without going through the comparator.
_GENE_ID_ALIASES = (
    "gene_id", "Gene", "gene", "feature_id",
    "ensembl_id", "ensembl_gene_id", "GeneID",
    "Unnamed: 0", "",
)


def _load_de(path: Path, gid: str, lfc: str, padj: str) -> pd.DataFrame:
    sep = "\t" if str(path).endswith((".tsv", ".tsv.gz", ".txt")) else ","
    df = pd.read_csv(path, sep=sep)
    if gid not in df.columns:
        for alias in _GENE_ID_ALIASES:
            if alias in df.columns:
                df = df.rename(columns={alias: gid})
                break
    if gid not in df.columns:
        raise SystemExit(f"{path}: no recognised gene-ID column "
                         f"(have: {list(df.columns)[:8]})")
    if lfc not in df.columns:
        raise SystemExit(f"{path}: missing log2FC column {lfc!r}")
    cols = [gid, lfc] + ([padj] if padj in df.columns else [])
    df = df[cols].copy()
    df[gid] = df[gid].astype(str).str.replace(r"\.\d+$", "", regex=True)
    df[lfc] = pd.to_numeric(df[lfc], errors="coerce")
    if padj in df.columns:
        df[padj] = pd.to_numeric(df[padj], errors="coerce")
    return df


def _panel(ax: plt.Axes, merged: pd.DataFrame, lfc: str, padj: str,
           alpha: float, min_lfc: float,
           title: str, subtitle: str, metrics: Dict[str, Any]) -> None:
    cand = merged[f"{lfc}_cand"].to_numpy()
    refv = merged[f"{lfc}_ref"].to_numpy()

    # Symmetric square axes on the 99.5th percentile of |log2FC|, with
    # a floor of 4 so even tightly-distributed cases get a sensible frame.
    lim = float(np.nanpercentile(np.abs(np.concatenate([cand, refv])), 99.5))
    lim = max(lim, 4.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # Reference grid + identity diagonal under the data
    ax.axhline(0, color="#cccccc", lw=0.8, zorder=0)
    ax.axvline(0, color="#cccccc", lw=0.8, zorder=0)
    ax.plot([-lim, lim], [-lim, lim],
            color="#666666", lw=1.0, linestyle="--", zorder=1,
            label="y = x")

    # Density (log-binned) over the full overlap.
    ax.hexbin(refv, cand, gridsize=55, mincnt=1, cmap="Blues",
              bins="log", linewidths=0.0, zorder=2)

    # Top-N overlay: genes that drive the Jaccard@200. Defined as
    # (padj < alpha) & (|log2FC| > min_lfc) on the reference side,
    # truncated to the same top_n the comparator uses.
    if f"{padj}_ref" in merged.columns:
        sig = merged[
            (merged[f"{padj}_ref"] < alpha) &
            (merged[f"{lfc}_ref"].abs() > min_lfc)
        ].copy()
        sig = sig.assign(_abs=sig[f"{lfc}_ref"].abs()) \
                 .sort_values("_abs", ascending=False) \
                 .head(int(metrics.get("top_n_used") or 200))
        if len(sig):
            ax.scatter(sig[f"{lfc}_ref"], sig[f"{lfc}_cand"],
                       s=8, c="#d04040", alpha=0.55,
                       edgecolors="none", zorder=3,
                       label=f"top-{len(sig)} reference DEGs")

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.18, linewidth=0.5)
    ax.set_xlabel("Reference log₂ FC", fontsize=10)
    ax.set_ylabel("FlowAgent log₂ FC", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=18)
    if subtitle:
        ax.text(0.5, 1.005, subtitle, ha="center", va="bottom",
                fontsize=8.5, color="#555", transform=ax.transAxes,
                fontstyle="italic")

    # Stats annotation in the top-left corner.
    bits = []
    rho = metrics.get("spearman_lfc")
    if rho is not None and not (isinstance(rho, float) and np.isnan(rho)):
        bits.append(f"ρ = {rho:.3f}")
    jacc = metrics.get("jaccard_top_n")
    if jacc is not None and not (isinstance(jacc, float) and np.isnan(jacc)):
        bits.append(f"J@200 = {jacc:.3f}")
    n_overlap = metrics.get("n_overlap")
    if n_overlap is not None:
        bits.append(f"n = {int(n_overlap):,}")
    if bits:
        ax.text(
            0.04, 0.965, "\n".join(bits), transform=ax.transAxes,
            ha="left", va="top", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#fffae0", edgecolor="#c2a047", lw=0.8),
            zorder=4,
        )

    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right", fontsize=7.8, frameon=False)


def _render_missing(ax: plt.Axes, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            fontsize=9, color="#a00000", transform=ax.transAxes,
            wrap=True)
    ax.axis("off")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--cases", default="config/fidelity_cases.yaml")
    ap.add_argument("--bulk-dir", default="results/fidelity_runs")
    ap.add_argument("--reference-base", default=".")
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--replicate", type=int, default=0)
    ap.add_argument("--out", default="figure_fidelity_concordance")
    ap.add_argument("--title",
                    default="FlowAgent autonomously reproduces published "
                            "RNA-seq differential expression")
    args = ap.parse_args()

    cases_path = (Path(args.cases) if Path(args.cases).is_absolute()
                  else _HERE / args.cases)
    cfg = load_yaml(cases_path)
    cases = [c for c in cfg.get("cases", [])
             if c.get("comparison") == "de_table"]
    if not cases:
        sys.exit(f"no de_table cases in {cases_path}")

    n = len(cases)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.4))
    if n == 1:
        axes = [axes]

    for ax, case in zip(axes, cases):
        slug = f"{case['id']}__{args.model}__rep{args.replicate}"
        cand_path = Path(args.bulk_dir) / slug / case["output_relpath"]
        ref_path  = Path(args.reference_base) / case["reference"]
        if not cand_path.exists():
            _render_missing(ax, f"candidate not found:\n{cand_path}")
            continue
        if not ref_path.exists():
            _render_missing(ax, f"reference not found:\n{ref_path}")
            continue

        params = case.get("params", {}) or {}
        gid     = params.get("gene_id_column", "gene_id")
        lfc     = params.get("log2fc_column",  "log2FoldChange")
        padj    = params.get("padj_column",    "padj")
        alpha   = float(params.get("alpha", 0.05))
        min_lfc = float(params.get("min_lfc", 1.0))

        try:
            cand = _load_de(cand_path, gid, lfc, padj)
            refr = _load_de(ref_path, gid, lfc, padj)
            merged = cand.merge(refr, on=gid, suffixes=("_cand", "_ref"))
            merged = merged.dropna(subset=[f"{lfc}_cand", f"{lfc}_ref"])
            metrics = compare_de_table(cand_path, ref_path, params)
        except SystemExit as exc:
            _render_missing(ax, str(exc))
            continue

        if "error" in metrics or len(merged) < 10:
            _render_missing(ax, metrics.get("error",
                            f"insufficient overlap: n={len(merged)}"))
            continue

        _panel(
            ax, merged, lfc, padj, alpha, min_lfc,
            _TITLE_MAP.get(case["id"], case["id"]),
            _SUBTITLE_MAP.get(case["id"], ""),
            metrics,
        )

    fig.suptitle(args.title, fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = Path(args.out)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {out}.pdf")
    print(f"wrote {out}.png")


if __name__ == "__main__":
    main()
