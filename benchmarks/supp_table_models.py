"""Supplementary Table 2 — model registry × empirical token / cost data.

Joins the static registry in ``config/models.yaml`` with per-cell metrics
from the latest merged planning run, producing a single TSV that gives
every model's provider, tier, context window, unit prices, and the
mean per-plan token / call / cost / pass-rate observed empirically.

Usage:
    python supp_table_models.py
    python supp_table_models.py --run results/planning/_merged/<TS>
    python supp_table_models.py --out results/figures/supp_table2_models.tsv

Writes both a TSV (machine-readable) and a Markdown rendering for the
manuscript supplement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

_HERE = Path(__file__).parent


def _latest_merged(base: Path) -> Optional[Path]:
    root = base / "planning" / "_merged"
    if not root.exists():
        return None
    runs = sorted([p for p in root.iterdir() if p.is_dir()],
                  key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


# Historical model IDs that appeared in earlier sweeps. The Gemini 1.5
# family was removed from the active registry but is still callable on
# Google's v1 API; the two renamed-preview entries reflect Google
# attaching an explicit ``-preview`` suffix mid-2026 to all 3.x models.
_LEGACY_REGISTRY = [
    {
        "model": "gemini-1.5-flash", "provider": "google",
        "family": "gemini-1.5", "tier": "legacy", "context_k": 1000,
        "input_per_1k": 0.000075, "output_per_1k": 0.000300,
    },
    {
        "model": "gemini-1.5-pro", "provider": "google",
        "family": "gemini-1.5", "tier": "legacy", "context_k": 2000,
        "input_per_1k": 0.00125, "output_per_1k": 0.00500,
    },
]
# Old ID → current canonical ID. Pricing/family/etc. are taken from the
# canonical entry in models.yaml; only the displayed model column keeps
# the historical name so it lines up with the metrics CSV.
_ID_ALIASES = {
    "gemini-3-flash":    "gemini-3-flash-preview",
    "gemini-3.1-pro":    "gemini-3.1-pro-preview",
}


def _load_registry(yaml_path: Path) -> pd.DataFrame:
    cfg = yaml.safe_load(yaml_path.read_text())
    rows = []
    for m in cfg.get("models", []):
        rows.append({
            "model":           m["id"],
            "provider":        m["provider"],
            "family":          m.get("family", ""),
            "tier":            m.get("tier", ""),
            "context_k":       m.get("context_k", float("nan")),
            "input_per_1k":    m.get("pricing", {}).get("input_per_1k", float("nan")),
            "output_per_1k":   m.get("pricing", {}).get("output_per_1k", float("nan")),
        })
    rows.extend(_LEGACY_REGISTRY)

    # Inject alias rows: copy canonical entries under the old ID
    by_id = {r["model"]: r for r in rows}
    for old_id, canon_id in _ID_ALIASES.items():
        if canon_id in by_id and old_id not in by_id:
            alias = dict(by_id[canon_id])
            alias["model"] = old_id
            alias["tier"]  = "legacy"  # historical alias, mark distinctly
            rows.append(alias)
    return pd.DataFrame(rows)


def _empirical(metrics_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    for c in ("prompt_tokens", "completion_tokens", "cost_usd",
              "llm_calls", "wall_seconds"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["overall_pass"] = df["overall_pass"].map(
        lambda v: v if isinstance(v, (bool, int, float))
        else str(v).strip().lower() == "true"
    ).astype(int)
    g = (df.groupby("model")
           .agg(n_cells=("overall_pass", "count"),
                pass_rate=("overall_pass", "mean"),
                mean_in=("prompt_tokens", "mean"),
                mean_out=("completion_tokens", "mean"),
                mean_calls=("llm_calls", "mean")
                    if "llm_calls" in df.columns else ("overall_pass", "size"),
                mean_cost=("cost_usd", "mean")
                    if "cost_usd" in df.columns else ("overall_pass", "size"),
                median_wall=("wall_seconds", "median")
                    if "wall_seconds" in df.columns else ("overall_pass", "size"))
           .reset_index())
    return g


_PROVIDER_ORDER  = {"openai": 0, "anthropic": 1, "google": 2, "ollama": 3}
_TIER_ORDER      = {"current": 0, "preview": 1, "legacy": 2, "deprecated": 3, "": 4}


def build_table(yaml_path: Path, metrics_csv: Path) -> pd.DataFrame:
    reg = _load_registry(yaml_path)
    obs = _empirical(metrics_csv)
    tab = reg.merge(obs, on="model", how="outer", indicator=True)

    # Annotate models that are in the registry but never ran in this sweep
    # (preview models added after the last sweep, etc.) and vice versa.
    tab["in_registry"] = tab["_merge"].isin(["both", "left_only"])
    tab["in_sweep"]    = tab["_merge"].isin(["both", "right_only"])
    tab.drop(columns="_merge", inplace=True)

    tab["_p_ord"] = tab["provider"].map(_PROVIDER_ORDER).fillna(99)
    tab["_t_ord"] = tab["tier"].map(_TIER_ORDER).fillna(99)
    tab = tab.sort_values(["_p_ord", "_t_ord", "model"]).drop(
        columns=["_p_ord", "_t_ord"]).reset_index(drop=True)

    tab["pass_rate"]    = tab["pass_rate"].astype(float).round(3)
    tab["mean_in"]      = tab["mean_in"].round(0)
    tab["mean_out"]     = tab["mean_out"].round(0)
    tab["mean_calls"]   = tab["mean_calls"].round(2)
    tab["mean_cost"]    = tab["mean_cost"].round(5)
    tab["median_wall"]  = tab["median_wall"].round(1)

    cols = [
        "model", "provider", "family", "tier", "context_k",
        "input_per_1k", "output_per_1k",
        "n_cells", "pass_rate",
        "mean_in", "mean_out", "mean_calls",
        "median_wall", "mean_cost",
        "in_registry", "in_sweep",
    ]
    return tab[cols]


def render_markdown(tab: pd.DataFrame) -> str:
    """Render Supplementary Table 2 as Markdown for the supplement."""
    headers = [
        "Model ID", "Provider", "Family", "Tier", "Ctx (k)",
        "Input $/1k", "Output $/1k",
        "N", "Pass rate",
        "Mean in", "Mean out", "Mean calls",
        "Median wall (s)", "Mean cost ($)",
    ]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    sub = tab[tab["in_sweep"]].copy()

    def _s(v):
        if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
            return "—"
        return str(v)

    for _, r in sub.iterrows():
        lines.append("| " + " | ".join([
            f"`{r['model']}`",
            _s(r["provider"]),
            _s(r["family"]),
            _s(r["tier"]),
            "—" if pd.isna(r["context_k"]) else f"{int(r['context_k'])}",
            "—" if pd.isna(r["input_per_1k"]) else f"{r['input_per_1k']:.5f}",
            "—" if pd.isna(r["output_per_1k"]) else f"{r['output_per_1k']:.5f}",
            "—" if pd.isna(r["n_cells"]) else f"{int(r['n_cells'])}",
            "—" if pd.isna(r["pass_rate"]) else f"{r['pass_rate']:.1%}",
            "—" if pd.isna(r["mean_in"]) else f"{int(r['mean_in']):,}",
            "—" if pd.isna(r["mean_out"]) else f"{int(r['mean_out']):,}",
            "—" if pd.isna(r["mean_calls"]) else f"{r['mean_calls']:.1f}",
            "—" if pd.isna(r["median_wall"]) else f"{r['median_wall']:.1f}",
            "—" if pd.isna(r["mean_cost"]) else f"${r['mean_cost']:.5f}",
        ]) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-base", default="results")
    ap.add_argument("--run", default=None,
                    help="Specific merged run dir (default: latest)")
    ap.add_argument("--models-yaml", default="config/models.yaml")
    ap.add_argument("--out", default="results/figures/supp_table2_models.tsv")
    args = ap.parse_args()

    base = _HERE / args.results_base if not Path(args.results_base).is_absolute() \
        else Path(args.results_base)

    if args.run:
        run_dir = Path(args.run)
    else:
        run_dir = _latest_merged(base)
        if run_dir is None:
            sys.exit(f"No merged planning run found under {base}/planning/_merged/")
    metrics_csv = run_dir / "metrics.csv"
    if not metrics_csv.exists():
        sys.exit(f"metrics.csv not found in {run_dir}")

    yaml_path = _HERE / args.models_yaml if not Path(args.models_yaml).is_absolute() \
        else Path(args.models_yaml)

    tab = build_table(yaml_path, metrics_csv)

    out_tsv = _HERE / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out_tsv, sep="\t", index=False)
    out_md = out_tsv.with_suffix(".md")
    out_md.write_text(render_markdown(tab))

    n_sweep = int(tab["in_sweep"].sum())
    n_reg   = int(tab["in_registry"].sum())
    print(f"[ok] supplementary table 2 → {out_tsv}")
    print(f"     {n_sweep} models in sweep, {n_reg} in registry")
    print(f"     source run: {run_dir}")
    print(f"[ok] markdown rendering → {out_md}")


if __name__ == "__main__":
    main()
