"""Figure generation for benchmark results.

Produces publication-ready PDFs + PNGs from benchmark metrics. Each plotting
function consumes a pandas.DataFrame (loaded from ``metrics.csv``) and
returns a matplotlib ``Figure``.

Style goals:
  * Colour-blind-safe palette (Okabe-Ito for providers)
  * Consistent sans-serif typography (Arial / Helvetica / DejaVu Sans)
  * 300 DPI PNG + vector PDF for every figure
  * Tight, constrained layouts with no wasted whitespace
  * Deterministic model ordering (legacy → mid → frontier)

Usage:
    python -m harness.plot --results=results
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker as mtick
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ── Publication-ready defaults ───────────────────────────────────

def _set_publication_style() -> None:
    """Apply matplotlib rcParams tuned for journal-figure quality."""
    plt.rcParams.update({
        # Typography: prefer Arial / Helvetica (widely accepted by journals)
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":             9.0,
        "axes.titlesize":       10.5,
        "axes.titleweight":     "bold",
        "axes.labelsize":        9.0,
        "xtick.labelsize":       8.0,
        "ytick.labelsize":       8.0,
        "legend.fontsize":       8.0,
        "figure.titlesize":     11.0,
        "figure.titleweight":   "bold",

        # Axes & ticks
        "axes.linewidth":        0.8,
        "axes.edgecolor":       "#374151",
        "axes.labelcolor":      "#111827",
        "xtick.color":          "#374151",
        "ytick.color":          "#374151",
        "xtick.major.width":     0.8,
        "ytick.major.width":     0.8,
        "xtick.major.size":      3.0,
        "ytick.major.size":      3.0,
        "axes.spines.top":      False,
        "axes.spines.right":    False,

        # Grid: light, horizontal only by default
        "axes.grid":             False,
        "grid.color":           "#e5e7eb",
        "grid.linewidth":        0.6,
        "grid.alpha":            0.8,

        # Figure
        "figure.dpi":           150,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":    0.05,
        "pdf.fonttype":         42,     # TrueType (editable in Illustrator)
        "ps.fonttype":          42,

        # Layout
        "figure.constrained_layout.use":       True,
        "figure.constrained_layout.h_pad":     0.04,
        "figure.constrained_layout.w_pad":     0.04,

        # Colours
        "patch.edgecolor":      "white",
        "patch.linewidth":       0.4,
    })


_set_publication_style()


# ── Colour palettes ──────────────────────────────────────────────

# Okabe–Ito — an 8-colour palette designed for colour-blind readability
# (https://jfly.uni-koeln.de/color/). Used for provider grouping.
_PROVIDER_COLOURS = {
    "anthropic": "#E69F00",  # orange
    "openai":    "#0072B2",  # blue
    "google":    "#009E73",  # bluish-green
    "ollama":    "#CC79A7",  # reddish-purple
    "other":     "#6b7280",  # slate
}

# Tier ordering for x-axis layout. Keeps backwards compatibility with the
# old three-tier scheme (legacy/mid/frontier) while also accepting the
# current models.yaml vocabulary (legacy/preview/current).
_TIER_ORDER = {
    "legacy": 0,
    "mid": 1,
    "preview": 1,
    "frontier": 2,
    "current": 2,
}

# Taxonomy palette — Okabe-Ito-derived. Green = safe refusal with
# correct diagnosis; amber = refusal with wrong diagnosis (coincidentally
# correct); red = confidently unsafe fix (the dangerous category);
# blue = tried but failed (at least the pipeline surfaces the error);
# grey = no engagement (silent / max-retries).
_TAXONOMY_CATEGORIES = [
    "correct_refusal", "misdiagnosed_refusal",
    "attempted_repair", "silent_failure", "unsafe_repair",
]
_TAXONOMY_COLOURS = {
    "correct_refusal":      "#15803d",  # green
    "misdiagnosed_refusal": "#E69F00",  # amber
    "attempted_repair":     "#0072B2",  # blue
    "silent_failure":       "#6b7280",  # slate grey
    "unsafe_repair":        "#b91c1c",  # deep red
}
_TAXONOMY_LABELS = {
    "correct_refusal":      "Correct refusal",
    "misdiagnosed_refusal": "Misdiagnosed refusal",
    "attempted_repair":     "Attempted repair",
    "silent_failure":       "Silent failure",
    "unsafe_repair":        "Unsafe repair",
}

# Custom divergent heat-map colormap: desaturated red → amber → green, tuned
# for readability in greyscale and for protanopia. Values 0–1 map to
# colours that remain distinguishable at 0.3 vs 0.7.
_HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "flowagent_rag",
    [
        (0.00, "#b91c1c"),  # deep red
        (0.25, "#ea580c"),  # red-orange
        (0.50, "#facc15"),  # amber
        (0.75, "#65a30d"),  # yellow-green
        (1.00, "#15803d"),  # deep green
    ],
    N=256,
)


def _provider_from_model(model: str) -> str:
    """Guess provider from the model id string."""
    m = model.lower()
    if "claude" in m:  return "anthropic"
    if "gpt" in m or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return "openai"
    if "gemini" in m:  return "google"
    if "llama" in m or "mistral" in m or "qwen" in m:
        return "ollama"
    return "other"


def _short_name(model: str) -> str:
    """Shorten model names for axis labels."""
    replacements = [
        # ── Anthropic (current) ────────────────────────────
        ("claude-opus-4-7",            "Opus 4.7"),
        ("claude-opus-4-6",            "Opus 4.6"),
        ("claude-opus-4-5",            "Opus 4.5"),
        ("claude-opus-4-1",            "Opus 4.1"),
        ("claude-opus-4",              "Opus 4"),
        ("claude-sonnet-4-6",          "Sonnet 4.6"),
        ("claude-sonnet-4-5",          "Sonnet 4.5"),
        ("claude-sonnet-4",            "Sonnet 4"),
        ("claude-haiku-4-5",           "Haiku 4.5"),
        ("claude-haiku-3-5",           "Haiku 3.5"),
        # ── Anthropic (legacy, dated snapshot IDs) ─────────
        ("claude-3-5-haiku-20241022",  "Haiku 3.5"),
        ("claude-3-5-sonnet-20241022", "Sonnet 3.5"),
        ("claude-3-haiku-20240307",    "Haiku 3"),
        ("claude-3-sonnet-20240229",   "Sonnet 3"),
        ("claude-sonnet-4-20250514",   "Sonnet 4"),
        # ── Google ─────────────────────────────────────────
        ("gemini-3.1-flash-lite-preview", "Gemini 3.1 Flash-Lite"),
        ("gemini-3.1-pro",             "Gemini 3.1 Pro"),
        ("gemini-3-flash",             "Gemini 3 Flash"),
        ("gemini-2.5-flash-lite",      "Gemini 2.5 Flash-Lite"),
        ("gemini-2.5-flash",           "Gemini 2.5 Flash"),
        ("gemini-2.5-pro",             "Gemini 2.5 Pro"),
        ("gemini-1.5-flash",           "Gemini 1.5 Flash"),
        ("gemini-1.5-pro",             "Gemini 1.5 Pro"),
        # ── OpenAI (current — GPT-5.4) ─────────────────────
        ("gpt-5.4-nano",               "GPT-5.4 nano"),
        ("gpt-5.4-mini",               "GPT-5.4 mini"),
        ("gpt-5.4-pro",                "GPT-5.4 Pro"),
        ("gpt-5.4",                    "GPT-5.4"),
        # ── OpenAI (current — GPT-4.1) ─────────────────────
        ("gpt-4.1-nano",               "GPT-4.1 nano"),
        ("gpt-4.1-mini",               "GPT-4.1 mini"),
        ("gpt-4.1",                    "GPT-4.1"),
        # ── OpenAI (reasoning) ─────────────────────────────
        ("o3-pro",                     "o3-pro"),
        ("o3-mini",                    "o3-mini"),
        ("o3",                         "o3"),
        ("o4-mini",                    "o4-mini"),
        ("o1-pro",                     "o1-pro"),
        ("o1",                         "o1"),
        # ── OpenAI (legacy) ────────────────────────────────
        ("gpt-4o-mini",                "GPT-4o mini"),
        ("gpt-4o",                     "GPT-4o"),
        ("gpt-4-turbo",                "GPT-4 Turbo"),
        ("gpt-3.5-turbo",              "GPT-3.5"),
        ("gpt-4",                      "GPT-4"),
    ]
    for long, short in replacements:
        if model == long:
            return short
    return model


# Models that emit reasoning / thinking tokens (billed as output). Used
# to annotate heatmap columns so reviewers can see at a glance which
# models pay the hidden-thinking tax on the cost-vs-quality tradeoff.
_REASONING_MODEL_IDS: Set[str] = {
    # OpenAI — GPT-5.4 family and the o-series
    "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.4-pro",
    "o1", "o1-pro", "o3", "o3-pro", "o3-mini", "o4-mini",
    # Anthropic — Opus 4.6 / 4.7 are documented thinking models
    "claude-opus-4-6", "claude-opus-4-7",
    # Google — Gemini 2.5 / 3.x have thinking modes, though Flash variants
    # run at effort=low by default. We leave them non-reasoning to keep
    # the annotation useful (otherwise every Google model except Flash-Lite
    # would flip to reasoning and the label loses information).
}


def _classify_model(model_id: str) -> Tuple[str, bool]:
    """Return ``(provider, is_reasoning)`` for a model ID.

    ``provider`` is one of ``openai`` / ``anthropic`` / ``google`` /
    ``ollama`` / ``other``. ``is_reasoning`` is True for models that
    emit hidden-thinking tokens (billed as output).
    """
    return _provider_from_model(model_id), model_id in _REASONING_MODEL_IDS


def _tier_of(model: str) -> str:
    """Best-effort inference of which tier a model belongs to, for ordering.

    Used only when the ``tier`` column is absent from the metrics CSV.
    Returns one of: ``legacy`` | ``mid`` | ``frontier`` — these are the
    plotting buckets, which map to ``_TIER_ORDER``. Pre-current-gen
    snapshots (2024 and earlier) are ``legacy``; mid-generation bridges
    (gpt-4o, Sonnet 3.5, Gemini 2.0/1.5 Pro, etc.) are ``mid``; the
    2026 flagships (GPT-5.4 family, Claude 4.5+/Sonnet 4.5+/Haiku 4.5,
    Gemini 2.5+/3.x, o3/o4 reasoning) are ``frontier``.
    """
    m = model.lower()
    # Legacy — pre-current-gen or explicitly superseded.
    legacy_markers = (
        "gpt-3.5-turbo", "gpt-4-turbo",
        "claude-3-haiku", "claude-3-sonnet-2024",
        "gemini-1.5-flash", "gemini-1.5-pro",
    )
    if any(k in m for k in legacy_markers):
        return "legacy"
    # Mid — bridging-generation models.
    mid_markers = (
        "claude-3-5-", "claude-haiku-3-5",
        "gpt-4o", "gpt-4o-mini",
        "o1", "o1-pro",
        "gemini-2.0-", "claude-sonnet-4-20250514", "claude-opus-4-20250514",
    )
    if any(k in m for k in mid_markers):
        return "mid"
    # Frontier — current flagships.
    return "frontier"


def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Wilson score 95% confidence interval for a proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0, centre - margin), min(1, centre + margin)


def _empty_figure(msg: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.5, 2.2))
    ax.text(0.5, 0.5, msg, ha="center", va="center", wrap=True,
            fontsize=9, color="#6b7280")
    ax.set_axis_off()
    return fig


def _style_value_axis(ax, *, x: bool = True) -> None:
    """Apply consistent grid + spine styling to a proportion/count axis."""
    if x:
        ax.xaxis.grid(True, linestyle="-", linewidth=0.6, alpha=0.35)
    else:
        ax.yaxis.grid(True, linestyle="-", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)


# ── Benchmark A: planning correctness ────────────────────────────

def _pass_rate_panel(ax, df_slice: pd.DataFrame, title: str,
                     show_xlabel: bool = True) -> None:
    """Horizontal bar chart of overall_pass rate with Wilson CIs.

    Bars are coloured by provider. Order: lowest rate on the bottom, highest
    on the top (so the strongest model is visually prominent).
    """
    if df_slice.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="#6b7280")
        ax.set_title(title, loc="left")
        return

    agg = (df_slice.groupby("model")["overall_pass"]
           .agg(["sum", "count"]).reset_index())
    agg.columns = ["model", "k", "n"]
    agg[["rate", "ci_lo", "ci_hi"]] = agg.apply(
        lambda r: pd.Series(_wilson_ci(int(r["k"]), int(r["n"]))),
        axis=1,
    )
    agg = agg.sort_values("rate", ascending=True).reset_index(drop=True)

    y       = np.arange(len(agg))
    colours = [_PROVIDER_COLOURS.get(_provider_from_model(m), "#6b7280")
               for m in agg["model"]]
    labels  = [_short_name(m) for m in agg["model"]]
    # Clip to ≥0 — when a Wilson CI is hit by the [0, 1] boundary cap the
    # subtraction can round to a tiny negative (e.g. -1e-17) and matplotlib
    # then refuses the whole xerr array.
    xerr_lo = np.clip((agg["rate"] - agg["ci_lo"]).values, 0.0, None)
    xerr_hi = np.clip((agg["ci_hi"] - agg["rate"]).values, 0.0, None)

    ax.barh(y, agg["rate"], color=colours, edgecolor="white",
            linewidth=0.6, height=0.72,
            xerr=[xerr_lo, xerr_hi], capsize=2.2,
            error_kw={"elinewidth": 0.8, "capthick": 0.8, "color": "#1f2937"})

    # Rate annotation: always just past the upper CI endpoint so it never
    # collides with the error-bar cap.
    for i, (rate, ci_hi) in enumerate(zip(agg["rate"], agg["ci_hi"])):
        x = min(ci_hi + 0.015, 1.14)
        ax.text(x, i, f"{rate:.0%}", va="center", ha="left",
                fontsize=7.5, color="#1f2937")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title(title, loc="left")
    _style_value_axis(ax, x=True)
    if show_xlabel:
        ax.set_xlabel("Pass rate  (95% Wilson CI)")


def _add_provider_legend(fig: plt.Figure, providers_present: List[str],
                         loc: str = "upper center",
                         bbox_to_anchor=(0.5, 1.02)) -> None:
    """Attach a compact provider legend to a figure."""
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=_PROVIDER_COLOURS[p],
                      ec="white", linewidth=0.4)
        for p in providers_present
    ]
    _PROVIDER_DISPLAY = {
        "anthropic": "Anthropic", "openai": "OpenAI",
        "google": "Google", "ollama": "Ollama", "other": "Other",
    }
    labels = [_PROVIDER_DISPLAY.get(p, p.capitalize()) for p in providers_present]
    fig.legend(handles, labels, loc=loc, ncol=len(labels),
               frameon=False, bbox_to_anchor=bbox_to_anchor,
               handlelength=1.2, handleheight=1.0, borderaxespad=0.0,
               columnspacing=1.4)


def planning_figure(df: pd.DataFrame) -> plt.Figure:
    """Two-panel bar chart: standard prompts vs hard prompts.

    Pass rate per model with 95% Wilson confidence intervals, bars coloured
    by provider.  Rows with a non-null ``error`` column are filtered out so
    they don't collapse the aggregate.
    """
    if "error" in df.columns:
        df = df[df["error"].isna()]
    if "model" not in df.columns or df.empty or "overall_pass" not in df.columns:
        reason = ("all cells errored (likely missing API keys)"
                  if "error" in df.columns
                  else "no planning metrics present in this run")
        return _empty_figure(f"No usable planning data.\n({reason})")

    df = df.copy()
    df["overall_pass"] = df["overall_pass"].map(
        lambda v: v if isinstance(v, (bool, int, float))
        else str(v).strip().lower() == "true"
    ).astype(int)

    has_hard = df["input_id"].str.startswith("hard_").any()

    n_models = df["model"].nunique()
    panel_height = max(3.0, 0.28 * n_models + 1.2)

    if has_hard:
        orig = df[~df["input_id"].str.startswith("hard_")]
        hard = df[ df["input_id"].str.startswith("hard_")]

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(7.2, panel_height), sharey=False,
        )
        _pass_rate_panel(ax1, orig, "a  Standard prompts")
        _pass_rate_panel(ax2, hard, "b  Hard prompts")
    else:
        fig, ax = plt.subplots(figsize=(5.2, panel_height))
        _pass_rate_panel(ax, df, "All prompts")

    providers_present = sorted(
        {_provider_from_model(m) for m in df["model"].unique()},
        key=lambda p: list(_PROVIDER_COLOURS).index(p)
        if p in _PROVIDER_COLOURS else 99,
    )
    _add_provider_legend(fig, providers_present,
                         bbox_to_anchor=(0.5, 1.015))

    fig.suptitle("Planning correctness across LLMs",
                 fontsize=11, fontweight="bold", y=1.06)
    return fig


_PROVIDER_RANK = {"openai": 0, "anthropic": 1, "google": 2,
                   "ollama": 3, "other": 4}


def _style_xtick_labels_by_provider(ax, columns) -> None:
    """Colour x-tick labels by provider and italicise reasoning models."""
    for tick, m in zip(ax.get_xticklabels(), columns):
        provider, is_reasoning = _classify_model(m)
        tick.set_color(_PROVIDER_COLOURS.get(provider, "#111827"))
        if is_reasoning:
            tick.set_fontstyle("italic")
            tick.set_fontweight("bold")


# Prompt-tier classification is *corpus-based*: prompts in the hard
# benchmark tier carry a ``hard_`` prefix in ``corpus/prompts.yaml``.
# These are a-priori designated as the stress-test subset, regardless
# of whether any given model happens to solve them at runtime. Row
# labels on the heatmaps colour-code that corpus tier — so a reviewer
# can see which rows belong to the hard-benchmark subset at a glance.
_CORPUS_TIER_COLOURS = {
    "hard":     "#b91c1c",   # deep red — corpus-designated hard prompts
    "standard": "#111827",   # dark grey/black — standard benchmark tier
}


def _classify_corpus_tier(input_id: str) -> str:
    """Return ``hard`` or ``standard`` based on the prompt's corpus tier."""
    return "hard" if str(input_id).startswith("hard_") else "standard"


def _style_ytick_labels_by_corpus(ax, prompt_ids) -> None:
    """Colour y-tick labels by corpus tier.

    ``prompt_ids`` is an iterable of ``input_id`` values in the same
    order as the tick positions on the y-axis. Hard-tier rows are
    bolded + coloured red; standard-tier rows use the default text style.
    """
    ticks = ax.get_yticklabels()
    for tick_label, prompt_id in zip(ticks, prompt_ids):
        tier = _classify_corpus_tier(prompt_id)
        tick_label.set_color(_CORPUS_TIER_COLOURS[tier])
        if tier == "hard":
            tick_label.set_fontweight("bold")


def _add_provider_reasoning_legend(fig, providers_present=None,
                                    show_difficulty: bool = True) -> None:
    """Small key explaining colour + italic + difficulty conventions."""
    from matplotlib.lines import Line2D
    handles = []
    if providers_present is None:
        providers_present = ["openai", "anthropic", "google"]
    labels_map = {"openai": "OpenAI", "anthropic": "Anthropic (Claude)",
                  "google": "Google (Gemini)", "ollama": "Local (Ollama)"}
    # Column-axis annotations: provider + reasoning
    for p in providers_present:
        handles.append(Line2D(
            [], [], marker="s", color="none",
            markerfacecolor=_PROVIDER_COLOURS.get(p, "#6b7280"),
            markersize=9, label=f"{labels_map.get(p, p.title())} (col)",
        ))
    handles.append(Line2D(
        [], [], marker="none", color="#111827", linestyle="",
        label="$\\bf{\\mathit{italic\\ bold}}$ = reasoning",
    ))
    # Row-axis annotations: corpus tier
    if show_difficulty:
        corpus_labels = [
            ("hard",     "Hard-corpus prompt (row)"),
            ("standard", "Standard-corpus prompt (row)"),
        ]
        for key, lbl in corpus_labels:
            handles.append(Line2D(
                [], [], marker="s", color="none",
                markerfacecolor=_CORPUS_TIER_COLOURS[key], markersize=9,
                label=lbl,
            ))
    fig.legend(
        handles=handles, loc="lower center",
        ncol=min(len(handles), 4), frameon=False, fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )


def planning_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Per-prompt × per-model pass-rate heatmap across the full corpus.

    Shows all 41 prompts. Row labels are colour-coded by corpus tier —
    **red bold** for prompts in the hard-benchmark subset (``hard_``
    prefix in ``corpus/prompts.yaml``), default weight for standard
    prompts. Within each corpus tier rows are ordered with lowest
    cross-model mean pass rate on top, so empirical difficulty is still
    visible through row ordering even though the tier labelling is
    corpus-defined.

    Columns are grouped by provider (OpenAI → Anthropic → Google) and
    ranked by pass rate within each provider; reasoning models are
    rendered in italic bold.
    """
    if "error" in df.columns:
        df = df[df["error"].isna()]
    if "model" not in df.columns or df.empty or "overall_pass" not in df.columns:
        return _empty_figure("No data for heatmap.")

    df = df.copy()
    df["overall_pass"] = df["overall_pass"].map(
        lambda v: v if isinstance(v, (bool, int, float))
        else str(v).strip().lower() == "true"
    ).astype(int)

    pivot = df.pivot_table(
        index="input_id", columns="model", values="overall_pass",
        aggfunc="mean",
    )

    # Row order: hard-corpus prompts on top (sorted by empirical
    # difficulty — hardest-to-plan first), then standard-corpus prompts
    # below (same ordering within tier). A single horizontal separator
    # between the two tiers anchors the visual split.
    row_means = pivot.mean(axis=1)
    hard_ids = [i for i in pivot.index if _classify_corpus_tier(i) == "hard"]
    std_ids  = [i for i in pivot.index if _classify_corpus_tier(i) == "standard"]
    hard_ids = sorted(hard_ids, key=lambda i: row_means[i])
    std_ids  = sorted(std_ids,  key=lambda i: row_means[i])
    pivot = pivot.loc[hard_ids + std_ids]

    # Column order: provider → reasoning-flag → descending pass rate.
    def _col_sort_key(m: str):
        provider, reasoning = _classify_model(m)
        return (_PROVIDER_RANK.get(provider, 9), not reasoning,
                -pivot[m].mean())
    col_order = sorted(pivot.columns, key=_col_sort_key)
    pivot = pivot[col_order]

    # Figure size: scale with matrix shape, clamp to journal widths.
    fig_w = max(6.0, min(0.55 * pivot.shape[1] + 2.4, 11.5))
    # 41 prompts shown; taller clamp than the old hard-only-18 version.
    fig_h = max(3.2, min(0.34 * pivot.shape[0] + 1.6, 14.0))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(pivot.values, aspect="auto", cmap=_HEATMAP_CMAP,
                   vmin=0, vmax=1, interpolation="nearest")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([_short_name(m) for m in pivot.columns],
                       rotation=35, ha="right", fontsize=8)
    _style_xtick_labels_by_provider(ax, pivot.columns)

    display_names = [
        n.removeprefix("hard_").replace("_", " ").title()
        for n in pivot.index
    ]
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(display_names, fontsize=8.5)

    # Corpus-tier colouring: red bold for hard-tier prompts, default for standard.
    _style_ytick_labels_by_corpus(ax, pivot.index)

    # Horizontal separator between the two corpus tiers.
    n_hard = sum(1 for i in pivot.index if _classify_corpus_tier(i) == "hard")
    if 0 < n_hard < pivot.shape[0]:
        ax.axhline(n_hard - 0.5, color="#111827", linewidth=1.2, alpha=0.6)

    # Thin white separators
    ax.set_xticks(np.arange(-.5, pivot.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, pivot.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)

    # Vertical separators between provider groups — help the eye anchor.
    prev_provider = None
    for j, m in enumerate(pivot.columns):
        p, _ = _classify_model(m)
        if prev_provider is not None and p != prev_provider:
            ax.axvline(j - 0.5, color="#111827", linewidth=1.2, alpha=0.6)
        prev_provider = p

    # Cell values are encoded by colour only — explicit % labels were
    # removed after they saturated the figure at 97–100% across most
    # rows and obscured the difficulty-tier row-labels. The colourbar
    # remains the quantitative key.

    ax.set_title(
        "Per-prompt pass rate across the full corpus  —  "
        "red bold labels: hard-benchmark subset (corpus tier)",
        loc="left", fontsize=10.5, fontweight="bold",
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.015,
                        ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=2, labelsize=7.5)
    cbar.set_label("Pass rate", fontsize=8.5)

    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)

    # Legend key for provider colours + reasoning italic.
    providers_present = []
    for m in pivot.columns:
        p, _ = _classify_model(m)
        if p not in providers_present:
            providers_present.append(p)
    _add_provider_reasoning_legend(fig, providers_present=providers_present)
    fig.subplots_adjust(bottom=0.22)
    return fig


def _load_model_tiers(models_yaml: Optional[Path] = None) -> Dict[str, str]:
    """Return ``{model_id: tier}`` from ``config/models.yaml``.

    Falls back to ``_tier_of(model)`` inference when the YAML can't be
    located — so the function still works against historical CSVs.
    """
    if models_yaml is None:
        here = Path(__file__).resolve().parent.parent  # benchmarks/
        models_yaml = here / "config" / "models.yaml"
    tiers: Dict[str, str] = {}
    try:
        import yaml  # local import to avoid module-level dep
        with models_yaml.open() as f:
            cfg = yaml.safe_load(f)
        for entry in cfg.get("models", []):
            if entry.get("id") and entry.get("tier"):
                tiers[entry["id"]] = entry["tier"]
    except Exception:
        pass
    return tiers


def planning_heatmap_by_tier(
    df: pd.DataFrame,
    models_yaml: Optional[Path] = None,
    *,
    struggle_threshold: float = 0.80,  # retained for API stability; unused
) -> plt.Figure:
    """Two-panel per-prompt × per-model heatmap split into current / legacy tiers.

    Hard prompts only. Rows are colour-coded by cross-model difficulty:
    **red** (mean pass < 50%), **amber** (50–90%), **green** (≥90%) — so
    reviewers can see at a glance which prompts stress the system
    regardless of model tier, and which are saturated across the corpus.

    Columns within each panel are sorted by descending mean pass rate.
    ``preview`` tier models (e.g. ``gemini-3.1-flash-lite-preview``) are
    grouped with ``current``; ``legacy`` models get their own panel.
    Models whose tier can't be resolved fall through to the ``current``
    panel so nothing is dropped from the figure.
    """
    if "error" in df.columns:
        df = df[df["error"].isna()]
    if df.empty or "model" not in df.columns or "overall_pass" not in df.columns:
        return _empty_figure("No data for tier-split heatmap.")

    df = df.copy()
    df["overall_pass"] = df["overall_pass"].map(
        lambda v: v if isinstance(v, (bool, int, float))
        else str(v).strip().lower() == "true"
    ).astype(int)

    pivot = df.pivot_table(
        index="input_id", columns="model", values="overall_pass",
        aggfunc="mean",
    )
    if pivot.empty:
        return _empty_figure("No prompt data for tier split.")

    # Row order: hard-corpus prompts on top, standard below. Within each
    # tier, sort by empirical mean (hardest first) so the gradient stays
    # legible even though the hard/standard split is corpus-defined.
    row_means = pivot.mean(axis=1)
    hard_ids = sorted(
        (i for i in pivot.index if _classify_corpus_tier(i) == "hard"),
        key=lambda i: row_means[i],
    )
    std_ids = sorted(
        (i for i in pivot.index if _classify_corpus_tier(i) == "standard"),
        key=lambda i: row_means[i],
    )
    pivot = pivot.loc[hard_ids + std_ids]

    # Column grouping by tier.
    tiers = _load_model_tiers(models_yaml)
    current_cols, legacy_cols = [], []
    for m in pivot.columns:
        tier = tiers.get(m, _tier_of(m))
        if tier in ("current", "preview", "frontier", "mid"):
            current_cols.append(m)
        elif tier == "legacy":
            legacy_cols.append(m)
        else:
            current_cols.append(m)  # fall through — don't drop unresolved

    # Within each panel sort by (provider, reasoning-flag, pass rate) so
    # models cluster by family and reasoning column-groups are obvious.
    def _panel_sort_key(m: str):
        provider, reasoning = _classify_model(m)
        return (_PROVIDER_RANK.get(provider, 9), not reasoning,
                -pivot[m].mean())
    current_cols = sorted(current_cols, key=_panel_sort_key)
    legacy_cols = sorted(legacy_cols, key=_panel_sort_key)

    pivot_cur = pivot[current_cols]
    pivot_leg = pivot[legacy_cols] if legacy_cols else None

    # Row-level stats for difficulty colouring (used by the inner
    # _render via closure): red=hard (<50%), amber=moderate,
    # green=easy (>=90%).
    row_means = pivot.mean(axis=1)

    n_cur = pivot_cur.shape[1]
    n_leg = pivot_leg.shape[1] if pivot_leg is not None else 0
    n_rows = pivot.shape[0]

    # Figure sizing: accommodate both panels side-by-side.
    panel_w_current = max(3.5, 0.42 * n_cur + 1.5)
    panel_w_legacy = max(2.5, 0.42 * n_leg + 1.5) if n_leg else 0
    fig_w = panel_w_current + panel_w_legacy + 2.0  # + colourbar + gutter
    fig_h = max(3.5, min(0.34 * n_rows + 1.7, 14.0))

    width_ratios = [n_cur, max(n_leg, 0.4)] if n_leg else [n_cur]
    fig, axes = plt.subplots(
        1, len(width_ratios), figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.08},
        squeeze=False,
    )
    axes = axes[0]
    ax_cur = axes[0]
    ax_leg = axes[1] if n_leg else None

    def _render(ax, data, *, show_ylabels: bool, title: str):
        im = ax.imshow(data.values, aspect="auto", cmap=_HEATMAP_CMAP,
                       vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks(range(data.shape[1]))
        ax.set_xticklabels(
            [_short_name(m) for m in data.columns],
            rotation=35, ha="right", fontsize=8,
        )
        # Provider colour + reasoning italic on x-tick labels.
        _style_xtick_labels_by_provider(ax, data.columns)

        ax.set_yticks(range(data.shape[0]))
        if show_ylabels:
            display_names = [
                n.removeprefix("hard_").replace("_", " ").title()
                for n in data.index
            ]
            ax.set_yticklabels(display_names, fontsize=8.5)
            # Corpus-tier colouring: red bold for hard-benchmark
            # prompts (``hard_`` prefix in the corpus YAML), default
            # weight for standard prompts.
            _style_ytick_labels_by_corpus(ax, data.index)
            # Horizontal separator between the hard and standard tier blocks.
            n_hard = sum(
                1 for i in data.index if _classify_corpus_tier(i) == "hard"
            )
            if 0 < n_hard < data.shape[0]:
                ax.axhline(n_hard - 0.5, color="#111827",
                           linewidth=1.2, alpha=0.6)
        else:
            ax.set_yticklabels([])
            # Still draw the separator on the right panel so rows
            # visually align across panels.
            n_hard = sum(
                1 for i in data.index if _classify_corpus_tier(i) == "hard"
            )
            if 0 < n_hard < data.shape[0]:
                ax.axhline(n_hard - 0.5, color="#111827",
                           linewidth=1.2, alpha=0.6)
        ax.set_xticks(np.arange(-.5, data.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, data.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(which="major", length=0)

        # Vertical separators between provider groups.
        prev_provider = None
        for j, m in enumerate(data.columns):
            p, _ = _classify_model(m)
            if prev_provider is not None and p != prev_provider:
                ax.axvline(j - 0.5, color="#111827", linewidth=1.2, alpha=0.6)
            prev_provider = p

        # No per-cell % annotations — colour alone encodes pass rate,
        # the shared colourbar is the key. Removing the annotations lets
        # the row-label difficulty colouring and column grouping breathe.

        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
        for spine in ("top", "right", "bottom", "left"):
            ax.spines[spine].set_visible(False)
        return im

    im = _render(
        ax_cur, pivot_cur,
        show_ylabels=True,
        title=f"Current / preview models  (n={n_cur})",
    )
    if ax_leg is not None:
        _render(
            ax_leg, pivot_leg,
            show_ylabels=False,
            title=f"Legacy models  (n={n_leg})",
        )

    # Shared colourbar
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.75, pad=0.015,
                        ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=2, labelsize=7.5)
    cbar.set_label("Pass rate", fontsize=8.5)

    fig.suptitle(
        "Per-prompt pass rate by model tier  —  "
        "red bold labels: corpus-designated hard benchmarks; "
        "normal labels: standard benchmarks",
        fontsize=10.5, fontweight="bold", x=0.02, ha="left",
    )

    # Provider + reasoning legend (same as single-panel heatmap).
    providers_present = []
    for m in list(pivot_cur.columns) + (
        list(pivot_leg.columns) if pivot_leg is not None else []
    ):
        p, _ = _classify_model(m)
        if p not in providers_present:
            providers_present.append(p)
    _add_provider_reasoning_legend(fig, providers_present=providers_present)
    fig.subplots_adjust(bottom=0.20)
    return fig


# ── Benchmark B: error recovery ──────────────────────────────────

# Legacy fall-back — populated from fault_tier column when present.
_UNRECOVERABLE_FAULTS = {
    "corrupt_fastq", "paired_single_mismatch",
    "empty_input_file", "binary_as_fastq",
}

_FAULT_LABELS = {
    # Easy
    "missing_wget":                "Missing binary (wget \u2192 curl)",
    "tool_typo":                   "Tool name typo (fastq_c)",
    "wrong_flag":                  "Wrong CLI flag (-x \u2192 -i)",
    "missing_output_dir":          "Missing output directory",
    "readonly_output":             "Read-only output directory",
    "path_with_spaces":            "Unquoted path with spaces",
    "multiqc_collision":           "Output file collision (no -f)",
    "samtools_subcommand_typo":    "Subcommand typo (samtools srot)",
    "missing_flag_value":          "Flag missing its value (-@)",
    "missing_mandatory_arg":       "Missing mandatory arg (bwa mem)",
    "cp_source_missing":           "cp source file not found",
    "deep_nonexistent_outdir":     "Redirect to unmade parent dirs",
    "ambiguous_flag":              "Ambiguous abbreviated flag",
    "undefined_env_var":           "Undefined env var (set -u)",
    "missing_python_module":       "Python module not installed",
    # Hard
    "stale_conda_pin":             "Stale conda version pin",
    "bam_unsorted_indexing":       "BAM not sorted for indexing",
    "missing_bwa_index":           "bwa mem before bwa index",
    "missing_faidx":               "Missing .fai index",
    "missing_sequence_dict":       "Missing GATK .dict file",
    "chromosome_naming_mismatch":  "chr1 vs 1 prefix mismatch",
    "htseq_needs_name_sort":       "htseq needs -n sorted BAM",
    "java_heap_oom":                "Java heap OOM (bump -Xmx)",
    "vcf_contig_mismatch":         "VCF contig mismatch on merge",
    # Unrecoverable
    "paired_single_mismatch":      "Paired/single mode mismatch",
    "corrupt_fastq":               "Corrupted input FASTQ",
    "empty_input_file":            "Empty input FASTQ",
    "binary_as_fastq":             "Binary data posing as FASTQ",
}


_TIER_ORDER_B = {"easy": 0, "hard": 1, "unrecoverable": 2}

# Colours used across recovery figures
_REC_GREEN = "#15803d"
_REC_RED   = "#b91c1c"
_REC_SLATE = "#475569"


def _tier_from_row(row, fallback_set: set) -> str:
    """Pick up the explicit ``fault_tier`` column; else infer from legacy set."""
    t = row.get("fault_tier") if hasattr(row, "get") else None
    if isinstance(t, str) and t in _TIER_ORDER_B:
        return t
    return "unrecoverable" if row["fault"] in fallback_set else "easy"


def recovery_figure(df: pd.DataFrame) -> plt.Figure:
    """Per-fault stacked bar chart, grouped into Easy / Hard / Unrecoverable.

    Recoverable faults (easy/hard): green = recovered, red = failed.
    Unrecoverable faults:           slate = correctly rejected, red =
                                    incorrectly attempted.
    """
    if "fault" not in df.columns:
        return _empty_figure("No recovery data.")

    df = df.copy()
    df["recovered"] = df["recovered"].map(
        lambda v: v if isinstance(v, bool)
        else str(v).strip().lower() == "true"
    )

    # Per-fault aggregation
    agg = (df.groupby("fault")["recovered"]
             .agg(["sum", "count"]).reset_index())
    agg.columns = ["fault", "n_recovered", "n_total"]
    agg["n_failed"] = agg["n_total"] - agg["n_recovered"]
    agg["rate"]     = agg["n_recovered"] / agg["n_total"]

    # Attach tier (prefer explicit column; fall back to legacy set)
    if "fault_tier" in df.columns:
        tier_by_fault = (df.groupby("fault")["fault_tier"]
                           .agg(lambda s: s.iloc[0]).to_dict())
    else:
        tier_by_fault = {}
    agg["tier"] = agg["fault"].map(
        lambda f: tier_by_fault.get(f)
        or ("unrecoverable" if f in _UNRECOVERABLE_FAULTS else "easy"))
    agg["label"] = agg["fault"].map(
        lambda f: _FAULT_LABELS.get(f, f.replace("_", " ").title()))

    # Sort: easy first (by rate desc), then hard, then unrecoverable
    agg["sort_key"] = agg.apply(
        lambda r: (_TIER_ORDER_B.get(r["tier"], 99), -r["rate"]),
        axis=1,
    )
    agg = agg.sort_values("sort_key").reset_index(drop=True)

    n_per_tier = agg.groupby("tier").size().to_dict()
    n_easy   = n_per_tier.get("easy", 0)
    n_hard   = n_per_tier.get("hard", 0)
    n_unrec  = n_per_tier.get("unrecoverable", 0)

    fig_h = max(3.2, 0.42 * len(agg) + 1.2)
    fig, ax = plt.subplots(figsize=(8.2, fig_h))

    for i, row in agg.iterrows():
        rate, fail = row["rate"], 1 - row["rate"]
        if row["tier"] == "unrecoverable":
            ax.barh(i, fail,  height=0.68, color=_REC_SLATE,
                    edgecolor="white", linewidth=0.5)
            if rate > 0:
                ax.barh(i, rate, height=0.68, left=fail, color=_REC_RED,
                        edgecolor="white", linewidth=0.5)
        else:
            ax.barh(i, rate,  height=0.68, color=_REC_GREEN,
                    edgecolor="white", linewidth=0.5)
            if fail > 0:
                ax.barh(i, fail, height=0.68, left=rate, color=_REC_RED,
                        edgecolor="white", linewidth=0.5)

        ax.text(1.015, i, f"{int(row['n_recovered'])}/{int(row['n_total'])}",
                va="center", fontsize=8, color="#1f2937")

    ax.set_yticks(np.arange(len(agg)))
    ax.set_yticklabels(agg["label"])
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Fraction of runs")
    ax.invert_yaxis()
    _style_value_axis(ax, x=True)

    # Tier separators + side-labels (placed well to the left of the tick labels)
    tier_colors = {"easy": _REC_GREEN, "hard": "#b45309",
                   "unrecoverable": _REC_SLATE}
    offset = 0
    for tier, count in (("easy", n_easy), ("hard", n_hard),
                        ("unrecoverable", n_unrec)):
        if count == 0:
            continue
        mid = offset + (count - 1) / 2
        ax.text(-0.28, mid,
                tier.capitalize() if tier != "unrecoverable" else "Unrecoverable",
                ha="right", va="center", fontsize=8.5, fontstyle="italic",
                color=tier_colors[tier], fontweight="semibold",
                transform=ax.get_yaxis_transform(),
                clip_on=False)
        offset += count
        if offset < len(agg):
            ax.axhline(offset - 0.5, color="#94a3b8",
                       linewidth=0.7, linestyle="--")

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=_REC_GREEN, edgecolor="white", label="Recovered (correct)"),
        Patch(facecolor=_REC_SLATE, edgecolor="white", label="Correctly rejected"),
        Patch(facecolor=_REC_RED,   edgecolor="white", label="Wrong outcome"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7.5,
              framealpha=0.95, edgecolor="#d1d5db")

    ax.set_title("LLM-driven error recovery", loc="left")
    return fig


def recovery_tier_summary_figure(df: pd.DataFrame) -> Optional[plt.Figure]:
    """Compact per-tier summary: recovery rate for easy vs. hard, plus the
    false-recovery rate on unrecoverable faults (where "recovered" is wrong).
    """
    if "fault" not in df.columns:
        return None
    df = df.copy()
    df["recovered"] = df["recovered"].map(
        lambda v: v if isinstance(v, bool)
        else str(v).strip().lower() == "true"
    )
    if "fault_tier" in df.columns:
        tier_col = df["fault_tier"]
    else:
        tier_col = df["fault"].map(
            lambda f: "unrecoverable" if f in _UNRECOVERABLE_FAULTS else "easy")
    df["tier"] = tier_col

    tiers = [t for t in ("easy", "hard", "unrecoverable")
             if (df["tier"] == t).any()]
    if not tiers:
        return None

    rows = []
    for t in tiers:
        sub = df[df["tier"] == t]
        k = int(sub["recovered"].sum())
        n = int(len(sub))
        rows.append((t, k, n, k / n if n else 0.0))
    tdf = pd.DataFrame(rows, columns=["tier", "k", "n", "rate"])

    fig, ax = plt.subplots(figsize=(5.2, max(1.6, 0.5 * len(tiers) + 0.8)))
    y = np.arange(len(tdf))
    labels = []
    colours = []
    for _, row in tdf.iterrows():
        if row["tier"] == "unrecoverable":
            # Invert the narrative: what we want here is NON-recovery
            correct = 1 - row["rate"]
            labels.append(f"Unrecoverable (correct reject = {correct:.0%})")
            colours.append(_REC_SLATE)
        else:
            labels.append(
                f"{row['tier'].capitalize()} (recovered = {row['rate']:.0%})"
            )
            colours.append(_REC_GREEN if row["tier"] == "easy" else "#b45309")

    displayed = np.where(
        tdf["tier"] == "unrecoverable", 1 - tdf["rate"], tdf["rate"]
    )
    ax.barh(y, displayed, color=colours, edgecolor="white",
            linewidth=0.6, height=0.6)
    for i, (v, row) in enumerate(zip(displayed, tdf.itertuples())):
        ax.text(v + 0.01, i, f"{int(row.k)}/{int(row.n)}",
                va="center", ha="left", fontsize=8, color="#1f2937")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Fraction of runs with the desired outcome")
    ax.set_title("Recovery outcome by fault tier", loc="left")
    _style_value_axis(ax, x=True)
    return fig


def recovery_per_fault_heatmap(df: pd.DataFrame) -> Optional[plt.Figure]:
    """Per-fault × per-model recovery heatmap.

    Rows are individual fault types, grouped by tier (easy → hard →
    unrecoverable) with horizontal separators. Columns are models,
    grouped by provider with reasoning models clustered first within
    each provider block (matching the planning-heatmap convention).

    Cell value is **outcome correctness**, unified across tiers so green
    is always "good":
      * Easy / Hard tiers: fraction of cells where the LLM ``recovered``.
      * Unrecoverable tier: fraction of cells where the LLM correctly
        ``rejected`` (i.e. ``recovered == False``).
    Colour is the shared red→amber→green ``_HEATMAP_CMAP``.
    """
    if df.empty or "fault" not in df.columns or "model" not in df.columns:
        return None

    df = df.copy()
    df["recovered"] = df["recovered"].map(
        lambda v: v if isinstance(v, bool)
        else str(v).strip().lower() == "true"
    )

    # Tier per fault — prefer explicit column, fall back to legacy set.
    if "fault_tier" in df.columns:
        tier_by_fault = (df.groupby("fault")["fault_tier"]
                           .agg(lambda s: s.iloc[0]).to_dict())
    else:
        tier_by_fault = {}
    def _tier_for(f):
        t = tier_by_fault.get(f)
        # ``tier_by_fault`` may yield NaN if the upstream CSV recorded
        # a missing tier; fall back to the legacy unrecoverable set.
        if isinstance(t, str) and t.strip():
            return t
        return "unrecoverable" if f in _UNRECOVERABLE_FAULTS else "easy"

    # Compute outcome-correctness: recovery for easy/hard, rejection for
    # unrecoverable. Unifies the colour scale so green = good across tiers.
    df["_tier"] = df["fault"].map(_tier_for)
    df["_correct"] = np.where(
        df["_tier"] == "unrecoverable",
        ~df["recovered"].astype(bool),
        df["recovered"].astype(bool),
    ).astype(int)

    pivot = df.pivot_table(
        index="fault", columns="model", values="_correct", aggfunc="mean"
    )
    if pivot.empty:
        return None

    # Row order: by tier, then by mean correctness within tier (worst-on-top
    # for easy/hard so reviewer sees the failure clusters; best-on-top for
    # unrecoverable since they all should be 100%).
    fault_tiers = {f: _tier_for(f) for f in pivot.index}
    fault_means = pivot.mean(axis=1)
    def _row_key(f):
        tier = fault_tiers[f]
        # easy=0, hard=1, unrecoverable=2 → tier ordering preserved
        # within each tier sort by ascending mean (worst at top of tier)
        return (_TIER_ORDER_B.get(tier, 9), fault_means[f])
    row_order = sorted(pivot.index, key=_row_key)
    pivot = pivot.loc[row_order]

    # Column order: provider → reasoning-flag → descending correctness mean
    def _col_key(m):
        provider, reasoning = _classify_model(m)
        return (_PROVIDER_RANK.get(provider, 9), not reasoning,
                -pivot[m].mean())
    col_order = sorted(pivot.columns, key=_col_key)
    pivot = pivot[col_order]

    n_rows, n_cols = pivot.shape
    fig_w = max(6.5, min(0.65 * n_cols + 3.5, 12.0))
    fig_h = max(4.0, min(0.32 * n_rows + 2.0, 14.0))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(pivot.values, aspect="auto", cmap=_HEATMAP_CMAP,
                   vmin=0, vmax=1, interpolation="nearest")

    # X-tick labels (models) with provider colour + reasoning italic
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([_short_name(m) for m in pivot.columns],
                       rotation=35, ha="right", fontsize=8.5)
    _style_xtick_labels_by_provider(ax, pivot.columns)

    # Y-tick labels (faults) with pretty names from _FAULT_LABELS
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(
        [_FAULT_LABELS.get(f, f.replace("_", " ").title()) for f in pivot.index],
        fontsize=8.5,
    )

    # Tier separators (horizontal) + tier-label markers on the left edge
    tier_sequence = [fault_tiers[f] for f in pivot.index]
    prev_tier = None
    for i, t in enumerate(tier_sequence):
        if prev_tier is not None and t != prev_tier:
            ax.axhline(i - 0.5, color="#111827", linewidth=1.4, alpha=0.7)
        prev_tier = t
    # Inline tier markers on the y-axis (shown left of the labels)
    tier_labels_seen = set()
    for i, t in enumerate(tier_sequence):
        if t not in tier_labels_seen:
            tier_labels_seen.add(t)
            display = {"easy": "Easy", "hard": "Hard",
                        "unrecoverable": "Unrecov."}.get(t, t.title())
            ax.text(-0.5, i - 0.4, display,
                    fontsize=8.5, fontweight="bold", style="italic",
                    color="#374151", ha="right", va="bottom")

    # Provider separators (vertical)
    prev_provider = None
    for j, m in enumerate(pivot.columns):
        p, _ = _classify_model(m)
        if prev_provider is not None and p != prev_provider:
            ax.axvline(j - 0.5, color="#111827", linewidth=1.2, alpha=0.6)
        prev_provider = p

    # Cell-grid white separators
    ax.set_xticks(np.arange(-.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)

    # No per-cell annotations — colour alone encodes the value, the
    # shared colourbar is the quantitative key. Raw n_correct/n_total
    # counts are still in the underlying CSV for spot-checks.

    ax.set_title(
        "Per-fault recovery correctness across models  —  "
        "green = correct outcome (recovered for easy/hard, rejected for unrecoverable)",
        loc="left", fontsize=10.5, fontweight="bold",
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.015,
                        ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=2, labelsize=7.5)
    cbar.set_label("Outcome correctness", fontsize=8.5)

    # Provider + reasoning legend (mirrors the planning heatmap)
    providers_present = []
    for m in pivot.columns:
        p, _ = _classify_model(m)
        if p not in providers_present:
            providers_present.append(p)
    _add_provider_reasoning_legend(
        fig, providers_present=providers_present, show_difficulty=False,
    )
    fig.subplots_adjust(bottom=0.18, left=0.28)
    return fig


def recovery_reasoning_split_figure(per_cell_df: pd.DataFrame) -> Optional[plt.Figure]:
    """Two-panel figure contrasting reasoning vs non-reasoning models on the
    unrecoverable-fault taxonomy.

    Panel a — per-model stacked bars, grouped into reasoning (top) and
    non-reasoning (bottom) blocks with a horizontal separator.
    Panel b — aggregated: one stacked bar per reasoning class (reasoning
    vs non-reasoning), letting readers see the headline difference in
    safe-refusal vs unsafe-repair rates between the two classes.

    Input is the ``per_cell.csv`` that ``recovery_taxonomy.py`` emits.
    Reasoning classification uses ``_REASONING_MODEL_IDS`` defined at
    the top of this module.
    """
    if per_cell_df.empty or "_category" not in per_cell_df.columns:
        return None
    if "fault_tier" in per_cell_df.columns:
        df = per_cell_df[per_cell_df["fault_tier"] == "unrecoverable"].copy()
    else:
        df = per_cell_df.copy()
    if df.empty or "_model" not in df.columns:
        return None

    df["_reasoning"] = df["_model"].map(
        lambda m: "reasoning" if m in _REASONING_MODEL_IDS else "non-reasoning"
    )

    # ── Per-model percentages ──────────────────────────────────────
    counts = (df.groupby(["_model", "_category"])
                .size().unstack("_category", fill_value=0))
    for c in _TAXONOMY_CATEGORIES:
        if c not in counts.columns:
            counts[c] = 0
    counts = counts[_TAXONOMY_CATEGORIES]
    totals = counts.sum(axis=1)
    pcts = counts.div(totals, axis=0) * 100

    # Reasoning tag per model (preserved across groupby)
    model_is_reasoning = (df.groupby("_model")["_reasoning"]
                            .first()
                            .eq("reasoning"))

    # ── Aggregated across reasoning class ──────────────────────────
    group_counts = (df.groupby(["_reasoning", "_category"])
                      .size().unstack("_category", fill_value=0))
    for c in _TAXONOMY_CATEGORIES:
        if c not in group_counts.columns:
            group_counts[c] = 0
    group_counts = group_counts[_TAXONOMY_CATEGORIES]
    group_totals = group_counts.sum(axis=1)
    group_pcts = group_counts.div(group_totals, axis=0) * 100

    # Order reasoning groups top-down
    group_order = [g for g in ("reasoning", "non-reasoning")
                   if g in group_pcts.index]

    # Model order within each reasoning block — by safe-refusal rate descending
    safe = pcts["correct_refusal"] + pcts["misdiagnosed_refusal"]
    reasoning_models = [m for m in pcts.index if model_is_reasoning.get(m, False)]
    nonreas_models   = [m for m in pcts.index if not model_is_reasoning.get(m, False)]
    reasoning_models = sorted(reasoning_models, key=lambda m: safe[m])
    nonreas_models   = sorted(nonreas_models,   key=lambda m: safe[m])
    # Reasoning at top; non-reasoning below.
    ordered_models = reasoning_models + nonreas_models
    n_reas = len(reasoning_models)

    _set_publication_style()
    height = max(2.8, 0.55 * len(ordered_models) + 2.2)
    fig, axes = plt.subplots(
        1, 2, figsize=(12, height),
        gridspec_kw={"width_ratios": [2.3, 1.0], "wspace": 0.12},
    )
    ax_per, ax_agg = axes

    # ── Panel a: per-model stacked bars ────────────────────────────
    y = np.arange(len(ordered_models))
    left = np.zeros(len(ordered_models))
    for cat in _TAXONOMY_CATEGORIES:
        vals = pcts.loc[ordered_models, cat].values
        ax_per.barh(y, vals, left=left, height=0.72,
                    color=_TAXONOMY_COLOURS[cat], edgecolor="white",
                    linewidth=0.6, label=_TAXONOMY_LABELS[cat])
        for i, v in enumerate(vals):
            if v >= 6:
                ax_per.text(left[i] + v / 2, y[i], f"{v:.0f}%",
                            ha="center", va="center", fontsize=8,
                            color="white", fontweight="bold")
        left += vals
    # n labels
    for i, m in enumerate(ordered_models):
        ax_per.text(101, i, f"n={int(totals.loc[m])}", va="center",
                    fontsize=7.5, color="#444")

    ax_per.set_yticks(y)
    ax_per.set_yticklabels([_short_name(m) for m in ordered_models])

    # Italic-bold reasoning labels + provider colour on y-ticks, matching
    # the planning-heatmap convention so the two figures read together.
    for tick, m in zip(ax_per.get_yticklabels(), ordered_models):
        provider, is_reasoning = _classify_model(m)
        tick.set_color(_PROVIDER_COLOURS.get(provider, "#111827"))
        if is_reasoning:
            tick.set_fontstyle("italic")
            tick.set_fontweight("bold")

    # Separator between reasoning block (top) and non-reasoning (below)
    if 0 < n_reas < len(ordered_models):
        ax_per.axhline(n_reas - 0.5, color="#111827",
                       linewidth=1.2, alpha=0.6)
        ax_per.text(-0.5, n_reas - 0.5 - 0.3, "↑ reasoning  ",
                    fontsize=8, style="italic", color="#6b7280",
                    ha="right", va="top")
        ax_per.text(-0.5, n_reas - 0.5 + 0.3, "↓ non-reasoning",
                    fontsize=8, style="italic", color="#6b7280",
                    ha="right", va="bottom")

    ax_per.set_xlim(0, 108)
    ax_per.set_xlabel("% of cells")
    ax_per.set_title("a  Per-model taxonomy (unrecoverable faults)",
                     loc="left", fontsize=11, fontweight="bold")
    ax_per.grid(axis="x", linestyle="--", alpha=0.35)
    for spine in ("top", "right"): ax_per.spines[spine].set_visible(False)

    # ── Panel b: aggregated reasoning-vs-non-reasoning ─────────────
    y2 = np.arange(len(group_order))
    left2 = np.zeros(len(group_order))
    for cat in _TAXONOMY_CATEGORIES:
        vals = group_pcts.loc[group_order, cat].values
        ax_agg.barh(y2, vals, left=left2, height=0.55,
                    color=_TAXONOMY_COLOURS[cat], edgecolor="white",
                    linewidth=0.6)
        for i, v in enumerate(vals):
            if v >= 6:
                ax_agg.text(left2[i] + v / 2, y2[i], f"{v:.0f}%",
                            ha="center", va="center", fontsize=8.5,
                            color="white", fontweight="bold")
        left2 += vals
    for i, g in enumerate(group_order):
        ax_agg.text(101, i, f"n={int(group_totals.loc[g])}", va="center",
                    fontsize=7.5, color="#444")
    ax_agg.set_yticks(y2)
    ax_agg.set_yticklabels([g.replace("-", "-\n") for g in group_order],
                            fontsize=10)
    ax_agg.set_xlim(0, 108)
    ax_agg.set_xlabel("% of cells")
    ax_agg.set_title("b  Aggregated by class",
                     loc="left", fontsize=11, fontweight="bold")
    ax_agg.grid(axis="x", linestyle="--", alpha=0.35)
    for spine in ("top", "right"): ax_agg.spines[spine].set_visible(False)

    fig.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=_TAXONOMY_COLOURS[c])
                 for c in _TAXONOMY_CATEGORIES],
        labels=[_TAXONOMY_LABELS[c] for c in _TAXONOMY_CATEGORIES],
        loc="lower center", bbox_to_anchor=(0.5, -0.03),
        ncol=len(_TAXONOMY_CATEGORIES), frameon=False, fontsize=9,
    )
    return fig


def recovery_taxonomy_figure(per_cell_df: pd.DataFrame) -> Optional[plt.Figure]:
    """Stacked-bar chart of the unrecoverable-tier response taxonomy.

    Input is the ``per_cell.csv`` that ``recovery_taxonomy.py`` emits —
    one row per cell with a ``_category`` column and a ``_model`` column.
    Output is one horizontal bar per model, stacked left-to-right by
    category percentage; models are ordered top-down by decreasing
    combined "safe refusal" rate (correct + misdiagnosed). The unsafe
    category is coloured red so the dangerous rows jump out visually
    even to a grayscale reader.
    """
    if per_cell_df.empty or "_category" not in per_cell_df.columns:
        return None
    # Restrict to unrecoverable tier (paranoia; the taxonomy script
    # already filters, but the CSV is flat).
    if "fault_tier" in per_cell_df.columns:
        df = per_cell_df[per_cell_df["fault_tier"] == "unrecoverable"].copy()
    else:
        df = per_cell_df.copy()
    if df.empty:
        return None

    counts = (
        df.groupby(["_model", "_category"])
          .size().unstack("_category", fill_value=0)
    )
    # Ensure every category column exists even if absent from the data
    for c in _TAXONOMY_CATEGORIES:
        if c not in counts.columns:
            counts[c] = 0
    counts = counts[_TAXONOMY_CATEGORIES]
    totals = counts.sum(axis=1)
    pcts = counts.div(totals, axis=0) * 100

    # Order models by safe-refusal rate descending (best first, on top)
    safe = pcts["correct_refusal"] + pcts["misdiagnosed_refusal"]
    ordered_models = safe.sort_values(ascending=True).index.tolist()

    _set_publication_style()
    height = max(2.2, 0.55 * len(ordered_models) + 1.0)
    fig, ax = plt.subplots(figsize=(8.5, height))

    y = np.arange(len(ordered_models))
    left = np.zeros(len(ordered_models))
    for cat in _TAXONOMY_CATEGORIES:
        vals = pcts.loc[ordered_models, cat].values
        ax.barh(y, vals, left=left, height=0.72,
                color=_TAXONOMY_COLOURS[cat], edgecolor="white",
                linewidth=0.6,
                label=_TAXONOMY_LABELS[cat])
        # In-bar percentage labels, only for segments >=6 % (readable)
        for i, v in enumerate(vals):
            if v >= 6:
                ax.text(left[i] + v / 2, y[i], f"{v:.0f}%",
                        ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold")
        left += vals

    # n label at right of each bar
    for i, m in enumerate(ordered_models):
        ax.text(101, i, f"n={int(totals.loc[m])}", va="center",
                fontsize=7.5, color="#444")

    ax.set_yticks(y)
    ax.set_yticklabels([_short_name(m) for m in ordered_models])
    ax.set_xlim(0, 108)
    ax.set_xlabel("% of cells")
    ax.set_title(
        "Recovery taxonomy on unrecoverable faults\n"
        "(ordered top → bottom by decreasing safe-refusal rate)",
        fontsize=11, pad=8,
    )
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Legend below the plot — compact, all five categories in a single row
    fig.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=_TAXONOMY_COLOURS[c])
                 for c in _TAXONOMY_CATEGORIES],
        labels=[_TAXONOMY_LABELS[c] for c in _TAXONOMY_CATEGORIES],
        loc="lower center", bbox_to_anchor=(0.5, -0.03),
        ncol=len(_TAXONOMY_CATEGORIES), frameon=False, fontsize=9,
    )
    # No manual subplots_adjust — _save uses ``bbox_inches='tight'``
    # which crops to include the legend automatically, and
    # _set_publication_style installs a constrained layout engine that
    # conflicts with subplots_adjust.
    return fig


# ── Benchmark E: head-to-head competitors ────────────────────────

# Competitor palette — fixed so the same agent keeps the same colour
# across every figure in the paper.
_COMPETITOR_COLOURS = {
    "flowagent":   "#0f766e",   # teal
    "biomaster":   "#a855f7",   # violet
    "autoba":      "#f97316",   # orange
    "cellagent":   "#2563eb",   # blue
    "other":       "#6b7280",
}


def competitors_figure(df: pd.DataFrame) -> Optional[plt.Figure]:
    """Side-by-side bar chart comparing competitors on overall_pass rate
    and cost-per-successful-plan.

    Expects columns: competitor, competitor_name, input_id, overall_pass,
    cost_usd (optional). A two-panel layout mirrors planning_cost_summary.
    """
    if "competitor" not in df.columns or df.empty:
        return None

    df = df.copy()
    # Filter out unavailable-competitor rows (they have error set and no real plan)
    if "error" in df.columns:
        avail = df[df["error"].isna() | (df["error"].astype(str) == "")]
        if avail.empty:
            return _empty_figure(
                "All competitors reported 'not available'.\n"
                "Install their upstream packages or set the CLI env var\n"
                "(see harness/competitors.py for install hints)."
            )
        df = avail

    df["overall_pass"] = df["overall_pass"].map(
        lambda v: v if isinstance(v, (bool, int, float))
        else str(v).strip().lower() == "true"
    ).astype(int)
    if "cost_usd" in df.columns:
        df["cost_usd"] = pd.to_numeric(df["cost_usd"], errors="coerce").fillna(0.0)

    g = (df.groupby(["competitor", "competitor_name"])
             .agg(n=("overall_pass", "count"),
                  passes=("overall_pass", "sum"),
                  total_cost=("cost_usd", "sum") if "cost_usd" in df.columns
                             else ("overall_pass", "count"),
                  mean_wall=("wall_seconds", "mean")
                             if "wall_seconds" in df.columns
                             else ("overall_pass", "mean"))
             .reset_index())
    if g.empty:
        return None

    g["pass_rate"] = g["passes"] / g["n"]
    if "cost_usd" in df.columns:
        g["cost_per_pass"] = g.apply(
            lambda r: (r["total_cost"] / r["passes"]) if r["passes"] > 0 else float("nan"),
            axis=1,
        )
    g = g.sort_values("pass_rate", ascending=True).reset_index(drop=True)

    # ── Compute Wilson CI per competitor ──
    def _ci(row):
        _, lo, hi = _wilson_ci(int(row["passes"]), int(row["n"]))
        return pd.Series({"ci_lo": lo, "ci_hi": hi})
    g[["ci_lo", "ci_hi"]] = g.apply(_ci, axis=1)

    has_cost = "cost_per_pass" in g.columns and g["cost_per_pass"].dropna().size

    # Layout
    if has_cost:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.6))
    else:
        fig, ax1 = plt.subplots(figsize=(5.4, 3.4))
        ax2 = None

    y = np.arange(len(g))
    cols = [_COMPETITOR_COLOURS.get(c, _COMPETITOR_COLOURS["other"])
            for c in g["competitor"]]
    labels = [r["competitor_name"] for _, r in g.iterrows()]

    # Panel a — pass rate with Wilson CIs
    xerr_lo = np.clip(g["pass_rate"] - g["ci_lo"], 0, None)
    xerr_hi = np.clip(g["ci_hi"] - g["pass_rate"], 0, None)
    ax1.barh(y, g["pass_rate"], color=cols, edgecolor="white",
             linewidth=0.6, height=0.68,
             xerr=[xerr_lo, xerr_hi], capsize=3,
             error_kw={"elinewidth": 0.8, "capthick": 0.8, "color": "#1f2937"})
    for i, (rate, ci_hi) in enumerate(zip(g["pass_rate"], g["ci_hi"])):
        x = min(ci_hi + 0.015, 1.14)
        ax1.text(x, i, f"{rate:.0%}",
                 va="center", ha="left", fontsize=8, color="#1f2937")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.set_xlim(0, 1.18)
    ax1.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax1.set_xlabel("Pass rate (95% Wilson CI)")
    ax1.set_title("a  Plan correctness" if has_cost else "Plan correctness",
                  loc="left")
    _style_value_axis(ax1, x=True)

    # Panel b — cost per successful plan
    if ax2 is not None:
        t = g.sort_values("cost_per_pass").reset_index(drop=True)
        yb = np.arange(len(t))
        cols2 = [_COMPETITOR_COLOURS.get(c, _COMPETITOR_COLOURS["other"])
                 for c in t["competitor"]]
        ax2.barh(yb, t["cost_per_pass"], color=cols2, edgecolor="white",
                 linewidth=0.6, height=0.68)
        for i, v in enumerate(t["cost_per_pass"]):
            if pd.isna(v):
                ax2.text(0.005, i, "n/a", va="center", ha="left",
                         fontsize=8, color="#9ca3af",
                         transform=ax2.get_yaxis_transform())
                continue
            label = (f"${v:,.4f}" if v < 0.01 else
                     f"${v:,.3f}" if v < 1 else
                     f"${v:,.2f}")
            ax2.text(v, i, f"  {label}", va="center", ha="left",
                     fontsize=8, color="#1f2937")
        ax2.set_yticks(yb)
        ax2.set_yticklabels([r["competitor_name"] for _, r in t.iterrows()])
        vmax = t["cost_per_pass"].dropna().max() if t["cost_per_pass"].dropna().size else 1
        ax2.set_xlim(0, vmax * 1.35 if vmax > 0 else 1)
        ax2.set_xlabel("USD per successful plan")
        ax2.set_title("b  Cost efficiency", loc="left")
        _style_value_axis(ax2, x=True)

    fig.suptitle("Head-to-head: FlowAgent vs. alternative agentic systems",
                 fontsize=11, fontweight="bold", y=1.03)
    return fig


def competitors_agentic_figure(df: pd.DataFrame) -> Optional[plt.Figure]:
    """Three-panel head-to-head of the agentic systems alone.

    Filters to flowagent / biomaster / autoba (excludes raw-LLM lanes) and
    renders:
      (a) Binary pass rate + Wilson 95% CI
      (b) Mean tool-completeness score — a partial-credit view where each
          cell contributes ``tools_present_fraction`` instead of 0/1.
          Addresses the reviewer comment that a 5-of-6 plan shouldn't
          score identically to a 0-of-6 plan.
      (c) Failure-mode composition — stacked bar of pass / plan_fail /
          error per system. Shows BioMaster's fragility pattern.

    Expects columns: competitor, competitor_name, overall_pass,
    tools_present_fraction (optional), error.
    """
    if "competitor" not in df.columns or df.empty:
        return None

    AGENTIC = {"flowagent", "biomaster", "autoba"}
    sub = df[df["competitor"].isin(AGENTIC)].copy()
    if sub.empty:
        return _empty_figure(
            "No flowagent/biomaster/autoba rows in the competitors sweep.\n"
            "Run: make competitors (with BIOMASTER_DIR / AUTOBA_DIR set)."
        )

    sub["overall_pass"] = sub["overall_pass"].map(
        lambda v: v if isinstance(v, (bool, int, float))
        else str(v).strip().lower() == "true"
    ).astype(int)

    # Classify each row. ``error`` in the merged CSV is NaN for successful
    # rows and sometimes the literal string "None" — both must be treated
    # as "no error". The naive ``if r.get("error"):`` check was truthy for
    # NaN and mis-labelled every cell as an upstream crash.
    def _has_error(err) -> bool:
        if err is None:
            return False
        if isinstance(err, float) and pd.isna(err):
            return False
        s = str(err).strip()
        return bool(s) and s.lower() not in ("none", "nan", "null")

    def _label(r):
        if _has_error(r.get("error")):
            return "error"
        if r["overall_pass"]:
            return "pass"
        return "plan_fail"
    sub["outcome"] = sub.apply(_label, axis=1)

    # Pretty names + consistent ordering
    NAME_MAP = {"flowagent": "FlowAgent", "biomaster": "BioMaster", "autoba": "AutoBA"}
    COLOURS  = {"flowagent": "#0072B2", "biomaster": "#E69F00", "autoba": "#009E73"}
    order = [c for c in ("flowagent", "autoba", "biomaster") if c in sub["competitor"].unique()]

    # ── Panel A: pass rate + Wilson CI ─────────────────────────────
    agg = (sub.groupby("competitor")
              .agg(n=("overall_pass", "size"),
                   passes=("overall_pass", "sum"))
              .reindex(order))
    agg["pass_rate"] = agg["passes"] / agg["n"]
    cis = [_wilson_ci(int(r.passes), int(r.n)) for r in agg.itertuples()]
    agg["lo"] = [c[1] for c in cis]
    agg["hi"] = [c[2] for c in cis]

    # ── Panel B: partial-credit tool-completeness ─────────────────
    has_frac = "tools_present_fraction" in sub.columns
    if has_frac:
        sub["tools_present_fraction"] = pd.to_numeric(
            sub["tools_present_fraction"], errors="coerce"
        )
        # Crashed cells have no plan and therefore no fraction — score as 0
        # so they don't inflate the partial-credit view.
        sub.loc[sub["outcome"] == "error", "tools_present_fraction"] = 0.0
        tools_mean = sub.groupby("competitor")["tools_present_fraction"].mean().reindex(order)
    else:
        tools_mean = None

    # ── Panel C: outcome composition ───────────────────────────────
    counts = (sub.groupby(["competitor", "outcome"])
                  .size().unstack(fill_value=0)
                  .reindex(order)
                  .reindex(columns=["pass", "plan_fail", "error"], fill_value=0))
    counts_frac = counts.div(counts.sum(axis=1), axis=0)

    # ── Layout ─────────────────────────────────────────────────────
    n_panels = 3 if has_frac else 2
    fig_w = 10.5 if has_frac else 7.5
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 4.0),
                             gridspec_kw={"wspace": 0.38})
    if n_panels == 2:
        ax_pr, ax_stack = axes
        ax_tpf = None
    else:
        ax_pr, ax_tpf, ax_stack = axes

    x = np.arange(len(order))
    bar_colours = [COLOURS[c] for c in order]
    labels = [NAME_MAP[c] for c in order]

    # Panel A
    ax_pr.bar(x, agg["pass_rate"].values, color=bar_colours,
              edgecolor="#111827", linewidth=0.7, width=0.7)
    err_lo = agg["pass_rate"].values - agg["lo"].values
    err_hi = agg["hi"].values - agg["pass_rate"].values
    ax_pr.errorbar(x, agg["pass_rate"].values,
                   yerr=[err_lo, err_hi],
                   fmt="none", ecolor="#111827", capsize=4, linewidth=1)
    for xi, (pr, n, p) in enumerate(zip(agg["pass_rate"], agg["n"], agg["passes"])):
        ax_pr.text(xi, pr + 0.035, f"{pr:.0%}\n({int(p)}/{int(n)})",
                   ha="center", va="bottom", fontsize=9, fontweight="semibold")
    ax_pr.set_xticks(x)
    ax_pr.set_xticklabels(labels, fontsize=10)
    ax_pr.set_ylim(0, 1.12)
    ax_pr.set_ylabel("Pass rate (binary)", fontsize=10)
    ax_pr.set_title("a  Plan correctness (binary pass/fail)",
                    loc="left", fontsize=11, fontweight="bold")
    for side in ("top", "right"): ax_pr.spines[side].set_visible(False)
    ax_pr.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax_pr.grid(axis="y", linestyle=":", color="#d1d5db", alpha=0.6)
    ax_pr.set_axisbelow(True)

    # Panel B — partial credit
    if ax_tpf is not None and tools_mean is not None:
        ax_tpf.bar(x, tools_mean.values, color=bar_colours,
                   edgecolor="#111827", linewidth=0.7, width=0.7)
        for xi, v in enumerate(tools_mean.values):
            if np.isnan(v):
                continue
            ax_tpf.text(xi, v + 0.035, f"{v:.0%}",
                        ha="center", va="bottom", fontsize=9, fontweight="semibold")
        ax_tpf.set_xticks(x)
        ax_tpf.set_xticklabels(labels, fontsize=10)
        ax_tpf.set_ylim(0, 1.12)
        ax_tpf.set_ylabel("Mean tool-completeness", fontsize=10)
        ax_tpf.set_title("b  Partial credit (mean expected-tool recovery)",
                         loc="left", fontsize=11, fontweight="bold")
        for side in ("top", "right"): ax_tpf.spines[side].set_visible(False)
        ax_tpf.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax_tpf.grid(axis="y", linestyle=":", color="#d1d5db", alpha=0.6)
        ax_tpf.set_axisbelow(True)

    # Panel C — stacked outcome composition
    OUTCOME_COLOURS = {
        "pass":      "#15803d",
        "plan_fail": "#E69F00",
        "error":     "#b91c1c",
    }
    OUTCOME_LABELS = {
        "pass":      "Pass",
        "plan_fail": "Plan failed (rubric)",
        "error":     "Upstream crash",
    }
    bottom = np.zeros(len(order))
    for out_key in ("pass", "plan_fail", "error"):
        vals = counts_frac[out_key].values
        ax_stack.bar(
            x, vals, bottom=bottom,
            color=OUTCOME_COLOURS[out_key],
            edgecolor="white", linewidth=0.7, width=0.7,
            label=OUTCOME_LABELS[out_key],
        )
        # Annotate non-trivial segments
        for xi, v in enumerate(vals):
            if v > 0.04:
                ax_stack.text(
                    xi, bottom[xi] + v / 2,
                    f"{int(counts[out_key].iloc[xi])}",
                    ha="center", va="center", fontsize=8.5,
                    fontweight="semibold",
                    color=("#111827" if out_key == "plan_fail" else "white"),
                )
        bottom += vals
    ax_stack.set_xticks(x)
    ax_stack.set_xticklabels(labels, fontsize=10)
    ax_stack.set_ylim(0, 1.0)
    ax_stack.set_ylabel("Fraction of cells", fontsize=10)
    panel_letter = "c" if has_frac else "b"
    ax_stack.set_title(f"{panel_letter}  Failure mode composition",
                       loc="left", fontsize=11, fontweight="bold")
    ax_stack.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax_stack.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28),
                    ncol=3, fontsize=8.5, frameon=False)
    for side in ("top", "right"): ax_stack.spines[side].set_visible(False)
    ax_stack.grid(axis="y", linestyle=":", color="#d1d5db", alpha=0.6)
    ax_stack.set_axisbelow(True)

    fig.suptitle(
        "Head-to-head across agentic bioinformatics systems  —  "
        "FlowAgent vs. AutoBA vs. BioMaster",
        fontsize=11.5, fontweight="bold", x=0.02, ha="left",
    )
    fig.subplots_adjust(bottom=0.22, top=0.86)
    return fig


def competitors_perprompt_figure(df: pd.DataFrame) -> Optional[plt.Figure]:
    """Heatmap: competitor × prompt pass rate (averaged over replicates)."""
    if "competitor" not in df.columns or df.empty:
        return None
    df = df.copy()
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"].astype(str) == "")]
        if df.empty:
            return None
    df["overall_pass"] = df["overall_pass"].map(
        lambda v: v if isinstance(v, (bool, int, float))
        else str(v).strip().lower() == "true"
    ).astype(int)

    pivot = df.pivot_table(index="input_id", columns="competitor",
                           values="overall_pass", aggfunc="mean")
    if pivot.empty:
        return None

    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index,
                      pivot.mean(axis=0).sort_values(ascending=True).index]

    fig_w = max(4.5, 0.9 * pivot.shape[1] + 2.0)
    fig_h = max(3.0, 0.4 * pivot.shape[0] + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(pivot.values, aspect="auto", cmap=_HEATMAP_CMAP,
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right", fontsize=9)
    display = [n.removeprefix("hard_").replace("_", " ").title()
               for n in pivot.index]
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(display, fontsize=8.5)
    ax.set_xticks(np.arange(-.5, pivot.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, pivot.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)

    # No per-cell annotations — colour alone encodes the value.

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.015,
                        ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=2, labelsize=8)
    cbar.set_label("Pass rate", fontsize=9)

    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)
    ax.set_title("Per-prompt pass rate by competitor", loc="left")
    return fig


# ── Benchmark C: generator fidelity ──────────────────────────────

def generation_figure(df: pd.DataFrame) -> plt.Figure:
    """Pass/fail matrix of (preset × generator × check)."""
    if "generator" not in df.columns or "plan_id" not in df.columns:
        return _empty_figure("No generation data.")

    checks = [c for c in (
        "validation_ok", "step_count_matches", "dag_isomorphic",
        "tools_preserved", "regression_launchdir_quoted",
    ) if c in df.columns]
    if not checks:
        return _empty_figure("No generator checks present.")

    check_labels = {
        "validation_ok":                "Plan validates",
        "step_count_matches":           "Step count matches",
        "dag_isomorphic":               "DAG isomorphic",
        "tools_preserved":              "Tools preserved",
        "regression_launchdir_quoted":  "Launch dir quoted",
    }

    df = df.copy()
    df["row"] = df["plan_id"] + " × " + df["generator"]
    heat = df.set_index("row")[checks].astype(float)

    fig_w = max(5.0, 0.7 * len(checks) + 3.4)
    fig_h = max(2.4, 0.33 * len(heat) + 1.2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(heat.values, aspect="auto", cmap=_HEATMAP_CMAP,
                   vmin=0, vmax=1, interpolation="nearest")

    ax.set_xticks(range(len(checks)))
    ax.set_xticklabels([check_labels.get(c, c) for c in checks],
                       rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(heat)))
    ax.set_yticklabels(heat.index, fontsize=8)

    # Cell markers — use glyphs present in Arial / Helvetica to avoid PDF
    # font-embedding warnings ("●" U+25CF and "×" U+00D7 are in every core
    # font; "✓" U+2713 is not in Arial).
    # No per-cell annotations — colour alone encodes the value.

    ax.set_xticks(np.arange(-.5, heat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, heat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)

    ax.set_title("Generator fidelity", loc="left")
    return fig


# ── Benchmark D: executor coverage ───────────────────────────────

def executor_matrix_figure(df: pd.DataFrame) -> plt.Figure:
    if "executor" not in df.columns:
        return _empty_figure("No executor data.")

    levels = [c for c in ("interface_ok", "mock_ok", "live_ok")
              if c in df.columns]
    if not levels:
        return _empty_figure("No executor levels present.")

    level_labels = {
        "interface_ok": "Interface",
        "mock_ok":      "Mock run",
        "live_ok":      "Live run",
    }
    mat = df.set_index("executor")[levels]

    def _score(v):
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("true", "1"):  return 1.0
            if s in ("false", "0"): return 0.0
            return 0.5
        if v is True or v == 1:  return 1.0
        if v is False or v == 0: return 0.0
        return 0.5

    mat_num = mat.map(_score) if hasattr(mat, "map") else mat.applymap(_score)

    fig_h = max(2.2, 0.42 * len(mat) + 0.8)
    fig, ax = plt.subplots(figsize=(5.0, fig_h))
    im = ax.imshow(mat_num.values, aspect="auto", cmap=_HEATMAP_CMAP,
                   vmin=0, vmax=1, interpolation="nearest")

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([level_labels.get(l, l) for l in levels], fontsize=8.5)
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index, fontsize=8.5)

    # No per-cell annotations — colour alone encodes the value.

    ax.set_xticks(np.arange(-.5, mat_num.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, mat_num.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)

    ax.set_title("Executor coverage", loc="left")
    return fig


# ── Cost analyses ────────────────────────────────────────────────

def _per_model_cost_table(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Aggregate per-model cost, token counts, pass-rate, cost-per-pass.

    Returns ``None`` if the dataframe has no cost column (e.g. mock run or
    pre-token-tracking data).
    """
    if "cost_usd" not in df.columns:
        return None
    df = df.copy()
    for c in ("prompt_tokens", "completion_tokens", "cost_usd"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["overall_pass"] = df["overall_pass"].map(
        lambda v: v if isinstance(v, (bool, int, float))
        else str(v).strip().lower() == "true"
    ).astype(int)

    g = (df.groupby("model")
            .agg(n=("overall_pass", "count"),
                 passes=("overall_pass", "sum"),
                 total_cost=("cost_usd", "sum"),
                 mean_cost=("cost_usd", "mean"),
                 mean_in=("prompt_tokens", "mean"),
                 mean_out=("completion_tokens", "mean"))
            .reset_index())
    if g.empty or g["total_cost"].sum() == 0:
        return None

    g["pass_rate"]         = g["passes"] / g["n"]
    # Cost per successful plan (∞ if zero passes → use NaN)
    g["cost_per_pass"] = g.apply(
        lambda r: (r["total_cost"] / r["passes"]) if r["passes"] > 0 else float("nan"),
        axis=1,
    )
    g["cost_per_100_plans"] = g["mean_cost"] * 100
    g = g.sort_values("cost_per_pass").reset_index(drop=True)
    return g


def cost_summary_figure(df: pd.DataFrame) -> Optional[plt.Figure]:
    """Two-panel cost view: (a) $ per 100 plans, (b) $ per successful plan.

    Bars sorted by cost-per-pass (cheapest-per-successful-plan first) and
    coloured by provider. Returns ``None`` if cost data is absent.
    """
    table = _per_model_cost_table(df)
    if table is None or table.empty:
        return None

    n_models = len(table)
    panel_h  = max(3.2, 0.28 * n_models + 1.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, panel_h), sharey=True)

    def _bar(ax, col: str, title: str, unit: str):
        t = table.sort_values(col).reset_index(drop=True)
        y = np.arange(len(t))
        cols = [_PROVIDER_COLOURS.get(_provider_from_model(m), "#6b7280")
                for m in t["model"]]
        ax.barh(y, t[col], color=cols, edgecolor="white",
                linewidth=0.6, height=0.72)
        ax.set_yticks(y)
        ax.set_yticklabels([_short_name(m) for m in t["model"]])
        for i, v in enumerate(t[col]):
            if pd.isna(v):
                ax.text(0.005, i, "n/a", va="center", ha="left",
                        fontsize=7.5, color="#9ca3af", transform=ax.get_yaxis_transform())
                continue
            label = (f"${v:,.3f}" if v < 1 else
                     f"${v:,.2f}" if v < 100 else f"${v:,.0f}")
            ax.text(v, i, f" {label}", va="center", ha="left",
                    fontsize=7.5, color="#1f2937")
        ax.set_xlabel(f"{title}  ({unit})")
        ax.set_title(title, loc="left")
        # Leave headroom for annotations
        vmax = t[col].dropna().max() if t[col].dropna().size else 1
        ax.set_xlim(0, vmax * 1.35 if vmax > 0 else 1)
        _style_value_axis(ax, x=True)

    _bar(ax1, "cost_per_100_plans", "a  Cost per 100 plans", "USD")
    _bar(ax2, "cost_per_pass",      "b  Cost per successful plan", "USD")

    providers = sorted(
        {_provider_from_model(m) for m in table["model"]},
        key=lambda p: list(_PROVIDER_COLOURS).index(p)
        if p in _PROVIDER_COLOURS else 99,
    )
    _add_provider_legend(fig, providers, bbox_to_anchor=(0.5, 1.015))
    fig.suptitle("Per-model cost benchmark", fontsize=11,
                 fontweight="bold", y=1.06)
    return fig


def cost_vs_quality_figure(df: pd.DataFrame) -> Optional[plt.Figure]:
    """Scatter of pass-rate vs. either mean wall time or mean cost.

    Prefers cost on the x-axis when available (more actionable than
    latency); falls back to wall time otherwise. Returns ``None`` if
    neither column is present.
    """
    if "overall_pass" not in df.columns or df.empty:
        return None

    df = df.copy()
    df["overall_pass"] = df["overall_pass"].map(
        lambda v: v if isinstance(v, (bool, int, float))
        else str(v).strip().lower() == "true"
    ).astype(int)

    # Prefer cost on the x-axis if available and non-zero
    use_cost = (
        "cost_usd" in df.columns and
        pd.to_numeric(df["cost_usd"], errors="coerce").fillna(0).sum() > 0
    )
    x_col    = "cost_usd" if use_cost else "wall_seconds"
    x_label  = ("Mean cost per plan (USD)" if use_cost
                else "Mean wall time per plan (s)")
    if x_col not in df.columns or df[x_col].isna().all():
        return None

    df[x_col] = pd.to_numeric(df[x_col], errors="coerce").fillna(0.0)
    g = (df.groupby("model")
            .agg(rate=("overall_pass", "mean"),
                 x=(x_col, "mean"),
                 n=("overall_pass", "count"))
            .reset_index())
    if g.empty:
        return None

    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    for _, row in g.iterrows():
        prov   = _provider_from_model(row["model"])
        colour = _PROVIDER_COLOURS.get(prov, "#6b7280")
        ax.scatter(row["x"], row["rate"], s=60, color=colour,
                   edgecolor="white", linewidth=0.8, zorder=3)

    # Pareto frontier: highest pass-rate at each cost/wall budget (left-to-right)
    frontier = g.sort_values("x").copy()
    best_rate = -1.0
    is_frontier = []
    for _, r in frontier.iterrows():
        on = r["rate"] > best_rate
        if on: best_rate = r["rate"]
        is_frontier.append(on)
    frontier["frontier"] = is_frontier

    must_label = {
        g.loc[g["x"].idxmin(), "model"],
        g.loc[g["rate"].idxmax(), "model"],
    }

    placed: List[Tuple[float, float]] = []
    x_range = max(g["x"].max() - g["x"].min(), 1e-9)
    for _, row in frontier.iterrows():
        if not (row["frontier"] or row["model"] in must_label):
            continue
        dy = 8
        for px, _ in placed:
            if abs(px - row["x"]) < 0.15 * x_range:
                dy = -10 if dy > 0 else 12
        ax.annotate(_short_name(row["model"]),
                    (row["x"], row["rate"]),
                    xytext=(6, dy), textcoords="offset points",
                    fontsize=7.5, color="#1f2937",
                    arrowprops=dict(arrowstyle="-", color="#9ca3af",
                                    linewidth=0.6, shrinkA=0, shrinkB=3))
        placed.append((row["x"], row["rate"]))

    ax.set_xlabel(x_label)
    ax.set_ylabel("Pass rate")

    if use_cost:
        ax.set_xscale("log")
        # Format as USD on a log scale
        import matplotlib.ticker as mtick
        def _fmt(x, _):
            if x <= 0: return ""
            if x >= 0.01: return f"${x:.2f}"
            if x >= 0.001: return f"${x:.3f}"
            return f"${x:.4f}"
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(_fmt))

    y_lo = max(0.0, g["rate"].min() - 0.08)
    y_hi = min(1.02, g["rate"].max() + 0.06)
    ax.set_ylim(y_lo, y_hi)
    span  = y_hi - y_lo
    step  = 0.05 if span < 0.3 else 0.1 if span < 0.6 else 0.25
    ticks = np.arange(0, 1.01, step)
    ax.set_yticks([t for t in ticks if y_lo <= t <= y_hi])
    ax.set_yticklabels([f"{t:.0%}" for t in ticks if y_lo <= t <= y_hi])
    ax.grid(True, linestyle="-", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)

    providers_present = sorted(
        {_provider_from_model(m) for m in g["model"]},
        key=lambda p: list(_PROVIDER_COLOURS).index(p)
        if p in _PROVIDER_COLOURS else 99,
    )
    _add_provider_legend(fig, providers_present,
                         loc="upper right",
                         bbox_to_anchor=(0.985, 0.985))

    ax.set_title("Pass rate vs cost" if use_cost else "Pass rate vs latency",
                 loc="left")
    return fig


# ── Wall-clock latency ───────────────────────────────────────────

def latency_figure(df: pd.DataFrame) -> Optional[plt.Figure]:
    """Two-panel wall-clock latency comparison across models.

    Panel a: sorted horizontal bar chart of median wall-time per model
    with IQR error bars, provider-coloured.
    Panel b: scatter of pass rate vs median wall-time (log-x) with
    Pareto frontier annotated — the speed/quality trade-off.

    Pairs with ``planning_cost_quality.pdf``: that figure answers
    "what does it cost?", this one answers "is it fast enough?".
    Returns ``None`` if ``wall_seconds`` is missing or all-zero.
    """
    if "wall_seconds" not in df.columns or df.empty:
        return None

    df = df.copy()
    df["wall_seconds"] = pd.to_numeric(df["wall_seconds"], errors="coerce")
    df = df[df["wall_seconds"].notna() & (df["wall_seconds"] > 0)]
    if df.empty:
        return None

    if "overall_pass" in df.columns:
        df["overall_pass"] = df["overall_pass"].map(
            lambda v: v if isinstance(v, (bool, int, float))
            else str(v).strip().lower() == "true"
        ).astype(int)

    g = (df.groupby("model")["wall_seconds"]
           .agg(median="median",
                q1=lambda s: float(np.percentile(s, 25)),
                q3=lambda s: float(np.percentile(s, 75)),
                count="count")
           .reset_index())
    if "overall_pass" in df.columns:
        rates = (df.groupby("model")["overall_pass"]
                   .mean().rename("rate").reset_index())
        g = g.merge(rates, on="model", how="left")
    else:
        g["rate"] = np.nan
    g = g.sort_values("median", ascending=True).reset_index(drop=True)

    fig_h = max(3.4, 0.26 * len(g) + 1.4)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.0, fig_h),
        gridspec_kw={"width_ratios": [1.0, 1.05], "wspace": 0.32},
    )

    # Panel a — sorted bar chart with IQR error bars
    y    = np.arange(len(g))
    cols = [_PROVIDER_COLOURS.get(_provider_from_model(m), "#6b7280")
            for m in g["model"]]
    err_lo = (g["median"] - g["q1"]).clip(lower=0).to_numpy()
    err_hi = (g["q3"] - g["median"]).clip(lower=0).to_numpy()
    ax1.barh(y, g["median"], xerr=[err_lo, err_hi],
             color=cols, edgecolor="white", linewidth=0.6, height=0.72,
             capsize=2, error_kw={"elinewidth": 0.8, "capthick": 0.8,
                                  "color": "#1f2937"})
    for i, (m, q3) in enumerate(zip(g["median"], g["q3"])):
        ax1.text(q3 + 0.02 * g["q3"].max(), i, f"{m:.0f}s",
                 va="center", ha="left", fontsize=7.5, color="#1f2937")

    ax1.set_yticks(y)
    ax1.set_yticklabels([_short_name(m) for m in g["model"]])
    ax1.set_xlabel("Median wall time per plan (s)  ·  IQR")
    ax1.set_title("a  Latency per model", loc="left", fontweight="bold")
    ax1.set_xlim(0, g["q3"].max() * 1.18)
    _style_value_axis(ax1, x=True)

    # Panel b — pass rate vs latency scatter with Pareto frontier highlighted
    if g["rate"].notna().any():
        plot_df = g.dropna(subset=["rate"]).copy()
        for _, row in plot_df.iterrows():
            prov   = _provider_from_model(row["model"])
            colour = _PROVIDER_COLOURS.get(prov, "#6b7280")
            ax2.scatter(row["median"], row["rate"], s=60, color=colour,
                        edgecolor="white", linewidth=0.8, zorder=3)

        # Frontier: lowest latency at each new-best pass-rate threshold
        frontier = plot_df.sort_values("median").copy()
        best_rate = -1.0
        is_frontier = []
        for _, r in frontier.iterrows():
            on = r["rate"] > best_rate
            if on:
                best_rate = r["rate"]
            is_frontier.append(on)
        frontier["frontier"] = is_frontier
        on_frontier = set(frontier.loc[frontier["frontier"], "model"])

        # Linear x when range spans < 1 decade — log compression hurts here
        x_lo, x_hi = plot_df["median"].min(), plot_df["median"].max()
        use_log = (x_hi / max(x_lo, 1e-9)) >= 10.0
        if use_log:
            ax2.set_xscale("log")
            ax2.set_xlabel("Median wall time per plan (s, log)")
        else:
            pad = 0.05 * max(x_hi - x_lo, 1.0)
            ax2.set_xlim(x_lo - pad, x_hi + pad)
            ax2.set_xlabel("Median wall time per plan (s)")
        ax2.xaxis.set_major_formatter(mtick.FuncFormatter(
            lambda v, _: f"{v:.0f}s"))

        ax2.set_ylabel("Pass rate")
        ax2.set_title("b  Speed vs quality trade-off",
                      loc="left", fontweight="bold")

        # Highlight frontier points with a heavier ring
        front_pts = plot_df[plot_df["model"].isin(on_frontier)]
        ax2.scatter(front_pts["median"], front_pts["rate"], s=110,
                    facecolors="none", edgecolor="#1f2937",
                    linewidth=1.2, zorder=4)

        y_lo = max(0.0, plot_df["rate"].min() - 0.04)
        y_hi = min(1.02, plot_df["rate"].max() + 0.04)
        ax2.set_ylim(y_lo, y_hi)
        span = y_hi - y_lo
        step = 0.02 if span < 0.15 else 0.05 if span < 0.3 else 0.1
        ticks = np.arange(0, 1.01, step)
        ax2.set_yticks([t for t in ticks if y_lo <= t <= y_hi])
        ax2.set_yticklabels([f"{t:.0%}" for t in ticks if y_lo <= t <= y_hi])
        ax2.grid(True, linestyle="-", linewidth=0.6, alpha=0.35)
        ax2.set_axisbelow(True)

        # Selective labelling — labelling all 32 models overcrowds the
        # panel. We mark: every frontier point, the slowest model, the
        # lowest-pass-rate model, and one representative per pass-rate
        # tier per provider (whichever sits at the latency extreme).
        to_label = set(on_frontier)
        to_label.add(plot_df.loc[plot_df["median"].idxmax(), "model"])
        to_label.add(plot_df.loc[plot_df["rate"].idxmin(), "model"])

        # Per (provider, rate-tier) pick the fastest + slowest representative
        rate_bin = (plot_df["rate"] * 200).round().astype(int)  # 0.5% bins
        plot_df["_rb"] = rate_bin
        for (_prov, _rb), sub in plot_df.groupby(
                [plot_df["model"].map(_provider_from_model), "_rb"]):
            to_label.add(sub.loc[sub["median"].idxmin(), "model"])
            to_label.add(sub.loc[sub["median"].idxmax(), "model"])

        x_range = max(x_hi - x_lo, 1.0)
        bin_w   = 0.05 * x_range
        labelled = plot_df[plot_df["model"].isin(to_label)].sort_values(
            ["rate", "median"], ascending=[False, True])
        buckets: Dict[int, int] = {}
        for _, row in labelled.iterrows():
            bk = int((row["median"] - x_lo) / max(bin_w, 1e-9))
            slot = buckets.get(bk, 0)
            buckets[bk] = slot + 1
            # Alternate above / below; spread further on later slots
            offsets = [(8, 8), (8, -12), (-8, 8), (-8, -12),
                       (10, 18), (10, -20), (-10, 18), (-10, -20)]
            dx, dy = offsets[slot % len(offsets)]
            ha = "left" if dx > 0 else "right"
            on_front = row["model"] in on_frontier
            ax2.annotate(_short_name(row["model"]),
                         (row["median"], row["rate"]),
                         xytext=(dx, dy), textcoords="offset points",
                         fontsize=7.0, ha=ha,
                         color="#111827" if on_front else "#4b5563",
                         fontweight="bold" if on_front else "normal",
                         arrowprops=dict(arrowstyle="-", color="#9ca3af",
                                         linewidth=0.5, shrinkA=0, shrinkB=2))
        plot_df.drop(columns="_rb", inplace=True, errors="ignore")
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "No pass-rate data", ha="center", va="center",
                 fontsize=10, color="#6b7280", transform=ax2.transAxes)

    providers = sorted(
        {_provider_from_model(m) for m in g["model"]},
        key=lambda p: list(_PROVIDER_COLOURS).index(p)
        if p in _PROVIDER_COLOURS else 99,
    )
    _add_provider_legend(fig, providers, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Wall-clock latency", fontsize=11,
                 fontweight="bold", y=1.06)
    return fig


# ── Turns-to-completion ──────────────────────────────────────────

def turns_to_completion_figure(df: pd.DataFrame) -> Optional[plt.Figure]:
    """Per-model bar chart of mean LLM calls per plan (with std bars).

    Useful complement to pass-rate: two models can have identical pass
    rates but very different turn counts, which dominate cost + latency.
    """
    if "llm_calls" not in df.columns or df.empty:
        return None
    df = df.copy()
    df["llm_calls"] = pd.to_numeric(df["llm_calls"], errors="coerce")
    df = df[df["llm_calls"].notna() & (df["llm_calls"] > 0)]
    if df.empty:
        return None

    g = (df.groupby("model")["llm_calls"]
           .agg(["mean", "std", "count"]).reset_index())
    g = g.sort_values("mean").reset_index(drop=True)

    fig_h = max(3.0, 0.26 * len(g) + 1.0)
    fig, ax = plt.subplots(figsize=(6.0, fig_h))

    y = np.arange(len(g))
    cols = [_PROVIDER_COLOURS.get(_provider_from_model(m), "#6b7280")
            for m in g["model"]]
    ax.barh(y, g["mean"], xerr=g["std"].fillna(0),
            color=cols, edgecolor="white", linewidth=0.6, height=0.72,
            capsize=2, error_kw={"elinewidth": 0.8, "capthick": 0.8,
                                 "color": "#1f2937"})
    for i, (m, s) in enumerate(zip(g["mean"], g["std"].fillna(0))):
        ax.text(m + s + 0.05, i, f"{m:.1f}", va="center", ha="left",
                fontsize=7.5, color="#1f2937")

    ax.set_yticks(y)
    ax.set_yticklabels([_short_name(m) for m in g["model"]])
    ax.set_xlabel("Mean LLM calls per plan (±1 SD)")
    ax.set_title("Turns to completion", loc="left")
    ax.set_xlim(0, (g["mean"] + g["std"].fillna(0)).max() * 1.22)
    _style_value_axis(ax, x=True)

    providers = sorted(
        {_provider_from_model(m) for m in g["model"]},
        key=lambda p: list(_PROVIDER_COLOURS).index(p)
        if p in _PROVIDER_COLOURS else 99,
    )
    _add_provider_legend(fig, providers, bbox_to_anchor=(0.5, 1.04))
    return fig


# ── Consistency across replicates ────────────────────────────────

def consistency_figure(df: pd.DataFrame) -> Optional[plt.Figure]:
    """Per-model agreement: fraction of (model, prompt) cells where all
    replicates yield the same ``overall_pass`` outcome.

    Low agreement at a given model indicates non-determinism / flakiness
    even when the mean pass rate is high. Returns ``None`` if the run
    only has one replicate per cell.
    """
    if df.empty or "overall_pass" not in df.columns or "replicate" not in df.columns:
        return None

    df = df.copy()
    df["overall_pass"] = df["overall_pass"].map(
        lambda v: v if isinstance(v, (bool, int, float))
        else str(v).strip().lower() == "true"
    ).astype(int)

    grp = df.groupby(["model", "input_id"])["overall_pass"]
    stats = grp.agg(["nunique", "count"]).reset_index()
    if (stats["count"] <= 1).all():
        return None  # only one replicate per cell — consistency undefined

    # ``nunique`` == 1 means all replicates agreed (either all pass or all fail)
    stats["unanimous"] = (stats["nunique"] == 1).astype(int)
    agree = (stats.groupby("model")["unanimous"]
                  .agg(["sum", "count"]).reset_index())
    agree["rate"] = agree["sum"] / agree["count"]
    agree = agree.sort_values("rate", ascending=True).reset_index(drop=True)

    fig_h = max(3.0, 0.26 * len(agree) + 1.0)
    fig, ax = plt.subplots(figsize=(6.0, fig_h))
    y = np.arange(len(agree))
    cols = [_PROVIDER_COLOURS.get(_provider_from_model(m), "#6b7280")
            for m in agree["model"]]
    ax.barh(y, agree["rate"], color=cols, edgecolor="white",
            linewidth=0.6, height=0.72)
    for i, v in enumerate(agree["rate"]):
        x = min(v + 0.015, 0.99)
        ax.text(x, i, f"{v:.0%}", va="center", ha="left",
                fontsize=7.5, color="#1f2937")

    ax.set_yticks(y)
    ax.set_yticklabels([_short_name(m) for m in agree["model"]])
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Fraction of prompts with unanimous replicate outcome")
    ax.set_title("Inter-replicate consistency", loc="left")
    _style_value_axis(ax, x=True)

    providers = sorted(
        {_provider_from_model(m) for m in agree["model"]},
        key=lambda p: list(_PROVIDER_COLOURS).index(p)
        if p in _PROVIDER_COLOURS else 99,
    )
    _add_provider_legend(fig, providers, bbox_to_anchor=(0.5, 1.04))
    return fig


# ── Hallucination rate ───────────────────────────────────────────

def hallucination_figure(df: pd.DataFrame) -> Optional[plt.Figure]:
    """Per-model hallucinated-tool fraction.

    Two panels:
      (a) fraction of plans with ≥1 hallucinated tool
      (b) mean hallucination rate per plan (hallucinated / total tools)

    Returns ``None`` if the hallucination metric is absent.
    """
    if "hallucination_rate" not in df.columns and "num_hallucinated_tools" not in df.columns:
        return None

    df = df.copy()
    if "num_hallucinated_tools" in df.columns:
        df["num_hallucinated_tools"] = pd.to_numeric(
            df["num_hallucinated_tools"], errors="coerce").fillna(0)
        df["any_hallucination"] = (df["num_hallucinated_tools"] > 0).astype(int)
    else:
        df["any_hallucination"] = 0
    if "hallucination_rate" in df.columns:
        df["hallucination_rate"] = pd.to_numeric(
            df["hallucination_rate"], errors="coerce").fillna(0)
    else:
        df["hallucination_rate"] = 0

    g = (df.groupby("model")
            .agg(frac_plans=("any_hallucination", "mean"),
                 mean_rate=("hallucination_rate", "mean"))
            .reset_index())
    if g.empty or (g["frac_plans"].sum() == 0 and g["mean_rate"].sum() == 0):
        return None

    g = g.sort_values("frac_plans").reset_index(drop=True)

    fig_h = max(3.2, 0.26 * len(g) + 1.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, fig_h), sharey=True)

    y = np.arange(len(g))
    cols = [_PROVIDER_COLOURS.get(_provider_from_model(m), "#6b7280")
            for m in g["model"]]

    ax1.barh(y, g["frac_plans"], color=cols, edgecolor="white",
             linewidth=0.6, height=0.72)
    for i, v in enumerate(g["frac_plans"]):
        ax1.text(v + 0.005, i, f"{v:.0%}", va="center", ha="left",
                 fontsize=7.5, color="#1f2937")
    ax1.set_yticks(y)
    ax1.set_yticklabels([_short_name(m) for m in g["model"]])
    ax1.set_xlim(0, max(g["frac_plans"].max() * 1.35, 0.05))
    ax1.set_xlabel("Fraction of plans with a hallucinated tool")
    ax1.set_title("a  Any hallucination", loc="left")
    _style_value_axis(ax1, x=True)

    ax2.barh(y, g["mean_rate"], color=cols, edgecolor="white",
             linewidth=0.6, height=0.72)
    for i, v in enumerate(g["mean_rate"]):
        ax2.text(v + 0.002, i, f"{v:.1%}", va="center", ha="left",
                 fontsize=7.5, color="#1f2937")
    ax2.set_xlim(0, max(g["mean_rate"].max() * 1.35, 0.02))
    ax2.set_xlabel("Mean hallucinated-tool fraction per plan")
    ax2.set_title("b  Mean rate per plan", loc="left")
    _style_value_axis(ax2, x=True)

    providers = sorted(
        {_provider_from_model(m) for m in g["model"]},
        key=lambda p: list(_PROVIDER_COLOURS).index(p)
        if p in _PROVIDER_COLOURS else 99,
    )
    _add_provider_legend(fig, providers, bbox_to_anchor=(0.5, 1.015))
    fig.suptitle("Tool hallucination benchmark", fontsize=11,
                 fontweight="bold", y=1.06)
    return fig


# ── Token usage ──────────────────────────────────────────────────

def token_usage_figure(df: pd.DataFrame) -> Optional[plt.Figure]:
    """Per-model token spend: stacked input+output bars, sorted cheap→expensive.

    The two-panel layout shows:
      (a) mean prompt / completion tokens per plan (stacked)
      (b) tokens per successful plan (dividing by pass rate)

    Returns ``None`` if token columns are absent.
    """
    if "prompt_tokens" not in df.columns or "completion_tokens" not in df.columns:
        return None

    df = df.copy()
    for c in ("prompt_tokens", "completion_tokens", "llm_calls"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["overall_pass"] = df["overall_pass"].map(
        lambda v: v if isinstance(v, (bool, int, float))
        else str(v).strip().lower() == "true"
    ).astype(int)

    g = (df.groupby("model")
            .agg(in_mean=("prompt_tokens", "mean"),
                 out_mean=("completion_tokens", "mean"),
                 n=("overall_pass", "count"),
                 passes=("overall_pass", "sum"))
            .reset_index())
    if g.empty or (g["in_mean"].sum() == 0 and g["out_mean"].sum() == 0):
        return None

    g["total"]         = g["in_mean"] + g["out_mean"]
    g["pass_rate"]     = g["passes"] / g["n"]
    g["tokens_per_pass"] = g.apply(
        lambda r: r["total"] / r["pass_rate"] if r["pass_rate"] > 0 else float("nan"),
        axis=1,
    )
    g = g.sort_values("total").reset_index(drop=True)

    fig_h = max(3.2, 0.26 * len(g) + 1.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, fig_h), sharey=True)

    y = np.arange(len(g))
    labels = [_short_name(m) for m in g["model"]]

    # Panel a: stacked input/output
    IN_COL  = "#94a3b8"     # neutral slate for prompt tokens
    OUT_COL = "#0f766e"     # teal for completion tokens
    ax1.barh(y, g["in_mean"],  color=IN_COL, edgecolor="white", linewidth=0.5,
             height=0.72, label="Prompt")
    ax1.barh(y, g["out_mean"], left=g["in_mean"], color=OUT_COL,
             edgecolor="white", linewidth=0.5, height=0.72, label="Completion")
    for i, row in g.iterrows():
        ax1.text(row["total"] + g["total"].max() * 0.01, i,
                 f"{int(row['total']):,}", va="center", ha="left",
                 fontsize=7.5, color="#1f2937")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.set_xlim(0, g["total"].max() * 1.22)
    ax1.set_xlabel("Mean tokens per plan")
    ax1.set_title("a  Tokens per plan (stacked)", loc="left")
    ax1.legend(loc="lower right", fontsize=7.5, frameon=False)
    _style_value_axis(ax1, x=True)

    # Panel b: tokens per successful plan
    t = g.sort_values("tokens_per_pass").reset_index(drop=True)
    yb = np.arange(len(t))
    cols = [_PROVIDER_COLOURS.get(_provider_from_model(m), "#6b7280")
            for m in t["model"]]
    ax2.barh(yb, t["tokens_per_pass"], color=cols, edgecolor="white",
             linewidth=0.6, height=0.72)
    for i, v in enumerate(t["tokens_per_pass"]):
        if pd.isna(v):
            ax2.text(0.005, i, "n/a", va="center", ha="left",
                     fontsize=7.5, color="#9ca3af",
                     transform=ax2.get_yaxis_transform())
            continue
        ax2.text(v + t["tokens_per_pass"].dropna().max() * 0.01, i,
                 f"{int(v):,}", va="center", ha="left",
                 fontsize=7.5, color="#1f2937")
    # Show the model ordering from this panel (not shared with panel a)
    ax2.set_yticks(yb)
    ax2.set_yticklabels([_short_name(m) for m in t["model"]])
    ax2.set_xlim(0, t["tokens_per_pass"].dropna().max() * 1.25)
    ax2.set_xlabel("Tokens per successful plan")
    ax2.set_title("b  Token efficiency (tokens ÷ pass rate)", loc="left")
    _style_value_axis(ax2, x=True)

    fig.suptitle("Token usage benchmark", fontsize=11,
                 fontweight="bold", y=1.03)
    return fig


# ── Saving helpers ───────────────────────────────────────────────

def _save(fig: plt.Figure, out_base: Path, *,
          pdf: bool = True, png: bool = True, svg: bool = False) -> None:
    """Save a figure in multiple vector / raster formats at publication DPI."""
    out_base.parent.mkdir(parents=True, exist_ok=True)
    if pdf:
        fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    if png:
        fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if svg:
        fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")


# ── CLI: regenerate all figures ──────────────────────────────────

def _latest_taxonomy_per_cell(results_root: Path) -> Optional[Path]:
    """Return the newest ``per_cell.csv`` under ``results/recovery/_taxonomy/``."""
    tdir = results_root / "recovery" / "_taxonomy"
    if not tdir.exists():
        return None
    subs = [
        p for p in tdir.iterdir()
        if p.is_dir() and (p / "per_cell.csv").exists()
    ]
    if not subs:
        return None
    return max(subs, key=lambda p: p.stat().st_mtime) / "per_cell.csv"


def _regen_recovery_taxonomy(results_root: Path) -> Optional[Path]:
    """Run ``recovery_taxonomy.py`` against every recovery run in-tree.

    Returns the path to the freshly written ``per_cell.csv`` or ``None``
    if the taxonomy script isn't importable (e.g. running the plot
    module from outside the ``benchmarks/`` tree).
    """
    import sys
    import subprocess
    tax_script = Path(__file__).resolve().parent.parent / "recovery_taxonomy.py"
    if not tax_script.exists():
        return None
    try:
        subprocess.run(
            [sys.executable, str(tax_script),
             "--runs", str(results_root / "recovery" / "*")],
            check=True, capture_output=True,
        )
    except Exception as e:
        print(f"[warn] recovery_taxonomy regen failed: {e}")
        return None
    return _latest_taxonomy_per_cell(results_root)


def _load_all_recovery_runs(results_root: Path) -> Optional[pd.DataFrame]:
    """Load and concat every ``recovery/<ts>/metrics.csv`` under ``results/``.

    Tags each row with the ``model`` taken from that run's
    ``manifest.json`` (metrics.csv rows don't carry the model id), then
    deduplicates by ``(model, fault, seed)`` keeping the newest run so a
    re-run replaces stale cells rather than duplicating them.

    Returns ``None`` if no runs exist.
    """
    rec_dir = results_root / "recovery"
    if not rec_dir.exists():
        return None
    frames: List[pd.DataFrame] = []
    for sub in sorted(rec_dir.iterdir(), key=lambda p: p.stat().st_mtime):
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        csv = sub / "metrics.csv"
        if not csv.exists() or csv.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if df.empty:
            continue
        # Tag model from manifest.json if the rows don't have one
        if "model" not in df.columns:
            model = "unknown"
            mf = sub / "manifest.json"
            if mf.exists():
                try:
                    import json
                    manifest = json.loads(mf.read_text())
                    models = [m.get("id") for m in manifest.get("models") or []]
                    if models:
                        model = models[0]
                except Exception:
                    pass
            df["model"] = model
        df["_source_run"] = sub.name
        frames.append(df)
    if not frames:
        return None
    merged = pd.concat(frames, ignore_index=True)
    # Deduplicate: later runs of the same cell replace earlier ones.
    # (Rows were appended in mtime order above, so keep="last" wins.)
    if {"model", "fault", "seed"}.issubset(merged.columns):
        merged = merged.drop_duplicates(
            subset=["model", "fault", "seed"], keep="last",
        ).reset_index(drop=True)
    return merged


def _latest(run_dir: Path) -> Optional[Path]:
    """Return the latest results dir for a given benchmark.

    For ``planning``, prefer ``_merged/<latest>`` if it exists, then
    ``<run>/rescored_<latest>``, then ``<run>`` itself.
    """
    if not run_dir.exists():
        return None
    merged = run_dir / "_merged"
    if merged.exists():
        merged_subs = [p for p in merged.iterdir() if p.is_dir()
                       and (p / "metrics.csv").exists()]
        if merged_subs:
            return max(merged_subs, key=lambda p: p.stat().st_mtime)

    subs = [p for p in run_dir.iterdir() if p.is_dir()
            and not p.name.startswith("_")]
    if not subs:
        return None
    latest_run = max(subs, key=lambda p: p.stat().st_mtime)
    rescored = sorted(
        (sub for sub in latest_run.iterdir()
         if sub.is_dir() and sub.name.startswith("rescored_")
         and (sub / "metrics.csv").exists()),
        key=lambda p: p.stat().st_mtime,
    )
    return rescored[-1] if rescored else latest_run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results",
                    help="Root results directory (default: results)")
    ap.add_argument("--svg", action="store_true",
                    help="Also emit SVG alongside PDF/PNG")
    args = ap.parse_args()

    root    = Path(args.results)
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for bench_name, fn in (
        ("planning",    planning_figure),
        ("recovery",    recovery_figure),
        ("generation",  generation_figure),
        ("executors",   executor_matrix_figure),
        ("competitors", competitors_figure),
    ):
        # Recovery is special: a sweep across models produces one dir per
        # model, and the per-fault figure is far more informative when
        # aggregated across all of them. For everything else, pick the
        # latest single run as before.
        if bench_name == "recovery":
            df = _load_all_recovery_runs(root)
            if df is None or df.empty:
                print(f"[skip] no results for {bench_name}")
                continue
        else:
            latest = _latest(root / bench_name)
            if latest is None:
                print(f"[skip] no results for {bench_name}")
                continue
            csv = latest / "metrics.csv"
            if not csv.exists() or csv.stat().st_size == 0:
                print(f"[skip] empty metrics for {bench_name}")
                continue
            df = pd.read_csv(csv)
            if df.empty:
                print(f"[skip] empty dataframe for {bench_name}")
                continue

        fig = fn(df)
        _save(fig, fig_dir / bench_name, svg=args.svg)
        plt.close(fig)
        print(f"[ok]   {bench_name} → {fig_dir/bench_name}.pdf")

        # Bonus tier-summary alongside the per-fault bar chart
        if bench_name == "recovery":
            tierfig = recovery_tier_summary_figure(df)
            if tierfig is not None:
                _save(tierfig, fig_dir / "recovery_tier_summary", svg=args.svg)
                plt.close(tierfig)
                print(f"[ok]   recovery_tier_summary → "
                      f"{fig_dir/'recovery_tier_summary'}.pdf")

            # Per-fault × per-model heatmap — the cross-model comparison
            heat = recovery_per_fault_heatmap(df)
            if heat is not None:
                _save(heat, fig_dir / "recovery_per_fault_heatmap",
                      svg=args.svg)
                plt.close(heat)
                print(f"[ok]   recovery_per_fault_heatmap → "
                      f"{fig_dir/'recovery_per_fault_heatmap'}.pdf")

            # Per-model recovery figures (one per model that ran).
            # ``recovery_figure`` and ``recovery_tier_summary_figure``
            # both accept a filtered df, so we just slice on the model
            # column. Filename uses the short-name slug for legibility.
            if "model" in df.columns:
                per_model_dir = fig_dir / "recovery_per_model"
                per_model_dir.mkdir(parents=True, exist_ok=True)
                for m in sorted(df["model"].dropna().unique()):
                    sub = df[df["model"] == m]
                    if sub.empty:
                        continue
                    # Sanitise the model id into a filesystem-safe slug.
                    # Replace ``.`` too — otherwise ``_save`` /
                    # ``with_suffix`` truncates everything after the
                    # last dot (gemini-2.5-flash → gemini-2.pdf).
                    slug = (
                        str(m).replace("/", "_").replace(":", "_")
                              .replace(" ", "_").replace(".", "_")
                    )
                    pmf = recovery_figure(sub)
                    if pmf is not None:
                        _save(pmf, per_model_dir / f"recovery_{slug}",
                              svg=args.svg)
                        plt.close(pmf)
                    pmt = recovery_tier_summary_figure(sub)
                    if pmt is not None:
                        _save(pmt,
                              per_model_dir / f"recovery_tier_{slug}",
                              svg=args.svg)
                        plt.close(pmt)
                print(f"[ok]   per-model recovery figures → "
                      f"{per_model_dir}/recovery_<model>.pdf "
                      f"(+ recovery_tier_<model>.pdf)")

            # Unrecoverable-tier taxonomy (correct / misdiagnosed /
            # unsafe / silent). Uses per_cell.csv from the latest
            # recovery_taxonomy.py run; regenerates if absent.
            tax_csv = _latest_taxonomy_per_cell(root)
            if tax_csv is None or not tax_csv.exists():
                tax_csv = _regen_recovery_taxonomy(root)
            if tax_csv is not None and tax_csv.exists():
                tax_df = pd.read_csv(tax_csv)
                taxfig = recovery_taxonomy_figure(tax_df)
                if taxfig is not None:
                    _save(taxfig, fig_dir / "recovery_taxonomy",
                          svg=args.svg)
                    plt.close(taxfig)
                    print(f"[ok]   recovery_taxonomy → "
                          f"{fig_dir/'recovery_taxonomy'}.pdf")

                split = recovery_reasoning_split_figure(tax_df)
                if split is not None:
                    _save(split, fig_dir / "recovery_reasoning_split",
                          svg=args.svg)
                    plt.close(split)
                    print(f"[ok]   recovery_reasoning_split → "
                          f"{fig_dir/'recovery_reasoning_split'}.pdf")

        # Companion heatmap for the head-to-head
        if bench_name == "competitors":
            heat = competitors_perprompt_figure(df)
            if heat is not None:
                _save(heat, fig_dir / "competitors_perprompt", svg=args.svg)
                plt.close(heat)
                print(f"[ok]   competitors_perprompt → "
                      f"{fig_dir/'competitors_perprompt'}.pdf")

            # Focused agentic-only head-to-head (FlowAgent / BioMaster / AutoBA)
            agentic = competitors_agentic_figure(df)
            if agentic is not None:
                _save(agentic, fig_dir / "competitors_agentic", svg=args.svg)
                plt.close(agentic)
                print(f"[ok]   competitors_agentic → "
                      f"{fig_dir/'competitors_agentic'}.pdf")

        # Planning bonus figures
        if bench_name == "planning":
            fig2 = planning_heatmap(df)
            _save(fig2, fig_dir / "planning_heatmap", svg=args.svg)
            plt.close(fig2)
            print(f"[ok]   planning_heatmap → {fig_dir/'planning_heatmap'}.pdf")

            fig2b = planning_heatmap_by_tier(df)
            _save(fig2b, fig_dir / "planning_heatmap_by_tier", svg=args.svg)
            plt.close(fig2b)
            print(f"[ok]   planning_heatmap_by_tier → {fig_dir/'planning_heatmap_by_tier'}.pdf")

            fig3 = cost_vs_quality_figure(df)
            if fig3 is not None:
                _save(fig3, fig_dir / "planning_cost_quality", svg=args.svg)
                plt.close(fig3)
                print(f"[ok]   planning_cost_quality → "
                      f"{fig_dir/'planning_cost_quality'}.pdf")

            fig3b = latency_figure(df)
            if fig3b is not None:
                _save(fig3b, fig_dir / "planning_latency", svg=args.svg)
                plt.close(fig3b)
                print(f"[ok]   planning_latency → "
                      f"{fig_dir/'planning_latency'}.pdf")

            fig4 = cost_summary_figure(df)
            if fig4 is not None:
                _save(fig4, fig_dir / "planning_cost_summary", svg=args.svg)
                plt.close(fig4)
                print(f"[ok]   planning_cost_summary → "
                      f"{fig_dir/'planning_cost_summary'}.pdf")

                table = _per_model_cost_table(df)
                if table is not None:
                    out_tsv = fig_dir / "planning_cost_summary.tsv"
                    out_tsv.write_text(table.to_csv(sep="\t", index=False))
                    print(f"[ok]   planning_cost_summary → {out_tsv}")

            for label, figfn in (
                ("planning_turns",         turns_to_completion_figure),
                ("planning_consistency",   consistency_figure),
                ("planning_hallucination", hallucination_figure),
                ("planning_tokens",        token_usage_figure),
            ):
                f = figfn(df)
                if f is not None:
                    _save(f, fig_dir / label, svg=args.svg)
                    plt.close(f)
                    print(f"[ok]   {label} → {fig_dir/label}.pdf")


if __name__ == "__main__":
    main()
