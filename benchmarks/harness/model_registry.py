"""Canonical model ids for merged CSVs and publication figures.

Registry labels (``gpt-5.5-mini``) can differ from the OpenAI API slug
actually invoked (``gpt-5.4-mini``). Figures and merged metrics should use
the canonical id so heatmaps do not show unreleased model names.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict

import pandas as pd


@lru_cache(maxsize=1)
def registry_plot_aliases() -> Dict[str, str]:
    """Map registry-only ids to the model id they represent on the API."""
    aliases: Dict[str, str] = {}
    try:
        from flowagent.core.providers.openai_models import _OPENAI_API_ALIASES
        aliases.update(_OPENAI_API_ALIASES)
    except ImportError:
        pass

    cfg = Path(__file__).resolve().parent.parent / "config" / "models.yaml"
    try:
        import yaml
        data = yaml.safe_load(cfg.read_text())
        for m in (data or {}).get("models", []) or []:
            mid = m.get("id")
            api_id = m.get("api_id")
            if mid and api_id and api_id != mid:
                aliases[mid] = api_id
    except Exception:
        pass
    return aliases


def canonical_model_id(model: str) -> str:
    return registry_plot_aliases().get(model, model)


def remap_model_column(df: pd.DataFrame, *, column: str = "model") -> pd.DataFrame:
    """Rewrite ``column`` using :func:`canonical_model_id`."""
    if column not in df.columns or df.empty:
        return df
    aliases = registry_plot_aliases()
    if not aliases:
        return df
    out = df.copy()
    out[column] = out[column].map(lambda m: aliases.get(m, m))
    return out
