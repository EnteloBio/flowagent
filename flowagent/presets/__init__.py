"""Pre-built, validated workflow templates for common bioinformatics analyses.

These bypass LLM generation entirely for standard workflows, offering
faster and more reproducible results.
"""

from .catalog import PRESET_CATALOG, get_preset, list_presets

__all__ = ["PRESET_CATALOG", "get_preset", "list_presets"]
