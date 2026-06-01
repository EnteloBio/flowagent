"""Env-flag gating for the command-level plan validator (todo T0).

Two independent toggles replace the monolithic
``FLOWAGENT_VALIDATOR_ENABLED``:

* ``FLOWAGENT_VALIDATOR_AUTOFIX`` (default **on**) — deterministic
  in-place rewrites (tool typos, archive nesting, ``\\n`` escapes).
  No LLM call; safe to keep on per Benchmark K evidence.

* ``FLOWAGENT_VALIDATOR_RETRY`` (default **off**) — run validation and,
  on failure, feed violations back to the LLM for another attempt.
  Shares the completeness-reflection retry budget on the primary plan
  path; Benchmark K found this unreliable and completeness-regressive.

Legacy ``FLOWAGENT_VALIDATOR_ENABLED`` still works when set explicitly:
``true`` enables both layers; ``false`` disables both. Granular flags
apply when the legacy variable is unset.

All readers re-parse the environment on every call so benchmark harnesses
can flip flags per cell without restarting the process.
"""

from __future__ import annotations

import os
from typing import Optional

_FALSE = frozenset({"0", "false", "no", "off"})


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSE


def validator_legacy_enabled() -> Optional[bool]:
    """Return legacy monolithic flag if set, else ``None``."""
    raw = os.environ.get("FLOWAGENT_VALIDATOR_ENABLED")
    if raw is None:
        return None
    return raw.strip().lower() not in _FALSE


def validator_autofix_enabled() -> bool:
    legacy = validator_legacy_enabled()
    if legacy is not None:
        return legacy
    return _env_bool("FLOWAGENT_VALIDATOR_AUTOFIX", default=True)


def validator_retry_enabled() -> bool:
    legacy = validator_legacy_enabled()
    if legacy is not None:
        return legacy
    return _env_bool("FLOWAGENT_VALIDATOR_RETRY", default=False)


def validator_any_enabled() -> bool:
    return validator_autofix_enabled() or validator_retry_enabled()
