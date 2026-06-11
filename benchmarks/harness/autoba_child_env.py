"""Environment defaults for AutoBA shim subprocesses (macOS / conda stability).

AutoBA imports ``torch`` at load time. Combined with NumPy/MKL/OpenMP that
can SIGSEGV (-11) or abort with duplicate ``libomp`` before Python prints the
JSON envelope. Call :func:`apply_autoba_defaults` at the very start of
``autoba_shim.main()`` and use :func:`autoba_subprocess_environ`` when
spawning the shim from :mod:`harness.competitors`.
"""

from __future__ import annotations

import os
from typing import Dict, MutableMapping


def apply_autoba_defaults(env: MutableMapping[str, str]) -> None:
    """Set conservative defaults (only if unset) on ``os.environ`` or a copy."""
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")


def autoba_subprocess_environ() -> Dict[str, str]:
    """Full env dict for ``asyncio.create_subprocess_exec(..., env=...)``."""
    env = dict(os.environ)
    apply_autoba_defaults(env)
    return env
