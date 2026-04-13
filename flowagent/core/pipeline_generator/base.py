"""Abstract base for pipeline code generators."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class PipelineGenerator(ABC):
    """Convert a WorkflowPlan dict into a portable pipeline file."""

    @abstractmethod
    def generate(self, workflow_plan: Dict[str, Any], *, output_dir: Optional[Path] = None) -> str:
        """Return the pipeline source code as a string.

        If *output_dir* is given, also write the file(s) to disk.
        """

    @abstractmethod
    def validate(self, code: str, *, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Dry-run / lint the generated pipeline code.

        Returns a dict with ``valid`` (bool), ``errors`` (list[str]),
        and ``warnings`` (list[str]).
        """

    @abstractmethod
    def default_filename(self) -> str:
        """Return the canonical filename (e.g. 'main.nf', 'Snakefile')."""
