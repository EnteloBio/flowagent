"""FlowAgent: Multi-agent framework for automating bioinformatics workflows.

Supports multiple LLM providers (OpenAI, Anthropic, Google Gemini, Ollama),
generates Nextflow / Snakemake pipelines, and executes via local, cgat-core
HPC, or Kubernetes backends.
"""

__version__ = "0.2.0"

from typing import Any, Dict, Optional

__all__ = ["FlowAgent", "__version__"]


class FlowAgent:
    """High-level facade for programmatic use of FlowAgent.

    Example::

        from flowagent import FlowAgent

        fa = FlowAgent(provider="openai", model="gpt-4.1")
        plan = await fa.plan("RNA-seq with kallisto", data_dir="./fastq/")
        plan_nf = fa.to_nextflow(plan, output_dir="pipeline/")
        result = await fa.execute(plan, executor="local")
        report = await fa.analyze(result["output_dir"])
    """

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        executor: str = "local",
    ):
        import os
        from .config.settings import Settings

        self._settings = Settings()
        if provider:
            os.environ["LLM_PROVIDER"] = provider
        if model:
            os.environ["LLM_MODEL"] = model
        if api_key:
            os.environ.setdefault("OPENAI_API_KEY", api_key)
            os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
            os.environ.setdefault("GOOGLE_API_KEY", api_key)
        self._executor = executor

    async def plan(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a workflow plan from natural language."""
        from .core.llm import LLMInterface
        llm = LLMInterface()
        return await llm.generate_workflow_plan(prompt)

    async def execute(self, plan_or_prompt, *, executor: Optional[str] = None,
                      checkpoint_dir: Optional[str] = None) -> Dict[str, Any]:
        """Execute a workflow plan or prompt."""
        from .core.workflow_manager import WorkflowManager
        wm = WorkflowManager(executor_type=executor or self._executor)
        if isinstance(plan_or_prompt, str):
            return await wm.execute_workflow(plan_or_prompt)
        from .core.agent_types import Workflow, WorkflowStep
        steps = [WorkflowStep(
            name=s["name"], command=s["command"],
            dependencies=s.get("dependencies", []),
        ) for s in plan_or_prompt.get("steps", [])]
        wf = Workflow(
            name=plan_or_prompt.get("name", plan_or_prompt.get("workflow_type", "workflow")),
            description=plan_or_prompt.get("description", ""),
            steps=steps,
            checkpoint_dir=checkpoint_dir,
        )
        return await wm.execute_workflow(wf)

    def to_nextflow(self, plan: Dict[str, Any], output_dir: str = "flowagent_pipeline_output") -> str:
        """Export a plan as a Nextflow pipeline. Returns the file path."""
        from pathlib import Path
        from .core.pipeline_generator import NextflowGenerator
        gen = NextflowGenerator()
        gen.generate(plan, output_dir=Path(output_dir))
        return str(Path(output_dir) / gen.default_filename())

    def to_snakemake(self, plan: Dict[str, Any], output_dir: str = "flowagent_pipeline_output") -> str:
        """Export a plan as a Snakemake pipeline. Returns the file path."""
        from pathlib import Path
        from .core.pipeline_generator import SnakemakeGenerator
        gen = SnakemakeGenerator()
        gen.generate(plan, output_dir=Path(output_dir))
        return str(Path(output_dir) / gen.default_filename())

    async def analyze(self, results_dir: str, save_report: bool = True) -> Dict[str, Any]:
        """Analyze workflow results in a directory."""
        from .workflow import analyze_workflow
        return await analyze_workflow(results_dir, save_report)

    @staticmethod
    def list_presets():
        """Return available preset workflow summaries."""
        from .presets.catalog import list_presets
        return list_presets()

    @staticmethod
    def get_preset(preset_id: str) -> Optional[Dict[str, Any]]:
        """Load a preset workflow plan by ID."""
        from .presets.catalog import get_preset
        return get_preset(preset_id)
