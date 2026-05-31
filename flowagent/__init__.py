"""FlowAgent: Multi-agent framework for automating bioinformatics workflows.

Supports multiple LLM providers (OpenAI, Anthropic, Google Gemini, Ollama),
generates Nextflow / Snakemake pipelines, and executes via local, cgat-core
HPC, or Kubernetes backends.

Recommended API
---------------
::

    from flowagent import Session

    async with Session(model="gpt-4.1", data_dir=".", executor="local") as fa:
        ctx  = await fa.inspect()
        plan = await fa.plan("RNA-seq → DESeq2", context=ctx)
        plan = await fa.validate(plan)
        run  = await fa.run(plan, checkpoint="workflow_state/")
        report = await fa.interpret(run.output_dir)

Sync shortcut (useful in notebooks)::

    run = Session.run_sync("RNA-seq with kallisto", executor="local")

Legacy API (backwards-compatible)::

    fa = FlowAgent(provider="openai", model="gpt-4.1")
    plan = await fa.plan("RNA-seq")
    result = await fa.execute(plan)
"""

from __future__ import annotations

__version__ = "0.2.0"

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

__all__ = ["Session", "FlowAgent", "RunResult", "PlanResult", "__version__"]


# ── Structured return objects ─────────────────────────────────

@dataclass
class PlanResult:
    """The result of a planning step."""
    plan: Dict[str, Any]
    steps: List[Dict[str, Any]] = field(default_factory=list)
    prompt: str = ""
    model: str = ""
    plan_path: Optional[str] = None
    markdown_path: Optional[str] = None

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PlanResult(name={self.plan.get('name', 'workflow')!r}, "
            f"steps={len(self.steps)}, plan_path={self.plan_path!r})"
        )


@dataclass
class RunResult:
    """The result of a workflow execution."""
    status: str
    output_dir: Optional[str] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)
    plan: Optional[Dict[str, Any]] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status in ("completed", "success")

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RunResult(status={self.status!r}, "
            f"output_dir={self.output_dir!r})"
        )


# ── Session API ───────────────────────────────────────────────

class Session:
    """High-level async session for programmatic use of FlowAgent.

    Recommended usage via async context manager::

        async with Session(model="gpt-4.1") as fa:
            plan = await fa.plan("RNA-seq with kallisto")
            run  = await fa.run(plan)

    Parameters
    ----------
    model
        LLM model identifier (e.g. ``"gpt-4.1"``, ``"claude-opus-4-5"``).
        Provider is inferred from the model prefix.
    provider
        Override the provider (``"openai"``, ``"anthropic"``, ``"google"``).
    api_key
        API key; if omitted reads from environment / ``.env``.
    executor
        Execution backend: ``"local"`` (default), ``"hpc"``, ``"kubernetes"``.
    data_dir
        Working directory that contains data/ and reference/ sub-folders.
        Defaults to current directory.
    on_step_start
        Optional callback ``f(step_name: str) -> None`` called before each step.
    on_recovery
        Optional callback ``f(step_name: str, error: str) -> None`` on recovery.
    on_token
        Optional callback ``f(token: str) -> None`` for streaming tokens.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        executor: str = "local",
        data_dir: Optional[str] = None,
        on_step_start: Optional[Callable[[str], None]] = None,
        on_recovery: Optional[Callable[[str, str], None]] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ):
        from .config.settings import Settings

        if model:
            os.environ["LLM_MODEL"] = model
            if not provider and "LLM_PROVIDER" not in os.environ:
                from .core.providers.registry import _infer_provider
                os.environ["LLM_PROVIDER"] = _infer_provider(model)
        if provider:
            os.environ["LLM_PROVIDER"] = provider
        if api_key:
            os.environ.setdefault("OPENAI_API_KEY", api_key)
            os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
            os.environ.setdefault("GOOGLE_API_KEY", api_key)

        self._executor = executor
        self._data_dir = Path(data_dir) if data_dir else Path(".")
        self._settings = Settings()

        # Event hooks
        self.on_step_start = on_step_start
        self.on_recovery = on_recovery
        self.on_token = on_token

    async def __aenter__(self) -> "Session":
        return self

    async def __aexit__(self, *_: Any) -> None:
        pass

    # ── Core API ───────────────────────────────────────────────

    async def inspect(self) -> Dict[str, Any]:
        """Scan the working directory and return a PipelineContext dict.

        Discovers FASTQ / BAM files, reference files, and asks the user
        about organism and genome build (when in an interactive terminal).
        Returns a dict representation of :class:`~flowagent.core.schemas.PipelineContext`.
        """
        from .core.pipeline_planner import gather_pipeline_context
        cwd_before = os.getcwd()
        try:
            os.chdir(self._data_dir)
            ctx = await gather_pipeline_context("", interactive=False)
        finally:
            os.chdir(cwd_before)
        return {
            "input_files": ctx.input_files,
            "paired_end": ctx.paired_end,
            "organism": ctx.organism,
            "genome_build": ctx.genome_build,
            "workflow_type": ctx.workflow_type,
            "reference_fasta": ctx.reference_fasta,
            "annotation_gtf": ctx.annotation_gtf,
            "needs_reference_download": ctx.needs_reference_download,
            "needs_annotation_download": ctx.needs_annotation_download,
        }

    async def plan(
        self,
        prompt: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        preset: Optional[str] = None,
        save_to: Optional[str] = None,
    ) -> PlanResult:
        """Generate a workflow plan from *prompt*.

        Parameters
        ----------
        prompt
            Natural-language description of the analysis.
        context
            Optional context dict (from :meth:`inspect`); if omitted the
            planner will scan the filesystem automatically.
        preset
            Use a preset as the base plan instead of LLM generation.
        save_to
            If given, write ``workflow.json`` + ``workflow.md`` to this directory.

        Returns
        -------
        PlanResult
            Structured object with ``.plan``, ``.steps``, and optional paths.
        """
        from .core.llm import LLMInterface
        from .core.pipeline_planner import gather_pipeline_context
        from .core.schemas import PipelineContext

        if preset:
            from .presets.catalog import get_preset, apply_context_to_preset
            raw_preset = get_preset(preset)
            if raw_preset is None:
                from .presets.catalog import list_presets
                names = ", ".join(p["id"] for p in list_presets())
                raise ValueError(f"Unknown preset '{preset}'. Available: {names}")
            if context is None:
                ctx = await gather_pipeline_context(prompt, interactive=False)
            else:
                ctx = PipelineContext(**{
                    k: v for k, v in context.items()
                    if k in PipelineContext.__dataclass_fields__
                })
            plan = apply_context_to_preset(raw_preset, ctx)
        else:
            llm = LLMInterface()
            if context is None:
                ctx_obj = await gather_pipeline_context(prompt, interactive=False)
            else:
                ctx_obj = PipelineContext(**{
                    k: v for k, v in context.items()
                    if k in PipelineContext.__dataclass_fields__
                })
            plan = await llm.generate_workflow_plan(prompt, context=ctx_obj)

        plan_path = None
        md_path = None
        if save_to:
            from .core.plan_store import save_plan
            p = save_plan(
                plan, Path(save_to),
                prompt=prompt,
                model=self._settings.LLM_MODEL,
            )
            plan_path = str(p)
            md_path = str(p.parent / "workflow.md")

        return PlanResult(
            plan=plan,
            steps=plan.get("steps", []),
            prompt=prompt,
            model=self._settings.LLM_MODEL,
            plan_path=plan_path,
            markdown_path=md_path,
        )

    async def validate(self, plan_or_result: "PlanResult | Dict[str, Any]") -> PlanResult:
        """Run completeness + command validator on a plan.

        Raises ``ValueError`` if the plan has unfixable structural errors.
        Returns a (possibly corrected) ``PlanResult``.
        """
        from .core.completeness import check_completeness

        plan = plan_or_result.plan if isinstance(plan_or_result, PlanResult) else plan_or_result

        issues = check_completeness(plan)
        if issues:
            # Non-fatal: log warnings but don't block
            import logging
            _log = logging.getLogger(__name__)
            for issue in issues:
                _log.warning("Plan completeness: %s", issue)

        if isinstance(plan_or_result, PlanResult):
            return plan_or_result
        return PlanResult(plan=plan, steps=plan.get("steps", []))

    async def run(
        self,
        plan_or_result: "PlanResult | Dict[str, Any]",
        *,
        checkpoint: Optional[str] = None,
        resume: bool = False,
    ) -> RunResult:
        """Execute a workflow plan.

        Parameters
        ----------
        plan_or_result
            A :class:`PlanResult` or raw plan dict.
        checkpoint
            Directory to write step checkpoints (enables ``resume``).
        resume
            If True, skip already-completed steps from a previous run.

        Returns
        -------
        RunResult
        """
        from .core.workflow_manager import WorkflowManager
        from .core.agent_types import Workflow, WorkflowStep

        plan = plan_or_result.plan if isinstance(plan_or_result, PlanResult) else plan_or_result

        wm = WorkflowManager(executor_type=self._executor)

        steps = [WorkflowStep(
            name=s["name"],
            command=s.get("command", ""),
            dependencies=s.get("dependencies", []),
            description=s.get("description", ""),
        ) for s in plan.get("steps", [])]

        wf = Workflow(
            name=plan.get("name", plan.get("workflow_type", "workflow")),
            description=plan.get("description", ""),
            steps=steps,
            checkpoint_dir=checkpoint,
        )

        raw = await wm.execute_workflow(wf)
        return RunResult(
            status=raw.get("status", "unknown"),
            output_dir=raw.get("output_dir"),
            steps=raw.get("steps", []),
            plan=plan,
            raw=raw,
        )

    async def interpret(
        self,
        results_dir: str,
        *,
        questions: Optional[List[str]] = None,
        save_report: bool = True,
    ) -> Dict[str, Any]:
        """Interpret workflow outputs using LLM analysis (Benchmark G style).

        Parameters
        ----------
        results_dir
            Path to the workflow results directory.
        questions
            Optional list of natural-language questions. If omitted, the
            analysis system generates a default set based on detected outputs.
        save_report
            If True, write a ``report.md`` into *results_dir*.

        Returns
        -------
        dict
            ``{"status": "success", "report": str, "agentic_report": str}``
        """
        from .workflow import analyze_workflow
        return await analyze_workflow(results_dir, save_report)

    def to_nextflow(
        self,
        plan_or_result: "PlanResult | Dict[str, Any]",
        output_dir: str = "flowagent_pipeline_output",
    ) -> str:
        """Export a plan as a Nextflow pipeline. Returns the file path."""
        from .core.pipeline_generator import NextflowGenerator
        plan = plan_or_result.plan if isinstance(plan_or_result, PlanResult) else plan_or_result
        gen = NextflowGenerator()
        gen.generate(plan, output_dir=Path(output_dir))
        return str(Path(output_dir) / gen.default_filename())

    def to_snakemake(
        self,
        plan_or_result: "PlanResult | Dict[str, Any]",
        output_dir: str = "flowagent_pipeline_output",
    ) -> str:
        """Export a plan as a Snakemake pipeline. Returns the file path."""
        from .core.pipeline_generator import SnakemakeGenerator
        plan = plan_or_result.plan if isinstance(plan_or_result, PlanResult) else plan_or_result
        gen = SnakemakeGenerator()
        gen.generate(plan, output_dir=Path(output_dir))
        return str(Path(output_dir) / gen.default_filename())

    @staticmethod
    def list_presets() -> List[Dict[str, str]]:
        """Return available preset workflow summaries."""
        from .presets.catalog import list_presets
        return list_presets()

    @staticmethod
    def get_preset(preset_id: str) -> Optional[Dict[str, Any]]:
        """Load a preset workflow plan by ID."""
        from .presets.catalog import get_preset
        return get_preset(preset_id)

    # ── Sync convenience wrapper ───────────────────────────────

    @classmethod
    def run_sync(
        cls,
        prompt: str,
        *,
        model: Optional[str] = None,
        executor: str = "local",
        checkpoint: Optional[str] = None,
        preset: Optional[str] = None,
        save_to: Optional[str] = None,
    ) -> RunResult:
        """Synchronous shortcut — plan + validate + run in one call.

        Useful for Jupyter notebooks and scripts that cannot use ``await``::

            result = Session.run_sync("RNA-seq with kallisto", executor="local")
            print(result.output_dir)
        """
        async def _inner():
            async with cls(model=model, executor=executor) as fa:
                plan = await fa.plan(prompt, preset=preset, save_to=save_to)
                plan = await fa.validate(plan)
                return await fa.run(plan, checkpoint=checkpoint)

        try:
            loop = asyncio.get_running_loop()
            # Already inside an event loop (e.g. Jupyter)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _inner())
                return future.result()
        except RuntimeError:
            return asyncio.run(_inner())


# ── Legacy FlowAgent facade (backwards-compatible) ────────────

class FlowAgent:
    """Legacy high-level facade — prefer :class:`Session` for new code.

    Example::

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
        self._session = Session(
            model=model,
            provider=provider,
            api_key=api_key,
            executor=executor,
        )
        self._executor = executor

    async def plan(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a workflow plan from natural language."""
        result = await self._session.plan(prompt)
        return result.plan

    async def execute(
        self,
        plan_or_prompt: Any,
        *,
        executor: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a workflow plan or prompt."""
        if executor and executor != self._executor:
            sess = Session(executor=executor)
        else:
            sess = self._session
        run = await sess.run(
            plan_or_prompt if isinstance(plan_or_prompt, dict) else {"steps": [], "name": plan_or_prompt},
            checkpoint=checkpoint_dir,
        )
        return run.raw or {"status": run.status, "output_dir": run.output_dir}

    def to_nextflow(self, plan: Dict[str, Any], output_dir: str = "flowagent_pipeline_output") -> str:
        return self._session.to_nextflow(plan, output_dir)

    def to_snakemake(self, plan: Dict[str, Any], output_dir: str = "flowagent_pipeline_output") -> str:
        return self._session.to_snakemake(plan, output_dir)

    async def analyze(self, results_dir: str, save_report: bool = True) -> Dict[str, Any]:
        return await self._session.interpret(results_dir, save_report=save_report)

    @staticmethod
    def list_presets() -> List[Dict[str, str]]:
        return Session.list_presets()

    @staticmethod
    def get_preset(preset_id: str) -> Optional[Dict[str, Any]]:
        return Session.get_preset(preset_id)
