"""Head-to-head competitor harness for Benchmark E.

This module defines a small pluggable interface so that third-party
agentic bioinformatics systems (BioMaster, AutoBA, CellAgent, …) can be
evaluated on the same prompt corpus + metric set as FlowAgent.

Design
------
Every competitor is a subclass of :class:`Competitor` that implements
:meth:`Competitor.plan` — given a natural-language prompt (plus shared
context), return a FlowAgent-compatible plan dict of the form::

    {
      "workflow_type": "rna_seq_kallisto" | "custom" | ...,
      "steps": [
        {"name": "...", "command": "...", "dependencies": [...],
         "outputs": [...], "description": "..."},
        ...
      ]
    }

That lets ``harness.metrics.score_plan`` evaluate the output the same
way we score FlowAgent plans, so the head-to-head comparison is fair.

If a competitor's upstream dependency is not installed, or it cannot be
adapted to return the shared plan schema, :meth:`Competitor.available`
returns ``False`` with a human-readable reason. Running
``bench_competitors.py`` then skips it and records a clear message in
the manifest — no silent failures.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOG = logging.getLogger(__name__)


# ── Shared plan schema ───────────────────────────────────────────

def _empty_plan(workflow_type: str = "custom") -> Dict[str, Any]:
    return {"workflow_type": workflow_type, "steps": []}


def _normalise_step(step: Dict[str, Any]) -> Dict[str, Any]:
    """Fill in required keys on a step dict (keeps scoring lenient)."""
    return {
        "name":         step.get("name") or step.get("id") or "step",
        "command":      step.get("command") or step.get("cmd") or "",
        "dependencies": step.get("dependencies") or step.get("deps") or [],
        "outputs":      step.get("outputs") or [],
        "description":  step.get("description") or "",
    }


def _normalise_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Accept a variety of agent-framework outputs and coerce to the
    FlowAgent plan schema so metrics.score_plan can evaluate them."""
    wf = (plan.get("workflow_type")
          or plan.get("pipeline_type")
          or plan.get("workflow")
          or "custom")
    if isinstance(wf, list):
        wf = wf[0] if wf else "custom"
    raw_steps = (plan.get("steps")
                 or plan.get("tasks")
                 or plan.get("pipeline")
                 or [])
    return {
        "workflow_type": str(wf),
        "steps": [_normalise_step(s) for s in raw_steps],
    }


# ── Base class ───────────────────────────────────────────────────

@dataclass
class CompetitorResult:
    plan: Dict[str, Any]
    wall_seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    cost_usd: float = 0.0
    raw_output: Optional[str] = None
    error: Optional[str] = None


class Competitor(ABC):
    """Interface every head-to-head competitor implements."""

    #: Short slug used in results CSV (e.g. ``"flowagent"``, ``"biomaster"``)
    id: str = ""

    #: Human-readable name for the figure label
    name: str = ""

    #: Citation / GitHub URL for the manuscript
    url: str = ""

    def available(self) -> Tuple[bool, str]:
        """Return ``(True, "")`` if the competitor can be invoked; else
        ``(False, <reason>)`` — used to skip with a clean message."""
        return True, ""

    @abstractmethod
    async def plan(self, prompt: str, *, context: Optional[Dict[str, Any]] = None
                   ) -> CompetitorResult:
        """Produce a FlowAgent-compatible plan for ``prompt``.

        Implementations should surface token / cost usage where possible
        via ``CompetitorResult``. A timeout is the caller's concern.
        """


# ── FlowAgent adapter (baseline / self) ──────────────────────────

class FlowAgentCompetitor(Competitor):
    """Baseline adapter — just forwards to ``LLMInterface.generate_workflow_plan``.

    Reuses the ``_TokenTracker`` wrapper pattern from ``bench_planning.py``
    so cost numbers are comparable across competitors.
    """

    id   = "flowagent"
    name = "FlowAgent"
    url  = "https://github.com/EnteloBio/flowagent"

    def __init__(self, model_cfg: Optional[Dict[str, Any]] = None):
        self.model_cfg = model_cfg or {}

    def available(self) -> Tuple[bool, str]:
        try:
            import flowagent  # noqa: F401
            return True, ""
        except Exception as e:
            return False, f"flowagent not importable: {e}"

    async def plan(self, prompt: str, *, context=None) -> CompetitorResult:
        from flowagent.core.llm import LLMInterface
        from flowagent.core.schemas import PipelineContext
        from harness.metrics import cost_usd

        # Reuse the _TokenTracker from bench_planning
        import sys
        bench_dir = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(bench_dir))
        from bench_planning import _TokenTracker  # noqa: E402

        ctx = PipelineContext(
            input_files=(context or {}).get(
                "input_files",
                ["HBR_Rep1_R1.fastq.gz", "HBR_Rep1_R2.fastq.gz"],
            ),
            paired_end=(context or {}).get("paired_end", True),
            organism=(context or {}).get("organism", "human"),
            genome_build=(context or {}).get("genome_build", "GRCh38"),
            workflow_type=(context or {}).get("workflow_type", ""),
        )

        t0 = time.perf_counter()
        llm = LLMInterface()
        tracker = _TokenTracker(llm.provider)
        llm.provider = tracker
        try:
            raw_plan = await llm.generate_workflow_plan(prompt, context=ctx)
            plan = _normalise_plan(raw_plan)
            return CompetitorResult(
                plan=plan,
                wall_seconds=time.perf_counter() - t0,
                prompt_tokens=tracker.prompt_tokens,
                completion_tokens=tracker.completion_tokens,
                llm_calls=tracker.call_count,
                cost_usd=cost_usd(tracker.prompt_tokens,
                                   tracker.completion_tokens,
                                   self.model_cfg),
            )
        except Exception as e:
            return CompetitorResult(
                plan=_empty_plan(),
                wall_seconds=time.perf_counter() - t0,
                error=f"{type(e).__name__}: {e}",
            )


# ── BioMaster adapter ────────────────────────────────────────────

# BioMaster: Multi-agent System for Automated Bioinformatics Analysis Workflow
# Su et al., bioRxiv 2025.01.23.634608
#
# Upstream is a script project (python run.py config.yaml) with no pip
# packaging. We drive it through ``biomaster_shim.py`` in this directory —
# a subprocess that synthesises a scratch config, runs BioMaster under a
# LangChain OpenAI callback to capture token/cost, and emits a richer JSON
# envelope (plan + tokens + cost + wall + error). Using a subprocess gives
# us process isolation (BioMaster's langchain globals and per-run output
# dirs don't collide across cells) plus a real kill-on-timeout.

_BIOMASTER_SHIM = Path(__file__).parent / "biomaster_shim.py"

_BIOMASTER_IMPORT_HINT = (
    "BioMaster is not configured. Two-step setup:\n"
    "  1. git clone <biomaster-repo> /path/to/BioMaster\n"
    "     cd /path/to/BioMaster && pip install -r requirements.txt\n"
    "  2. export BIOMASTER_DIR=/path/to/BioMaster\n"
    "The harness then drives BioMaster via harness/biomaster_shim.py.\n"
    "Alternatively, set BIOMASTER_CLI to your own executable that accepts\n"
    "'--prompt <text>' and prints the shim's JSON envelope on stdout.\n"
    "Upstream: https://www.biorxiv.org/content/10.1101/2025.01.23.634608"
)


class BioMasterCompetitor(Competitor):
    """Adapter for Su et al., 2025 — drives BioMaster via a subprocess shim.

    The shim handles config synthesis, token accounting, and output
    parsing (see ``biomaster_shim.py``). :meth:`available` reports False
    with a setup hint if neither ``$BIOMASTER_DIR`` nor ``$BIOMASTER_CLI``
    is set.
    """

    id   = "biomaster"
    name = "BioMaster"
    url  = "https://www.biorxiv.org/content/10.1101/2025.01.23.634608"

    def __init__(self, model: str = "gpt-4.1"):
        self.model = model
        # Explicit CLI override takes precedence over the bundled shim.
        self._cli = os.environ.get("BIOMASTER_CLI")
        self._biomaster_dir = os.environ.get("BIOMASTER_DIR")

    def _effective_cli(self) -> Optional[List[str]]:
        """Resolve the command to invoke. Returns the argv list or None."""
        if self._cli and Path(self._cli).exists() and os.access(self._cli, os.X_OK):
            return [self._cli]
        if self._biomaster_dir and _BIOMASTER_SHIM.exists():
            bm = Path(self._biomaster_dir).expanduser()
            if (bm / "agents" / "Biomaster.py").exists():
                return [sys.executable, str(_BIOMASTER_SHIM),
                        "--biomaster-dir", str(bm)]
        return None

    def available(self) -> Tuple[bool, str]:
        return (True, "") if self._effective_cli() else (False, _BIOMASTER_IMPORT_HINT)

    async def plan(self, prompt: str, *, context=None) -> CompetitorResult:
        ok, why = self.available()
        if not ok:
            return CompetitorResult(
                plan=_empty_plan(),
                wall_seconds=0.0,
                error=f"not-available: {why.splitlines()[0]}",
            )

        t0 = time.perf_counter()
        try:
            envelope = await self._invoke_shim(prompt, context)
        except Exception as e:
            return CompetitorResult(
                plan=_empty_plan(),
                wall_seconds=time.perf_counter() - t0,
                error=f"cli-adapter: {type(e).__name__}: {e}",
            )

        plan = _normalise_plan(envelope.get("plan") or {})
        return CompetitorResult(
            plan=plan,
            wall_seconds=float(envelope.get("wall_seconds") or
                               (time.perf_counter() - t0)),
            prompt_tokens=int(envelope.get("prompt_tokens") or 0),
            completion_tokens=int(envelope.get("completion_tokens") or 0),
            llm_calls=int(envelope.get("llm_calls") or 0),
            cost_usd=float(envelope.get("cost_usd") or 0.0),
            raw_output=json.dumps(envelope)[:20_000],
            error=envelope.get("error"),
        )

    async def _invoke_shim(self, prompt: str, context) -> Dict[str, Any]:
        """Run the shim (or user-provided CLI) and parse its JSON envelope.

        Context is optional; if ``input_files`` is present we forward it as
        BioMaster's ``datalist`` so the plan is generated against concrete
        input files rather than a generic default.
        """
        argv = self._effective_cli()
        assert argv is not None
        files: List[str] = []
        if context and context.get("input_files"):
            files = [f"{p}: input file" for p in context["input_files"]]
        argv = [*argv, "--prompt", prompt,
                "--files", json.dumps(files),
                "--model", self.model]
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        out = stdout.decode(errors="replace")
        # The shim always prints a JSON envelope on stdout, but a custom
        # $BIOMASTER_CLI may intermix chatter. Try to parse the whole
        # stream first; on failure, scan backwards for the last line that
        # is a well-formed JSON object.
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            pass
        for line in reversed(out.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        raise RuntimeError(
            f"BioMaster shim exited {proc.returncode}, "
            f"no JSON envelope in stdout ({len(out)}B). "
            f"stderr: {stderr.decode(errors='replace')[:400]}"
        )


# ── AutoBA adapter ───────────────────────────────────────────────

# AutoBA / Auto-BioinfoGPT: Zhou et al., 2023.
# Upstream is a script project (``python app.py --config cfg.yaml ...``) with
# no pip packaging. Same subprocess-shim strategy as BioMaster — the shim
# synthesises an AutoBA-native config, invokes ``app.py`` via runpy under
# an openai-client token patch, and emits a JSON envelope matching the
# BioMaster shape so this adapter can reuse most of the scaffolding.

_AUTOBA_SHIM = Path(__file__).parent / "autoba_shim.py"

_AUTOBA_IMPORT_HINT = (
    "AutoBA is not configured. Two-step setup:\n"
    "  1. git clone <autoba-repo> /path/to/AutoBA\n"
    "     cd /path/to/AutoBA && pip install openai pyyaml torch \\\n"
    "         llama-index-core llama-index-embeddings-openai \\\n"
    "         llama-index-embeddings-huggingface\n"
    "  2. export AUTOBA_DIR=/path/to/AutoBA\n"
    "Upstream: https://github.com/JoshuaChou2018/Auto-BioinfoGPT"
)


class AutoBACompetitor(Competitor):
    """Adapter for Zhou et al., 2023 — drives AutoBA via ``autoba_shim.py``.

    Mirrors the BioMaster adapter in structure: the shim handles config
    synthesis, token accounting (via an ``openai`` SDK patch), and JSON
    envelope emission. :meth:`available` reports False with a setup hint
    if neither ``$AUTOBA_DIR`` nor ``$AUTOBA_CLI`` is set.
    """

    id   = "autoba"
    name = "AutoBA"
    url  = "https://github.com/JoshuaChou2018/Auto-BioinfoGPT"

    def __init__(self, model: str = "gpt-4.1"):
        self.model = model
        self._cli = os.environ.get("AUTOBA_CLI")
        self._autoba_dir = os.environ.get("AUTOBA_DIR")

    def _effective_cli(self) -> Optional[List[str]]:
        if self._cli and Path(self._cli).exists() and os.access(self._cli, os.X_OK):
            return [self._cli]
        if self._autoba_dir and _AUTOBA_SHIM.exists():
            ab = Path(self._autoba_dir).expanduser()
            if (ab / "app.py").exists():
                return [sys.executable, str(_AUTOBA_SHIM),
                        "--autoba-dir", str(ab)]
        return None

    def available(self) -> Tuple[bool, str]:
        return (True, "") if self._effective_cli() else (False, _AUTOBA_IMPORT_HINT)

    async def plan(self, prompt: str, *, context=None) -> CompetitorResult:
        ok, why = self.available()
        if not ok:
            return CompetitorResult(
                plan=_empty_plan(), wall_seconds=0.0,
                error=f"not-available: {why.splitlines()[0]}",
            )

        t0 = time.perf_counter()
        try:
            envelope = await self._invoke_shim(prompt, context)
        except Exception as e:
            return CompetitorResult(
                plan=_empty_plan(), wall_seconds=time.perf_counter() - t0,
                error=f"cli-adapter: {type(e).__name__}: {e}",
            )

        plan = _normalise_plan(envelope.get("plan") or {})
        return CompetitorResult(
            plan=plan,
            wall_seconds=float(envelope.get("wall_seconds") or
                               (time.perf_counter() - t0)),
            prompt_tokens=int(envelope.get("prompt_tokens") or 0),
            completion_tokens=int(envelope.get("completion_tokens") or 0),
            llm_calls=int(envelope.get("llm_calls") or 0),
            cost_usd=float(envelope.get("cost_usd") or 0.0),
            raw_output=json.dumps(envelope)[:20_000],
            error=envelope.get("error"),
        )

    async def _invoke_shim(self, prompt: str, context) -> Dict[str, Any]:
        argv = self._effective_cli()
        assert argv is not None
        files: List[str] = []
        if context and context.get("input_files"):
            files = [f"{p}: input file" for p in context["input_files"]]
        argv = [*argv, "--prompt", prompt,
                "--files", json.dumps(files),
                "--model", self.model]
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        out = stdout.decode(errors="replace")
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            pass
        for line in reversed(out.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        raise RuntimeError(
            f"AutoBA shim exited {proc.returncode}, "
            f"no JSON envelope in stdout ({len(out)}B). "
            f"stderr: {stderr.decode(errors='replace')[:400]}"
        )


# ── Raw-LLM baseline ─────────────────────────────────────────────

# Zero-shot "LLM only" baseline — proves whether FlowAgent's planning
# scaffolding (preset catalogue, reference resolution, pattern
# extraction, JSON repair, error-recovery loop) adds value over asking
# a frontier model to emit the workflow plan directly. Reviewers always
# ask for this. Without it, a positive FlowAgent result could just be
# the underlying LLM doing all the work.
#
# The system prompt matches FlowAgent's own output schema so
# ``harness.metrics.score_plan`` can grade raw-LLM output on the same
# rubric. No context gathering, no repair loop, no presets — exactly
# one provider call per cell.

_RAW_LLM_SYSTEM_PROMPT = """You are a bioinformatics pipeline planner.
Given a user's request, produce a JSON workflow plan using this schema:

{
  "name": "<workflow_name>",
  "description": "<short description>",
  "workflow_type": "<rna_seq_kallisto | rna_seq_star | rna_seq_hisat | chip_seq | atac_seq | variant_calling | single_cell_10x | single_cell_kb | qc_only | custom>",
  "steps": [
    {
      "name": "<unique_snake_case_id>",
      "command": "<runnable shell command>",
      "dependencies": ["<prior step names>"],
      "outputs": ["<declared output paths>"]
    }
  ]
}

Rules:
- Steps must be in topological order.
- ``dependencies`` must reference names of prior steps exactly.
- Commands should be runnable shell pipelines using standard
  bioinformatics tools (fastqc, kallisto, salmon, STAR, bwa, samtools,
  macs2, cellranger, kb-python, deseq2 via Rscript, multiqc, etc.).
- Include every step needed to go from raw input to the requested output.

Return ONLY the JSON object. No markdown fences. No commentary."""


_PROVIDER_ENV_VARS = {
    "openai":    "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google":    "GOOGLE_API_KEY",
    "ollama":    None,
}


def _strip_code_fences(text: str) -> str:
    """Drop ```json … ``` wrappers if the model insisted on them."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines)
    return t.strip()


class RawLLMCompetitor(Competitor):
    """Zero-shot raw-LLM baseline — one provider call, no scaffolding.

    Establishes the ceiling of what the underlying model can do WITHOUT
    FlowAgent's planning infrastructure. If ``raw_gpt-5.4`` produces plans
    of comparable quality to FlowAgent on the same scoring rubric, the
    scaffolding isn't adding value; if it's measurably worse, it is.
    """

    def __init__(self, model_id: str, models_yaml_cfg: Optional[Dict[str, Any]] = None):
        self.model_id = model_id
        # Slug-safe id for results CSV; keep model id human-readable in name.
        self.id = f"raw_{model_id}"
        self.name = f"Raw LLM ({model_id})"
        self.url = ""
        # Used for pricing lookup. Accept either a single {id: ..., pricing: ...}
        # dict, or a full models.yaml cfg dict {"models": [...]}.
        self._cfg_full = models_yaml_cfg or {}

    def _model_cfg(self) -> Dict[str, Any]:
        if "pricing" in self._cfg_full:
            return self._cfg_full
        for m in self._cfg_full.get("models", []):
            if m.get("id") == self.model_id:
                return m
        return {}

    def _provider_name(self) -> str:
        # Local import to avoid hard coupling at module load.
        from flowagent.core.providers.registry import _infer_provider
        return _infer_provider(self.model_id)

    def available(self) -> Tuple[bool, str]:
        env_var = _PROVIDER_ENV_VARS.get(self._provider_name())
        if env_var and not os.environ.get(env_var):
            return False, f"{env_var} not set for raw {self.model_id}"
        return True, ""

    async def plan(self, prompt: str, *, context=None) -> CompetitorResult:
        from flowagent.core.providers import create_provider
        from harness.metrics import cost_usd
        bench_dir = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(bench_dir))
        from bench_planning import _TokenTracker  # noqa: E402

        provider_name = self._provider_name()
        env_var = _PROVIDER_ENV_VARS.get(provider_name)
        api_key = os.environ.get(env_var) if env_var else None

        t0 = time.perf_counter()
        try:
            provider = create_provider(
                provider_name, model=self.model_id, api_key=api_key,
            )
            tracker = _TokenTracker(provider)
            resp = await tracker.chat(
                [
                    {"role": "system", "content": _RAW_LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model=self.model_id,
            )
            content = _strip_code_fences(resp.content or "")
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as je:
                return CompetitorResult(
                    plan=_empty_plan(),
                    wall_seconds=time.perf_counter() - t0,
                    prompt_tokens=tracker.prompt_tokens,
                    completion_tokens=tracker.completion_tokens,
                    llm_calls=tracker.call_count,
                    cost_usd=cost_usd(
                        tracker.prompt_tokens, tracker.completion_tokens,
                        self._model_cfg(),
                    ),
                    error=f"JSONDecodeError: {je}",
                )
            plan = _normalise_plan(parsed)
            return CompetitorResult(
                plan=plan,
                wall_seconds=time.perf_counter() - t0,
                prompt_tokens=tracker.prompt_tokens,
                completion_tokens=tracker.completion_tokens,
                llm_calls=tracker.call_count,
                cost_usd=cost_usd(
                    tracker.prompt_tokens, tracker.completion_tokens,
                    self._model_cfg(),
                ),
            )
        except Exception as e:
            return CompetitorResult(
                plan=_empty_plan(),
                wall_seconds=time.perf_counter() - t0,
                error=f"{type(e).__name__}: {e}",
            )


# ── Registry ─────────────────────────────────────────────────────

# Default frontier models for the raw-LLM baseline lanes. Chosen to
# span providers (OpenAI, Anthropic, Google) without including the most
# expensive premium-reasoning tier.
DEFAULT_RAW_FRONTIER_MODELS = (
    "gpt-5.4",
    "claude-opus-4-7",
    "gemini-2.5-pro",
)


def build_registry(
    model_cfg: Optional[Dict[str, Any]] = None,
    raw_models: Optional[List[str]] = None,
    models_yaml_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Competitor]:
    """Construct the default competitor registry.

    Parameters
    ----------
    model_cfg : dict, optional
        Single-model cfg passed to FlowAgent / BioMaster / AutoBA lanes
        (they all share one "driver model" for token accounting).
    raw_models : list[str], optional
        Model IDs to add as zero-shot raw-LLM baselines. Each becomes a
        separate competitor keyed ``raw_<model_id>``. When omitted, no
        raw lanes are added — call with ``raw_models=DEFAULT_RAW_FRONTIER_MODELS``
        for the manuscript comparison.
    models_yaml_cfg : dict, optional
        Full ``config/models.yaml`` contents, used to look up pricing
        for the raw-LLM lanes.
    """
    model_id = (model_cfg or {}).get("id", "gpt-4.1")
    # Ordering: third-party competitors first, FlowAgent last. Means any
    # adapter / shim issues surface before the (known-good) FlowAgent
    # baseline is spent on, so a broken sweep fails fast.
    reg: Dict[str, Competitor] = {
        "autoba":     AutoBACompetitor(model=model_id),
        "biomaster":  BioMasterCompetitor(model=model_id),
        "flowagent":  FlowAgentCompetitor(model_cfg=model_cfg),
    }
    for raw_id in (raw_models or []):
        key = f"raw_{raw_id}"
        reg[key] = RawLLMCompetitor(model_id=raw_id, models_yaml_cfg=models_yaml_cfg)
    return reg
