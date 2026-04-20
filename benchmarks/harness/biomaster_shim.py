#!/usr/bin/env python3
"""Subprocess shim that makes BioMaster (Su et al., 2025) callable from the
head-to-head benchmark harness.

BioMaster is distributed as a script project — `python run.py config.yaml`.
It has no pip packaging and writes its PLAN to ``<output_dir>/<id>_PLAN.json``,
with per-step shell scripts at ``<output_dir>/<id>_Step_N.sh``. This shim:

  1. Synthesises a scratch config that sets BioMaster's ``data.goal`` to the
     incoming prompt and ``data.files`` to any provided input files.
  2. Wraps the whole invocation in a LangChain ``get_openai_callback`` so we
     capture token usage and cost for apples-to-apples comparison.
  3. Invokes ``execute_PLAN`` then ``execute_TASK`` with ``excutor=False`` —
     BioMaster generates the plan + shell scripts without executing them.
  4. Reads back ``<id>_PLAN.json`` + the generated ``<id>_Step_N.sh`` scripts,
     maps the result into FlowAgent's ``{workflow_type, steps[]}`` schema,
     and emits a single JSON object on stdout with cost/wall/error fields.

Usage (called by ``BioMasterCompetitor._invoke_cli`` via subprocess)::

    python biomaster_shim.py --prompt "Run RNA-seq with kallisto"

Options::

    --prompt <str>          Required. BioMaster ``goal``.
    --files <json-array>    Optional. List of "path: description" strings.
    --biomaster-dir <path>  Path to the BioMaster clone. Defaults to
                            $BIOMASTER_DIR.
    --model <str>           Main + tool model (default: env OPENAI_MODEL,
                            else "gpt-4.1").
    --api-key / --base-url  Override env OPENAI_API_KEY / OPENAI_BASE_URL.
    --embedding-model <str> Defaults to "text-embedding-3-small".
    --timeout <sec>         Per-stage timeout (not a hard kill).

The parent harness applies the real timeout with ``asyncio.wait_for``; this
script just needs to exit cleanly when it finishes so the subprocess can be
killed on timeout.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Reuse the harness's dotenv loader so direct invocation (not only driven
# by bench_competitors.py) picks up benchmarks/.env without a shell export.
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from harness.runner import _load_dotenv_once  # type: ignore
    _load_dotenv_once()
except Exception:
    pass


def _die(msg: str, *, error_type: str = "shim-error", exit_code: int = 2) -> None:
    """Emit a JSON error envelope on stdout and exit non-zero."""
    print(json.dumps({
        "plan": {"workflow_type": "custom", "steps": []},
        "prompt_tokens": 0, "completion_tokens": 0,
        "llm_calls": 0, "cost_usd": 0.0, "wall_seconds": 0.0,
        "error": f"{error_type}: {msg}",
    }))
    sys.exit(exit_code)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--files", default="[]",
                    help='JSON array of "path: description" strings')
    ap.add_argument("--biomaster-dir",
                    default=os.environ.get("BIOMASTER_DIR", ""))
    ap.add_argument("--model",
                    default=os.environ.get("OPENAI_MODEL", "gpt-4.1"))
    ap.add_argument("--api-key",
                    default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--base-url",
                    default=os.environ.get("OPENAI_BASE_URL",
                                           "https://api.openai.com/v1"))
    ap.add_argument("--embedding-model", default="text-embedding-3-small")
    ap.add_argument("--keep-output", action="store_true",
                    help="Do not delete the scratch output dir on exit")
    return ap.parse_args()


def _map_step(raw: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Project a BioMaster PLAN step onto FlowAgent's step schema.

    BioMaster writes generated shell scripts only after ``execute_TASK``
    completes — which hits an upstream UnboundLocalError under
    ``executor: false``. We therefore use the PLAN.json metadata itself:
    ``tools`` (tool-name string) as the ``command`` source so
    ``score_plan``'s tool matcher sees what BioMaster *planned* to use,
    ``description`` for context, and ``output_filename`` for outputs.
    """
    step_number = raw.get("step_number", idx + 1)
    tools = raw.get("tools", "")
    if isinstance(tools, list):
        tools = " ".join(tools)
    return {
        "name":         f"step_{step_number}",
        "command":      str(tools),
        "dependencies": [f"step_{step_number - 1}"] if step_number > 1 else [],
        "outputs":      raw.get("output_filename") or [],
        "description":  raw.get("description") or "",
    }


# Ordered tool-name → FlowAgent workflow_type mapping. Specific wins over
# general — the first match wins, so domain-specific signatures (bismark
# for methylation, kraken for metagenomics) come before broad aligners.
# The slugs mirror the canonical values in corpus/prompts.yaml.
_WORKFLOW_TYPE_RULES: List[Tuple[str, List[str]]] = [
    ("methylation",         ["bismark", "methylkit", "methyldackel"]),
    ("metagenomics",        ["kraken", "bracken", "metaphlan"]),
    ("amplicon",            ["dada2", "qiime"]),
    ("small_rna",           ["mirdeep", "mirbase"]),
    ("hic",                 ["pairtools", "hicpro", "juicer", "hic-pro"]),
    ("cnv",                 ["cnvkit", "qdnaseq"]),
    ("structural_variants", ["manta", "delly", "lumpy", "sniffles"]),
    ("long_read",           ["nanopolish", "medaka", "minimap2"]),
    ("assembly",            ["hifiasm", "flye", "canu", "spades", "unicycler"]),
    ("sc_rna_seq",          ["cellranger", "starsolo", "kb-python", "kb ref",
                             "kb count"]),
    ("cut_and_run",         ["seacr"]),
    ("chip_seq",            ["macs2", "macs3"]),
    ("variant_calling",     ["mutect2", "haplotypecaller", "gatk", "bcftools",
                             "deepvariant", "strelka"]),
    ("rna_seq_kallisto",    ["kallisto"]),
    ("rna_seq_salmon",      ["salmon"]),
    ("rna_seq_hisat2",      ["hisat2"]),
    ("rna_seq_star",        ["star"]),
]


def _classify_workflow_type(plan: Dict[str, Any]) -> str:
    """Infer a FlowAgent-canonical workflow_type from a BioMaster plan.

    BioMaster has no workflow taxonomy — its plans are just ordered steps.
    FlowAgent's scoring (harness/metrics.py::type_matches) is asymmetric:
    ``custom`` is a wildcard on the *expected* side, not the actual side,
    so emitting ``custom`` fails the ``type_correct`` gate on strictly
    typed prompts (rnaseq_kallisto_basic, chipseq_macs2, …).

    This classifier gives BioMaster the equivalent benefit FlowAgent earns
    by labelling its plan — a post-hoc mapper over the tools BioMaster
    itself chose. Does not fabricate capability; only labels what's there.
    """
    haystack_parts = []
    for step in plan.get("steps", []):
        haystack_parts.append(step.get("command") or "")
        haystack_parts.append(step.get("description") or "")
    haystack = " ".join(haystack_parts).lower()
    if not haystack.strip():
        return "custom"
    for slug, needles in _WORKFLOW_TYPE_RULES:
        if any(n in haystack for n in needles):
            return slug
    return "custom"


def _write_scratch_config(*, biomaster_dir: Path, scratch_dir: Path,
                          run_id: str, args: argparse.Namespace,
                          datalist: List[str], goal: str) -> Path:
    """Materialise a BioMaster-compatible config.yaml in the scratch dir.

    The ``id`` is a fresh uuid-derived slug so concurrent cells don't collide
    on output filenames. ``executor: false`` so BioMaster plans+writes shell
    scripts but does not run them (we only want the plan for scoring).
    """
    cfg = {
        "api": {
            "main": {"key": args.api_key, "base_url": args.base_url},
            "embedding": {"key": args.api_key, "base_url": args.base_url},
            "ollama": {"enabled": False,
                       "base_url": "http://localhost:11434"},
        },
        "models": {
            "main": args.model,
            "tool": args.model,
            "embedding": args.embedding_model,
        },
        "biomaster": {
            "executor": False,
            "id": run_id,
            "generate_plan": True,
            "use_ollama": False,
        },
        "data": {"files": datalist, "goal": goal},
    }
    try:
        import yaml  # BioMaster already requires this
    except ImportError:
        _die("PyYAML not available in this env (BioMaster needs it too)",
             error_type="missing-dep")
    cfg_path = scratch_dir / "config.yaml"
    with cfg_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return cfg_path


def main() -> None:
    args = _parse_args()
    t0 = time.perf_counter()

    if not args.biomaster_dir:
        _die("Set --biomaster-dir or $BIOMASTER_DIR to the BioMaster clone",
             error_type="config")
    biomaster_dir = Path(args.biomaster_dir).expanduser().resolve()
    if not (biomaster_dir / "agents" / "Biomaster.py").exists():
        _die(f"{biomaster_dir} does not look like a BioMaster clone "
             "(missing agents/Biomaster.py)", error_type="config")

    try:
        datalist = json.loads(args.files)
        if not isinstance(datalist, list):
            raise ValueError("--files must decode to a list of strings")
    except Exception as e:
        _die(f"--files must be a JSON array: {e}", error_type="config")

    # Fresh id + scratch output dir. uuid4 collision-free for concurrent cells.
    run_id = uuid.uuid4().hex[:10]
    scratch_dir = Path(tempfile.mkdtemp(prefix=f"biomaster_{run_id}_"))
    output_dir = scratch_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = _write_scratch_config(
        biomaster_dir=biomaster_dir, scratch_dir=scratch_dir,
        run_id=run_id, args=args, datalist=datalist, goal=args.prompt,
    )

    # BioMaster resolves `./output`, `./chroma_db`, and `./doc` relative to
    # cwd. Run from the scratch dir so scratch writes (output, chroma_db,
    # token.txt) don't pollute the clone; symlink the clone's read-only
    # reference dirs (doc/ holds Plan_Knowledge.json + Task_Knowledge.json)
    # so BioMaster's hardcoded ``self.doc_dir = "doc"`` still resolves.
    prev_cwd = os.getcwd()
    for ref in ("doc",):
        src = biomaster_dir / ref
        if src.exists():
            try:
                (scratch_dir / ref).symlink_to(src)
            except OSError:
                # Fall back to copy if symlinks aren't supported on the fs
                shutil.copytree(src, scratch_dir / ref)
    os.chdir(scratch_dir)
    sys.path.insert(0, str(biomaster_dir))

    prompt_tokens = completion_tokens = llm_calls = 0
    cost_usd = 0.0
    error: Optional[str] = None
    plan: Dict[str, Any] = {"workflow_type": "custom", "steps": []}

    # BioMaster and its dependencies (chromadb telemetry, langchain, the
    # Biomaster class itself) pepper stdout with status prints. Redirect
    # all that noise to stderr so stdout stays clean for our JSON envelope.
    with contextlib.redirect_stdout(sys.stderr):
        try:
            # Import the token-accounting context manager now. Everything
            # else (including BioMaster itself) is imported inside
            # ``runpy.run_path`` below — the same way ``python run.py``
            # would import them if a user ran BioMaster straight after a
            # fresh clone. No direct references to agents.Biomaster here.
            try:
                from langchain_community.callbacks import get_openai_callback
            except Exception:
                from langchain.callbacks import get_openai_callback  # type: ignore
            import runpy

            # Ape ``python run.py config.yaml`` exactly: set argv, cwd,
            # and sys.path as that invocation would, then hand off to
            # runpy. Wrapping with the OpenAI callback lets us capture
            # token use without modifying BioMaster's code path.
            run_py = biomaster_dir / "run.py"
            if not run_py.exists():
                raise RuntimeError(f"{run_py} not found")
            prev_argv = sys.argv
            sys.argv = [str(run_py), str(cfg_path)]
            try:
                with get_openai_callback() as cb:
                    try:
                        runpy.run_path(str(run_py), run_name="__main__")
                    except SystemExit:
                        pass  # clean script exits are fine
                    except Exception as run_exc:
                        # Upstream BioMaster has an UnboundLocalError in
                        # ``execute_TASK`` when ``executor: false``. PLAN.json
                        # is written by ``execute_PLAN`` *before* that crash,
                        # so we keep going and score the plan — recording
                        # the crash as a soft error so it's visible in the
                        # manifest.
                        error = (f"run.py raised: "
                                 f"{type(run_exc).__name__}: {run_exc}")
                    prompt_tokens = int(getattr(cb, "prompt_tokens", 0) or 0)
                    completion_tokens = int(
                        getattr(cb, "completion_tokens", 0) or 0)
                    llm_calls = int(
                        getattr(cb, "successful_requests", 0) or 0)
                    cost_usd = float(getattr(cb, "total_cost", 0.0) or 0.0)
            finally:
                sys.argv = prev_argv

            # run.py writes PLAN.json under ./output/<id>_PLAN.json (relative
            # to cwd, which is our scratch dir). If execute_PLAN's internal
            # JSON parsing failed, upstream silently returns None without
            # saving — treat that as a fair "BioMaster failed to plan"
            # result, not a shim error.
            plan_path = Path("output") / f"{run_id}_PLAN.json"
            if not plan_path.exists():
                if error is None:
                    error = ("execute_PLAN produced no PLAN.json "
                             "(upstream LLM output wasn't valid JSON)")
                raw_steps: List[Dict[str, Any]] = []
            else:
                # Tolerate BioMaster's two observed PLAN.json shapes:
                # the canonical ``{"plan": [...]}`` wrapper, or a bare
                # top-level list of steps.
                raw_plan = json.loads(plan_path.read_text(encoding="utf-8"))
                if isinstance(raw_plan, list):
                    raw_steps = raw_plan
                elif isinstance(raw_plan, dict):
                    raw_steps = raw_plan.get("plan") or []
                else:
                    raw_steps = []
                if not isinstance(raw_steps, list):
                    raw_steps = []
                raw_steps = [s for s in raw_steps if isinstance(s, dict)]
            steps = [_map_step(s, i) for i, s in enumerate(raw_steps)]
            plan = {
                "workflow_type": _classify_workflow_type({"steps": steps}),
                "steps": steps,
            }
            # If we have a plan to score, drop any soft error from the
            # upstream ``execute_TASK`` UnboundLocalError — it's expected
            # and irrelevant once PLAN.json is in hand. Preserve errors
            # only when scoring has nothing to work with.
            if steps and error and error.startswith("run.py raised"):
                error = None

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-800:]}"
        finally:
            os.chdir(prev_cwd)
            if not args.keep_output:
                shutil.rmtree(scratch_dir, ignore_errors=True)

    print(json.dumps({
        "plan": plan,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "llm_calls": llm_calls,
        "cost_usd": cost_usd,
        "wall_seconds": time.perf_counter() - t0,
        "error": error,
    }))


if __name__ == "__main__":
    main()
