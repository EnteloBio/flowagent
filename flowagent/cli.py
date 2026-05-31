"""Command line interface for FlowAgent."""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from .config.settings import Settings

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FlowAgent: AI-powered bioinformatics workflow assistant.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Scaffold a new project directory
    flowagent init my_rnaseq

    # Plan only — writes workflow.json + workflow.md for review
    flowagent plan "RNA-seq with kallisto on data/*.fastq.gz"
    flowagent plan "ChIP-seq hg38" --preset chipseq --out chipseq_plan/

    # Run a frozen plan (no re-LLM)
    flowagent run workflow.json
    flowagent run workflow.json --executor hpc --hpc-system slurm

    # Check status of a running / completed workflow
    flowagent status
    flowagent status --checkpoint-dir workflow_state/

    # Tail logs for a specific step
    flowagent logs kallisto_quant
    flowagent logs --checkpoint-dir workflow_state/ fastqc

    # Diff two plans
    flowagent diff workflow_v1.json workflow_v2.json

    # Ask the agent anything (default -- smart routing)
    flowagent prompt "Check which bioinformatics tools are installed"
    flowagent prompt "Run RNA-seq analysis on FASTQ files in data/"

    # Force a specific mode
    flowagent prompt "Run kallisto on my data" --workflow
    flowagent prompt "What files are here?" --agent

    # Use a preset (skips LLM planning)
    flowagent prompt "run it" --preset rnaseq-kallisto

    # Generate a pipeline file
    flowagent prompt "RNA-seq with HISAT2" --pipeline-format nextflow --no-execute

    # Analyze results
    flowagent prompt "analyze" --analysis-dir results/

    # Web interface
    flowagent serve

    # MCP tool server (for Cursor / VS Code)
    flowagent mcp serve
    """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ── init ────────────────────────────────────────────────────
    init_parser = subparsers.add_parser(
        "init", help="Scaffold a new FlowAgent project directory",
    )
    init_parser.add_argument(
        "directory", nargs="?", default=".",
        help="Target directory (default: current directory)",
    )
    init_parser.add_argument(
        "--preset", default=None,
        help="Copy a sample preset plan into the directory (e.g. rnaseq-kallisto)",
    )

    # ── plan ────────────────────────────────────────────────────
    plan_parser = subparsers.add_parser(
        "plan",
        help="Generate a workflow plan and save it for review (no execution)",
    )
    plan_parser.add_argument("prompt", help="Natural language description of the analysis")
    plan_parser.add_argument(
        "--out", default=".", metavar="DIR",
        help="Directory to write workflow.json + workflow.md (default: .)",
    )
    plan_parser.add_argument("--preset", default=None, help="Base plan on a preset instead of LLM")
    plan_parser.add_argument(
        "--non-interactive", action="store_true",
        help="Skip interactive reference questions; use defaults",
    )
    plan_parser.add_argument(
        "--model", default=None,
        help="LLM model override (e.g. gpt-4.1, claude-sonnet-4-6)",
    )
    plan_parser.add_argument("--no-dag", dest="no_dag", action="store_true")
    plan_parser.add_argument(
        "--samplesheet", default=None,
        help="Path to samplesheet CSV (nf-core format: sample,fastq_1,fastq_2,condition)",
    )

    # ── run ─────────────────────────────────────────────────────
    run_parser = subparsers.add_parser(
        "run", help="Execute a frozen plan (workflow.json) without re-planning",
    )
    run_parser.add_argument(
        "plan_path", nargs="?", default="workflow.json",
        help="Path to workflow.json (or directory containing it). Default: workflow.json",
    )
    run_parser.add_argument(
        "--executor", choices=["local", "cgat", "hpc", "kubernetes", "nextflow", "snakemake"],
        default=None,
    )
    run_parser.add_argument("--hpc-system", choices=["slurm", "sge", "torque"], default=None)
    run_parser.add_argument("--checkpoint-dir", default="workflow_state",
                            help="Directory for step checkpoints (default: workflow_state)")
    run_parser.add_argument("--resume", action="store_true", help="Skip already-completed steps")

    # ── status ──────────────────────────────────────────────────
    status_parser = subparsers.add_parser(
        "status", help="Show status of a workflow run",
    )
    status_parser.add_argument(
        "--checkpoint-dir", default="workflow_state",
        help="Checkpoint directory to inspect (default: workflow_state)",
    )
    status_parser.add_argument(
        "--plan", default="workflow.json",
        help="Plan file to cross-reference step names (default: workflow.json)",
    )

    # ── logs ────────────────────────────────────────────────────
    logs_parser = subparsers.add_parser(
        "logs", help="Print logs for a workflow step",
    )
    logs_parser.add_argument("step", nargs="?", default=None,
                             help="Step name to show (omit to list available steps)")
    logs_parser.add_argument(
        "--checkpoint-dir", default="workflow_state",
        help="Checkpoint directory (default: workflow_state)",
    )
    logs_parser.add_argument(
        "--tail", type=int, default=0, metavar="N",
        help="Show only the last N lines (default: all)",
    )

    # ── diff ────────────────────────────────────────────────────
    diff_parser = subparsers.add_parser(
        "diff", help="Diff two plan files",
    )
    diff_parser.add_argument("plan_a", help="First plan (path to workflow.json or directory)")
    diff_parser.add_argument("plan_b", help="Second plan (path to workflow.json or directory)")

    # ── prompt (existing) ───────────────────────────────────────
    prompt_parser = subparsers.add_parser(
        "prompt", help="Send a prompt to FlowAgent",
    )
    prompt_parser.add_argument("prompt", help="Natural language prompt")

    # Mode overrides (mutually exclusive)
    mode_group = prompt_parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--agent", action="store_true",
        help="Force agent mode (interactive tool-calling loop)",
    )
    mode_group.add_argument(
        "--workflow", action="store_true",
        help="Force workflow mode (plan-then-execute pipeline)",
    )

    # Workflow options
    prompt_parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    prompt_parser.add_argument("--force-resume", action="store_true", help="Resume and re-run all steps")
    prompt_parser.add_argument("--checkpoint-dir", help="Checkpoint directory")
    prompt_parser.add_argument("--analysis-dir", help="Directory to analyze")
    prompt_parser.add_argument(
        "--pipeline-format", choices=["nextflow", "snakemake"], default=None,
        help="Export as Nextflow or Snakemake pipeline",
    )
    prompt_parser.add_argument("--profile", default=None, help="Nextflow profile")
    prompt_parser.add_argument("--no-execute", action="store_true", help="Generate pipeline without running")
    prompt_parser.add_argument(
        "--executor", choices=["local", "cgat", "hpc", "kubernetes", "nextflow", "snakemake"],
        default=None, help="Execution backend",
    )
    prompt_parser.add_argument("--hpc-system", choices=["slurm", "sge", "torque"], default=None)
    prompt_parser.add_argument("--preset", default=None, help="Use a preset workflow (e.g. rnaseq-kallisto)")
    prompt_parser.add_argument(
        "--non-interactive", action="store_true",
        help="Skip interactive questions; use defaults (human/GRCh38/Ensembl)",
    )
    prompt_parser.add_argument(
        "--model", default=None,
        help="LLM model to use (e.g. claude-sonnet-4-5-20250929, gpt-4.1). "
             "Overrides LLM_MODEL; provider is auto-detected from the model prefix.",
    )
    prompt_parser.add_argument(
        "--no-dag", dest="no_dag", action="store_true",
        help="DAG-blind planner ablation: instruct the LLM to emit a flat "
             "ordered list of steps without any 'dependencies' field. "
             "Used by benchmarks/bench_ablation.py to test whether DAG "
             "awareness in the prompt improves bioinformatics plan quality.",
    )
    prompt_parser.add_argument(
        "--samplesheet", default=None,
        help="Path to samplesheet CSV (nf-core format: sample,fastq_1,fastq_2,condition)",
    )
    prompt_parser.add_argument(
        "--validate-output", default=None, metavar="CASE",
        help="After running, score outputs against a fidelity reference case "
             "(e.g. gse52778_dex_de). Requires benchmarks/ to be on sys.path.",
    )
    prompt_parser.add_argument(
        "--interpret", action="store_true",
        help="After running, run LLM interpretation on analysis outputs "
             "(Benchmark G questions). Requires --analysis-dir or auto-detected results/.",
    )

    # ── validate-output ─────────────────────────────────────────
    vo_parser = subparsers.add_parser(
        "validate-output",
        help="Score workflow outputs against a fidelity reference (Benchmark F)",
    )
    vo_parser.add_argument(
        "results_dir",
        help="Directory containing the workflow outputs to validate",
    )
    vo_parser.add_argument(
        "--case", default=None,
        help="Fidelity case ID to score against (e.g. gse52778_dex_de). "
             "Omit to auto-detect from results_dir contents.",
    )
    vo_parser.add_argument(
        "--cases-file", default=None,
        help="Path to fidelity_cases.yaml (default: built-in benchmarks/config/fidelity_cases.yaml)",
    )
    vo_parser.add_argument(
        "--references-dir", default=None,
        help="Directory containing reference outputs (default: benchmarks/references/)",
    )

    # ── interpret ───────────────────────────────────────────────
    interp_parser = subparsers.add_parser(
        "interpret",
        help="Run LLM interpretation on workflow outputs (Benchmark G style)",
    )
    interp_parser.add_argument(
        "results_dir",
        help="Directory containing workflow outputs to interpret",
    )
    interp_parser.add_argument(
        "--model", default=None,
        help="LLM model for interpretation (e.g. gpt-4.1)",
    )
    interp_parser.add_argument(
        "--no-save", action="store_true",
        help="Do not write report.md to results_dir",
    )

    # ── serve ───────────────────────────────────────────────────
    serve_parser = subparsers.add_parser("serve", help="Start the web interface")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)

    # ── mcp ─────────────────────────────────────────────────────
    mcp_parser = subparsers.add_parser(
        "mcp", help="Model Context Protocol tool server",
    )
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command")
    mcp_serve = mcp_subparsers.add_parser(
        "serve", help="Start an MCP server exposing FlowAgent tools",
    )
    mcp_serve.add_argument("--host", default="127.0.0.1")
    mcp_serve.add_argument("--port", type=int, default=8765)
    mcp_serve.add_argument("--stdio", action="store_true",
                           help="Use stdio transport (for Cursor/VS Code)")

    return parser, parser.parse_args()


_WORKFLOW_FALLBACK_PHRASES = (
    "run rna-seq", "run rnaseq", "run chipseq", "run chip-seq",
    "run atacseq", "run atac-seq", "run variant calling",
    "run fastqc", "run kallisto", "run hisat", "run star ",
    "run salmon", "run bowtie", "run bwa ", "run cellranger",
    "execute pipeline", "execute workflow", "run pipeline",
    "run workflow", "process fastq", "align reads",
    "quantify transcripts", "call peaks", "call variants",
    "download from geo", "download from sra",
)


_ROUTING_SYSTEM_PROMPT = (
    "You route prompts for a bioinformatics CLI. Classify the user's prompt "
    "as exactly one word: 'workflow' or 'agent'.\n"
    "- 'workflow' = a multi-step bioinformatics pipeline to plan and execute: "
    "RNA-seq / ChIP-seq / ATAC-seq / scRNA-seq / variant calling pipelines, "
    "GEO/SRA downloads followed by processing, differential expression with "
    "DESeq2/edgeR, peak calling with MACS2, alignment with STAR/HISAT2/BWA, "
    "quantification with kallisto/salmon, and similar end-to-end tasks.\n"
    "- 'agent' = an exploratory, conversational, or single-action question "
    "that doesn't warrant a planned pipeline: e.g. 'what bioinformatics "
    "tools are installed?', 'list files in data/', 'explain this FASTQ "
    "header', 'summarise this MultiQC report'.\n"
    "Reply with a single lowercase word — workflow or agent. No punctuation, "
    "no explanation."
)


async def _should_use_workflow(prompt: str, args) -> bool:
    """Decide whether a prompt should go through the workflow manager.

    Uses an LLM classifier against the configured model (``--model`` /
    ``LLM_MODEL``) so natural phrasings route correctly without requiring
    the user to remember ``--workflow``. Falls back to a small keyword list
    only if the classifier call fails (API down, quota, etc.), so routing
    still works offline or against a broken endpoint.
    """
    # Explicit flags always win.
    if args.workflow:
        return True
    if args.agent:
        return False
    # These flags imply workflow mode.
    if args.preset or args.pipeline_format or args.resume or args.checkpoint_dir:
        return True
    if args.analysis_dir:
        return False  # analysis is its own path

    # LLM-based routing. One tiny classification call (~60 in / 1 out) against
    # the configured model; negligible cost and latency compared to the
    # workflow itself, and robust against phrasing variations the keyword
    # list can't handle.
    try:
        from .core.llm import LLMInterface
        llm = LLMInterface()
        response = await llm._call_openai(
            [
                {"role": "system", "content": _ROUTING_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            timeout=30,
        )
        label = (response or "").strip().lower().split()[:1]
        label = label[0] if label else ""
        # Strip trailing punctuation from the label for robustness.
        label = label.rstrip(".,!?;:\"'")
        if label in ("workflow", "agent"):
            logger.info("LLM router classified prompt as '%s'", label)
            return label == "workflow"
        logger.warning(
            "LLM router returned unexpected response %r; falling back to keyword match",
            response,
        )
    except Exception as exc:
        logger.warning(
            "LLM router unavailable (%s); falling back to keyword match", exc,
        )

    # Fallback: legacy keyword-phrase match for when the LLM call can't run.
    p = prompt.lower()
    return any(phrase in p for phrase in _WORKFLOW_FALLBACK_PHRASES)


async def _run_agent_cli(prompt: str):
    """Run the agent loop from the CLI, printing results to stdout."""
    from .core.agent_loop import run_agent_loop
    from .core.providers import create_provider

    s = Settings()
    api_key = s.active_api_key or s.OPENAI_API_KEY
    if not api_key:
        logger.error("No API key configured. Set OPENAI_API_KEY in .env")
        sys.exit(1)

    provider = create_provider(
        s.LLM_PROVIDER, model=s.LLM_MODEL,
        api_key=api_key, base_url=s.LLM_BASE_URL,
    )

    def on_token(token: str):
        print(token, end="", flush=True)

    logger.info("Agent mode (tool-calling loop)")
    result = await run_agent_loop(provider, prompt, on_token=on_token)

    # If streaming didn't fire, print the full response
    if result.get("response") and not any(
        tc.get("name") for tc in result.get("tool_calls", [])
    ):
        print(result["response"])

    tool_calls = result.get("tool_calls", [])
    if tool_calls:
        print(f"\n\n[Agent used {len(tool_calls)} tool call(s) over {result.get('iterations', 0)} iteration(s)]")


async def main(
    prompt: str,
    args,
):
    """Main entry point for the CLI."""
    try:
        if getattr(args, "model", None):
            os.environ["LLM_MODEL"] = args.model
            # Infer provider from model prefix unless user set it explicitly.
            if "LLM_PROVIDER" not in os.environ:
                from .core.providers.registry import _infer_provider
                os.environ["LLM_PROVIDER"] = _infer_provider(args.model)

        if getattr(args, "no_dag", False):
            os.environ["LLM_DAG_AWARE"] = "false"
            logger.info(
                "DAG-blind planner mode enabled (LLM_DAG_AWARE=false)"
            )

        # --preset shortcut
        if args.preset:
            from .presets.catalog import get_preset, list_presets, apply_context_to_preset, expand_with_samplesheet
            from .core.pipeline_planner import gather_pipeline_context
            plan = get_preset(args.preset)
            if plan is None:
                names = ", ".join(p["id"] for p in list_presets())
                raise ValueError(f"Unknown preset '{args.preset}'. Available: {names}")
            logger.info("Using preset workflow: %s", plan["name"])

            samplesheet_path = getattr(args, "samplesheet", None)
            # Run planning phase to resolve references
            ctx = await gather_pipeline_context(
                prompt,
                interactive=not getattr(args, "non_interactive", False),
                samplesheet=samplesheet_path,
            )
            plan = apply_context_to_preset(plan, ctx)
            if samplesheet_path:
                plan = expand_with_samplesheet(plan, samplesheet_path)

            from .core.workflow_manager import WorkflowManager
            from .core.agent_types import Workflow, WorkflowStep
            wm = WorkflowManager(executor_type=args.executor or Settings().EXECUTOR_TYPE)
            steps = [WorkflowStep(
                name=s["name"], command=s["command"],
                dependencies=s.get("dependencies", []),
                description=s.get("description", ""),
            ) for s in plan["steps"]]
            wf = Workflow(name=plan["name"], description=plan.get("description", ""), steps=steps)
            result = await wm.execute_workflow(wf)
            print(f"\nWorkflow '{plan['name']}' finished: {result.get('status', 'unknown')}")
            if result.get("output_dir"):
                print(f"Output: {result['output_dir']}")
            return

        # --analysis-dir shortcut
        if args.analysis_dir:
            from .workflow import analyze_workflow
            logger.info("Analyzing workflow results in %s", args.analysis_dir)
            results = await analyze_workflow(args.analysis_dir)
            if results["status"] == "success":
                print("\n" + results["report"])
                if results.get("agentic_report"):
                    print("\n" + results["agentic_report"])
            else:
                print(f"Analysis failed: {results.get('error', 'Unknown error')}")
            return

        # --pipeline-format shortcut
        if args.pipeline_format:
            s = Settings()
            fmt = args.pipeline_format
            logger.info("Generating %s pipeline from prompt", fmt)
            from .core.llm import LLMInterface
            from .core.pipeline_generator import NextflowGenerator, SnakemakeGenerator
            from .core.pipeline_planner import gather_pipeline_context

            llm = LLMInterface()
            context = await gather_pipeline_context(
                prompt,
                interactive=not getattr(args, "non_interactive", False),
            )
            workflow_plan = await llm.generate_workflow_plan(prompt, context=context)
            output_dir = Path("flowagent_pipeline_output")
            gen = NextflowGenerator() if fmt == "nextflow" else SnakemakeGenerator()
            code = gen.generate(workflow_plan, output_dir=output_dir)
            print(f"\nGenerated {gen.default_filename()} in {output_dir}/")

            vresult = gen.validate(code, output_dir=output_dir)
            for warn in vresult.get("warnings", []):
                logger.warning("Validation: %s", warn)
            for err in vresult.get("errors", []):
                logger.error("Validation: %s", err)

            if args.no_execute:
                print("Pipeline generated (--no-execute). Run it manually.")
            else:
                import shutil
                runner = "nextflow" if fmt == "nextflow" else "snakemake"
                if not shutil.which(runner):
                    print(f"\n{runner} is not installed. Install it to execute the pipeline:")
                    if runner == "snakemake":
                        print(f"  conda install -c bioconda -c conda-forge snakemake")
                    else:
                        print(f"  conda install -c bioconda nextflow")
                    print(f"\nOr re-run with --no-execute to just generate the file.")
                else:
                    from .core.executor_factory import ExecutorFactory
                    executor = ExecutorFactory.create(fmt, profile=args.profile or s.PIPELINE_PROFILE)
                    # Run from the user's cwd (where input files are), not the output dir
                    pipeline_file = str(output_dir.resolve() / gen.default_filename())
                    step = {"name": f"{fmt}_run", "pipeline_file": pipeline_file, "cwd": os.getcwd()}
                    logger.info("Executing %s pipeline: %s", fmt, pipeline_file)
                    result = await executor.execute_step(step)

                    # ── Error recovery loop for pipeline execution ────
                    max_recovery = 3
                    for recovery_attempt in range(1, max_recovery + 1):
                        if result["status"] == "completed":
                            break

                        error_detail = result.get("stderr", "") or result.get("stdout", "")
                        logger.error("Pipeline failed (attempt %d/%d):\n%s",
                                     recovery_attempt, max_recovery, error_detail[-2000:])
                        logger.info("Attempting LLM error recovery...")

                        recovery_prompt = (
                            f"A {fmt} pipeline failed during execution. "
                            "Analyse the error and return a corrected workflow plan.\n\n"
                            f"Error output:\n{error_detail[-3000:]}\n\n"
                            f"Original workflow plan:\n{json.dumps(workflow_plan, indent=2)}\n\n"
                            "Fix ONLY the failing step command. Keep the primary "
                            "bioinformatics tool (fastqc, kallisto, multiqc, etc.) as "
                            "the FIRST token of the command so the pipeline generator "
                            "can route to the correct conda environment / container. "
                            "Do not prepend 'mkdir -p' to commands whose tool does not "
                            "create its output directory itself — rely on the existing "
                            "create_results_dirs step instead.\n\n"
                            "Common fixes:\n"
                            "- 'command not found' (exit 127): substitute an equivalent "
                            "tool (curl -fSL -o <file> <url> for wget, etc.)\n"
                            "- 'No such file or directory' for an OUTPUT dir: depend on "
                            "the directory-creation step rather than prepending mkdir.\n"
                            "- 'Specified output directory does not exist' (FastQC): "
                            "the output directory must exist before FastQC runs — add "
                            "a dependency on the directory-creation step.\n"
                            "- multiqc missing output / created _1.html / _2.html: add "
                            "'-f' flag so it overwrites existing files, and use '-n "
                            "multiqc_report' to force the exact output name.\n"
                            "Return the COMPLETE corrected workflow plan as JSON (same "
                            "schema as above). Return ONLY the JSON, no other text."
                        )
                        try:
                            raw = await llm._call_openai([
                                {"role": "system", "content": (
                                    "You are a bioinformatics pipeline debugging expert. "
                                    "Return only valid JSON."
                                )},
                                {"role": "user", "content": recovery_prompt},
                            ])
                            fixed_plan = json.loads(llm._clean_llm_response(raw))

                            # Regenerate and re-run
                            code = gen.generate(fixed_plan, output_dir=output_dir)
                            logger.info("Regenerated %s with LLM fix (attempt %d)",
                                        gen.default_filename(), recovery_attempt)
                            workflow_plan = fixed_plan
                            result = await executor.execute_step(step)
                        except Exception as rec_err:
                            logger.warning("Recovery attempt %d failed: %s",
                                           recovery_attempt, rec_err)
                            break

                    if result["status"] == "completed":
                        logger.info("Pipeline completed successfully")
                        if result.get("stdout"):
                            print(result["stdout"][-2000:])
                    else:
                        error_detail = result.get("stderr", "") or result.get("stdout", "")
                        logger.error("Pipeline failed after %d recovery attempts:\n%s",
                                     max_recovery, error_detail[-2000:])
            return

        # Smart routing: agent loop vs workflow manager
        if await _should_use_workflow(prompt, args):
            # Workflow path
            if args.resume and not args.checkpoint_dir:
                raise ValueError("--checkpoint-dir is required with --resume")
            if args.executor:
                os.environ["EXECUTOR_TYPE"] = args.executor
            if args.hpc_system:
                os.environ["HPC_SYSTEM"] = args.hpc_system
            s = Settings()
            exec_type = args.executor or s.EXECUTOR_TYPE
            logger.info("Workflow mode (executor=%s)", exec_type)
            from .workflow import run_workflow
            await run_workflow(prompt, args.checkpoint_dir, args.resume, args.force_resume)
        else:
            # Agent loop (default)
            await _run_agent_cli(prompt)

    except Exception as e:
        logger.error("Operation failed: %s", e)
        raise


def _cmd_init(args) -> None:
    """Scaffold a new FlowAgent project directory."""
    from .core.plan_store import save_plan
    from .presets.catalog import get_preset, list_presets

    target = Path(args.directory).resolve()
    target.mkdir(parents=True, exist_ok=True)
    for subdir in ("data", "reference", "results", "workflow_state"):
        (target / subdir).mkdir(exist_ok=True)

    # .env template
    env_path = target / ".env"
    if not env_path.exists():
        env_path.write_text(
            "# FlowAgent environment configuration\n"
            "# LLM_PROVIDER=openai\n"
            "# LLM_MODEL=gpt-4.1\n"
            "OPENAI_API_KEY=\n"
            "ANTHROPIC_API_KEY=\n"
        )

    # samplesheet template
    ss_path = target / "samplesheet.csv"
    if not ss_path.exists():
        ss_path.write_text("sample,fastq_1,fastq_2,condition\n")

    # Optional preset plan
    if args.preset:
        preset = get_preset(args.preset)
        if preset is None:
            names = ", ".join(p["id"] for p in list_presets())
            print(f"Unknown preset '{args.preset}'. Available: {names}")
            sys.exit(1)
        plan_path = save_plan(preset, target)
        print(f"[init] preset plan saved → {plan_path}")

    print(f"[init] project scaffolded in {target}")
    print("  data/           — place raw FASTQ / BAM files here")
    print("  reference/      — genome / transcriptome / annotation")
    print("  results/        — outputs will be written here")
    print("  workflow_state/ — step checkpoints and logs")
    print("  .env            — API keys and settings")
    print("  samplesheet.csv — edit to describe your samples")
    if args.preset:
        print("  workflow.json   — review the plan before running")
        print("  workflow.md     — human-readable plan summary")
    print("\nNext steps:")
    if args.preset:
        print(f"  flowagent run workflow.json")
    else:
        print(f'  flowagent plan "describe your analysis"')


async def _cmd_plan(args) -> None:
    """Generate a workflow plan and save it for review."""
    from .core.plan_store import save_plan
    from .config.settings import Settings

    if getattr(args, "model", None):
        os.environ["LLM_MODEL"] = args.model
        if "LLM_PROVIDER" not in os.environ:
            from .core.providers.registry import _infer_provider
            os.environ["LLM_PROVIDER"] = _infer_provider(args.model)

    if getattr(args, "no_dag", False):
        os.environ["LLM_DAG_AWARE"] = "false"

    out_dir = Path(args.out).resolve()
    s = Settings()

    samplesheet_path = getattr(args, "samplesheet", None)

    if args.preset:
        from .presets.catalog import get_preset, list_presets, apply_context_to_preset, expand_with_samplesheet
        from .core.pipeline_planner import gather_pipeline_context
        preset = get_preset(args.preset)
        if preset is None:
            names = ", ".join(p["id"] for p in list_presets())
            print(f"Unknown preset '{args.preset}'. Available: {names}")
            sys.exit(1)
        ctx = await gather_pipeline_context(
            args.prompt,
            interactive=not getattr(args, "non_interactive", False),
            samplesheet=samplesheet_path,
        )
        plan = apply_context_to_preset(preset, ctx)
        if samplesheet_path:
            plan = expand_with_samplesheet(plan, samplesheet_path)
    else:
        from .core.llm import LLMInterface
        from .core.pipeline_planner import gather_pipeline_context
        llm = LLMInterface()
        ctx = await gather_pipeline_context(
            args.prompt,
            interactive=not getattr(args, "non_interactive", False),
            samplesheet=samplesheet_path,
        )
        plan = await llm.generate_workflow_plan(args.prompt, context=ctx)

    plan_path = save_plan(
        plan, out_dir,
        prompt=args.prompt,
        model=os.environ.get("LLM_MODEL", s.LLM_MODEL),
    )
    md_path = plan_path.parent / "workflow.md"
    n = len(plan.get("steps", []))
    print(f"[plan] {n} steps → {plan_path}")
    print(f"[plan] review   → {md_path}")
    print("\nRun the plan with:")
    print(f"  flowagent run {plan_path}")


async def _cmd_run(args) -> None:
    """Execute a frozen plan without re-planning."""
    from .core.plan_store import load_plan
    from .core.agent_types import Workflow, WorkflowStep
    from .core.workflow_manager import WorkflowManager

    plan_path = Path(args.plan_path).resolve()
    plan = load_plan(plan_path)

    if args.executor:
        os.environ["EXECUTOR_TYPE"] = args.executor
    if args.hpc_system:
        os.environ["HPC_SYSTEM"] = args.hpc_system

    s = Settings()
    executor = args.executor or s.EXECUTOR_TYPE
    wm = WorkflowManager(executor_type=executor)

    steps = [WorkflowStep(
        name=step["name"],
        command=step.get("command", ""),
        dependencies=step.get("dependencies", []),
        description=step.get("description", ""),
    ) for step in plan.get("steps", [])]

    wf = Workflow(
        name=plan.get("name", "workflow"),
        description=plan.get("description", ""),
        steps=steps,
        checkpoint_dir=args.checkpoint_dir,
    )

    print(f"[run] executing {len(steps)} steps from {plan_path.name}")
    result = await wm.execute_workflow(wf)
    status = result.get("status", "unknown")
    print(f"[run] finished: {status}")
    if result.get("output_dir"):
        print(f"[run] output: {result['output_dir']}")


def _cmd_status(args) -> None:
    """Show status of workflow steps from checkpoint directory."""
    checkpoint_dir = Path(args.checkpoint_dir)
    plan_path = Path(args.plan) if hasattr(args, "plan") else Path("workflow.json")

    if not checkpoint_dir.exists():
        print(f"No checkpoint directory found at '{checkpoint_dir}'.")
        print("Run 'flowagent run workflow.json' first.")
        return

    # Load plan to get canonical step order
    step_order: list = []
    if plan_path.exists():
        try:
            import json as _json
            plan = _json.loads(plan_path.read_text())
            step_order = [s["name"] for s in plan.get("steps", [])]
        except Exception:
            pass

    # Read checkpoint files
    statuses: dict = {}
    for ck in sorted(checkpoint_dir.glob("*.json")):
        try:
            import json as _json
            data = _json.loads(ck.read_text())
            name = data.get("step_name") or ck.stem
            statuses[name] = data
        except Exception:
            pass

    if not statuses:
        print(f"No checkpoints found in '{checkpoint_dir}'.")
        return

    all_steps = step_order + [n for n in statuses if n not in step_order]
    _STATUS_ICONS = {
        "completed": "✓", "failed": "✗", "running": "⟳",
        "skipped": "–", "pending": "·",
    }

    print(f"\nWorkflow status ({checkpoint_dir}):\n")
    for name in all_steps:
        if name not in statuses:
            print(f"  · {name:40s}  pending")
            continue
        data = statuses[name]
        st = data.get("status", "unknown")
        icon = _STATUS_ICONS.get(st, "?")
        ts = data.get("end_time") or data.get("start_time") or ""
        ts_str = f"  [{ts[:19]}]" if ts else ""
        print(f"  {icon} {name:40s}  {st}{ts_str}")
    print()


def _cmd_logs(args) -> None:
    """Print logs for a specific step."""
    checkpoint_dir = Path(args.checkpoint_dir)
    step_name = args.step
    tail_n = args.tail

    log_dirs = [checkpoint_dir, checkpoint_dir / "logs", Path("workflow_state/logs")]

    if step_name is None:
        # List available step log files
        found = []
        for d in log_dirs:
            if d.exists():
                found.extend(d.glob("*.log"))
                found.extend(d.glob("*.txt"))
        if not found:
            print(f"No log files found in '{checkpoint_dir}'.")
        else:
            print("Available step logs:")
            for f in sorted(set(found)):
                print(f"  {f.stem}")
        return

    # Find the log file for the requested step
    candidates = []
    for d in log_dirs:
        for ext in ("log", "txt", "out", "stderr"):
            p = d / f"{step_name}.{ext}"
            if p.exists():
                candidates.append(p)

    if not candidates:
        # Fall back to checkpoint JSON stdout/stderr
        ck = checkpoint_dir / f"{step_name}.json"
        if ck.exists():
            import json as _json
            data = _json.loads(ck.read_text())
            for key in ("stdout", "stderr", "output"):
                content = data.get(key, "")
                if content:
                    lines = content.splitlines()
                    if tail_n:
                        lines = lines[-tail_n:]
                    print(f"--- {key} ---")
                    print("\n".join(lines))
            return
        print(f"No logs found for step '{step_name}' in '{checkpoint_dir}'.")
        return

    log_path = candidates[0]
    lines = log_path.read_text(errors="replace").splitlines()
    if tail_n:
        lines = lines[-tail_n:]
    print("\n".join(lines))


def _cmd_validate_output(args) -> None:
    """Score workflow outputs against a fidelity reference (Benchmark F)."""
    import subprocess
    import shutil
    from pathlib import Path

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    # Locate benchmark script
    bench_root = Path(__file__).parent.parent / "benchmarks"
    bench_script = bench_root / "bench_fidelity.py"
    if not bench_script.exists():
        print(
            "benchmarks/bench_fidelity.py not found. "
            "The FlowAgent benchmarks must be available alongside the package."
        )
        sys.exit(1)

    cases_file = args.cases_file or str(bench_root / "config" / "fidelity_cases.yaml")
    refs_dir = args.references_dir or str(bench_root / "references")

    cmd = [
        sys.executable, str(bench_script),
        "--cases", cases_file,
        "--candidate-dir", str(results_dir),
        "--reference-dir", refs_dir,
    ]
    if args.case:
        cmd += ["--case", args.case]

    print(f"[validate-output] scoring {results_dir.name} ...")
    try:
        result = subprocess.run(cmd, check=False, cwd=str(bench_root))
        sys.exit(result.returncode)
    except FileNotFoundError:
        print("Python interpreter not found. This is unexpected.")
        sys.exit(1)


async def _cmd_interpret(args) -> None:
    """Run LLM interpretation on workflow outputs."""
    if getattr(args, "model", None):
        os.environ["LLM_MODEL"] = args.model
        if "LLM_PROVIDER" not in os.environ:
            from .core.providers.registry import _infer_provider
            os.environ["LLM_PROVIDER"] = _infer_provider(args.model)

    from .workflow import analyze_workflow
    results_dir = Path(args.results_dir).resolve()
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    save_report = not getattr(args, "no_save", False)
    print(f"[interpret] analysing {results_dir} ...")
    result = await analyze_workflow(str(results_dir), save_report=save_report)

    if result.get("status") == "success":
        print("\n" + result.get("report", ""))
        if result.get("agentic_report"):
            print("\n" + result["agentic_report"])
    else:
        print(f"Interpretation failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


def _cmd_diff(args) -> None:
    """Diff two plan files."""
    from .core.plan_store import load_plan, diff_plans

    path_a = Path(args.plan_a)
    path_b = Path(args.plan_b)

    plan_a = load_plan(path_a)
    plan_b = load_plan(path_b)

    name_a = plan_a.get("name", path_a.name)
    name_b = plan_b.get("name", path_b.name)
    print(f"Diff: {name_a}  →  {name_b}\n")
    print(diff_plans(plan_a, plan_b))


def run():
    """Run the CLI."""
    parser, args = parse_args()

    try:
        if args.command == "init":
            _cmd_init(args)

        elif args.command == "plan":
            asyncio.run(_cmd_plan(args))

        elif args.command == "run":
            asyncio.run(_cmd_run(args))

        elif args.command == "status":
            _cmd_status(args)

        elif args.command == "logs":
            _cmd_logs(args)

        elif args.command == "diff":
            _cmd_diff(args)

        elif args.command == "validate-output":
            _cmd_validate_output(args)

        elif args.command == "interpret":
            asyncio.run(_cmd_interpret(args))

        elif args.command == "prompt":
            asyncio.run(main(prompt=args.prompt, args=args))

        elif args.command == "mcp":
            if getattr(args, "mcp_command", None) == "serve":
                from .mcp_server import run_mcp_server
                run_mcp_server(
                    host=args.host,
                    port=args.port,
                    stdio=args.stdio,
                )
            else:
                # Print mcp subparser help
                parser.parse_args(["mcp", "--help"])

        elif args.command == "serve":
            os.environ["USER_EXECUTION_DIR"] = os.getcwd()
            try:
                import uvicorn
            except ImportError:
                logger.error("uvicorn required: pip install 'flowagent[web]'")
                sys.exit(1)

            logger.info("Starting FlowAgent web UI on %s:%s", args.host, args.port)
            try:
                uvicorn.run("flowagent.web:app", host=str(args.host), port=int(args.port), log_level="info")
            except KeyboardInterrupt:
                logger.info("Shutting down...")
        else:
            parser.print_help()
            sys.exit(1)

    except Exception as e:
        logger.error("Operation failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    run()
