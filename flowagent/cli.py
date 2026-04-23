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
    """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

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

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the web interface")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)

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

        # --preset shortcut
        if args.preset:
            from .presets.catalog import get_preset, list_presets, apply_context_to_preset
            from .core.pipeline_planner import gather_pipeline_context
            plan = get_preset(args.preset)
            if plan is None:
                names = ", ".join(p["id"] for p in list_presets())
                raise ValueError(f"Unknown preset '{args.preset}'. Available: {names}")
            logger.info("Using preset workflow: %s", plan["name"])

            # Run planning phase to resolve references
            ctx = await gather_pipeline_context(
                prompt,
                interactive=not getattr(args, "non_interactive", False),
            )
            plan = apply_context_to_preset(plan, ctx)

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


def run():
    """Run the CLI."""
    parser, args = parse_args()

    try:
        if args.command == "prompt":
            asyncio.run(main(prompt=args.prompt, args=args))

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
