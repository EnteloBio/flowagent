"""Command line interface for FlowAgent."""

import argparse
import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .workflow import analyze_workflow, run_workflow
from .config.settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FlowAgent: Multi-agent framework for automating bioinformatics workflows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    1. Run a workflow (shell commands, local):
       flowagent prompt "Analyze RNA-seq data using Kallisto..."

    2. Generate a Nextflow pipeline and execute it:
       flowagent prompt "Run RNA-seq with HISAT2" --pipeline-format nextflow --profile docker

    3. Generate a Snakemake pipeline (no auto-execute):
       flowagent prompt "Variant calling pipeline" --pipeline-format snakemake --no-execute

    4. Use cgat-core for HPC execution:
       flowagent prompt "Process ChIP-seq data" --executor cgat

    5. Analyze results:
       flowagent prompt "analyze workflow results" --analysis-dir results

    6. Start web interface:
       flowagent serve --host 0.0.0.0 --port 8000
    """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Prompt command
    prompt_parser = subparsers.add_parser(
        "prompt", help="Execute a workflow or analyze results"
    )
    prompt_parser.add_argument("prompt", help="Workflow prompt")
    prompt_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume workflow from checkpoint",
    )
    prompt_parser.add_argument(
        "--force-resume",
        action="store_true",
        help="Force resuming workflow and run all steps, regardless of completion status",
    )
    prompt_parser.add_argument(
        "--checkpoint-dir",
        help="Directory for workflow checkpoints",
    )
    prompt_parser.add_argument(
        "--analysis-dir",
        help="Directory containing workflow results to analyze",
    )

    # Pipeline generation options
    prompt_parser.add_argument(
        "--pipeline-format",
        choices=["shell", "nextflow", "snakemake"],
        default=None,
        help="Pipeline output format (default: from settings or 'shell')",
    )
    prompt_parser.add_argument(
        "--profile",
        default=None,
        help="Nextflow profile (local, docker, singularity, slurm)",
    )
    prompt_parser.add_argument(
        "--no-execute",
        action="store_true",
        help="Generate pipeline files without executing them",
    )

    # Executor options
    prompt_parser.add_argument(
        "--executor",
        choices=["local", "cgat", "hpc", "kubernetes", "nextflow", "snakemake"],
        default=None,
        help="Execution backend (default: from settings or 'local')",
    )
    prompt_parser.add_argument(
        "--hpc-system",
        choices=["slurm", "sge", "torque"],
        default=None,
        help="HPC scheduler (used with --executor hpc)",
    )

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the web interface")
    serve_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server to",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to",
    )

    return parser, parser.parse_args()


async def main(
    prompt: str,
    resume: bool = False,
    force_resume: bool = False,
    checkpoint_dir: Optional[str] = None,
    analysis_dir: Optional[str] = None,
    pipeline_format: Optional[str] = None,
    profile: Optional[str] = None,
    no_execute: bool = False,
    executor: Optional[str] = None,
    hpc_system: Optional[str] = None,
):
    """Main entry point for the CLI."""
    try:
        if analysis_dir:
            logger.info(f"Analyzing workflow results in {analysis_dir}")
            results = await analyze_workflow(analysis_dir)

            if results["status"] == "success":
                print("\nStandard Analysis Report:")
                print("=" * 80)
                print(results["report"])
                print("=" * 80)

                print("\nAgentic Analysis Report:")
                print("=" * 80)
                print(results["agentic_report"])
                print("=" * 80)

                if results.get("report_files"):
                    print(
                        f"\nAnalysis reports saved to: {', '.join(results['report_files'].values())}"
                    )
            else:
                print(f"Analysis failed: {results.get('error', 'Unknown error')}")

        else:
            if resume and not checkpoint_dir:
                raise ValueError(
                    "Checkpoint directory must be provided if resume is set to True"
                )

            # Apply CLI overrides to settings
            s = Settings()
            fmt = pipeline_format or s.PIPELINE_FORMAT
            exec_type = executor or s.EXECUTOR_TYPE

            # Pipeline generation path
            if fmt in ("nextflow", "snakemake"):
                logger.info("Generating %s pipeline from prompt…", fmt)
                from .core.llm import LLMInterface
                from .core.pipeline_generator import NextflowGenerator, SnakemakeGenerator

                llm = LLMInterface()
                workflow_plan = await llm.generate_workflow_plan(prompt)

                output_dir = Path("flowagent_pipeline_output")
                if fmt == "nextflow":
                    gen = NextflowGenerator()
                else:
                    gen = SnakemakeGenerator()

                code = gen.generate(workflow_plan, output_dir=output_dir)
                print(f"\nGenerated {gen.default_filename()} in {output_dir}/")

                # Validate
                vresult = gen.validate(code, output_dir=output_dir)
                if vresult["errors"]:
                    for err in vresult["errors"]:
                        logger.error("Validation error: %s", err)
                if vresult["warnings"]:
                    for w in vresult["warnings"]:
                        logger.warning("Validation: %s", w)

                # Execute unless --no-execute
                if not no_execute and vresult.get("valid", True):
                    from .core.executor_factory import ExecutorFactory
                    pipeline_executor = ExecutorFactory.create(
                        fmt,
                        profile=profile or s.PIPELINE_PROFILE,
                    )
                    step = {
                        "name": f"{fmt}_run",
                        "pipeline_file": str(output_dir / gen.default_filename()),
                        "cwd": str(output_dir),
                    }
                    result = await pipeline_executor.execute_step(step)
                    if result["status"] == "completed":
                        logger.info("Pipeline execution completed successfully")
                    else:
                        logger.error("Pipeline execution failed:\n%s", result.get("stderr", ""))
                elif no_execute:
                    print("Pipeline generated (--no-execute). Run it manually.")
            else:
                # Classic shell-command workflow — WorkflowManager reads EXECUTOR_TYPE / HPC_SYSTEM from env
                if executor:
                    os.environ["EXECUTOR_TYPE"] = executor
                if hpc_system:
                    os.environ["HPC_SYSTEM"] = hpc_system
                logger.info("Starting workflow (executor=%s)", exec_type)
                await run_workflow(prompt, checkpoint_dir, resume, force_resume)

    except Exception as e:
        logger.error(f"Operation failed: {str(e)}")
        raise


def run():
    """Run the CLI."""
    parser, args = parse_args()

    try:
        if args.command == "prompt":
            asyncio.run(
                main(
                    prompt=args.prompt,
                    resume=args.resume,
                    force_resume=args.force_resume,
                    checkpoint_dir=args.checkpoint_dir,
                    analysis_dir=args.analysis_dir,
                    pipeline_format=args.pipeline_format,
                    profile=args.profile,
                    no_execute=args.no_execute,
                    executor=args.executor,
                    hpc_system=args.hpc_system,
                )
            )
        elif args.command == "serve":
            # Start new web interface
            web_path = Path(__file__).parent / "web.py"

            env = os.environ.copy()
            env["USER_EXECUTION_DIR"] = os.getcwd()

            project_dir = Path(__file__).parent.parent
            os.chdir(project_dir)

            try:
                subprocess.run(
                    [
                        "chainlit",
                        "run",
                        web_path.absolute(),
                        "--host",
                        str(args.host),
                        "--port",
                        str(args.port),
                    ],
                    env=env,
                )
            except KeyboardInterrupt:
                logger.info("Shutting down web interface...")
        else:
            # Show help if no command specified
            parser.print_help()
            sys.exit(1)

    except Exception as e:
        logger.error(f"Operation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    run()
