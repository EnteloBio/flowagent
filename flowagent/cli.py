"""Command line interface for FlowAgent."""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import argparse

from .workflow import run_workflow, analyze_workflow
from .api import start_server

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
        description="FlowAgent: A modern framework for RNA-seq analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    1. Run a workflow:
       flowagent prompt "Analyze RNA-seq data using Kallisto..." --checkpoint-dir workflow_state

    2. Analyze results:
       flowagent prompt "analyze workflow results" --analysis-dir results

    3. Start web interface:
       flowagent serve --host 0.0.0.0 --port 8000
    """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Prompt command
    prompt_parser = subparsers.add_parser("prompt", help="Execute a workflow or analyze results")
    prompt_parser.add_argument("prompt", help="Workflow prompt")
    prompt_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume workflow from checkpoint",
    )
    prompt_parser.add_argument(
        "--checkpoint-dir",
        help="Directory for workflow checkpoints",
    )
    prompt_parser.add_argument(
        "--analysis-dir",
        help="Directory containing workflow results to analyze",
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
    
    return parser.parse_args()

async def main(prompt: str, resume: bool = False, checkpoint_dir: str = None, analysis_dir: str = None):
    """Main entry point for the CLI."""
    try:
        if analysis_dir:
            # This is an analysis request
            logger.info(f"Analyzing workflow results in {analysis_dir}")
            results = await analyze_workflow(analysis_dir)
            
            if results["status"] == "success":
                print(results["report"])
                if results.get("report_file"):
                    print(f"\nAnalysis report saved to: {results['report_file']}")
            else:
                print(f"Analysis failed: {results.get('error', 'Unknown error')}")
                
        else:
            # This is a workflow execution request
            if resume and not checkpoint_dir:
                raise ValueError("Checkpoint directory must be provided if resume is set to True")
                
            logger.info("Starting new workflow")
            await run_workflow(prompt, checkpoint_dir, resume)
            
    except Exception as e:
        logger.error(f"Operation failed: {str(e)}")
        raise

def run():
    """Run the CLI."""
    args = parse_args()
    
    try:
        if args.command == "prompt":
            # Run workflow or analysis
            asyncio.run(
                main(
                    prompt=args.prompt,
                    resume=args.resume,
                    checkpoint_dir=args.checkpoint_dir,
                    analysis_dir=args.analysis_dir,
                )
            )
        elif args.command == "serve":
            # Start web interface
            start_server(args.host, args.port)
        else:
            # Show help if no command specified
            parse_args().print_help()
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Operation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run()
