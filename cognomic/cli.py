"""Command line interface for Cognomic."""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import click
import uvicorn

from .workflow import run_workflow, analyze_workflow
from .api import start_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

@click.group()
def cli():
    """Cognomic CLI."""
    pass

@cli.command()
@click.argument('prompt')
@click.option('--checkpoint-dir', type=click.Path(), help='Directory for workflow state and checkpoints')
@click.option('--analysis-dir', type=click.Path(), help='Directory containing workflow results to analyze')
@click.option('--resume/--no-resume', default=False, help='Resume workflow from checkpoint')
@click.option('--save-report/--no-save-report', default=True, help='Save analysis report to file')
def prompt(prompt: str, checkpoint_dir: str = None, analysis_dir: str = None, resume: bool = False, save_report: bool = True):
    """Run Cognomic workflow or analyze results.
    
    There are two main modes:
    
    1. Run a workflow:
       cognomic prompt "run rna-seq analysis" --checkpoint-dir=workflow_state
       
    2. Analyze existing results:
       cognomic prompt "run analyze workflow results" --analysis-dir=results
    """
    try:
        if "analyze" in prompt.lower() and analysis_dir:
            asyncio.run(analyze_workflow(analysis_dir, save_report))
        else:
            asyncio.run(run_workflow(prompt, checkpoint_dir, resume))
    except Exception as e:
        logger.error(f"Operation failed: {str(e)}")
        sys.exit(1)

@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind the server to')
@click.option('--port', default=8000, help='Port to bind the server to')
def serve(host: str, port: int):
    """Serve the Cognomic HTTP API."""
    start_server(host, port)

if __name__ == "__main__":
    cli()
