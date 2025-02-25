"""CLI interface for Cognomic."""

import asyncio
import logging
import json
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

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Cognomic: A modern framework for RNA-seq analysis.
    
    Example commands:
    
    1. Run a workflow:
       cognomic prompt "Analyze RNA-seq data using Kallisto..."
    
    2. Analyze results:
       cognomic prompt "analyze workflow results" --analysis-dir=results
    
    3. Start web interface:
       cognomic serve
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        sys.exit(1)

@cli.command()
@click.argument('prompt')
@click.option('--checkpoint-dir', type=click.Path(), help='Directory for workflow state and checkpoints')
@click.option('--analysis-dir', type=click.Path(), help='Directory containing workflow results to analyze')
@click.option('--resume/--no-resume', default=False, help='Resume workflow from checkpoint')
@click.option('--save-report/--no-save-report', default=True, help='Save analysis report to file')
def prompt(prompt: str, checkpoint_dir: str = None, analysis_dir: str = None, resume: bool = False, save_report: bool = True):
    """Execute a workflow or analyze results based on natural language prompt.
    
    Examples:
    
    1. Run RNA-seq analysis:
       cognomic prompt "Analyze RNA-seq data in my fastq.gz files using Kallisto. 
       The fastq files are in current directory and I want to use Homo_sapiens.GRCh38.cdna.all.fa 
       as reference. Generate QC reports and save everything in results/rna_seq_analysis." 
       --checkpoint-dir workflow_state
       
    2. Analyze results:
       cognomic prompt "analyze workflow results" --analysis-dir=results
    """
    try:
        if analysis_dir:
            # Run analysis workflow
            results = asyncio.run(analyze_workflow(prompt, Path(analysis_dir), save_report))
            click.echo(results)
        else:
            # Run normal workflow
            results = asyncio.run(run_workflow(prompt, checkpoint_dir, resume))
            click.echo(results)
            
    except Exception as e:
        logger.error(f"Operation failed: {str(e)}")
        sys.exit(1)

@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind the server to')
@click.option('--port', default=8000, help='Port to bind the server to')
def serve(host: str, port: int):
    """Start the Cognomic web interface.
    
    Example:
    cognomic serve --port 8080
    """
    start_server(host, port)

if __name__ == "__main__":
    cli()
