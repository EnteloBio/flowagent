"""Command line interface for Cognomic."""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import click

from .core.workflow_manager import WorkflowManager
from .core.llm import LLMInterface
from .analysis.report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

async def analyze_workflow(analysis_dir: str, save_report: bool = True) -> None:
    """Analyze workflow results from a directory."""
    try:
        report_gen = ReportGenerator()
        report = await report_gen.generate_analysis_report(Path(analysis_dir))
        
        # Save report to file if requested
        if save_report:
            report_file = Path(analysis_dir) / "analysis_report.md"
            with open(report_file, 'w') as f:
                f.write(report)
            logger.info(f"Analysis report saved to: {report_file}")
        
        # Print report to console
        print(report)
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

async def run_workflow(prompt: str, checkpoint_dir: str = None, resume: bool = False) -> None:
    """Run workflow from prompt."""
    try:
        # Initialize LLM interface
        llm = LLMInterface()
        
        # Initialize workflow manager
        workflow_manager = WorkflowManager()
        
        # Set up checkpoint directory
        checkpoint_path = Path(checkpoint_dir) if checkpoint_dir else Path('workflow_state')
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        if resume:
            logger.info(f"Resuming workflow from {checkpoint_path}")
            result = await workflow_manager.resume_workflow(prompt, str(checkpoint_path))
        else:
            logger.info("Starting new workflow")
            result = await workflow_manager.execute_workflow(prompt)
            
        logger.info("Workflow completed successfully!")
        
    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}")
        logger.info(f"You can resume the workflow using:")
        logger.info(f"cognomic '{prompt}' --resume --checkpoint-dir={checkpoint_dir or 'workflow_state'}")
        raise

@click.command()
@click.argument('prompt')
@click.option('--checkpoint-dir', type=click.Path(), help='Directory for workflow state and checkpoints')
@click.option('--analysis-dir', type=click.Path(), help='Directory containing workflow results to analyze')
@click.option('--resume/--no-resume', default=False, help='Resume workflow from checkpoint')
@click.option('--save-report/--no-save-report', default=True, help='Save analysis report to file')
def run(prompt: str, checkpoint_dir: str = None, analysis_dir: str = None, resume: bool = False, save_report: bool = True):
    """Run Cognomic workflow or analyze results.
    
    There are two main modes:
    
    1. Run a workflow:
       cognomic "run rna-seq analysis" --checkpoint-dir=workflow_state
       
    2. Analyze existing results:
       cognomic "analyze workflow results" --analysis-dir=results
    """
    try:
        if "analyze" in prompt.lower() and analysis_dir:
            asyncio.run(analyze_workflow(analysis_dir, save_report))
        else:
            asyncio.run(run_workflow(prompt, checkpoint_dir, resume))
    except Exception as e:
        logger.error(f"Operation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run()
