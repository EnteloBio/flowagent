"""Command line interface for Cognomic."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import click

from .core.workflow_manager import WorkflowManager
from .core.llm import LLMInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

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
@click.option('--resume/--no-resume', default=False, help='Resume workflow from checkpoint')
def cli(prompt: str, checkpoint_dir: str = None, resume: bool = False):
    """Run Cognomic workflow from natural language prompt."""
    try:
        asyncio.run(run_workflow(prompt, checkpoint_dir, resume))
    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    cli()
