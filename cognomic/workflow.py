import asyncio
import logging
from pathlib import Path
from .core.workflow_manager import WorkflowManager
from .core.llm import LLMInterface
from .analysis.report_generator import ReportGenerator

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
        if resume and not checkpoint_dir:
            raise ValueError("Checkpoint directory must be provided if resume is set to True")
        
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
        logger.info(f"cognomic 'prompt {prompt}' --resume --checkpoint-dir={checkpoint_dir or 'workflow_state'}")
        raise
