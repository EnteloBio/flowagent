"""Command line interface for Cognomic."""

import argparse
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

async def run_workflow(prompt: str) -> None:
    """Run workflow from prompt."""
    try:
        # Initialize LLM interface
        llm = LLMInterface()
        
        # Initialize workflow manager
        workflow_manager = WorkflowManager(llm=llm)
        
        # Execute workflow
        result = await workflow_manager.execute_workflow(prompt)
        
        # Process results
        if result["status"] == "success":
            logger.info("Workflow completed successfully!")
            
            # Log results location
            if "archive_path" in result:
                logger.info(f"Results available in: {result['archive_path']}")
            
            # Log generated artifacts
            artifacts = []
            for step_result in result["results"]:
                if step_result["status"] == "success" and "result" in step_result:
                    for output in step_result["result"].get("outputs", []):
                        if "path" in output:
                            artifacts.append(output["path"])
            
            if artifacts:
                logger.info("Generated artifacts:")
                for artifact in artifacts:
                    logger.info(f"  - {artifact}")
            
            # Log any issues
            if result.get("report", {}).get("issues"):
                logger.warning("Issues found:")
                for issue in result["report"]["issues"]:
                    logger.warning(f"  - {issue}")
            
            # Log recommendations
            if result.get("report", {}).get("recommendations"):
                logger.info("Recommendations:")
                for rec in result["report"]["recommendations"]:
                    logger.info(f"  - {rec}")
                    
        else:
            logger.error(f"Workflow failed: {result.get('error', 'Unknown error')}")
            if result.get("diagnosis"):
                logger.error("Error diagnosis:")
                logger.error(f"  {result['diagnosis'].get('description', 'Unknown error')}")
                if result["diagnosis"].get("suggestions"):
                    logger.info("Suggestions:")
                    for suggestion in result["diagnosis"]["suggestions"]:
                        logger.info(f"  - {suggestion}")
    
    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}")
        raise

@click.group()
def cli():
    """Cognomic CLI."""
    pass

@cli.command()
@click.argument('prompt')
def run(prompt: str):
    """Run workflow from prompt."""
    try:
        asyncio.run(run_workflow(prompt))
    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}")
        sys.exit(1)

@cli.command()
@click.argument('query', type=str)
@click.option('--output-dir', type=click.Path(exists=True, file_okay=False, dir_okay=True), help='Directory containing workflow outputs')
def analyze(query: str, output_dir: str):
    """Analyze workflow outputs and generate a report."""
    async def run_analysis():
        try:
            from .analysis.report_generator import ReportGenerator
            generator = ReportGenerator()
            report = await generator.generate_analysis_report(Path(output_dir), query)
            
            click.echo("\nAnalysis Report:")
            click.echo("-" * 80)
            click.echo(report)
            click.echo("-" * 80)
            
        except Exception as e:
            click.echo(f"\nError: {str(e)}", err=True)
    
    # Run the async analysis in an event loop
    asyncio.run(run_analysis())

if __name__ == "__main__":
    cli()
