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
@click.argument('query')
@click.option('--output-dir', '-o', default=None, help='Directory containing outputs to analyze. Uses current directory if not specified.')
def analyze(query: str, output_dir: str = None):
    """Run LLM analysis on any tool outputs in the specified directory."""
    from .analysis.report_generator import ReportGenerator
    from pathlib import Path
    import logging
    import asyncio
    
    # Set debug logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Use current directory if no output dir specified
    output_dir = Path(output_dir) if output_dir else Path.cwd()
    
    async def run_analysis():
        try:
            # Run analysis
            generator = ReportGenerator()
            result = await generator.analyze_tool_outputs(output_dir)
            
            # Print results
            if result and result.get('status') == 'success':
                print("\nAnalysis Report:")
                print("-" * 80)
                print(result.get('analysis', 'No analysis available'))
                print("-" * 80)
            else:
                error_msg = result.get('message') if result else 'Unknown error occurred'
                print(f"\nError: {error_msg}")
                
        except Exception as e:
            print(f"Error running analysis: {str(e)}")
    
    # Run the async analysis in an event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_analysis())
    finally:
        loop.close()

if __name__ == "__main__":
    cli()
