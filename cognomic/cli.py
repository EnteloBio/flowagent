"""Command line interface for Cognomic."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

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

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Cognomic: Intelligent RNA-seq Analysis Pipeline")
    parser.add_argument("prompt", help="Natural language description of the analysis to perform")
    args = parser.parse_args()
    
    try:
        asyncio.run(run_workflow(args.prompt))
    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
