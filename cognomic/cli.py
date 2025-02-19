import asyncio
import argparse
import logging
from pathlib import Path
from .core.workflow_manager import WorkflowManager
from .core.knowledge import initialize_knowledge_base
from .utils.logging import setup_logging

logger = logging.getLogger(__name__)

async def run_workflow(prompt: str):
    """Run workflow based on natural language prompt"""
    # Initialize knowledge base
    knowledge_db = initialize_knowledge_base()
    
    # Create workflow manager
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    manager = WorkflowManager(knowledge_db, output_dir)
    
    try:
        logger.info(f"Planning workflow from prompt: {prompt}")
        results = await manager.execute_from_prompt(prompt)
        
        logger.info("Workflow completed successfully!")
        logger.info(f"Results available in: {output_dir}")
        logger.info("Generated artifacts:")
        for name, path in results['artifacts'].items():
            logger.info(f"  {name}: {path}")
            
        return results
        
    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}")
        raise

def main():
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(
        description='Run bioinformatics workflows using natural language.'
    )
    
    parser.add_argument(
        'prompt',
        type=str,
        help='Natural language description of the workflow to run'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Directory for workflow outputs (default: output)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)
    
    try:
        # Run workflow
        asyncio.run(run_workflow(args.prompt))
    except KeyboardInterrupt:
        logger.info("Workflow interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}")
        exit(1)

if __name__ == '__main__':
    main()
