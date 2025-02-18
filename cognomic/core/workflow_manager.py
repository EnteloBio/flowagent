"""
Workflow Manager for Cognomic
"""

import asyncio
import importlib
from typing import Any, Dict, Type
from pydantic import BaseModel
from ..utils.logging import get_logger
from ..workflows.pseudobulk import PseudoBulkWorkflow, PseudoBulkParams

logger = get_logger(__name__)

class WorkflowManager:
    """Manager class for handling different types of workflows."""

    WORKFLOW_MAPPING = {
        'pseudobulk': (PseudoBulkWorkflow, PseudoBulkParams),
        'rnaseq': (None, None),  # To be implemented
    }

    @classmethod
    def get_workflow(cls, workflow_type: str) -> tuple[Type[Any], Type[BaseModel]]:
        """Get the workflow class and params class for the specified workflow type."""
        if workflow_type not in cls.WORKFLOW_MAPPING:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
        
        workflow_class, params_class = cls.WORKFLOW_MAPPING[workflow_type]
        if workflow_class is None:
            raise NotImplementedError(f"Workflow {workflow_type} is not yet implemented")
        
        return workflow_class, params_class

    @classmethod
    def create_workflow(cls, workflow_type: str, params: Dict[str, Any]) -> Any:
        """Create and return a workflow instance of the specified type."""
        workflow_class, params_class = cls.get_workflow(workflow_type)
        
        # Convert dict params to proper params object
        workflow_params = params_class(**params)
        
        # Create and return workflow instance
        return workflow_class(params=workflow_params)

    @classmethod
    async def run_workflow_async(cls, workflow_type: str, params: Dict[str, Any]) -> None:
        """Create and execute a workflow of the specified type asynchronously."""
        try:
            logger.info(f"Creating workflow of type: {workflow_type}")
            workflow = cls.create_workflow(workflow_type, params)
            
            logger.info(f"Executing workflow: {workflow_type}")
            await workflow.execute()
            
            logger.info(f"Workflow {workflow_type} completed successfully")
        except Exception as e:
            logger.error(f"Error running workflow {workflow_type}: {e}")
            raise

    @classmethod
    def run_workflow(cls, workflow_type: str, params: Dict[str, Any]) -> None:
        """Create and execute a workflow of the specified type."""
        asyncio.run(cls.run_workflow_async(workflow_type, params))
