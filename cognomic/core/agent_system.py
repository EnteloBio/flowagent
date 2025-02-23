"""Agent system for managing workflow execution."""

import json
import logging
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime

from ..config.settings import settings
from ..utils.logging import get_logger
from .tool_tracker import ToolTracker
from .agent_types import WorkflowStep, AgentType
from .llm import LLMInterface

logger = get_logger(__name__)

class PLAN_agent:
    """Agent responsible for workflow planning."""
    
    def __init__(self, llm):
        self.llm = llm
        self.logger = get_logger(__name__)
    
    async def plan(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create a detailed execution plan for a task."""
        try:
            return await self.llm.generate_workflow_plan(task)
        except Exception as e:
            self.logger.error(f"Error in planning: {str(e)}")
            raise

class TASK_agent:
    """Agent responsible for task execution."""
    
    def __init__(self, llm):
        self.llm = llm
        self.logger = get_logger(__name__)
        self.tool_tracker = ToolTracker(llm)
    
    async def execute(self, step: WorkflowStep) -> Dict[str, Any]:
        """Execute a single task according to the plan."""
        try:
            return await self.tool_tracker.execute_tool(
                step.tool,
                step.action,
                step.parameters
            )
        except Exception as e:
            self.logger.error(f"Error executing step {step.name}: {str(e)}")
            raise

class DEBUG_agent:
    """Agent responsible for debugging and error handling."""
    
    def __init__(self, llm):
        self.llm = llm
        self.logger = get_logger(__name__)
    
    async def analyze_error(self, error: str, context: Dict[str, Any]) -> str:
        """Analyze an error and provide debugging suggestions."""
        try:
            return await self.llm.analyze_error(error, context)
        except Exception as e:
            self.logger.error(f"Error in debugging: {str(e)}")
            raise

class AgentSystem:
    """Coordinates agents and tools for workflow execution."""
    
    def __init__(self, llm: Optional[LLMInterface] = None):
        """Initialize agent system."""
        self.logger = get_logger(__name__)
        self.llm = llm or LLMInterface()
        self.tool_tracker = ToolTracker()
        
    async def execute_workflow(self, workflow_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute workflow steps in order."""
        try:
            results = []
            
            # Execute each step
            for step in workflow_plan["steps"]:
                try:
                    self.logger.info(f"Executing step {step['name']}")
                    result = await self.tool_tracker.execute_tool(step, self.llm)
                    results.append({
                        "step": step["name"],
                        "status": "success",
                        "result": result
                    })
                except Exception as e:
                    self.logger.error(f"Error executing step {step['name']}: {str(e)}")
                    results.append({
                        "step": step["name"],
                        "status": "failed",
                        "error": str(e)
                    })
                    
                    # Try to diagnose the error
                    try:
                        diagnosis = await self.llm.analyze_error(str(e), {
                            "step": step,
                            "error": str(e)
                        })
                        self.logger.info(f"Error diagnosis: {diagnosis}")
                    except Exception as diag_error:
                        self.logger.error(f"Error diagnosis failed: {str(diag_error)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            raise
