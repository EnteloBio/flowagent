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
        self.tool_tracker = ToolTracker()  # Initialize without llm parameter
    
    async def execute(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single task according to the plan."""
        try:
            # Generate command for the step
            command = await self.llm.generate_command(step)
            step["command"] = command
            
            # Execute the command using tool tracker
            result = await self.tool_tracker.execute_tool(step, self.llm)  # Pass llm here instead
            
            # Add result to step
            step["result"] = result
            return step
            
        except Exception as e:
            self.logger.error(f"Error executing step {step.get('name', 'unknown')}: {str(e)}")
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
        self.task_agent = TASK_agent(self.llm)
        self.plan_agent = PLAN_agent(self.llm)
    
    async def execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step."""
        try:
            self.logger.info(f"Executing step: {step['name']}")
            result = await self.task_agent.execute(step)
            self.logger.info(f"Completed step: {step['name']}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to execute step {step['name']}: {str(e)}")
            raise
    
    async def execute_workflow(self, workflow_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow steps using DAG-based parallel execution."""
        try:
            from .workflow_dag import WorkflowDAG
            
            # Create workflow DAG
            dag = WorkflowDAG()
            
            # Add steps to DAG with dependencies
            for step in workflow_plan["steps"]:
                dependencies = step.get("dependencies", [])
                dag.add_step(step, dependencies)
            
            # Execute workflow using parallel execution
            results = await dag.execute_parallel(self.execute_step)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            raise
