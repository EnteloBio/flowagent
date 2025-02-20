"""Dynamic workflow implementation that can adapt to any bioinformatics task."""

from typing import Dict, Any, List, Optional
import os
from pathlib import Path
import json
from pydantic import BaseModel, Field

from .base import WorkflowBase, WorkflowRegistry
from ..utils.logging import get_logger
from ..core.llm import LLMInterface
from ..core.agent_system import TASK_agent

logger = get_logger(__name__)

class DynamicWorkflowParams(BaseModel):
    """Dynamic parameters that can adapt to any workflow type."""
    
    description: str = Field(..., description="Natural language description of the workflow")
    input_files: List[str] = Field(..., description="List of input files")
    output_dir: str = Field(..., description="Output directory")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")
    required_tools: List[str] = Field(default_factory=list, description="Required tools")

class DynamicWorkflow(WorkflowBase):
    """A flexible workflow that can adapt to any bioinformatics task."""
    
    name = "dynamic"
    description = "Dynamic workflow that adapts to any bioinformatics task"
    
    def __init__(self, params: Dict[str, Any], task_agent: TASK_agent):
        """Initialize dynamic workflow."""
        super().__init__(params, task_agent)
        self.params = DynamicWorkflowParams(**params)
        self.llm = LLMInterface()
        self.workflow_plan = None
        self.agents = {}
        
    async def _plan_workflow(self) -> Dict[str, Any]:
        """Use LLM to plan the workflow based on description."""
        prompt = f"""
        Plan a bioinformatics workflow for the following task:
        {self.params.description}

        Input files: {self.params.input_files}
        Additional parameters: {json.dumps(self.params.parameters, indent=2)}

        1. Analyze the input files and task requirements
        2. Determine required tools and processing steps
        3. Create a detailed execution plan
        4. Define validation criteria

        Return a complete workflow plan as JSON with:
        1. Required tools and their versions
        2. Processing steps in order
        3. Parameters for each step
        4. Expected outputs
        5. Validation criteria
        """
        
        return await self.llm.plan_workflow("dynamic", {
            "description": self.params.description,
            "input_files": self.params.input_files,
            "parameters": self.params.parameters
        })

    async def _create_agents(self, required_tools: List[str]):
        """Dynamically create execution agents for required tools."""
        for tool in required_tools:
            agent_class = f"{tool.capitalize()}Agent"
            # Check if agent class exists in our registry
            if hasattr(self.task_agent, agent_class):
                self.agents[tool] = getattr(self.task_agent, agent_class)()
            else:
                # Create a generic agent that can execute the tool
                self.agents[tool] = GenericToolAgent(self.task_agent, tool)

    async def validate_params(self) -> bool:
        """Validate workflow parameters."""
        # Check input files exist
        for file in self.params.input_files:
            if not os.path.exists(file):
                self.logger.error(f"Input file not found: {file}")
                return False
        
        # Create output directory
        os.makedirs(self.params.output_dir, exist_ok=True)
        
        # Get workflow plan
        try:
            self.workflow_plan = await self._plan_workflow()
            
            # Update required tools based on plan
            self.params.required_tools = self.workflow_plan.get("required_tools", [])
            
            # Create execution agents
            await self._create_agents(self.params.required_tools)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error planning workflow: {str(e)}")
            return False
    
    async def execute(self) -> Dict[str, Any]:
        """Execute the dynamic workflow."""
        if not self.workflow_plan:
            raise ValueError("Workflow not planned. Run validate_params first.")
            
        results = {}
        
        try:
            # Execute each step in the plan
            for step in self.workflow_plan["steps"]:
                step_name = step["name"]
                tool = step["tool"]
                action = step["action"]
                parameters = step["parameters"]
                
                self.logger.info(f"Executing step: {step_name}")
                
                # Add global parameters
                parameters.update({
                    "output_dir": os.path.join(self.params.output_dir, step_name),
                    **self.params.parameters.get(step_name, {})
                })
                
                # Execute step
                if tool in self.agents:
                    result = await self.agents[tool].execute({
                        "action": action,
                        **parameters
                    })
                    results[step_name] = result
                else:
                    raise ValueError(f"No agent available for tool: {tool}")
                    
                self.logger.info(f"Completed step: {step_name}")
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error executing workflow: {str(e)}")
            
            # Try to get recovery suggestions from LLM
            error_analysis = await self.llm.analyze_error(e, {
                "step": step if "step" in locals() else None,
                "results": results
            })
            
            if error_analysis.get("recoverable", False):
                self.logger.info("Attempting to recover from error...")
                # TODO: Implement recovery logic
                pass
                
            raise
    
    async def validate_results(self, results: Dict[str, Any]) -> bool:
        """Validate workflow results."""
        if not self.workflow_plan:
            return False
            
        try:
            # Check each validation criterion from the plan
            for check in self.workflow_plan["validation"]["output_checks"]:
                if isinstance(check, str):
                    # Simple file existence check
                    if not os.path.exists(os.path.join(self.params.output_dir, check)):
                        self.logger.error(f"Missing required output: {check}")
                        return False
                elif isinstance(check, dict):
                    # Complex validation check
                    check_type = check.get("type")
                    check_params = check.get("parameters", {})
                    
                    if check_type == "file_exists":
                        path = os.path.join(self.params.output_dir, check_params["path"])
                        if not os.path.exists(path):
                            self.logger.error(f"Missing required output: {path}")
                            return False
                    elif check_type == "file_content":
                        # TODO: Implement content validation
                        pass
                    elif check_type == "custom":
                        # TODO: Implement custom validation
                        pass
                        
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating results: {str(e)}")
            return False

class GenericToolAgent:
    """Generic agent that can execute any command-line tool."""
    
    def __init__(self, task_agent: TASK_agent, tool_name: str):
        """Initialize generic tool agent."""
        self.task_agent = task_agent
        self.tool_name = tool_name
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given parameters."""
        action = params.pop("action")
        
        # Convert parameters to command-line arguments
        args = []
        for key, value in params.items():
            if isinstance(value, bool):
                if value:
                    args.append(f"--{key}")
            else:
                args.append(f"--{key} {value}")
                
        command = f"{self.tool_name} {action} {' '.join(args)}"
        
        # Execute command
        result = await self.task_agent.run_command(command)
        
        return {
            "command": command,
            "output": result.stdout,
            "error": result.stderr,
            "exit_code": result.returncode
        }
        
    async def validate(self, result: Dict[str, Any]) -> bool:
        """Validate tool execution result."""
        return result.get("exit_code", 1) == 0

# Register the workflow
WorkflowRegistry.register(DynamicWorkflow)
