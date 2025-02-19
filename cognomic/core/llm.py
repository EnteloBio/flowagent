"""LLM interface for workflow planning and execution."""

from typing import Dict, Any, List
import json
import os
from pathlib import Path

from ..utils.logging import get_logger
from ..workflows.base import WorkflowRegistry

logger = get_logger(__name__)

class LLMInterface:
    """Interface for LLM-based workflow planning and execution."""
    
    def __init__(self, api_key: str = None):
        """Initialize LLM interface."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found")
            
        self.logger = get_logger(__name__)
    
    def _get_workflow_planning_prompt(self, workflow_name: str) -> str:
        """Get workflow-specific planning prompt."""
        try:
            workflow_class = WorkflowRegistry.get_workflow(workflow_name)
            return workflow_class.get_workflow_prompt()
        except ValueError as e:
            available_workflows = [w["name"] for w in WorkflowRegistry.list_workflows()]
            raise ValueError(f"Unknown workflow: {workflow_name}. Available workflows: {available_workflows}")
    
    async def plan_workflow(self, workflow_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Plan workflow execution using LLM."""
        try:
            # Get workflow-specific prompt
            prompt = self._get_workflow_planning_prompt(workflow_name)
            
            # Add input data to prompt
            prompt += f"\nInput data:\n{json.dumps(input_data, indent=2)}"
            
            # Get LLM response
            response = await self._get_llm_response(prompt)
            
            # Parse and validate response
            try:
                workflow_plan = json.loads(response)
                self._validate_workflow_plan(workflow_plan)
                return workflow_plan
            except json.JSONDecodeError:
                raise ValueError("LLM response is not valid JSON")
            except ValueError as e:
                raise ValueError(f"Invalid workflow plan: {str(e)}")
                
        except Exception as e:
            self.logger.error(f"Error planning workflow: {str(e)}")
            raise
    
    def _validate_workflow_plan(self, plan: Dict[str, Any]) -> None:
        """Validate workflow plan structure."""
        required_keys = ["inputs", "steps", "outputs", "validation"]
        for key in required_keys:
            if key not in plan:
                raise ValueError(f"Missing required key in workflow plan: {key}")
                
        if not isinstance(plan["steps"], list):
            raise ValueError("Steps must be a list")
            
        for step in plan["steps"]:
            required_step_keys = ["name", "tool", "action", "type", "parameters"]
            for key in required_step_keys:
                if key not in step:
                    raise ValueError(f"Missing required key in step: {key}")
    
    async def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM API."""
        # TODO: Implement actual LLM API call
        # For now, return a basic workflow plan
        return json.dumps({
            "inputs": {
                "required_files": ["*.fastq.gz"],
                "parameters": {
                    "threads": "Number of threads to use",
                    "memory": "Memory limit"
                }
            },
            "steps": [
                {
                    "name": "quality_control",
                    "tool": "fastqc",
                    "action": "analyze",
                    "type": "qc",
                    "parameters": {}
                }
            ],
            "outputs": {
                "qc_reports": "fastqc_output"
            },
            "validation": {
                "required_files": ["*_fastqc.html"],
                "output_checks": ["Check FastQC reports exist"]
            }
        })
    
    async def analyze_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze error and suggest recovery steps."""
        # TODO: Implement error analysis
        return {
            "error_type": type(error).__name__,
            "message": str(error),
            "recoverable": False,
            "suggestions": ["Check input files exist", "Verify tool installations"]
        }
