"""LLM interface for workflow planning and execution."""

from typing import Dict, Any, List
import json
import os
from pathlib import Path
import openai
from openai import AsyncOpenAI

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
            
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = "gpt-4"  # Using GPT-4 for better reasoning
        self.logger = get_logger(__name__)
    
    def _get_workflow_planning_prompt(self, workflow_name: str) -> str:
        """Get workflow-specific planning prompt."""
        if workflow_name == "dynamic":
            return """You are a bioinformatics workflow planner. Your task is to create a detailed workflow plan for processing biological data.

Instructions:
1. Analyze the input description and files to understand the task requirements
2. Identify required bioinformatics tools and their versions
3. Design a step-by-step workflow that accomplishes the task
4. Include appropriate quality control and validation steps

Return a JSON object with exactly this structure:
{
    "required_tools": [
        {"name": "tool_name", "version": "version", "purpose": "description"}
    ],
    "steps": [
        {
            "name": "step_name",
            "tool": "tool_name",
            "action": "action",
            "type": "step_type",
            "parameters": {
                "param1": "value1"
            },
            "description": "step description"
        }
    ],
    "outputs": {
        "output_name": {
            "path": "relative/path",
            "description": "output description"
        }
    },
    "validation": {
        "output_checks": [
            {
                "type": "file_exists",
                "parameters": {"path": "expected/file/path"}
            },
            {
                "type": "file_content",
                "parameters": {
                    "path": "file/path",
                    "content_type": "content type",
                    "validation_rules": ["rule1", "rule2"]
                }
            }
        ]
    }
}

Return only valid JSON without explanation."""
        else:
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
            prompt += f"\n\nInput data:\n{json.dumps(input_data, indent=2)}"
            
            # Get LLM response
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Plan the workflow for the given input data."}
                ],
                temperature=0.2,  # Lower temperature for more consistent outputs
                max_tokens=2000
            )
            
            # Parse and validate response
            try:
                workflow_plan = json.loads(response.choices[0].message.content.strip())
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
        required_keys = ["required_tools", "steps", "outputs", "validation"]
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
    
    async def analyze_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze error and suggest recovery steps."""
        prompt = f"""Analyze this workflow error and suggest recovery steps:

Error: {str(error)}
Error Type: {type(error).__name__}
Step: {json.dumps(context.get('step', {}), indent=2)}
Results so far: {json.dumps(context.get('results', {}), indent=2)}

Determine if the error is recoverable and suggest specific recovery steps.
Return a JSON object with:
{{
    "error_type": "error classification",
    "message": "error description",
    "recoverable": true/false,
    "recovery_steps": [
        {{
            "action": "action to take",
            "parameters": {{}}
        }}
    ],
    "suggestions": [
        "user-friendly suggestion"
    ]
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Analyze the error and suggest recovery steps."}
                ],
                temperature=0.2
            )
            
            return json.loads(response.choices[0].message.content.strip())
            
        except Exception as e:
            self.logger.error(f"Error analyzing error: {str(e)}")
            return {
                "error_type": type(error).__name__,
                "message": str(error),
                "recoverable": False,
                "suggestions": ["Check input files exist", "Verify tool installations"]
            }
