"""Base classes for workflow management."""

from typing import Dict, Any, Type, List
import abc
import logging
from ..utils.logging import get_logger
from ..core.agent_types import TASK_agent
from ..agents.llm_agent import LLMAgent

logger = get_logger(__name__)

class WorkflowBase(abc.ABC):
    """Base class for all workflows."""
    
    name: str = "base"  # Override in subclass
    description: str = "Base workflow"  # Override in subclass
    required_tools: List[str] = []  # List of required tools e.g. ["kallisto", "fastqc"]
    
    def __init__(self, params: Dict[str, Any], task_agent: TASK_agent):
        """Initialize workflow with parameters."""
        self.params = params
        self.task_agent = task_agent
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Try to initialize LLM agent, but make it optional
        try:
            self.llm_agent = LLMAgent()
            self.has_llm = True
        except Exception as e:
            self.logger.warning(f"Could not initialize LLM agent: {str(e)}")
            self.logger.warning("Continuing without LLM assistance")
            self.has_llm = False
    
    @classmethod
    def get_workflow_prompt(cls) -> str:
        """Get the workflow-specific planning prompt."""
        return f"""Create a workflow plan for {cls.description}. The workflow should:

1. Validate input data and parameters
2. Execute required analysis steps using {', '.join(cls.required_tools)}
3. Generate appropriate quality control metrics
4. Save results in organized output directories

Return a JSON object with exactly this structure:
{{
    "inputs": {{
        "required_files": ["file patterns"],
        "parameters": {{"param_name": "param_description"}}
    }},
    "steps": [
        {{
            "name": "Step Name",
            "tool": "tool_name",
            "action": "action",
            "type": "step_type",
            "parameters": {{}}
        }}
    ],
    "outputs": {{
        "output_name": "output_directory"
    }},
    "validation": {{
        "required_files": ["files to check"],
        "output_checks": ["expected outputs"]
    }}
}}

Return only valid JSON without explanation."""

    @abc.abstractmethod
    async def validate_params(self) -> bool:
        """Validate workflow parameters."""
        pass
    
    @abc.abstractmethod
    async def execute(self) -> Dict[str, Any]:
        """Execute the workflow."""
        pass
    
    @abc.abstractmethod
    async def validate_results(self, results: Dict[str, Any]) -> bool:
        """Validate workflow results."""
        pass

class WorkflowRegistry:
    """Registry for available workflows."""
    
    _workflows: Dict[str, Type[WorkflowBase]] = {}
    
    @classmethod
    def register(cls, workflow_class: Type[WorkflowBase]):
        """Register a workflow class."""
        cls._workflows[workflow_class.name] = workflow_class
        logger.info(f"Registered workflow: {workflow_class.name}")
        
    @classmethod
    def get_workflow(cls, name: str) -> Type[WorkflowBase]:
        """Get workflow class by name."""
        if name not in cls._workflows:
            raise ValueError(f"Unknown workflow: {name}")
        return cls._workflows[name]
    
    @classmethod
    def list_workflows(cls) -> List[Dict[str, str]]:
        """List available workflows."""
        return [
            {"name": wf.name, "description": wf.description}
            for wf in cls._workflows.values()
        ]
