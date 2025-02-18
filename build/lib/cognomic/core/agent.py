"""Base agent implementation for Cognomic."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from langchain_core.language_models import BaseLLM
from langchain_core.prompts import BasePromptTemplate
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config.settings import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class AgentState(BaseModel):
    """Agent state model."""
    
    id: str
    status: str
    memory: Dict[str, Any] = {}
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""

    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        prompt_template: BasePromptTemplate,
        max_retries: int = settings.MAX_RETRIES,
    ) -> None:
        """Initialize base agent."""
        self.name = name
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_retries = max_retries
        self.state = AgentState(id=name, status="initialized")

    @abstractmethod
    async def plan(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Plan the execution of a task."""
        pass

    @abstractmethod
    async def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a planned task."""
        pass

    @abstractmethod
    async def validate(self, result: Dict[str, Any]) -> bool:
        """Validate the results of task execution."""
        pass

    @retry(
        stop=stop_after_attempt(settings.MAX_RETRIES),
        wait=wait_exponential(multiplier=settings.RETRY_DELAY),
        reraise=True,
    )
    async def _call_llm(self, prompt: str) -> str:
        """Call LLM with retry logic."""
        try:
            response = await self.llm.agenerate([prompt])
            return response.generations[0][0].text
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            self.state.error = str(e)
            raise

    async def update_state(self, **kwargs: Any) -> None:
        """Update agent state."""
        for key, value in kwargs.items():
            setattr(self.state, key, value)
        logger.debug(f"Agent {self.name} state updated: {self.state}")


class PlanningAgent(BaseAgent):
    """Agent responsible for workflow planning."""

    async def plan(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create a detailed execution plan for a task."""
        prompt = self.prompt_template.format(task=task)
        response = await self._call_llm(prompt)
        
        plan = self._parse_plan(response)
        await self.update_state(status="planning_complete")
        return plan

    async def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute is not applicable for PlanningAgent."""
        raise NotImplementedError("PlanningAgent does not execute tasks")

    async def validate(self, result: Dict[str, Any]) -> bool:
        """Validate the generated plan."""
        # Implement plan validation logic
        return True

    def _parse_plan(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into a structured plan."""
        # Implement plan parsing logic
        return {"steps": [], "dependencies": {}}


class ExecutionAgent(BaseAgent):
    """Agent responsible for task execution."""

    async def plan(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Planning is not applicable for ExecutionAgent."""
        raise NotImplementedError("ExecutionAgent does not plan tasks")

    async def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single task according to the plan."""
        await self.update_state(status="executing")
        
        try:
            # Implement task execution logic
            result = await self._execute_task(plan)
            await self.update_state(status="execution_complete")
            return result
        except Exception as e:
            await self.update_state(status="failed", error=str(e))
            raise

    async def validate(self, result: Dict[str, Any]) -> bool:
        """Validate task execution results."""
        # Implement result validation logic
        return True

    async def _execute_task(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific task."""
        # Implement specific task execution logic
        return {"status": "success"}
