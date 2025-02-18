"""Workflow orchestration system for Cognomic."""
import asyncio
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel

from ..config.settings import settings
from ..utils.logging import get_logger
from .agent import BaseAgent

logger = get_logger(__name__)


class WorkflowStep(BaseModel):
    """Model for a workflow step."""
    
    id: UUID
    name: str
    agent: str
    dependencies: List[UUID] = []
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class Workflow(BaseModel):
    """Model for a complete workflow."""
    
    id: UUID
    name: str
    steps: List[WorkflowStep]
    status: str = "pending"
    created_at: str
    updated_at: str
    metadata: Dict[str, Any] = {}


class WorkflowOrchestrator:
    """Orchestrates the execution of workflows."""

    def __init__(self) -> None:
        """Initialize the orchestrator."""
        self.workflows: Dict[UUID, Workflow] = {}
        self.agents: Dict[str, BaseAgent] = {}
        self.semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_WORKFLOWS)

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator."""
        self.agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")

    async def create_workflow(
        self, name: str, steps: List[Dict[str, Any]]
    ) -> Workflow:
        """Create a new workflow."""
        workflow_id = uuid4()
        workflow_steps = []

        for step_data in steps:
            step = WorkflowStep(
                id=uuid4(),
                name=step_data["name"],
                agent=step_data["agent"],
                dependencies=[UUID(dep) for dep in step_data.get("dependencies", [])],
            )
            workflow_steps.append(step)

        workflow = Workflow(
            id=workflow_id,
            name=name,
            steps=workflow_steps,
            created_at=str(datetime.utcnow()),
            updated_at=str(datetime.utcnow()),
        )

        self.workflows[workflow_id] = workflow
        logger.info(f"Created workflow: {workflow.id} - {workflow.name}")
        return workflow

    async def execute_workflow(self, workflow_id: UUID) -> Workflow:
        """Execute a workflow."""
        async with self.semaphore:
            workflow = self.workflows[workflow_id]
            workflow.status = "running"
            
            try:
                # Create execution groups based on dependencies
                execution_groups = self._create_execution_groups(workflow.steps)
                
                # Execute groups in sequence
                for group in execution_groups:
                    await self._execute_step_group(group, workflow)
                
                workflow.status = "completed"
                logger.info(f"Workflow completed: {workflow_id}")
                
            except Exception as e:
                workflow.status = "failed"
                logger.error(f"Workflow failed: {workflow_id} - {str(e)}")
                raise
                
            finally:
                workflow.updated_at = str(datetime.utcnow())
                
            return workflow

    async def _execute_step_group(
        self, group: List[WorkflowStep], workflow: Workflow
    ) -> None:
        """Execute a group of parallel steps."""
        tasks = []
        for step in group:
            if not self._are_dependencies_met(step, workflow.steps):
                continue
                
            task = self._execute_step(step)
            tasks.append(task)

        await asyncio.gather(*tasks)

    async def _execute_step(self, step: WorkflowStep) -> None:
        """Execute a single workflow step."""
        agent = self.agents.get(step.agent)
        if not agent:
            raise ValueError(f"Agent not found: {step.agent}")

        try:
            step.status = "running"
            result = await agent.execute({"step_id": step.id})
            
            if await agent.validate(result):
                step.status = "completed"
                step.result = result
            else:
                raise ValueError("Step validation failed")
                
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            raise

    def _create_execution_groups(
        self, steps: List[WorkflowStep]
    ) -> List[List[WorkflowStep]]:
        """Create groups of steps that can be executed in parallel."""
        groups = []
        remaining_steps = steps.copy()

        while remaining_steps:
            group = [
                step for step in remaining_steps
                if all(dep in [s.id for s in steps if s.status == "completed"]
                      for dep in step.dependencies)
            ]
            
            if not group:
                raise ValueError("Circular dependency detected")
                
            groups.append(group)
            remaining_steps = [s for s in remaining_steps if s not in group]

        return groups

    def _are_dependencies_met(
        self, step: WorkflowStep, all_steps: List[WorkflowStep]
    ) -> bool:
        """Check if all dependencies for a step are met."""
        return all(
            any(dep_step.id == dep_id and dep_step.status == "completed"
                for dep_step in all_steps)
            for dep_id in step.dependencies
        )

    def get_workflow_status(self, workflow_id: UUID) -> Dict[str, Any]:
        """Get the current status of a workflow."""
        workflow = self.workflows[workflow_id]
        return {
            "id": workflow.id,
            "name": workflow.name,
            "status": workflow.status,
            "steps": [
                {
                    "id": step.id,
                    "name": step.name,
                    "status": step.status,
                    "error": step.error
                }
                for step in workflow.steps
            ],
            "updated_at": workflow.updated_at
        }
