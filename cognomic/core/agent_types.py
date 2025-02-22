"""Type definitions for agent system."""

from typing import Dict, Any, Protocol, runtime_checkable, Optional
from dataclasses import dataclass
import shutil
import logging

logger = logging.getLogger(__name__)

@dataclass
class WorkflowStep:
    """Represents a single step in a workflow."""
    name: str
    tool: str
    action: str
    parameters: Dict[str, Any]
    type: str = 'command'
    description: Optional[str] = None

@runtime_checkable
class TASK_agent(Protocol):
    """Protocol for task execution agents."""
    
    async def execute_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Execute a workflow step."""
        ...
    
    async def execute_command(self, command: str) -> Dict[str, Any]:
        """Execute a shell command."""
        ...

    def _check_disk_space(self, output_dir: str) -> bool:
        """Check if there is sufficient disk space in the output directory.
        
        Args:
            output_dir: Directory to check disk space for
            
        Returns:
            bool: True if sufficient space available, False otherwise
        """
        try:
            # Get disk usage statistics
            usage = shutil.disk_usage(output_dir)
            
            # Ensure at least 1GB free space
            min_free_space = 1 * 1024 * 1024 * 1024  # 1GB in bytes
            return usage.free >= min_free_space
            
        except Exception as e:
            logger.error(f"Error checking disk space: {str(e)}")
            return False

@runtime_checkable
class PLAN_agent(Protocol):
    """Protocol for workflow planning agents."""
    
    async def plan_from_prompt(self, prompt: str) -> Dict[str, Any]:
        """Parse natural language prompt and create workflow plan."""
        ...
    
    async def decompose_workflow(self, prompt: str) -> Dict[str, Any]:
        """Convert prompt into workflow steps.
        
        Returns:
            Dict containing:
            - steps: List[Dict] with each step having:
                - name: str
                - tool: str
                - action: str
                - parameters: Dict[str, Any]
                - type: str (optional)
                - description: str (optional)
        """
        ...

@runtime_checkable
class DEBUG_agent(Protocol):
    """Protocol for debugging agents."""
    
    async def diagnose_failure(self, error_context: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnose workflow failure and suggest recovery steps."""
        ...
