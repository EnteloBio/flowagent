"""Common types used by the agent system."""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

class AgentType(Enum):
    """Types of agents in the system."""
    PLAN = "plan"
    TASK = "task" 
    DEBUG = "debug"

@dataclass
class WorkflowStep:
    """A single step in the workflow."""
    name: str
    tool: str
    action: str
    parameters: Dict[str, Any]
    status: str = "pending"
    error: Optional[str] = None
    output: Optional[Dict[str, Any]] = None
