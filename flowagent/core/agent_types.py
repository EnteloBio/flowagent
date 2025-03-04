"""Common types used by the agent system."""

from enum import Enum
from dataclasses import dataclass, field
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
    description: str = ""
    tool: str = ""
    command: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    critical: bool = True
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    error: Optional[str] = None
    output: Optional[Dict[str, Any]] = None
    action: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

@dataclass
class Workflow:
    """A workflow containing multiple steps."""
    name: str
    description: str = ""
    steps: List[WorkflowStep] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    output_dir: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    status: str = "pending"
