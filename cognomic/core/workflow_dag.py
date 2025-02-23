"""DAG-based workflow execution system."""

from typing import Dict, List, Set, Any
import networkx as nx
import asyncio
from dataclasses import dataclass
from .agent_types import WorkflowStep
from ..utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class DAGNode:
    """Represents a node in the workflow DAG."""
    id: str
    step: WorkflowStep
    status: str = "pending"  # pending, running, completed, failed
    result: Any = None

class WorkflowDAG:
    """Manages workflow execution as a Directed Acyclic Graph."""
    
    def __init__(self):
        """Initialize the workflow DAG."""
        self.graph = nx.DiGraph()
        self.logger = get_logger(__name__)
        
    def add_step(self, step: WorkflowStep, dependencies: List[str] = None) -> None:
        """Add a step to the workflow graph with optional dependencies."""
        node = DAGNode(id=step["name"], step=step)
        self.graph.add_node(node.id, node=node)
        
        if dependencies:
            for dep in dependencies:
                if dep in self.graph:
                    self.graph.add_edge(dep, node.id)
                else:
                    self.logger.warning(f"Dependency {dep} not found for step {node.id}")
    
    def get_ready_steps(self) -> List[DAGNode]:
        """Get all steps that are ready to execute (all dependencies completed)."""
        ready_steps = []
        for node_id in self.graph.nodes():
            node = self.graph.nodes[node_id]["node"]
            if node.status == "pending":
                predecessors = list(self.graph.predecessors(node_id))
                if all(self.graph.nodes[pred]["node"].status == "completed" 
                      for pred in predecessors):
                    ready_steps.append(node)
        return ready_steps
    
    def update_step_status(self, step_id: str, status: str, result: Any = None) -> None:
        """Update the status and result of a step."""
        node = self.graph.nodes[step_id]["node"]
        node.status = status
        if result is not None:
            node.result = result
    
    def get_all_results(self) -> Dict[str, Any]:
        """Get results from all completed steps."""
        results = {}
        for node_id in self.graph.nodes():
            node = self.graph.nodes[node_id]["node"]
            if node.status == "completed" and node.result is not None:
                results[node_id] = node.result
        return results
    
    async def execute_parallel(self, execute_step_fn) -> Dict[str, Any]:
        """Execute the workflow graph in parallel where possible."""
        pending_nodes = set(self.graph.nodes())
        running_tasks = set()
        results = {}
        
        while pending_nodes or running_tasks:
            # Get steps that are ready to execute
            ready_steps = self.get_ready_steps()
            
            # Start new tasks for ready steps
            for step in ready_steps:
                if step.id in pending_nodes:
                    task = asyncio.create_task(execute_step_fn(step.step))
                    running_tasks.add(task)
                    pending_nodes.remove(step.id)
                    self.update_step_status(step.id, "running")
            
            # Wait for any task to complete
            if running_tasks:
                done, running_tasks = await asyncio.wait(
                    running_tasks, 
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process completed tasks
                for task in done:
                    try:
                        result = await task
                        step_id = result["name"]
                        self.update_step_status(step_id, "completed", result)
                        results[step_id] = result
                    except Exception as e:
                        self.logger.error(f"Step execution failed: {str(e)}")
                        # Mark the failed step and its descendants as failed
                        failed_step = next(
                            (node for node in self.graph.nodes() 
                             if self.graph.nodes[node]["node"].status == "running"),
                            None
                        )
                        if failed_step:
                            self.update_step_status(failed_step, "failed")
                            for descendant in nx.descendants(self.graph, failed_step):
                                self.update_step_status(descendant, "failed")
                        raise
            
            # Small delay to prevent busy waiting
            await asyncio.sleep(0.1)
        
        return results
