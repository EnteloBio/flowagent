"""Directed Acyclic Graph for workflow management."""

import networkx as nx
from typing import Dict, Any, List, Callable, Optional
import asyncio
import logging

from ..utils.logging import get_logger
from .executors import BaseExecutor, LocalExecutor, CGATExecutor

logger = get_logger(__name__)

class WorkflowDAG:
    """Manages workflow as a Directed Acyclic Graph."""
    
    def __init__(self, executor_type: str = "local"):
        """Initialize workflow DAG.
        
        Args:
            executor_type: Type of executor to use ("local" or "cgat")
        """
        self.graph = nx.DiGraph()
        self.executor = (
            CGATExecutor() if executor_type == "cgat" 
            else LocalExecutor()
        )
        logger.info(f"Initialized WorkflowDAG with {executor_type} executor")
    
    def add_step(self, step: Dict[str, Any], dependencies: List[str] = None):
        """Add a step to the workflow graph."""
        self.graph.add_node(step["name"], step=step)
        
        if dependencies:
            for dep in dependencies:
                if dep not in self.graph:
                    raise ValueError(f"Dependency {dep} not found in graph")
                self.graph.add_edge(dep, step["name"])
                
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Dependencies would create a cycle in the graph")
    
    async def execute_parallel(self, execute_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute workflow steps in parallel respecting dependencies.
        
        Args:
            execute_fn: Optional function to execute steps. If not provided,
                       uses the configured executor.
        """
        try:
            # Use provided execute function or executor
            execute = execute_fn or self.executor.execute_step
            
            # Track job information
            jobs = {}
            
            # Execute steps in topological order
            for step_name in nx.topological_sort(self.graph):
                step = self.graph.nodes[step_name]["step"]
                
                # Get dependency job IDs
                step["dependencies"] = [
                    jobs[dep]["job_id"] 
                    for dep in self.graph.predecessors(step_name)
                    if dep in jobs and "job_id" in jobs[dep]
                ]
                
                # Execute step
                logger.info(f"Executing step: {step_name}")
                result = await execute(step)
                jobs[step_name] = result
                
                if result.get("status") == "failed":
                    raise Exception(f"Step {step_name} failed: {result.get('stderr', '')}")
            
            # Wait for all jobs to complete
            results = await self.executor.wait_for_completion(jobs)
            
            return {
                "status": "success",
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Step execution failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "results": jobs
            }
