"""Directed Acyclic Graph for workflow management."""

import networkx as nx
from typing import Dict, Any, List, Callable, Optional
import asyncio
import logging

from ..utils.logging import get_logger
from .executors import BaseExecutor, LocalExecutor, CGATExecutor, KubernetesExecutor

logger = get_logger(__name__)

class WorkflowDAG:
    """Manages workflow as a Directed Acyclic Graph."""
    
    def __init__(self, executor_type: str = "local"):
        """Initialize workflow DAG.
        
        Args:
            executor_type: Type of executor to use ("local", "cgat", or "kubernetes")
        """
        self.graph = nx.DiGraph()
        
        # Map executor types to executor classes
        executor_map = {
            "local": LocalExecutor,
            "cgat": CGATExecutor,
            "kubernetes": KubernetesExecutor if "KubernetesExecutor" in globals() else LocalExecutor
        }
        
        # Get executor class
        executor_class = executor_map.get(executor_type, LocalExecutor)
        self.executor = executor_class()
        
        logger.info(f"Initialized WorkflowDAG with {executor_type} executor")
    
    def add_step(self, step: Dict[str, Any], dependencies: List[str] = None):
        """Add a step to the workflow graph.
        
        Args:
            step: Step configuration dictionary
            dependencies: List of step names this step depends on
        """
        # Ensure step has required fields
        if "name" not in step:
            raise ValueError("Step must have a name")
        if "command" not in step:
            raise ValueError("Step must have a command")
            
        # Ensure resources are properly formatted
        if "resources" in step:
            resources = step["resources"]
            if isinstance(resources, dict):
                # Convert numeric values to integers
                for key in ["memory_mb", "cpus", "time_min"]:
                    if key in resources:
                        try:
                            resources[key] = int(resources[key])
                        except (ValueError, TypeError):
                            resources[key] = 4000 if key == "memory_mb" else 1 if key == "cpus" else 60
            else:
                # Set default resources if not a dict
                step["resources"] = {
                    "memory_mb": 4000,
                    "cpus": 1,
                    "time_min": 60,
                    "profile": "default"
                }
        else:
            # Set default resources if not present
            step["resources"] = {
                "memory_mb": 4000,
                "cpus": 1,
                "time_min": 60,
                "profile": "default"
            }
            
        # Add step to graph
        self.graph.add_node(step["name"], step=step)
        
        # Add dependencies
        if dependencies:
            for dep in dependencies:
                if dep not in self.graph:
                    raise ValueError(f"Dependency {dep} not found in graph")
                self.graph.add_edge(dep, step["name"])
                
        # Verify DAG remains acyclic
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
