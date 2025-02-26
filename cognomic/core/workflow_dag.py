"""Directed Acyclic Graph for workflow management."""

import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional
import asyncio
import logging

from ..utils.logging import get_logger
from .executors import BaseExecutor, LocalExecutor, HPCExecutor

logger = get_logger(__name__)

class WorkflowDAG:
    """Manages workflow as a Directed Acyclic Graph."""
    
    def __init__(self, executor_type: str = "local"):
        """Initialize workflow DAG.
        
        Args:
            executor_type: Type of executor to use ("local" or "hpc")
        """
        self.graph = nx.DiGraph()
        self.executor = (
            HPCExecutor() if executor_type == "hpc" 
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
            
    def visualize(self, output_file: Path) -> Path:
        """Generate a visualization of the workflow DAG.
        
        Args:
            output_file: Path to save the visualization file
            
        Returns:
            Path to the generated visualization file
        """
        try:
            if not self.graph.nodes:
                logger.warning("No nodes in workflow graph to visualize")
                return None
                
            # Create parent directory if it doesn't exist
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Clear any existing plot
            plt.clf()
            
            # Set up the plot with title
            plt.figure(figsize=(12, 8))
            plt.title("Workflow DAG", pad=20, size=16)
            
            # Use hierarchical layout for better workflow visualization
            pos = nx.spring_layout(self.graph, k=2, iterations=50)
            
            # Draw nodes with status-based colors
            node_colors = []
            completed_nodes = []
            failed_nodes = []
            pending_nodes = []
            start_nodes = []
            
            for node in self.graph.nodes:
                step = self.graph.nodes[node]["step"]
                status = step.get("status", "pending")
                
                # Check if this is a starting node (no incoming edges)
                is_start_node = self.graph.in_degree(node) == 0
                
                if is_start_node:
                    color = "#FFD700"  # Gold
                    start_nodes.append(node)
                elif status == "completed":
                    color = "#90EE90"  # Light green
                    completed_nodes.append(node)
                elif status == "failed":
                    color = "#FF6B6B"  # Light red
                    failed_nodes.append(node)
                else:
                    color = "#ADD8E6"  # Light blue
                    pending_nodes.append(node)
                node_colors.append(color)
            
            # Draw nodes
            nx.draw_networkx_nodes(self.graph, pos,
                                 node_color=node_colors,
                                 node_size=2000,
                                 alpha=0.9)
            
            # Draw edges with better arrows
            nx.draw_networkx_edges(self.graph, pos,
                                 edge_color='gray',
                                 arrows=True,
                                 arrowsize=20,
                                 arrowstyle='->',
                                 width=1.5)
            
            # Add labels with better font
            nx.draw_networkx_labels(self.graph, pos,
                                  font_size=10,
                                  font_weight='bold')
            
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor='#FFD700', markersize=15, label='Start'),
                plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='#90EE90', markersize=15, label='Completed'),
                plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='#FF6B6B', markersize=15, label='Failed'),
                plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='#ADD8E6', markersize=15, label='Pending')
            ]
            plt.legend(handles=legend_elements, loc='upper left', 
                      bbox_to_anchor=(1.05, 1), fontsize=10)
            
            # Add margin around the plot
            plt.margins(0.2)
            
            # Save the plot
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            plt.close()
            
            logger.info(f"Saved workflow visualization to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to visualize workflow: {e}")
            return None
    
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
                
                # Update step status in graph
                self.graph.nodes[step_name]["step"]["status"] = result.get("status", "pending")
                
                if result.get("status") == "failed":
                    raise Exception(f"Step {step_name} failed: {result.get('stderr', '')}")
            
            # Wait for all jobs to complete
            results = await self.executor.wait_for_completion(jobs)
            
            # Update final status for all steps
            for step_name, result in results.items():
                self.graph.nodes[step_name]["step"]["status"] = result.get("status", "completed")
            
            return {
                "status": "success",
                "results": results
            }
            
        except Exception as e:
            # Mark remaining steps as failed
            for step_name in self.graph.nodes:
                if step_name not in jobs:
                    self.graph.nodes[step_name]["step"]["status"] = "failed"
                    
            logger.error(f"Step execution failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "results": jobs
            }
