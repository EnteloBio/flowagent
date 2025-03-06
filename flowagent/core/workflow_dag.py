"""Directed Acyclic Graph for workflow management."""

import os
import logging
import json
import datetime
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from enum import Enum
from pathlib import Path

# Monkey patch to fix numpy alltrue deprecation in networkx
# This addresses the issue with np.alltrue being removed in NumPy 2.0
import networkx.drawing.nx_pylab as nx_pylab
if hasattr(nx_pylab, 'np') and hasattr(nx_pylab.np, 'alltrue'):
    # Replace alltrue with all in the nx_pylab module
    nx_pylab.np.alltrue = nx_pylab.np.all

import asyncio
import logging
import traceback

from ..utils.logging import get_logger
from .executors import BaseExecutor, LocalExecutor, HPCExecutor, KubernetesExecutor

logger = get_logger(__name__)

class WorkflowDAG:
    """Manages workflow as a Directed Acyclic Graph."""
    
    def __init__(self, executor_type: str = "local"):
        """Initialize workflow DAG.
        
        Args:
            executor_type: Type of executor to use ("local", "hpc", or "kubernetes")
        """
        self.graph = nx.DiGraph()
        
        # Initialize appropriate executor
        if executor_type == "kubernetes":
            self.executor = KubernetesExecutor()
        elif executor_type == "hpc":
            self.executor = HPCExecutor()
        else:
            self.executor = LocalExecutor()
            
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
            Path to the generated visualization file or None if visualization fails
        """
        if output_file is None:
            logger.warning("No output file specified for workflow DAG visualization")
            return None
            
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
                try:
                    # Get the step, handle case where key doesn't exist
                    step = self.graph.nodes[node].get("step")
                    if step is None:
                        logger.warning(f"Node {node} has no step data, skipping status check")
                        color = "#D3D3D3"  # Light gray for unknown status
                        pending_nodes.append(node)
                        node_colors.append(color)
                        continue
                    
                    # Check if step is a dictionary or an object
                    if isinstance(step, dict):
                        status = step.get("status", "pending")
                    else:
                        # Handle case where step is an object (like WorkflowStep)
                        status = getattr(step, "status", "pending")
                    
                    # Check if this is a starting node (no incoming edges)
                    is_start_node = self.graph.in_degree(node) == 0
                    
                    if is_start_node:
                        color = "#FFD700"  # Gold
                        start_nodes.append(node)
                    elif status == "completed":
                        color = "#90EE90"  # Light green
                        completed_nodes.append(node)
                    elif status in ["failed", "error"]:
                        color = "#FFA07A"  # Light salmon
                        failed_nodes.append(node)
                    else:
                        color = "#ADD8E6"  # Light blue
                        pending_nodes.append(node)
                    
                    node_colors.append(color)
                except Exception as e:
                    logger.warning(f"Error processing node {node}: {str(e)}")
                    color = "#D3D3D3"  # Light gray for unknown status
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
                          markerfacecolor='#FFA07A', markersize=15, label='Failed'),
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
            logger.error(f"Failed to visualize workflow: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
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
                try:
                    result = await execute(step)
                    jobs[step_name] = result
                    
                    # Update step status in graph
                    self.graph.nodes[step_name]["step"]["status"] = result.get("status", "pending")
                    
                    if result.get("status") == "failed":
                        error_msg = result.get("stderr", "")
                        cmd = step.get("command", "")
                        logger.error(f"Step {step_name} failed:\nCommand: {cmd}\nError: {error_msg}")
                        
                        # Check for common error patterns
                        if "command not found" in error_msg:
                            logger.error(f"Tool '{cmd.split()[0]}' not found. Please ensure it is installed and in your PATH")
                        elif "permission denied" in error_msg.lower():
                            logger.error("Permission denied. Check file/directory permissions")
                        elif "no such file" in error_msg.lower():
                            logger.error("Required input file not found. Check file paths and names")
                            
                        raise Exception(f"Step {step_name} failed: {error_msg}")
                        
                except Exception as step_error:
                    logger.error(f"Error executing step {step_name}: {str(step_error)}")
                    self.graph.nodes[step_name]["step"]["status"] = "failed"
                    raise
            
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
            # Mark remaining steps as failed and log detailed error
            failed_step = None
            for step_name in self.graph.nodes:
                status = self.graph.nodes[step_name]["step"].get("status", "")
                if status == "failed":
                    failed_step = step_name
                    break
                elif status != "completed":
                    self.graph.nodes[step_name]["step"]["status"] = "cancelled"
            
            if failed_step:
                step = self.graph.nodes[failed_step]["step"]
                logger.error(f"Workflow failed at step '{failed_step}':")
                logger.error(f"Command: {step.get('command', 'N/A')}")
                logger.error(f"Error: {str(e)}")
                logger.error("Dependencies:")
                for dep in self.graph.predecessors(failed_step):
                    dep_status = self.graph.nodes[dep]["step"].get("status", "unknown")
                    logger.error(f"  - {dep}: {dep_status}")
            
            return {
                "status": "failed",
                "error": str(e),
                "failed_step": failed_step
            }
