"""Workflow manager for coordinating LLM-based workflow execution."""

from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import asyncio
import networkx as nx
import json
import os

from ..utils.logging import get_logger
from ..utils import file_utils
from .agent_system import AgentSystem
from .llm import LLMInterface
from .agent_types import WorkflowStep
from .workflow_dag import WorkflowDAG

logger = get_logger(__name__)

class WorkflowManager:
    """Manages workflow execution and coordination."""
    
    def __init__(self):
        """Initialize workflow manager."""
        self.logger = get_logger(__name__)
        self.llm = LLMInterface()
        self.agent_system = AgentSystem(self.llm)
        self.cwd = os.getcwd()
        self.logger.info(f"Initial working directory: {self.cwd}")

    async def execute_workflow(self, prompt: str) -> Dict[str, Any]:
        """Execute workflow from prompt."""
        try:
            self.logger.info("Planning workflow steps...")
            workflow_plan = await self.llm.generate_workflow_plan(prompt)
            
            # Create output directories
            for step in workflow_plan["steps"]:
                output_dir = step["parameters"].get("output_dir")
                if output_dir:
                    file_utils.ensure_directory(output_dir)
                    self.logger.info(f"Created output directory: {output_dir}")
            
            # Create workflow DAG
            dag = WorkflowDAG()
            
            # Add steps to DAG with dependencies
            for step in workflow_plan["steps"]:
                dependencies = step.get("dependencies", [])
                dag.add_step(step, dependencies)
            
            # Execute workflow using parallel execution
            results = await dag.execute_parallel(self.agent_system.execute_step)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            raise

    async def resume_workflow(self, prompt: str, checkpoint_dir: str) -> Dict[str, Any]:
        """Resume workflow execution from checkpoint."""
        try:
            # Load checkpoint state
            checkpoint_file = os.path.join(checkpoint_dir, "workflow_state.json")
            if not os.path.exists(checkpoint_file):
                self.logger.warning(f"No checkpoint found at {checkpoint_file}, starting new workflow")
                return await self.execute_workflow(prompt)
                
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                
            self.logger.info(f"Resuming workflow from checkpoint: {checkpoint_file}")
            
            # Get completed steps
            completed_steps = set(step["name"] for step in checkpoint.get("results", []) 
                               if step["status"] == "success")
            
            # Get workflow plan
            workflow_plan = await self.llm.generate_workflow_plan(prompt)
            
            # Filter out completed steps
            remaining_steps = [step for step in workflow_plan["steps"] 
                             if step["name"] not in completed_steps]
            
            if not remaining_steps:
                self.logger.info("All steps already completed")
                return checkpoint
                
            self.logger.info(f"Resuming with {len(remaining_steps)} remaining steps")
            
            # Execute remaining steps
            results = checkpoint.get("results", [])
            
            # Execute remaining steps using agent system
            remaining_results = await self.agent_system.execute_workflow({"steps": remaining_steps})
            results.extend(remaining_results)
            
            # Update checkpoint after each batch
            checkpoint["results"] = results
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            
            # Generate final report
            report = await self._generate_analysis_report(results)
            checkpoint["report"] = report
            
            # Save final state
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
                
            return checkpoint
            
        except Exception as e:
            self.logger.error(f"Error resuming workflow: {str(e)}")
            raise

    def _build_dag(self, workflow_steps: List[WorkflowStep]) -> nx.DiGraph:
        """Build DAG from workflow steps."""
        dag = nx.DiGraph()
        for step in workflow_steps:
            dag.add_node(step.name, step=step, status='pending')
            for dep in step.dependencies:
                dag.add_edge(dep, step.name)
        return dag

    def get_execution_plan(self, dag: nx.DiGraph) -> list:
        """Get ordered list of task batches that can be executed in parallel."""
        execution_plan = []
        remaining_nodes = set(dag.nodes())
        
        while remaining_nodes:
            # Find nodes with no incomplete dependencies
            ready_nodes = {
                node for node in remaining_nodes
                if not any(pred in remaining_nodes for pred in dag.predecessors(node))
            }
            
            if not ready_nodes:
                # There are nodes left but none are ready - there must be a cycle
                raise ValueError("Cycle detected in workflow DAG")
            
            execution_plan.append(list(ready_nodes))
            remaining_nodes -= ready_nodes
        
        return execution_plan

    async def _generate_analysis_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate analysis report from workflow results."""
        try:
            # Check if we have any results
            if not results:
                return {
                    "status": None,
                    "quality": "unknown",
                    "issues": [{
                        "severity": "high",
                        "description": "No tool outputs available",
                        "impact": "Lack of tool outputs makes it impossible to assess the quality or draw any meaningful conclusions from the analysis.",
                        "solution": "Check the workflow configuration to ensure that tools are properly executed and their outputs are captured for analysis."
                    }],
                    "warnings": [{
                        "severity": "high",
                        "description": "Missing tool outputs",
                        "impact": "Without tool outputs, it is not possible to verify the analysis results or troubleshoot any potential issues.",
                        "solution": "Review the workflow execution logs to identify any errors or issues that might have prevented tool outputs from being generated."
                    }],
                    "recommendations": [{
                        "type": "quality",
                        "description": "Ensure all tools in the workflow are properly configured and executed to generate necessary outputs.",
                        "reason": "Having complete tool outputs is essential for quality assessment and interpretation of the analysis results."
                    }]
                }
            
            # Analyze results
            status = all(r["status"] == "success" for r in results)
            quality = "good" if status else "poor"
            
            issues = []
            warnings = []
            recommendations = []
            
            for result in results:
                if result["status"] != "success":
                    issues.append({
                        "severity": "high",
                        "description": f"Step '{result['step']}' failed",
                        "error": result.get("error", "Unknown error"),
                        "diagnosis": result.get("diagnosis", {})
                    })
            
            return {
                "status": "success" if status else "failed",
                "quality": quality,
                "issues": issues,
                "warnings": warnings,
                "recommendations": recommendations
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate analysis report: {str(e)}")
            return None
