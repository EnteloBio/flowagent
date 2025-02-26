"""Workflow manager for coordinating LLM-based workflow execution."""

from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import asyncio
import networkx as nx
import json
import os
from cognomic.config.settings import Settings

from ..utils.logging import get_logger
from ..utils import file_utils
from ..utils.dependency_manager import DependencyManager
from .llm import LLMInterface
from .agent_types import WorkflowStep
from .workflow_dag import WorkflowDAG
from ..agents.agentic.analysis_system import AgenticAnalysisSystem

logger = get_logger(__name__)

class WorkflowManager:
    """Manages workflow execution and coordination."""
    
    def __init__(self, executor_type: Optional[str] = None):
        """Initialize workflow manager.
        
        Args:
            executor_type: Type of executor to use ("local" or "cgat"). 
                         If None, uses EXECUTOR_TYPE from settings.
        """
        self.logger = get_logger(__name__)
        self.llm = LLMInterface()
        self.dependency_manager = DependencyManager()
        self.analysis_system = AgenticAnalysisSystem()
        
        # Get settings
        self.settings = Settings()
        
        # Use provided executor_type or get from settings
        self.executor_type = executor_type or self.settings.EXECUTOR_TYPE
        if self.executor_type not in ["local", "cgat"]:
            self.logger.warning(f"Invalid executor type '{self.executor_type}'. Defaulting to 'local'")
            self.executor_type = "local"
            
        self.cwd = os.getcwd()
        self.logger.info(f"Initial working directory: {self.cwd}")
        self.logger.info(f"Using {self.executor_type} executor")

    async def execute_workflow(self, prompt: str) -> Dict[str, Any]:
        """Execute workflow from prompt."""
        try:
            self.logger.info("Planning workflow steps...")
            workflow_plan = await self.llm.generate_workflow_plan(prompt)
            
            # Check and install required dependencies using LLM analysis
            self.logger.info("Analyzing and installing required dependencies...")
            if not await self.dependency_manager.ensure_workflow_dependencies(workflow_plan):
                raise ValueError("Failed to ensure all required workflow dependencies")
            
            # Get output directory from workflow steps
            output_dir = None
            steps = workflow_plan.get("steps", [])
            
            # Look for create_directories step first
            for step in steps:
                if step.get("name", "").lower() == "create_directories":
                    cmd = step.get("command", "")
                    if "mkdir" in cmd and "-p" in cmd:
                        # Extract base output directory from mkdir command
                        parts = cmd.split()
                        for part in parts:
                            if "results/" in part:
                                dirs = part.split()
                                output_dir = dirs[0].split("/")[0:2]  # Get base output dir
                                output_dir = "/".join(output_dir)
                                break
            
            # Fallback: check other steps for output directory pattern
            if not output_dir:
                for step in steps:
                    cmd = step.get("command", "")
                    if "results/" in cmd:
                        parts = cmd.split()
                        for part in parts:
                            if "results/" in part:
                                dirs = part.split("/")[0:2]  # Get base output dir
                                output_dir = "/".join(dirs)
                                break
                    if output_dir:
                        break
            
            if not output_dir:
                raise ValueError("No output directory found in workflow steps")
                
            # Resolve output directory path
            output_dir = str(Path(output_dir).resolve())
            self.logger.info(f"Using output directory from workflow steps: {output_dir}")
            
            # Create output directory
            file_utils.ensure_directory(output_dir)
            
            # Create workflow DAG with specified executor
            self.dag = WorkflowDAG(executor_type=self.executor_type)
            
            # Add steps to DAG with dependencies
            for step in workflow_plan["steps"]:
                dependencies = step.get("dependencies", [])
                self.dag.add_step(step, dependencies)
            
            # Execute workflow using parallel execution
            results = await self.dag.execute_parallel()
            
            # Add output directory to results
            results["output_directory"] = output_dir
            
            # Analyze results using agentic system
            self.logger.info(f"Analyzing results in directory: {output_dir}")
            analysis = await self.analyze_results(results)
            
            # Combine workflow results with analysis
            return {
                "workflow_results": results,
                "analysis": analysis,
                "output_directory": output_dir
            }
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            raise

    async def analyze_results(self, workflow_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze workflow results using agentic system."""
        try:
            self.logger.info("Starting agentic analysis of workflow results...")
            
            # Get results directory from workflow results
            results_dir = None
            
            # Method 1: Try primary output directory
            if "primary_output_dir" in workflow_results:
                results_dir = Path(workflow_results["primary_output_dir"])
            
            # Method 2: Try output directories list
            if not results_dir and "output_directories" in workflow_results:
                output_dirs = workflow_results["output_directories"]
                if output_dirs:
                    results_dir = Path(output_dirs[0])
            
            # Method 3: Try output_directory field
            if not results_dir:
                results_dir = Path(workflow_results.get("output_directory", "results"))
            
            # Validate directory exists
            if not results_dir or not results_dir.exists():
                self.logger.warning("No valid output directory found for analysis")
                return {
                    "status": "error",
                    "error": "No valid output directory found"
                }
                
            self.logger.info(f"Analyzing results in directory: {results_dir}")
            analysis_data = await self.analysis_system._prepare_analysis_data(results_dir)
            
            # Run analysis agents
            quality_analysis = await self.analysis_system.quality_agent.analyze(analysis_data)
            quant_analysis = await self.analysis_system.quantification_agent.analyze(analysis_data)
            tech_analysis = await self.analysis_system.technical_agent.analyze(analysis_data)
            
            self.logger.info("Agentic analysis completed successfully")
            
            return {
                "status": "success",
                "quality": quality_analysis,
                "quantification": quant_analysis,
                "technical": tech_analysis,
                "data": analysis_data
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

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
                               if step["status"] == "completed")
            
            # Get workflow plan
            workflow_plan = await self.llm.generate_workflow_plan(prompt)
            
            # Filter out completed steps
            remaining_steps = [step for step in workflow_plan["steps"] 
                             if step["name"] not in completed_steps]
            
            if not remaining_steps:
                self.logger.info("All steps already completed")
                return checkpoint
                
            self.logger.info(f"Resuming with {len(remaining_steps)} remaining steps")
            
            # Create workflow DAG for remaining steps
            self.dag = WorkflowDAG(executor_type=self.executor_type)
            
            # Add remaining steps to DAG
            for step in remaining_steps:
                dependencies = [dep for dep in step.get("dependencies", [])
                              if dep not in completed_steps]
                self.dag.add_step(step, dependencies)
            
            # Execute remaining steps
            results = await self.dag.execute_parallel()
            
            # Analyze results using agentic system
            analysis = await self.analyze_results(results)
            
            # Combine workflow results with analysis
            return {
                "workflow_results": results,
                "analysis": analysis
            }
            
        except Exception as e:
            self.logger.error(f"Failed to resume workflow: {str(e)}")
            raise

    async def _prepare_step(self, step: Dict[str, Any], workflow_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare a workflow step for execution."""
        try:
            # Extract basic step info
            step_name = step.get("name", "unknown")
            command = step.get("command", "")
            dependencies = step.get("dependencies", [])
            
            # Get resource requirements
            resources = step.get("resources", {})
            profile = resources.get("profile", "default")
            
            # Create the step dictionary
            prepared_step = {
                "name": step_name,
                "command": command,
                "status": "pending",
                "dependencies": dependencies,
                "profile": profile,
                "output": "",
                "error": None,
                "start_time": None,
                "end_time": None
            }
            
            return prepared_step
            
        except Exception as e:
            self.logger.error(f"Error preparing step {step.get('name', 'unknown')}: {str(e)}")
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
