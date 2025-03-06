"""Workflow manager for coordinating LLM-based workflow execution."""

from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging
import asyncio
import networkx as nx
import json
import os
import sys
from datetime import datetime
from flowagent.config.settings import Settings

from ..utils.logging import get_logger
from ..utils import file_utils
from ..utils.dependency_manager import DependencyManager
from .llm import LLMInterface
from .executor import Executor
from .agent_types import WorkflowStep, Workflow
from .workflow_dag import WorkflowDAG
from .smart_resume import detect_completed_steps, filter_workflow_steps
from ..agents.agentic.analysis_system import AgenticAnalysisSystem
from .api_usage import APIUsageTracker

logger = get_logger(__name__)

class WorkflowManager:
    """Manages workflow execution and coordination."""
    
    def __init__(self, executor_type: str = "local"):
        """Initialize the workflow manager.
        
        Args:
            executor_type: Type of executor to use (local, slurm, etc.)
        """
        self.logger = get_logger(__name__)
        self.llm = LLMInterface()
        self.logger.debug("LLMInterface initialized. Checking available methods...")
        if hasattr(self.llm, 'generate_workflow_plan'):
            self.logger.debug("generate_workflow_plan method is available.")
        else:
            self.logger.error("generate_workflow_plan method is NOT available.")
        self.dependency_manager = DependencyManager()
        self.executor_type = executor_type
        self.executor = Executor(executor_type)
        
        # Get initial working directory
        self.initial_cwd = os.getcwd()
        self.logger.info(f"Initial working directory: {self.initial_cwd}")
        self.logger.info(f"Using {executor_type} executor")
        
        # Get settings
        self.settings = Settings()
        
        # Validate executor type
        valid_executors = ["local", "cgat", "kubernetes"]
        if self.executor_type not in valid_executors:
            self.logger.warning(f"Invalid executor type '{self.executor_type}'. Defaulting to 'local'")
            self.executor_type = "local"
        
        # Special handling for Kubernetes executor
        if self.executor_type == "kubernetes" and not self.settings.KUBERNETES_ENABLED:
            self.logger.warning("Kubernetes executor requested but not enabled in settings. Defaulting to 'local'")
            self.executor_type = "local"
            
        self.analysis_system = AgenticAnalysisSystem()
        
    async def execute_workflow(self, prompt_or_workflow: Union[str, Workflow]) -> Dict[str, Any]:
        """Execute workflow from prompt or workflow object."""
        try:
            # Check if input is a workflow object or a prompt string
            if isinstance(prompt_or_workflow, Workflow):
                workflow = {
                    "name": prompt_or_workflow.name,
                    "description": prompt_or_workflow.description,
                    "steps": [vars(step) for step in prompt_or_workflow.steps]
                }
                workflow_name = workflow["name"]
                workflow_steps = workflow["steps"]
                prompt = None
            else:
                prompt = prompt_or_workflow
                workflow_data = await self.llm.generate_workflow_plan(prompt)
                self.logger.debug("generate_workflow_plan method called successfully.")
                workflow = workflow_data.get("workflow", {})
                workflow_name = workflow.get("name", "Unnamed workflow")
                workflow_steps = workflow.get("steps", [])
            
            # Create output directory
            output_dir = os.path.join(self.initial_cwd, "flowagent_output", workflow_name.replace(" ", "_"))
            os.makedirs(output_dir, exist_ok=True)
            
            # Save workflow to output directory
            workflow_file = os.path.join(output_dir, "workflow.json")
            with open(workflow_file, "w") as f:
                json.dump(workflow, f, indent=2)
            
            self.logger.info(f"Saved workflow to {workflow_file}")
            
            # Execute workflow steps
            self.logger.info(f"Executing workflow: {workflow_name}")
            
            # Start workflow API usage tracking
            workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.llm.api_usage_tracker.start_workflow(workflow_id, workflow_name)
            
            results = []
            for i, step in enumerate(workflow_steps):
                step_name = step.get("name", f"Step {i+1}")
                self.logger.info(f"Executing step {i+1}/{len(workflow_steps)}: {step_name}")
                
                # Execute step
                step_result = await self.executor.execute_step(step, output_dir, cwd=self.initial_cwd)
                
                # Add step name to the result for easier reference
                step_result["step_name"] = step_name
                
                results.append(step_result)
                
                # Check if step failed
                if step_result.get("status") == "error":
                    error_msg = step_result.get('error', '')
                    stderr = step_result.get('stderr', '')
                    
                    self.logger.error(f"Step {step_name} failed: {error_msg}")
                    if stderr:
                        self.logger.error(f"STDERR: {stderr}")
                    
                    # Check if this is a critical step
                    if step.get("critical", False):
                        self.logger.error(f"Critical step {step_name} failed, but continuing with workflow")
                    else:
                        self.logger.warning(f"Non-critical step {step_name} failed, continuing workflow")
                else:
                    self.logger.info(f"Step {step_name} completed successfully")
            
            # Generate workflow report
            report = self._generate_workflow_report(workflow, results, output_dir)
            
            # Determine overall workflow status
            workflow_status = "success"
            for result in results:
                if result.get("status") == "error":
                    # Check if this is a critical step
                    step_name = result.get("step_name", "unknown")
                    for step in workflow_steps:
                        if step.get("name") == step_name and step.get("critical", True):
                            workflow_status = "failed"
                            break
            
            # Generate workflow DAG visualization
            workflow_steps_with_status = []
            for i, step in enumerate(workflow_steps):
                # Check if step is a dictionary or a WorkflowStep object
                if isinstance(step, dict):
                    # Create a WorkflowStep from dictionary
                    step_copy = WorkflowStep(
                        name=step.get("name", f"Step {i+1}"),
                        description=step.get("description", ""),
                        command=step.get("command", ""),
                        dependencies=step.get("dependencies", [])
                    )
                else:
                    # Create a copy of the WorkflowStep object
                    step_copy = WorkflowStep(
                        name=step.name,
                        description=step.description,
                        command=step.command,
                        dependencies=step.dependencies
                    )
                
                # Set status based on execution results
                result_step = next((r for r in results if r.get("step_name") == step_copy.name), None)
                if result_step:
                    step_copy.status = result_step.get("status", "pending")
                else:
                    # For steps not found in results, check if they were skipped due to smart resume
                    # If the step name is in completed_steps (smart resume), mark it as completed
                    if hasattr(workflow, 'completed_steps') and step_copy.name in workflow.completed_steps:
                        step_copy.status = "completed"
                        self.logger.debug(f"Setting status of step {step_copy.name} to 'completed' based on smart resume")
                
                workflow_steps_with_status.append(step_copy)
            
            dag_image_path = self._save_workflow_dag(workflow_steps_with_status, output_dir)
            
            # End workflow API usage tracking
            workflow_usage = self.llm.api_usage_tracker.end_workflow(workflow_id)
            self.llm.api_usage_tracker.display_usage(workflow_id)
            
            # Return results
            result_dict = {
                "status": workflow_status,
                "workflow": workflow,
                "results": results,
                "report": report,
                "output_dir": output_dir,
                "api_usage": workflow_usage
            }
            
            # Only add dag_visualization if it was successfully created
            if dag_image_path:
                result_dict["dag_visualization"] = str(dag_image_path)
                
            return result_dict
        
        except Exception as e:
            self.logger.error(f"Error executing workflow: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
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
            
            # Start analysis API usage tracking
            analysis_name = f"Analysis of {str(results_dir)}"
            analysis_workflow_id = self.llm.api_usage_tracker.start_analysis_tracking(
                analysis_name=analysis_name,
                workflow_output_dir=str(results_dir)
            )
            
            # Perform analysis
            analysis_results = await self.analysis_system.analyze_results(workflow_results)
            
            # End analysis API usage tracking
            analysis_usage = self.llm.api_usage_tracker.end_workflow(analysis_workflow_id)
            
            # Add API usage to analysis results
            analysis_results["api_usage"] = analysis_usage
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing workflow results: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    async def plan_workflow(self, prompt: str) -> Dict[str, Any]:
        """Plan a workflow based on a natural language prompt without executing it.
        
        Args:
            prompt: Natural language prompt describing the workflow
            
        Returns:
            dict: Workflow plan
        """
        try:
            # Plan the workflow
            self.logger.info("Planning workflow steps...")
            workflow_plan = await self.llm.generate_workflow_plan(prompt)
            self.logger.debug("generate_workflow_plan method called successfully.")
            
            # Check dependencies with a timeout
            self.logger.info("Checking dependencies...")
            try:
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Dependency checking timed out")
                
                # Set a 30-second timeout for dependency checking
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)
                
                try:
                    all_installed, available_but_failed_install = await self.dependency_manager.ensure_workflow_dependencies(workflow_plan)
                finally:
                    # Cancel the alarm
                    signal.alarm(0)
                
                if not all_installed:
                    if available_but_failed_install:
                        # Some dependencies failed to install but are available in the environment
                        missing_deps = [dep for dep in workflow_plan.get("dependencies", {}).get("tools", []) 
                                      if isinstance(dep, dict) and dep["name"] not in available_but_failed_install
                                      or not isinstance(dep, dict) and dep not in available_but_failed_install]
                        if missing_deps:
                            self.logger.warning(f"Some dependencies could not be installed and are not available: {missing_deps}")
                            self.logger.warning("Continuing with workflow execution as some tools were found in the environment.")
                        else:
                            self.logger.info("All required tools are available in the environment despite installation failures.")
                    else:
                        # No dependencies are available, cannot proceed
                        self.logger.error("Dependencies not installed. Cannot execute workflow.")
                        return {"status": "error", "message": "Dependencies not installed. Cannot execute workflow."}
            
            except TimeoutError as e:
                self.logger.warning(f"Dependency checking timed out: {str(e)}")
                self.logger.warning("Continuing with workflow execution without full dependency verification")
            
            except Exception as e:
                self.logger.error(f"Error during dependency checking: {str(e)}")
                self.logger.warning("Continuing with workflow execution despite dependency checking error")
            
            return workflow_plan
            
        except Exception as e:
            self.logger.error(f"Failed to plan workflow: {str(e)}")
            raise

    async def resume_workflow(self, prompt: str, checkpoint_dir: str, force_resume: bool = False) -> Dict[str, Any]:
        """Resume workflow execution from checkpoint.
        
        Args:
            prompt: Natural language prompt describing the workflow
            checkpoint_dir: Directory to load workflow checkpoint from
            force_resume: If True, skip smart resume and run all steps
            
        Returns:
            dict: Workflow execution results
        """
        try:
            self.logger.info(f"Resuming workflow from checkpoint: {checkpoint_dir}")
            
            # Load checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.json")
            if not os.path.exists(checkpoint_path):
                self.logger.error(f"Checkpoint file not found: {checkpoint_path}")
                return None
            
            try:
                checkpoint = load_json(checkpoint_path)
                
                workflow_plan = checkpoint.get("workflow_plan", {})
                output_dir = checkpoint.get("output_dir", os.path.abspath("results"))
                
                # Ensure output directory exists
                os.makedirs(output_dir, exist_ok=True)
                
                # Detect completed steps only if force_resume is False
                completed_steps = set()
                if not force_resume:
                    # Convert to step_dicts format for smart resume
                    step_dicts = []
                    for step in workflow_plan.get("steps", []):
                        step_dict = {
                            "name": step.get("name", ""),
                            "command": step.get("command", ""),
                            "description": step.get("description", ""),
                            "dependencies": step.get("dependencies", [])
                        }
                        step_dicts.append(step_dict)
                    
                    # Detect completed steps
                    from flowagent.core.smart_resume import detect_completed_steps
                    completed_steps = detect_completed_steps(step_dicts)
                    if completed_steps:
                        self.logger.info(f"Detected {len(completed_steps)} completed steps: {', '.join(completed_steps)}")
                    else:
                        self.logger.info("No completed steps detected, starting from the beginning")
                else:
                    self.logger.info("Force resume enabled, running all steps regardless of completion status")
            
            except Exception as e:
                self.logger.error(f"Failed to load checkpoint: {str(e)}")
                return None
            
            # Get workflow name and description
            workflow_name = workflow_plan.get("name", "Unnamed workflow")
            workflow_description = workflow_plan.get("description", "")
            
            # Create workflow object
            workflow = Workflow(
                name=workflow_name,
                description=workflow_description,
                steps=[],
                dependencies=workflow_plan.get("dependencies", {}).get("tools", []),
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir
            )
            
            # Create steps
            for step_data in workflow_plan.get("steps", []):
                # Get resource requirements for the step
                resource_requirements = self._get_resource_requirements(step_data.get("tool", ""))
                
                # Create step object
                step = WorkflowStep(
                    name=step_data.get("name", ""),
                    description=step_data.get("description", ""),
                    tool=step_data.get("tool", ""),
                    command=step_data.get("command", ""),
                    args=step_data.get("args", {}),
                    dependencies=step_data.get("dependencies", []),
                    output_files=step_data.get("output_files", []),
                    critical=step_data.get("critical", True),
                    resource_requirements=resource_requirements
                )
                
                workflow.steps.append(step)
            
            # If a checkpoint directory is provided, use smart resume functionality
            if checkpoint_dir:
                self.logger.info("Using smart resume functionality")
                # Convert WorkflowStep objects to dictionaries for smart resume
                step_dicts = []
                for step in workflow.steps:
                    step_dict = {
                        "name": step.name,
                        "command": step.command,
                        "dependencies": step.dependencies
                    }
                    step_dicts.append(step_dict)
                
                # Detect completed steps
                completed_steps = detect_completed_steps(step_dicts)
                if completed_steps:
                    self.logger.info(f"Detected {len(completed_steps)} completed steps: {', '.join(completed_steps)}")
                
                # Store completed steps in the workflow object for later use in DAG visualization
                workflow.completed_steps = completed_steps
                
                # Filter workflow steps
                filtered_steps = filter_workflow_steps(step_dicts, completed_steps)
                
                # Update workflow steps based on filtered steps
                if len(filtered_steps) < len(step_dicts):
                    self.logger.info(f"Filtered {len(step_dicts) - len(filtered_steps)} steps, will run {len(filtered_steps)} steps")
                    # Keep only the steps that are in the filtered list
                    workflow.steps = [step for step in workflow.steps if any(step.name == fs["name"] for fs in filtered_steps)]
            
            # Execute the workflow
            result = await self.execute_workflow(workflow)
            
            # Generate analysis report if workflow was successful
            if result.get("status") == "success":
                self.logger.info("Workflow completed successfully")
                
                # Generate a report if requested
                if workflow_plan.get("generate_report", False):
                    self.logger.info("Generating report...")
                    from ..analysis.report_generator import ReportGenerator
                    report_generator = ReportGenerator()
                    report_path = await report_generator.generate_report(workflow_plan, result)
                    result["analysis_report"] = report_path
            
            # Generate workflow DAG visualization
            workflow_steps_with_status = []
            for i, step in enumerate(workflow.steps):
                # Check if step is a dictionary or a WorkflowStep object
                if isinstance(step, dict):
                    # Create a WorkflowStep from dictionary
                    step_copy = WorkflowStep(
                        name=step.get("name", f"Step {i+1}"),
                        description=step.get("description", ""),
                        command=step.get("command", ""),
                        dependencies=step.get("dependencies", [])
                    )
                else:
                    # Create a copy of the WorkflowStep object
                    step_copy = WorkflowStep(
                        name=step.name,
                        description=step.description,
                        command=step.command,
                        dependencies=step.dependencies
                    )
                
                # Set status based on execution results
                result_step = next((r for r in result.get("results", []) if r.get("step_name") == step_copy.name), None)
                if result_step:
                    step_copy.status = result_step.get("status", "pending")
                else:
                    # For steps not found in results, check if they were skipped due to smart resume
                    # If the step name is in completed_steps (smart resume), mark it as completed
                    if hasattr(workflow, 'completed_steps') and step_copy.name in workflow.completed_steps:
                        step_copy.status = "completed"
                        self.logger.debug(f"Setting status of step {step_copy.name} to 'completed' based on smart resume")
                
                workflow_steps_with_status.append(step_copy)
            
            dag_image_path = self._save_workflow_dag(workflow_steps_with_status, output_dir)
            
            # Return results
            result_dict = {
                "status": result.get("status", "unknown"),
                "output_dir": str(output_dir),
                "steps": [step_to_dict(step) for step in workflow_steps_with_status],
                "dag_image": dag_image_path and str(dag_image_path),
                "report": result.get("report", {})
            }
            
            # Display API usage statistics if available
            if hasattr(self.llm, "api_usage_tracker") and self.llm.api_usage_tracker:
                self.logger.info("Displaying API usage statistics...")
                # Force display API usage summary to console
                self.llm.api_usage_tracker.display_usage(workflow_id=f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            return result_dict
            
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

    def _save_workflow_dag(self, workflow_steps: List[WorkflowStep], output_dir: str) -> Optional[Path]:
        """Generate and save workflow DAG visualization.
        
        Args:
            workflow_steps: List of workflow steps
            output_dir: Directory to save the DAG image
            
        Returns:
            Path to the saved DAG image, or None if visualization failed
        """
        try:
            # Check if output_dir is None
            if output_dir is None:
                self.logger.warning("No output directory specified for workflow DAG visualization")
                return None
                
            # Check if workflow_steps is None or empty
            if not workflow_steps:
                self.logger.warning("No workflow steps provided for DAG visualization")
                return None
                
            # Create output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
            # Build the DAG
            try:
                dag = self._build_dag(workflow_steps)
                if not dag or not dag.nodes:
                    self.logger.warning("Built DAG is empty, nothing to visualize")
                    return None
            except Exception as e:
                self.logger.error(f"Error building DAG: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                return None
            
            # Update status of each step in the DAG based on workflow results
            for step in workflow_steps:
                try:
                    if step.name in dag.nodes:
                        # Get status from step, defaulting to 'pending'
                        status = 'pending'
                        if hasattr(step, 'status'):
                            status = step.status
                        elif isinstance(step, dict) and 'status' in step:
                            status = step['status']
                        
                        # Update the node's step status
                        # Convert WorkflowStep objects to dictionaries for compatibility with the visualizer
                        if isinstance(dag.nodes[step.name]['step'], dict):
                            dag.nodes[step.name]['step']['status'] = status
                        else:
                            # Convert WorkflowStep to dict if it's not already
                            step_as_dict = {
                                'name': dag.nodes[step.name]['step'].name,
                                'status': status,
                                'command': dag.nodes[step.name]['step'].command if hasattr(dag.nodes[step.name]['step'], 'command') else '',
                                'dependencies': dag.nodes[step.name]['step'].dependencies if hasattr(dag.nodes[step.name]['step'], 'dependencies') else []
                            }
                            dag.nodes[step.name]['step'] = step_as_dict
                except Exception as e:
                    self.logger.warning(f"Error updating step {getattr(step, 'name', 'unknown')} in DAG: {str(e)}")
                    continue
            
            # Create WorkflowDAG instance and visualize
            try:
                workflow_dag = WorkflowDAG(executor_type=self.executor_type)
                workflow_dag.graph = dag
                
                # Save visualization to output directory
                dag_image_path = Path(os.path.join(output_dir, "workflow_dag.png"))
                result_path = workflow_dag.visualize(dag_image_path)
                
                if result_path:
                    self.logger.info(f"Saved workflow DAG visualization to {result_path}")
                    return result_path
                else:
                    self.logger.warning("Failed to save workflow DAG visualization")
                    return None
            except Exception as e:
                self.logger.error(f"Error visualizing DAG: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                return None
        except Exception as e:
            self.logger.error(f"Failed to save workflow DAG visualization: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

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

    def _generate_workflow_report(self, workflow: Dict[str, Any], results: List[Dict[str, Any]], output_dir: str) -> Dict[str, Any]:
        """Generate a report for the workflow execution."""
        # Get workflow name and description
        workflow_name = workflow.get("name", "Unnamed workflow")
        workflow_description = workflow.get("description", "")
        
        # Calculate workflow statistics
        total_steps = len(results)
        successful_steps = sum(1 for r in results if r.get("status") == "success")
        failed_steps = sum(1 for r in results if r.get("status") == "error")
        skipped_steps = sum(1 for r in results if r.get("status") == "skipped")
        
        # Calculate execution time
        total_execution_time = sum(r.get("execution_time", 0) for r in results if "execution_time" in r)
        
        # Get dependency check results
        all_tools = set()
        for step in workflow.get("steps", []):
            all_tools.update(step.get("tools", []))
        
        dependency_results = self.dependency_manager.check_dependencies(workflow.get("steps", []))
        available_tools = dependency_results.get("available", [])
        missing_tools = dependency_results.get("missing", [])
        
        # Generate step reports
        step_reports = []
        for i, (step, result) in enumerate(zip(workflow.get("steps", []), results)):
            step_name = step.get("name", f"Step {i+1}")
            step_status = result.get("status", "unknown")
            step_execution_time = result.get("execution_time", 0)
            
            # Get tools for this step
            step_tools = step.get("tools", [])
            missing_step_tools = [tool for tool in step_tools if tool in missing_tools]
            
            step_report = {
                "name": step_name,
                "status": step_status,
                "execution_time": step_execution_time,
                "command": step.get("command", ""),
                "tools": step_tools,
                "missing_tools": missing_step_tools,
                "output": result.get("stdout", ""),
                "error": result.get("stderr", ""),
                "critical": step.get("critical", False)
            }
            
            step_reports.append(step_report)
        
        # Generate report
        report = {
            "workflow_name": workflow_name,
            "workflow_description": workflow_description,
            "execution_summary": {
                "total_steps": total_steps,
                "successful_steps": successful_steps,
                "failed_steps": failed_steps,
                "skipped_steps": skipped_steps,
                "total_execution_time": total_execution_time
            },
            "dependency_summary": {
                "total_tools": len(all_tools),
                "available_tools": available_tools,
                "missing_tools": missing_tools,
                "availability_percentage": len(available_tools) / len(all_tools) * 100 if all_tools else 100
            },
            "steps": step_reports,
            "output_directory": output_dir
        }
        
        # Save report to output directory
        report_file = os.path.join(output_dir, "report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        # Generate HTML report
        html_report = self._generate_html_report(report)
        html_report_file = os.path.join(output_dir, "report.html")
        with open(html_report_file, "w") as f:
            f.write(html_report)
        
        report["report_file"] = report_file
        report["html_report_file"] = html_report_file
        
        return report

    async def plan_and_execute_workflow(self, prompt, output_dir=None, checkpoint_dir=None):
        """Plan and execute a workflow based on a natural language prompt.
        
        Args:
            prompt: Natural language prompt describing the workflow
            output_dir: Directory to store workflow outputs
            checkpoint_dir: Directory to store workflow checkpoints
            
        Returns:
            dict: Workflow execution results
        """
        try:
            self.logger.info(f"Initial working directory: {os.getcwd()}")
            self.logger.info(f"Using {self.executor_type} executor")
            
            # Plan the workflow
            self.logger.info("Planning workflow steps...")
            workflow_plan = await self.llm.generate_workflow_plan(prompt)
            self.logger.debug("generate_workflow_plan method called successfully.")
            
            # Check if all dependencies are installed
            self.logger.info("Checking dependencies...")
            all_installed, available_but_failed_install = await self.dependency_manager.ensure_workflow_dependencies(workflow_plan)
            
            if not all_installed:
                if available_but_failed_install:
                    # Some dependencies failed to install but are available in the environment
                    missing_deps = [dep for dep in workflow_plan.get("dependencies", {}).get("tools", []) if dep not in available_but_failed_install]
                    if missing_deps:
                        self.logger.warning(f"Some dependencies could not be installed and are not available: {missing_deps}")
                        self.logger.warning("Continuing with workflow execution as some tools were found in the environment.")
                    else:
                        self.logger.info("All required tools are available in the environment despite installation failures.")
                else:
                    # No dependencies are available, cannot proceed
                    self.logger.error("Dependencies not installed. Cannot execute workflow.")
                    return {"status": "error", "message": "Dependencies not installed. Cannot execute workflow."}
            
            # Get output directory from workflow steps
            if output_dir is None:
                for step in workflow_plan.get("steps", []):
                    if "output_dir" in step:
                        output_dir = step["output_dir"]
                        break
            
            # Create workflow object
            workflow = Workflow(
                name=workflow_plan.get("name", "Unnamed Workflow"),
                description=workflow_plan.get("description", ""),
                steps=[],
                dependencies=workflow_plan.get("dependencies", {}).get("tools", []),
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir
            )
            
            # Create steps
            for step_data in workflow_plan.get("steps", []):
                # Get resource requirements for the step
                resource_requirements = self._get_resource_requirements(step_data.get("tool", ""))
                
                # Create step object
                step = WorkflowStep(
                    name=step_data.get("name", ""),
                    description=step_data.get("description", ""),
                    tool=step_data.get("tool", ""),
                    command=step_data.get("command", ""),
                    args=step_data.get("args", {}),
                    dependencies=step_data.get("dependencies", []),
                    output_files=step_data.get("output_files", []),
                    critical=step_data.get("critical", True),
                    resource_requirements=resource_requirements
                )
                
                workflow.steps.append(step)
            
            # If a checkpoint directory is provided, use smart resume functionality
            if checkpoint_dir:
                self.logger.info("Using smart resume functionality")
                # Convert WorkflowStep objects to dictionaries for smart resume
                step_dicts = []
                for step in workflow.steps:
                    step_dict = {
                        "name": step.name,
                        "command": step.command,
                        "dependencies": step.dependencies
                    }
                    step_dicts.append(step_dict)
                
                # Detect completed steps
                completed_steps = detect_completed_steps(step_dicts)
                if completed_steps:
                    self.logger.info(f"Detected {len(completed_steps)} completed steps: {', '.join(completed_steps)}")
                
                # Store completed steps in the workflow object for later use in DAG visualization
                workflow.completed_steps = completed_steps
                
                # Filter workflow steps
                filtered_steps = filter_workflow_steps(step_dicts, completed_steps)
                
                # Update workflow steps based on filtered steps
                if len(filtered_steps) < len(step_dicts):
                    self.logger.info(f"Filtered {len(step_dicts) - len(filtered_steps)} steps, will run {len(filtered_steps)} steps")
                    # Keep only the steps that are in the filtered list
                    workflow.steps = [step for step in workflow.steps if any(step.name == fs["name"] for fs in filtered_steps)]
            
            # Execute the workflow
            result = await self.execute_workflow(workflow)
            
            # Generate analysis report if workflow was successful
            if result.get("status") == "success":
                self.logger.info("Workflow completed successfully")
                
                # Generate a report if requested
                if workflow_plan.get("generate_report", False):
                    self.logger.info("Generating report...")
                    from ..analysis.report_generator import ReportGenerator
                    report_generator = ReportGenerator()
                    report_path = await report_generator.generate_report(workflow_plan, result)
                    result["analysis_report"] = report_path
            
            # Generate workflow DAG visualization
            workflow_steps_with_status = []
            for i, step in enumerate(workflow.steps):
                # Check if step is a dictionary or a WorkflowStep object
                if isinstance(step, dict):
                    # Create a WorkflowStep from dictionary
                    step_copy = WorkflowStep(
                        name=step.get("name", f"Step {i+1}"),
                        description=step.get("description", ""),
                        command=step.get("command", ""),
                        dependencies=step.get("dependencies", [])
                    )
                else:
                    # Create a copy of the WorkflowStep object
                    step_copy = WorkflowStep(
                        name=step.name,
                        description=step.description,
                        command=step.command,
                        dependencies=step.dependencies
                    )
                
                # Set status based on execution results
                result_step = next((r for r in result.get("results", []) if r.get("step_name") == step_copy.name), None)
                if result_step:
                    step_copy.status = result_step.get("status", "pending")
                else:
                    # For steps not found in results, check if they were skipped due to smart resume
                    # If the step name is in completed_steps (smart resume), mark it as completed
                    if hasattr(workflow, 'completed_steps') and step_copy.name in workflow.completed_steps:
                        step_copy.status = "completed"
                        self.logger.debug(f"Setting status of step {step_copy.name} to 'completed' based on smart resume")
                
                workflow_steps_with_status.append(step_copy)
            
            dag_image_path = self._save_workflow_dag(workflow_steps_with_status, output_dir)
            
            # Only add dag_visualization if it was successfully created
            if dag_image_path:
                result["dag_visualization"] = str(dag_image_path)
            
            # Display API usage statistics if available
            if hasattr(self.llm, "api_usage_tracker") and self.llm.api_usage_tracker:
                self.logger.info("Displaying API usage statistics...")
                # Force display API usage summary to console
                self.llm.api_usage_tracker.display_usage(workflow_id=f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}

    def _extract_geo_accession(self, prompt: str) -> Optional[str]:
        """Extract GEO accession number from prompt.
        
        Args:
            prompt: User prompt
            
        Returns:
            GEO accession number or None if not found
        """
        import re
        
        # Define regex pattern for GEO accession numbers
        geo_pattern = r'(?:GSE|GDS|GSM)\d+'
        
        # Search for GEO accession numbers in the prompt
        match = re.search(geo_pattern, prompt)
        
        if match:
            return match.group(0)
        
        return None
    
    def _add_geo_download_steps(self, geo_accession: str, workflow_steps: List[Dict[str, Any]], output_dir: str) -> List[Dict[str, Any]]:
        """Add GEO download steps to workflow.
        
        Args:
            geo_accession: GEO accession number
            workflow_steps: Original workflow steps
            output_dir: Output directory for the workflow
            
        Returns:
            Updated workflow steps with GEO download steps added
        """
        # Create data directory
        data_dir = os.path.join(output_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Define GEO download steps
        geo_steps = [
            {
                "name": f"Get SRR IDs for {geo_accession}",
                "description": f"Retrieve SRR IDs for GEO accession {geo_accession}",
                "command": f"esearch -db sra -query '{geo_accession}[Accession]' | efetch -format runinfo > {data_dir}/{geo_accession}_runinfo.csv",
                "tools": ["esearch", "efetch"],
                "critical": True
            },
            {
                "name": f"Extract SRR IDs from {geo_accession} runinfo",
                "description": "Extract SRR IDs from runinfo CSV file",
                "command": f"tail -n +2 {data_dir}/{geo_accession}_runinfo.csv | cut -d',' -f1 > {data_dir}/{geo_accession}_srr_ids.txt",
                "tools": ["tail", "cut"],
                "critical": True
            },
            {
                "name": f"Download SRA files for {geo_accession}",
                "description": "Download SRA files using prefetch",
                "command": f"cat {data_dir}/{geo_accession}_srr_ids.txt | xargs -I{{}} prefetch {{}} --output-directory {data_dir}",
                "tools": ["prefetch"],
                "critical": True
            },
            {
                "name": f"Convert SRA to FASTQ for {geo_accession}",
                "description": "Convert SRA files to FASTQ format",
                "command": f"cat {data_dir}/{geo_accession}_srr_ids.txt | xargs -I{{}} fasterq-dump {{}} -O {data_dir}",
                "tools": ["fasterq-dump"],
                "critical": True
            },
            {
                "name": f"Compress FASTQ files for {geo_accession}",
                "description": "Compress FASTQ files with gzip",
                "command": f"find {data_dir} -name '*.fastq' -exec gzip {{}} \\;",
                "tools": ["gzip", "find"],
                "critical": False
            }
        ]
        
        # Add GEO download steps to the beginning of the workflow
        return geo_steps + workflow_steps
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML report from workflow report."""
        # Generate HTML report
        html_report = "<html><body>"
        html_report += "<h1>Workflow Report</h1>"
        html_report += "<h2>Workflow Summary</h2>"
        html_report += f"<p>Workflow Name: {report['workflow_name']}</p>"
        html_report += f"<p>Workflow Description: {report['workflow_description']}</p>"
        html_report += "<h2>Execution Summary</h2>"
        html_report += f"<p>Total Steps: {report['execution_summary']['total_steps']}</p>"
        html_report += f"<p>Successful Steps: {report['execution_summary']['successful_steps']}</p>"
        html_report += f"<p>Failed Steps: {report['execution_summary']['failed_steps']}</p>"
        html_report += f"<p>Skipped Steps: {report['execution_summary']['skipped_steps']}</p>"
        html_report += f"<p>Total Execution Time: {report['execution_summary']['total_execution_time']} seconds</p>"
        html_report += "<h2>Dependency Summary</h2>"
        html_report += f"<p>Total Tools: {report['dependency_summary']['total_tools']}</p>"
        html_report += f"<p>Available Tools: {', '.join(report['dependency_summary']['available_tools'])}</p>"
        html_report += f"<p>Missing Tools: {', '.join(report['dependency_summary']['missing_tools'])}</p>"
        html_report += f"<p>Availability Percentage: {report['dependency_summary']['availability_percentage']}%</p>"
        html_report += "<h2>Step Reports</h2>"
        for step in report["steps"]:
            html_report += f"<h3>Step {step['name']}</h3>"
            html_report += f"<p>Status: {step['status']}</p>"
            html_report += f"<p>Execution Time: {step['execution_time']} seconds</p>"
            html_report += f"<p>Command: {step['command']}</p>"
            html_report += f"<p>Tools: {', '.join(step['tools'])}</p>"
            html_report += f"<p>Missing Tools: {', '.join(step['missing_tools'])}</p>"
            html_report += f"<p>Output: {step['output']}</p>"
            html_report += f"<p>Error: {step['error']}</p>"
            html_report += f"<p>Critical: {step['critical']}</p>"
        html_report += "</body></html>"
        
        return html_report

    def _get_resource_requirements(self, tool_name: str) -> Dict[str, Any]:
        """Get resource requirements for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Dictionary of resource requirements
        """
        # Default resource requirements
        default_requirements = {
            "cpu": 1,
            "memory": "1G",
            "time": "1:00:00"
        }
        
        # Tool-specific resource requirements
        tool_requirements = {
            # RNA-seq tools
            "star": {"cpu": 8, "memory": "32G", "time": "4:00:00"},
            "kallisto": {"cpu": 4, "memory": "8G", "time": "2:00:00"},
            "salmon": {"cpu": 4, "memory": "8G", "time": "2:00:00"},
            "stringtie": {"cpu": 4, "memory": "8G", "time": "2:00:00"},
            
            # Alignment tools
            "bwa": {"cpu": 8, "memory": "16G", "time": "4:00:00"},
            "bowtie": {"cpu": 4, "memory": "8G", "time": "2:00:00"},
            "bowtie2": {"cpu": 4, "memory": "8G", "time": "2:00:00"},
            "hisat2": {"cpu": 4, "memory": "8G", "time": "2:00:00"},
            
            # Variant calling tools
            "gatk": {"cpu": 4, "memory": "16G", "time": "4:00:00"},
            "samtools": {"cpu": 2, "memory": "4G", "time": "2:00:00"},
            "bcftools": {"cpu": 2, "memory": "4G", "time": "2:00:00"},
            
            # QC tools
            "fastqc": {"cpu": 1, "memory": "2G", "time": "1:00:00"},
            "multiqc": {"cpu": 1, "memory": "2G", "time": "0:30:00"},
            
            # SRA tools
            "prefetch": {"cpu": 1, "memory": "2G", "time": "4:00:00"},
            "fasterq-dump": {"cpu": 4, "memory": "8G", "time": "4:00:00"},
            "fastq-dump": {"cpu": 2, "memory": "4G", "time": "4:00:00"},
            
            # Single-cell tools
            "cellranger": {"cpu": 16, "memory": "64G", "time": "24:00:00"},
            "kb": {"cpu": 8, "memory": "32G", "time": "8:00:00"},
            "velocyto": {"cpu": 8, "memory": "32G", "time": "8:00:00"}
        }
        
        # Convert tool name to lowercase for case-insensitive matching
        tool_name_lower = tool_name.lower()
        
        # Return tool-specific requirements or default
        return tool_requirements.get(tool_name_lower, default_requirements)
