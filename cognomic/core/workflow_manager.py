"""Workflow manager for coordinating LLM-based workflow execution."""

from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import asyncio
import networkx as nx
import json
import os

from .agent_system import PLAN_agent, TASK_agent, DEBUG_agent, WorkflowStep
from .llm import LLMInterface
from .tool_tracker import ToolTracker

logger = logging.getLogger(__name__)

class WorkflowManager:
    """Manages workflow execution and coordination."""
    
    def __init__(self, llm: Optional[LLMInterface] = None, logger=None):
        """Initialize workflow manager."""
        self.llm = llm or LLMInterface()
        self.logger = logger or logging.getLogger(__name__)
        self.cwd = os.getcwd()
        self.logger.info(f"Initial working directory: {self.cwd}")
        
        # Initialize agents
        self.plan_agent = PLAN_agent(llm=self.llm)
        self.task_agent = TASK_agent(llm=self.llm)
        self.debug_agent = DEBUG_agent(llm=self.llm)

    async def execute_workflow(self, prompt: str) -> Dict[str, Any]:
        """Execute workflow from prompt."""
        try:
            self.logger.info("Planning workflow steps...")
            workflow_steps = await self.plan_agent.decompose_workflow(prompt)
            
            # Create output directories
            for step in workflow_steps:
                output_dir = step.parameters.get("output_dir")
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    self.logger.info(f"Created output directory: {output_dir}")
            
            # Set working directory for task agent
            self.task_agent.cwd = self.cwd
            self.logger.info(f"Set task agent working directory to: {self.cwd}")
            
            # Build DAG
            dag = self._build_dag(workflow_steps)
            
            # Execute workflow in DAG order
            execution_plan = self.get_execution_plan(dag)
            results = []
            for batch in execution_plan:
                batch_results = []
                for task_id in batch:
                    node_data = dag.nodes[task_id]
                    try:
                        self.logger.info(f"Executing {task_id}")
                        result = await self.task_agent.execute_step(node_data['step'])
                        batch_results.append({
                            "step": node_data['step'].name,
                            "status": "success",
                            "result": result
                        })
                    except Exception as e:
                        self.logger.error(f"Task {task_id} failed: {str(e)}")
                        batch_results.append({
                            "step": node_data['step'].name,
                            "status": "failed",
                            "error": str(e)
                        })
                results.extend(batch_results)
            
            # Archive results
            self.logger.info("Archiving workflow results...")
            archive_path = os.path.join("results", "workflow_archive.json")
            os.makedirs(os.path.dirname(archive_path), exist_ok=True)
            with open(archive_path, "w") as f:
                json.dump(results, f, indent=2)
            
            # Generate analysis report
            self.logger.info("Generating analysis report...")
            report = await self._generate_analysis_report(results)
            
            return {
                "status": "success",
                "results": results,
                "report": report,
                "archive_path": archive_path
            }
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            try:
                diagnosis = await self.debug_agent.diagnose_error(e, {
                    "error": str(e),
                    "context": {
                        "working_directory": self.cwd,
                        "prompt": prompt
                    }
                })
                self.logger.info(f"Error diagnosis: {diagnosis}")
            except Exception as diag_error:
                self.logger.error(f"Error diagnosis failed: {str(diag_error)}")
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
