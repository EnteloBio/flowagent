import asyncio
from pathlib import Path
from typing import Dict, Any, List
import logging
import os
import json
from .agent_system import PLAN_agent, TASK_agent, DEBUG_agent, WorkflowStep

logger = logging.getLogger(__name__)

class WorkflowManager:
    """Manager for workflow execution and coordination."""
    
    def __init__(self, llm=None, logger=None):
        """Initialize workflow manager."""
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)
        self.cwd = os.getcwd()
        self.logger.info(f"Initial working directory: {self.cwd}")
        
        # Initialize agents
        self.plan_agent = PLAN_agent(llm=llm)
        self.task_agent = TASK_agent(llm=llm)
        self.debug_agent = DEBUG_agent(llm=llm)

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
            
            # Execute each step
            results = []
            for step in workflow_steps:
                self.logger.info(f"Executing step: {step.name}")
                try:
                    result = await self.task_agent.execute_step(step)
                    results.append({
                        "step": step.name,
                        "status": "success",
                        "result": result
                    })
                except Exception as e:
                    self.logger.error(f"Step execution failed: {str(e)}")
                    # Try to diagnose the error
                    try:
                        diagnosis = await self.debug_agent.diagnose_error(e, {
                            "step": {
                                "name": step.name,
                                "tool": step.tool,
                                "action": step.action,
                                "type": step.type,
                                "parameters": step.parameters
                            },
                            "error": str(e),
                            "context": {
                                "working_directory": self.cwd,
                                "step_parameters": step.parameters
                            }
                        })
                        results.append({
                            "step": step.name,
                            "status": "failed",
                            "error": str(e),
                            "diagnosis": diagnosis
                        })
                    except Exception as diag_error:
                        self.logger.error(f"Error diagnosis failed: {str(diag_error)}")
                        results.append({
                            "step": step.name,
                            "status": "failed",
                            "error": str(e),
                            "diagnosis": None
                        })
            
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
