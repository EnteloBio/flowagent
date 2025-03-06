"""API usage tracking for FlowAgent."""

import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set

from ..utils.logging import get_logger


class APIUsageTracker:
    """Tracks API usage metrics including calls and tokens per workflow."""

    def __init__(self):
        """Initialize the API usage tracker."""
        self.logger = get_logger(__name__)
        self.workflows = {}
        self.current_workflow_id = None
        
    def start_workflow(self, workflow_id: str, workflow_name: str) -> None:
        """Start tracking a new workflow.
        
        Args:
            workflow_id: Unique identifier for the workflow
            workflow_name: Name of the workflow
        """
        self.current_workflow_id = workflow_id
        self.workflows[workflow_id] = {
            "workflow_id": workflow_id,
            "workflow_name": workflow_name,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "api_calls": [],
            "total_calls": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "models_used": set(),
            "duration_seconds": 0,
            "steps_tracked": set(),
            "steps_token_usage": {},
            "is_analysis": workflow_id.startswith("analysis_"),
            "parent_workflow_id": None  # Track parent workflow for analysis contexts
        }
        self.logger.info(f"Started tracking API usage for workflow: {workflow_name} ({workflow_id})")
        
    def end_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """End tracking for a workflow and return usage statistics.
        
        Args:
            workflow_id: Unique identifier for the workflow
            
        Returns:
            Dict containing API usage statistics
        """
        if workflow_id not in self.workflows:
            self.logger.warning(f"Cannot end tracking for unknown workflow: {workflow_id}")
            return {}
            
        workflow = self.workflows[workflow_id]
        end_time = datetime.now()
        workflow["end_time"] = end_time.isoformat()
        
        # Calculate duration
        if workflow["start_time"]:
            start_time = datetime.fromisoformat(workflow["start_time"])
            duration = end_time - start_time
            workflow["duration_seconds"] = duration.total_seconds()
        
        # If this is an analysis workflow, merge its stats with the parent workflow
        if workflow["is_analysis"] and workflow["parent_workflow_id"]:
            parent_id = workflow["parent_workflow_id"]
            if parent_id in self.workflows:
                parent = self.workflows[parent_id]
                # Merge API calls
                parent["api_calls"].extend(workflow["api_calls"])
                # Update totals
                parent["total_calls"] += workflow["total_calls"]
                parent["total_prompt_tokens"] += workflow["total_prompt_tokens"]
                parent["total_completion_tokens"] += workflow["total_completion_tokens"]
                parent["total_tokens"] += workflow["total_tokens"]
                # Merge models used
                parent["models_used"].update(workflow["models_used"])
                # Merge steps tracked
                parent["steps_tracked"].update(workflow["steps_tracked"])
                # Merge step token usage
                for step, usage in workflow["steps_token_usage"].items():
                    if step in parent["steps_token_usage"]:
                        parent_usage = parent["steps_token_usage"][step]
                        parent_usage["total_calls"] += usage["total_calls"]
                        parent_usage["total_tokens"] += usage["total_tokens"]
                        parent_usage["prompt_tokens"] += usage["prompt_tokens"]
                        parent_usage["completion_tokens"] += usage["completion_tokens"]
                        parent_usage["models"].update(usage["models"])
                    else:
                        parent["steps_token_usage"][step] = usage.copy()
                
                self.logger.info(f"Merged analysis workflow {workflow_id} stats into parent workflow {parent_id}")
        
        # Convert all sets to lists for JSON serialization
        workflow["models_used"] = list(workflow["models_used"])
        workflow["steps_tracked"] = list(workflow["steps_tracked"])
        
        # Convert step model sets to lists
        for step_name, step_data in workflow["steps_token_usage"].items():
            if isinstance(step_data["models"], set):
                step_data["models"] = list(step_data["models"])
        
        self.logger.info(f"Completed tracking API usage for workflow: {workflow['workflow_name']} ({workflow_id})")
        self.logger.info(f"API usage summary: {workflow['total_calls']} calls, {workflow['total_tokens']} tokens")
        
        if self.current_workflow_id == workflow_id:
            self.current_workflow_id = None
            
        return workflow
        
    def get_workflow_usage(self, workflow_id: str) -> Dict[str, Any]:
        """Get usage statistics for a specific workflow.
        
        Args:
            workflow_id: Unique identifier for the workflow
            
        Returns:
            Dict containing API usage statistics
        """
        if workflow_id not in self.workflows:
            self.logger.warning(f"Cannot get usage for unknown workflow: {workflow_id}")
            return {}
            
        workflow = self.workflows[workflow_id].copy()
        
        # Convert any sets to lists for JSON serialization
        workflow = self._prepare_data_for_serialization(workflow)
            
        return workflow
        
    def record_api_call(self, model: str, prompt_tokens: int, completion_tokens: int, 
                       workflow_id: Optional[str] = None, step_name: Optional[str] = None, 
                       purpose: Optional[str] = None) -> None:
        """Record an API call with token usage.
        
        Args:
            model: The model name used for the call
            prompt_tokens: Number of prompt tokens used
            completion_tokens: Number of completion tokens used
            workflow_id: Optional workflow ID (defaults to current workflow)
            step_name: Optional step name for context
            purpose: Optional description of the call's purpose
        """
        wf_id = workflow_id or self.current_workflow_id
        
        if not wf_id:
            # If there's no active workflow, create a special "analysis" workflow context
            # This allows API calls during report generation and analysis to be tracked
            analysis_workflow_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.logger.info(f"No active workflow found, creating analysis context: {analysis_workflow_id}")
            
            # Link this analysis workflow to the most recent non-analysis workflow
            parent_workflow_id = None
            for wf_id, wf in self.workflows.items():
                if not wf["is_analysis"]:
                    parent_workflow_id = wf_id
                    break
            
            self.start_workflow(analysis_workflow_id, "Report and Analysis Generation")
            if parent_workflow_id:
                self.workflows[analysis_workflow_id]["parent_workflow_id"] = parent_workflow_id
                self.logger.info(f"Linked analysis workflow {analysis_workflow_id} to parent workflow {parent_workflow_id}")
            
            wf_id = analysis_workflow_id
            
        if wf_id not in self.workflows:
            self.logger.warning(f"Cannot record API call: Unknown workflow {wf_id}")
            return
            
        call_info = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "step_name": step_name,
            "purpose": purpose
        }
        
        # Update workflow statistics
        workflow = self.workflows[wf_id]
        workflow["api_calls"].append(call_info)
        workflow["total_calls"] += 1
        workflow["total_prompt_tokens"] += prompt_tokens
        workflow["total_completion_tokens"] += completion_tokens
        workflow["total_tokens"] += (prompt_tokens + completion_tokens)
        workflow["models_used"].add(model)
        
        # Track per-step token usage
        if step_name:
            # Add step to tracking set
            workflow["steps_tracked"].add(step_name)
            
            # Initialize step tracking if not present
            if step_name not in workflow["steps_token_usage"]:
                workflow["steps_token_usage"][step_name] = {
                    "total_calls": 0,
                    "total_tokens": 0,
                    "prompt_tokens": 0, 
                    "completion_tokens": 0,
                    "models": set()
                }
                
            # Update step statistics
            step_stats = workflow["steps_token_usage"][step_name]
            step_stats["total_calls"] += 1
            step_stats["total_tokens"] += (prompt_tokens + completion_tokens)
            step_stats["prompt_tokens"] += prompt_tokens
            step_stats["completion_tokens"] += completion_tokens
            step_stats["models"].add(model)
        
        self.logger.debug(f"Recorded API call for {wf_id}: {prompt_tokens}+{completion_tokens}={prompt_tokens + completion_tokens} tokens")
        
    def get_current_workflow_usage(self) -> Dict[str, Any]:
        """Get usage statistics for the current workflow.
        
        Returns:
            Dict containing API usage statistics or empty dict if no active workflow
        """
        if not self.current_workflow_id:
            return {}
        return self.get_workflow_usage(self.current_workflow_id)
        
    def start_analysis_tracking(self, analysis_name: str, workflow_output_dir: str = None, results_dir: str = None) -> str:
        """Start tracking a new analysis workflow.
        
        Args:
            analysis_name: Name of the analysis workflow
            workflow_output_dir: Directory containing workflow outputs to analyze (preferred)
            results_dir: Directory containing results to analyze (legacy parameter)
            
        Returns:
            ID of the created analysis workflow
        """
        # Handle both parameter names for backward compatibility
        results_directory = workflow_output_dir if workflow_output_dir is not None else results_dir
        
        analysis_workflow_id = f"analysis_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Link this analysis workflow to the most recent non-analysis workflow
        # Sort workflows by start_time to find the most recent one
        parent_workflow_id = None
        non_analysis_workflows = [(wf_id, wf) for wf_id, wf in self.workflows.items() if not wf["is_analysis"]]
        
        if non_analysis_workflows:
            # Sort by start_time (most recent first)
            sorted_workflows = sorted(
                non_analysis_workflows,
                key=lambda x: x[1]["start_time"] if isinstance(x[1]["start_time"], datetime) else datetime.fromisoformat(x[1]["start_time"]),
                reverse=True
            )
            parent_workflow_id = sorted_workflows[0][0]
        
        self.start_workflow(analysis_workflow_id, analysis_name)
        
        # Set analysis workflow properties
        workflow = self.workflows[analysis_workflow_id]
        workflow["is_analysis"] = True
        workflow["results_dir"] = results_directory
        
        if parent_workflow_id:
            workflow["parent_workflow_id"] = parent_workflow_id
            self.logger.info(f"Linked analysis workflow {analysis_workflow_id} to parent workflow {parent_workflow_id}")
        
        return analysis_workflow_id
    
    def save_usage_report(self, workflow_id: str, output_dir: str) -> Optional[str]:
        """Save usage report to file.
        
        Args:
            workflow_id: Unique identifier for the workflow
            output_dir: Directory to save the report
            
        Returns:
            Path to the saved report file or None if failed
        """
        if workflow_id not in self.workflows:
            self.logger.warning(f"Cannot save report for unknown workflow: {workflow_id}")
            return None
            
        try:
            os.makedirs(output_dir, exist_ok=True)
            report_path = os.path.join(output_dir, f"api_usage_{workflow_id}.json")
            
            # Deep copy the workflow data and ensure it's JSON serializable
            workflow_data = self.workflows[workflow_id].copy()
            
            # Calculate cost estimate and add to report data
            cost_estimate = self.calculate_cost_estimate(workflow_id)
            workflow_data["cost_estimate"] = cost_estimate
            
            # Ensure all data is JSON serializable
            workflow_data = self._prepare_data_for_serialization(workflow_data)
            
            # Add report metadata
            workflow_data["report_generated"] = datetime.now().isoformat()
            workflow_data["report_version"] = "1.1.0"  # Version for tracking report format changes
            
            with open(report_path, 'w') as f:
                json.dump(workflow_data, f, indent=2)
                
            self.logger.info(f"Saved API usage report to: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Failed to save API usage report: {str(e)}")
            return None
            
    def calculate_cost_estimate(self, workflow_id: str) -> Dict[str, Any]:
        """Calculate approximate cost estimate for a workflow.
        
        Note: This provides rough estimates based on public pricing and
        should not be used for billing purposes.
        
        Args:
            workflow_id: Unique identifier for the workflow
            
        Returns:
            Dict containing cost estimates
        """
        # If workflow_id is actually a workflow dict (directly passed from another method)
        if isinstance(workflow_id, dict):
            workflow = workflow_id
        elif workflow_id not in self.workflows:
            return {"error": "Workflow not found"}
        else:
            workflow = self.workflows[workflow_id]
        
        # Simplified pricing model (as of March 2025)
        # These should be updated regularly or moved to a configuration file
        model_pricing = {
            "gpt-4o": {"input": 0.01, "output": 0.03},  # per 1K tokens
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            # Add other models as needed
        }
        
        total_cost = 0.0
        costs_by_model = {}
        
        for call in workflow["api_calls"]:
            model = call["model"]
            prompt_tokens = call["prompt_tokens"]
            completion_tokens = call["completion_tokens"]
            
            # Use default pricing if model not found
            pricing = model_pricing.get(model, {"input": 0.01, "output": 0.03})
            
            input_cost = (prompt_tokens / 1000) * pricing["input"]
            output_cost = (completion_tokens / 1000) * pricing["output"]
            call_cost = input_cost + output_cost
            
            if model not in costs_by_model:
                costs_by_model[model] = 0
            
            costs_by_model[model] += call_cost
            total_cost += call_cost
        
        return {
            "total_cost_usd": round(total_cost, 4),
            "costs_by_model": {model: round(cost, 4) for model, cost in costs_by_model.items()},
            "disclaimer": "This is a rough estimate and should not be used for billing purposes."
        }
        
    def display_usage(self, workflow_id: Optional[str] = None) -> None:
        """Display usage statistics for a workflow.
        
        Args:
            workflow_id: Optional unique identifier for the workflow.
                         If None and only one workflow exists, that workflow will be used.
        """
        # If no workflow_id provided but we have workflows, use the most recent one
        if workflow_id is None:
            if not self.workflows:
                self.logger.warning("No workflows to display usage for")
                return
            workflow_id = list(self.workflows.keys())[-1]
            
        # Handle case where a workflow dict is passed directly
        if isinstance(workflow_id, dict):
            usage = workflow_id
        elif workflow_id not in self.workflows:
            self.logger.warning(f"Cannot display usage for unknown workflow: {workflow_id}")
            return
        else:
            usage = self.workflows[workflow_id]
        
        # Calculate cost estimate
        cost_info = self.calculate_cost_estimate(usage)
        
        # Display summary
        print("\n==== API Usage Summary ====")
        print(f"Workflow: {usage.get('workflow_name', 'Unknown')} ({workflow_id if not isinstance(workflow_id, dict) else 'current'})")
        print(f"Duration: {usage.get('duration_seconds', 0):.2f} seconds")
        print(f"Total API Calls: {usage.get('total_calls', 0)}")
        print(f"Total Tokens: {usage.get('total_tokens', 0)} (Prompt: {usage.get('total_prompt_tokens', 0)}, Completion: {usage.get('total_completion_tokens', 0)})")
        print(f"Models Used: {', '.join(usage.get('models_used', []))}")
        if cost_info and 'total_cost_usd' in cost_info:
            print(f"Estimated Cost: ${cost_info['total_cost_usd']:.6f} USD")
        print("==========================\n")
        
    def track_api_usage(self, purpose: str, step_name: Optional[str] = None) -> None:
        """Track API usage for non-API operations like analysis phases.
        
        This allows tracking API usage equivalents for operations that don't directly
        make API calls but should be tracked as part of the workflow.
        
        Args:
            purpose: Description of the operation (e.g., "quality_analysis")
            step_name: Optional step name for context
        """
        if not self.current_workflow_id:
            self.logger.warning("Cannot track API usage: No active workflow")
            return
            
        # Add a record to track that this operation occurred
        # We don't increment token counts since this isn't an actual API call
        workflow = self.workflows[self.current_workflow_id]
        
        # Add the operation to the steps tracked
        if step_name:
            workflow["steps_tracked"].add(step_name)
            
        # Log the operation
        self.logger.debug(f"Tracked non-API operation: {purpose}")
        
    def add_tokens(
        self, model: str, prompt_tokens: int, completion_tokens: int,
        step_name: Optional[str] = None, purpose: Optional[str] = None
    ) -> None:
        """Add token usage to the current workflow.
        
        A more streamlined version of record_api_call that assumes current workflow context.
        
        Args:
            model: The model name used for the call
            prompt_tokens: Number of prompt tokens used
            completion_tokens: Number of completion tokens used
            step_name: Optional step name for context
            purpose: Optional description of the call's purpose
        """
        # Simply delegate to record_api_call with current workflow ID
        self.record_api_call(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            workflow_id=self.current_workflow_id,
            step_name=step_name,
            purpose=purpose
        )
        
    def _prepare_data_for_serialization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data structure for JSON serialization by converting non-serializable types.
        
        Args:
            data: Any data structure to prepare for serialization
            
        Returns:
            Data structure with all non-serializable types converted to serializable ones
        """
        if isinstance(data, dict):
            result = {}
            for k, v in data.items():
                result[k] = self._prepare_data_for_serialization(v)
            return result
        elif isinstance(data, list):
            return [self._prepare_data_for_serialization(item) for item in data]
        elif isinstance(data, set):
            return [self._prepare_data_for_serialization(item) for item in data]
        elif isinstance(data, (str, int, float, bool, type(None))):
            return data
        elif hasattr(data, 'isoformat') and callable(getattr(data, 'isoformat')):
            # Handle datetime objects
            return data.isoformat()
        else:
            # Convert other objects to strings for safe serialization
            try:
                return str(data)
            except Exception:
                return f"<Unserializable object of type {type(data).__name__}>"
