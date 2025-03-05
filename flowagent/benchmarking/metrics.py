"""Metrics collection for FlowAgent benchmarks.

This module provides functionality for extracting and analyzing metrics 
from FlowAgent workflow executions for benchmarking purposes.
"""

import os
import json
import time
import psutil
from typing import Dict, List, Optional, Any, Union
import logging
import re

from ..utils.logging import get_logger

logger = get_logger(__name__)


class BenchmarkMetrics:
    """Collects and analyzes metrics for FlowAgent benchmarks."""
    
    def __init__(self):
        """Initialize the metrics collector."""
        self.logger = get_logger(__name__)
    
    def extract_metrics(self, 
                        result: Dict[str, Any],
                        duration: float,
                        checkpoint_dir: Optional[str] = None) -> Dict[str, Any]:
        """Extract metrics from a workflow execution result.
        
        Args:
            result: Workflow execution result
            duration: Execution duration in seconds
            checkpoint_dir: Directory containing workflow checkpoints
            
        Returns:
            Dict with metrics
        """
        metrics = {
            "duration": duration,
            "status": result.get("status", "unknown"),
            "step_count": 0,
            "successful_steps": 0,
            "failed_steps": 0,
            "total_steps": 0,
            "completed_steps_detected": 0,
            "steps_skipped": 0,
            "completed_steps_detected_ratio": 0.0,
            "steps_skipped_ratio": 0.0,
            "memory_usage": self._get_memory_usage(),
            "cpu_usage": self._get_cpu_usage()
        }
        
        # Extract step metrics
        if "steps" in result:
            metrics["total_steps"] = len(result["steps"])
            metrics["successful_steps"] = sum(1 for step in result["steps"] if step.get("status") == "success")
            metrics["failed_steps"] = sum(1 for step in result["steps"] if step.get("status") == "error")
        
        # Extract smart resume metrics from log file if checkpoint_dir is provided
        if checkpoint_dir:
            smart_resume_metrics = self._extract_smart_resume_metrics(checkpoint_dir)
            metrics.update(smart_resume_metrics)
        
        # Calculate ratios
        if metrics["total_steps"] > 0:
            metrics["completed_steps_detected_ratio"] = metrics["completed_steps_detected"] / metrics["total_steps"]
            metrics["steps_skipped_ratio"] = metrics["steps_skipped"] / metrics["total_steps"]
        
        return metrics
    
    def _extract_smart_resume_metrics(self, checkpoint_dir: str) -> Dict[str, Any]:
        """Extract smart resume metrics from log files.
        
        Args:
            checkpoint_dir: Directory containing workflow checkpoints
            
        Returns:
            Dict with smart resume metrics
        """
        metrics = {
            "completed_steps_detected": 0,
            "steps_skipped": 0
        }
        
        # Look for log files in the checkpoint directory
        log_files = []
        if os.path.exists(checkpoint_dir):
            log_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.log')]
        
        # If no log files found in checkpoint dir, look in current directory
        if not log_files:
            log_files = [f for f in os.listdir('.') if f.endswith('.log')]
        
        # Process log files to extract smart resume metrics
        for log_file in log_files:
            log_path = os.path.join(checkpoint_dir, log_file) if os.path.exists(os.path.join(checkpoint_dir, log_file)) else log_file
            
            try:
                with open(log_path, 'r') as f:
                    log_content = f.read()
                    
                    # Extract completed steps detected
                    completed_steps_match = re.search(r'Detected (\d+) completed steps', log_content)
                    if completed_steps_match:
                        metrics["completed_steps_detected"] = int(completed_steps_match.group(1))
                    
                    # Extract steps skipped
                    skipped_steps_match = re.search(r'Filtered (\d+) steps', log_content)
                    if skipped_steps_match:
                        metrics["steps_skipped"] = int(skipped_steps_match.group(1))
            
            except Exception as e:
                self.logger.warning(f"Error extracting metrics from log file {log_path}: {str(e)}")
        
        return metrics
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage of the process.
        
        Returns:
            Memory usage in MB
        """
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert to MB
        except Exception as e:
            self.logger.warning(f"Error getting memory usage: {str(e)}")
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage of the process.
        
        Returns:
            CPU usage percentage
        """
        try:
            process = psutil.Process(os.getpid())
            return process.cpu_percent(interval=0.1)
        except Exception as e:
            self.logger.warning(f"Error getting CPU usage: {str(e)}")
            return 0.0
            
    def analyze_command_quality(self, commands: List[str]) -> Dict[str, Any]:
        """Analyze the quality of generated commands.
        
        Args:
            commands: List of commands to analyze
            
        Returns:
            Dict with command quality metrics
        """
        metrics = {
            "command_count": len(commands),
            "avg_command_length": 0,
            "tools_used": {},
            "flags_used": {},
            "error_indicators": 0
        }
        
        total_length = 0
        error_indicators = ["error", "failed", "not found", "cannot", "invalid"]
        
        for command in commands:
            # Calculate length
            total_length += len(command)
            
            # Count tools used
            tools = ["fastqc", "kallisto", "multiqc", "bowtie", "bwa", "samtools", "bcftools", "featureCounts", "htseq", "salmon"]
            for tool in tools:
                if tool in command:
                    metrics["tools_used"][tool] = metrics["tools_used"].get(tool, 0) + 1
            
            # Count flags used
            flags = re.findall(r'-[-\w]+', command)
            for flag in flags:
                metrics["flags_used"][flag] = metrics["flags_used"].get(flag, 0) + 1
            
            # Check for error indicators
            if any(indicator in command.lower() for indicator in error_indicators):
                metrics["error_indicators"] += 1
        
        # Calculate average command length
        if commands:
            metrics["avg_command_length"] = total_length / len(commands)
        
        return metrics
    
    def compare_models(self, 
                       model_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Compare metrics across different models.
        
        Args:
            model_results: Dictionary mapping model names to lists of results
            
        Returns:
            Dict with comparison metrics
        """
        comparison = {
            "models": list(model_results.keys()),
            "metrics": {},
            "rankings": {}
        }
        
        # Define metrics to compare (higher is better unless specified)
        metrics_to_compare = {
            "success_rate": "higher",
            "duration": "lower",
            "completed_steps_detected_ratio": "higher",
            "steps_skipped_ratio": "higher",
            "memory_usage": "lower",
            "cpu_usage": "lower",
            "error_indicators": "lower"
        }
        
        # Calculate aggregate metrics for each model
        for model, results in model_results.items():
            model_metrics = {}
            
            # Calculate success rate
            successful_runs = sum(1 for r in results if r.get("status") == "success")
            model_metrics["success_rate"] = successful_runs / len(results) if results else 0
            
            # Calculate average values for other metrics
            for metric in ["duration", "completed_steps_detected_ratio", "steps_skipped_ratio", 
                          "memory_usage", "cpu_usage", "error_indicators"]:
                values = [r.get("metrics", {}).get(metric, 0) for r in results 
                         if "metrics" in r and metric in r.get("metrics", {})]
                model_metrics[metric] = sum(values) / len(values) if values else 0
            
            comparison["metrics"][model] = model_metrics
        
        # Rank models for each metric
        for metric, direction in metrics_to_compare.items():
            if all(metric in comparison["metrics"][model] for model in comparison["models"]):
                if direction == "higher":
                    # Higher is better
                    ranked_models = sorted(comparison["models"], 
                                         key=lambda m: comparison["metrics"][m].get(metric, 0), 
                                         reverse=True)
                else:
                    # Lower is better
                    ranked_models = sorted(comparison["models"], 
                                         key=lambda m: comparison["metrics"][m].get(metric, float('inf')))
                
                comparison["rankings"][metric] = {
                    "best": ranked_models[0],
                    "ranking": ranked_models
                }
        
        return comparison
