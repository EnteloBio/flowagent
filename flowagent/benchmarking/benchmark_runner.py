"""Benchmark runner for FlowAgent.

This module provides a way to run benchmarks comparing different LLM models
when executing the same workflows with FlowAgent.
"""

import asyncio
import os
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ..core.workflow_manager import WorkflowManager
from ..config.settings import Settings
from ..utils.logging import get_logger
from .metrics import BenchmarkMetrics
from .visualization import BenchmarkVisualizer

logger = get_logger(__name__)

class BenchmarkRunner:
    """Runner for FlowAgent benchmarks across different LLM models."""
    
    def __init__(self, benchmark_dir: Optional[str] = None):
        """Initialize the benchmark runner.
        
        Args:
            benchmark_dir: Directory to store benchmark results
        """
        self.logger = get_logger(__name__)
        self.settings = Settings()
        
        # Set up benchmark directory
        if benchmark_dir:
            self.benchmark_dir = Path(benchmark_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.benchmark_dir = Path(f"benchmarks/run_{timestamp}")
        
        # Create benchmark directory if it doesn't exist
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Benchmark results will be stored in: {self.benchmark_dir}")
        
        # Initialize benchmark metrics and visualizer
        self.metrics = BenchmarkMetrics()
        self.visualizer = BenchmarkVisualizer(output_dir=str(self.benchmark_dir))
        
    async def run_benchmark(self, 
                      prompt: str, 
                      models: List[str], 
                      repetitions: int = 3,
                      output_base_dir: Optional[str] = None,
                      description: Optional[str] = None) -> Dict[str, Any]:
        """Run a benchmark across multiple models.
        
        Args:
            prompt: The workflow prompt to execute
            models: List of models to benchmark (e.g., ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"])
            repetitions: Number of times to repeat each workflow execution
            output_base_dir: Base directory for workflow outputs
            description: Description of the benchmark
            
        Returns:
            Dict with benchmark results
        """
        benchmark_id = str(uuid.uuid4())
        benchmark_start_time = time.time()
        
        # Create benchmark metadata
        benchmark_metadata = {
            "id": benchmark_id,
            "prompt": prompt,
            "models": models,
            "repetitions": repetitions,
            "description": description or "FlowAgent model benchmark",
            "start_time": datetime.now().isoformat(),
            "flowagent_version": self.settings.APP_VERSION,
        }
        
        # Save benchmark metadata
        metadata_file = self.benchmark_dir / f"metadata_{benchmark_id}.json"
        with open(metadata_file, "w") as f:
            json.dump(benchmark_metadata, f, indent=2)
        
        # Initialize results storage
        results = {}
        
        # Run benchmark for each model
        for model in models:
            self.logger.info(f"Running benchmark with model: {model}")
            model_results = []
            
            for rep in range(1, repetitions + 1):
                self.logger.info(f"  Repetition {rep}/{repetitions}")
                
                # Create model-specific output directory
                if output_base_dir:
                    output_dir = os.path.join(output_base_dir, f"{model.replace('-', '_')}_{rep}")
                else:
                    output_dir = os.path.join(self.benchmark_dir, f"output_{model.replace('-', '_')}_{rep}")
                
                # Create checkpoint directory for smart resume metrics
                checkpoint_dir = os.path.join(self.benchmark_dir, f"checkpoint_{model.replace('-', '_')}_{rep}")
                
                # Override the OPENAI_MODEL setting temporarily
                original_model = self.settings.OPENAI_MODEL
                self.settings.OPENAI_MODEL = model
                
                # Create a new workflow manager instance with the specified model
                workflow_manager = WorkflowManager(executor_type=self.settings.EXECUTOR_TYPE)
                
                # Capture start time
                start_time = time.time()
                
                try:
                    # Execute the workflow
                    result = await workflow_manager.plan_and_execute_workflow(
                        prompt=prompt,
                        output_dir=output_dir,
                        checkpoint_dir=checkpoint_dir
                    )
                    
                    # Capture end time and calculate duration
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    # Extract and store metrics
                    metrics = self.metrics.extract_metrics(
                        result=result,
                        duration=duration,
                        checkpoint_dir=checkpoint_dir
                    )
                    
                    model_results.append({
                        "repetition": rep,
                        "status": result.get("status", "unknown"),
                        "duration": duration,
                        "metrics": metrics,
                        "result": result
                    })
                    
                    self.logger.info(f"  Completed in {duration:.2f} seconds with status: {result.get('status', 'unknown')}")
                
                except Exception as e:
                    self.logger.error(f"Error running benchmark with model {model}: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    
                    model_results.append({
                        "repetition": rep,
                        "status": "error",
                        "error": str(e),
                        "duration": time.time() - start_time
                    })
                
                finally:
                    # Restore original model setting
                    self.settings.OPENAI_MODEL = original_model
            
            # Store results for this model
            results[model] = model_results
        
        # Calculate overall benchmark duration
        benchmark_duration = time.time() - benchmark_start_time
        
        # Update and save benchmark metadata with completion info
        benchmark_metadata.update({
            "end_time": datetime.now().isoformat(),
            "duration": benchmark_duration,
            "status": "completed"
        })
        
        with open(metadata_file, "w") as f:
            json.dump(benchmark_metadata, f, indent=2)
        
        # Save detailed results
        results_file = self.benchmark_dir / f"results_{benchmark_id}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Generate summary report
        summary = self.generate_summary(results, benchmark_metadata)
        summary_file = self.benchmark_dir / f"summary_{benchmark_id}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Generate visualization
        visualizations = self.generate_visualizations(results, benchmark_id)
        
        self.logger.info(f"Benchmark completed in {benchmark_duration:.2f} seconds")
        self.logger.info(f"Results saved to {self.benchmark_dir}")
        
        return {
            "benchmark_id": benchmark_id,
            "results": results,
            "summary": summary,
            "benchmark_dir": str(self.benchmark_dir)
        }
    
    def generate_summary(self, results: Dict[str, List[Dict]], metadata: Dict) -> Dict[str, Any]:
        """Generate a summary of benchmark results.
        
        Args:
            results: Benchmark results by model
            metadata: Benchmark metadata
            
        Returns:
            Dict with summary statistics
        """
        summary = {
            "benchmark_id": metadata["id"],
            "prompt": metadata["prompt"],
            "models": metadata["models"],
            "repetitions": metadata["repetitions"],
            "model_summaries": {},
            "comparison": {}
        }
        
        model_metrics = {}
        
        # Process results for each model
        for model, model_results in results.items():
            # Initialize metrics for this model
            model_metrics[model] = {
                "success_rate": 0,
                "avg_duration": 0,
                "completed_steps_detected_ratio": 0,
                "steps_skipped_ratio": 0,
                "memory_usage": 0,
                "cpu_usage": 0,
                "total_steps": 0
            }
            
            # Calculate success rate
            successful_runs = sum(1 for r in model_results if r.get("status") == "success")
            model_metrics[model]["success_rate"] = successful_runs / len(model_results) if model_results else 0
            
            # Calculate average duration for successful runs
            successful_durations = [r.get("duration", 0) for r in model_results if r.get("status") == "success"]
            model_metrics[model]["avg_duration"] = sum(successful_durations) / len(successful_durations) if successful_durations else 0
            
            # Calculate average metrics for successful runs
            successful_metrics = [r.get("metrics", {}) for r in model_results if r.get("status") == "success" and "metrics" in r]
            
            if successful_metrics:
                for metric_name in ["completed_steps_detected_ratio", "steps_skipped_ratio", "memory_usage", "cpu_usage", "total_steps"]:
                    values = [m.get(metric_name, 0) for m in successful_metrics if metric_name in m]
                    model_metrics[model][metric_name] = sum(values) / len(values) if values else 0
        
        # Add model summaries to the overall summary
        summary["model_summaries"] = model_metrics
        
        # Compare models
        if len(model_metrics) > 1:
            # Find the best model for each metric
            best_models = {}
            
            for metric in ["success_rate", "avg_duration", "completed_steps_detected_ratio", "steps_skipped_ratio"]:
                if metric == "avg_duration" or metric == "cpu_usage" or metric == "memory_usage":
                    # Lower is better for these metrics
                    best_model = min(model_metrics.items(), key=lambda x: x[1].get(metric, float('inf')) or float('inf'))
                else:
                    # Higher is better for these metrics
                    best_model = max(model_metrics.items(), key=lambda x: x[1].get(metric, 0) or 0)
                
                best_models[f"best_{metric}"] = {
                    "model": best_model[0],
                    "value": best_model[1].get(metric, 0)
                }
            
            summary["comparison"] = best_models
        
        return summary
    
    def generate_visualizations(self, results: Dict[str, List[Dict]], benchmark_id: str) -> Dict[str, str]:
        """Generate visualizations of benchmark results.
        
        Args:
            results: Benchmark results by model
            benchmark_id: Unique identifier for this benchmark run
            
        Returns:
            Dict mapping visualization names to file paths
        """
        self.logger.info("Generating benchmark visualizations...")
        
        # Extract model comparison data
        model_comparison = {}
        if len(results) > 1:
            # Calculate comparison metrics
            model_comparison = self.metrics.compare_models(results)
        
        # Use the visualizer to generate all visualizations
        visualizations = self.visualizer.visualize_all(results, model_comparison)
        
        # Save paths to visualization files in the benchmark metadata
        viz_metadata_file = self.benchmark_dir / f"visualizations_{benchmark_id}.json"
        with open(viz_metadata_file, "w") as f:
            json.dump(visualizations, f, indent=2)
        
        self.logger.info(f"Visualizations saved to {self.benchmark_dir}")
        
        # Create a DataFrame from the results for additional analysis
        data = []
        for model, model_results in results.items():
            for result in model_results:
                row = {
                    "model": model,
                    "repetition": result.get("repetition", 0),
                    "status": result.get("status", "unknown"),
                    "duration": result.get("duration", 0)
                }
                
                # Add metrics if available
                if "metrics" in result:
                    for metric_name, metric_value in result["metrics"].items():
                        row[metric_name] = metric_value
                
                data.append(row)
        
        if data:
            df = pd.DataFrame(data)
            csv_file = self.benchmark_dir / f"data_{benchmark_id}.csv"
            df.to_csv(csv_file, index=False)
            self.logger.info(f"Data saved to {csv_file}")
        
        return visualizations
