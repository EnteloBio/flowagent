"""Visualization utilities for FlowAgent benchmarks.

This module provides visualization capabilities for analyzing and comparing
benchmark results across different LLM models.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)


class BenchmarkVisualizer:
    """Visualizes benchmark results."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """Initialize the benchmark visualizer.
        
        Args:
            output_dir: Directory to save visualization results
        """
        self.output_dir = output_dir
        self.logger = get_logger(__name__)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set default visualization style
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({'font.size': 12})
    
    def visualize_all(self, 
                      model_results: Dict[str, List[Dict[str, Any]]], 
                      comparison: Dict[str, Any]) -> Dict[str, str]:
        """Generate all visualizations for the benchmark results.
        
        Args:
            model_results: Dictionary mapping model names to lists of results
            comparison: Results of model comparison
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        visualizations = {}
        
        # Create individual visualizations
        visualizations["execution_time"] = self.visualize_execution_time(model_results)
        visualizations["success_rate"] = self.visualize_success_rate(model_results)
        visualizations["smart_resume"] = self.visualize_smart_resume_metrics(model_results)
        visualizations["resource_usage"] = self.visualize_resource_usage(model_results)
        visualizations["model_comparison"] = self.visualize_model_comparison(comparison)
        
        return visualizations
    
    def visualize_execution_time(self, 
                                model_results: Dict[str, List[Dict[str, Any]]]) -> str:
        """Visualize execution time across models.
        
        Args:
            model_results: Dictionary mapping model names to lists of results
            
        Returns:
            Path to the generated visualization file
        """
        # Prepare data
        data = []
        for model, results in model_results.items():
            for result in results:
                if "metrics" in result and "duration" in result["metrics"]:
                    data.append({
                        "model": model,
                        "duration": result["metrics"]["duration"],
                        "workflow": result.get("workflow_type", "Unknown")
                    })
        
        if not data:
            self.logger.warning("No duration data available for visualization")
            return ""
            
        df = pd.DataFrame(data)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(x="model", y="duration", data=df)
        ax = sns.swarmplot(x="model", y="duration", data=df, color="0.25")
        
        plt.title("Execution Time by Model")
        plt.xlabel("Model")
        plt.ylabel("Duration (seconds)")
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(self.output_dir, "execution_time.png")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def visualize_success_rate(self, 
                              model_results: Dict[str, List[Dict[str, Any]]]) -> str:
        """Visualize success rate across models.
        
        Args:
            model_results: Dictionary mapping model names to lists of results
            
        Returns:
            Path to the generated visualization file
        """
        # Prepare data
        success_rates = {}
        for model, results in model_results.items():
            if not results:
                continue
                
            successful_runs = sum(1 for r in results if r.get("status") == "success")
            success_rates[model] = successful_runs / len(results)
            
        if not success_rates:
            self.logger.warning("No success rate data available for visualization")
            return ""
            
        models = list(success_rates.keys())
        rates = list(success_rates.values())
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=models, y=rates)
        
        # Add value labels on bars
        for i, v in enumerate(rates):
            ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
        
        plt.title("Success Rate by Model")
        plt.xlabel("Model")
        plt.ylabel("Success Rate")
        plt.ylim(0, 1.1)  # Set y-axis limits
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(self.output_dir, "success_rate.png")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def visualize_smart_resume_metrics(self, 
                                      model_results: Dict[str, List[Dict[str, Any]]]) -> str:
        """Visualize smart resume metrics across models.
        
        Args:
            model_results: Dictionary mapping model names to lists of results
            
        Returns:
            Path to the generated visualization file
        """
        # Prepare data
        data = []
        for model, results in model_results.items():
            for result in results:
                if "metrics" in result:
                    metrics = result["metrics"]
                    if "completed_steps_detected_ratio" in metrics and "steps_skipped_ratio" in metrics:
                        data.append({
                            "model": model,
                            "completed_steps_detected_ratio": metrics["completed_steps_detected_ratio"],
                            "steps_skipped_ratio": metrics["steps_skipped_ratio"]
                        })
        
        if not data:
            self.logger.warning("No smart resume data available for visualization")
            return ""
            
        df = pd.DataFrame(data)
        
        # Reshape data for grouped bar chart
        df_melted = pd.melt(df, 
                          id_vars=["model"], 
                          value_vars=["completed_steps_detected_ratio", "steps_skipped_ratio"],
                          var_name="metric", 
                          value_name="value")
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x="model", y="value", hue="metric", data=df_melted)
        
        plt.title("Smart Resume Effectiveness by Model")
        plt.xlabel("Model")
        plt.ylabel("Ratio")
        plt.ylim(0, 1.1)  # Set y-axis limits
        plt.legend(title="Metric", loc="upper right")
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(self.output_dir, "smart_resume_metrics.png")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def visualize_resource_usage(self, 
                                model_results: Dict[str, List[Dict[str, Any]]]) -> str:
        """Visualize resource usage across models.
        
        Args:
            model_results: Dictionary mapping model names to lists of results
            
        Returns:
            Path to the generated visualization file
        """
        # Prepare data
        data = []
        for model, results in model_results.items():
            for result in results:
                if "metrics" in result:
                    metrics = result["metrics"]
                    if "memory_usage" in metrics and "cpu_usage" in metrics:
                        data.append({
                            "model": model,
                            "memory_usage": metrics["memory_usage"],
                            "cpu_usage": metrics["cpu_usage"]
                        })
        
        if not data:
            self.logger.warning("No resource usage data available for visualization")
            return ""
            
        df = pd.DataFrame(data)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Memory usage subplot
        sns.boxplot(x="model", y="memory_usage", data=df, ax=ax1)
        ax1.set_title("Memory Usage by Model")
        ax1.set_xlabel("Model")
        ax1.set_ylabel("Memory Usage (MB)")
        
        # CPU usage subplot
        sns.boxplot(x="model", y="cpu_usage", data=df, ax=ax2)
        ax2.set_title("CPU Usage by Model")
        ax2.set_xlabel("Model")
        ax2.set_ylabel("CPU Usage (%)")
        
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(self.output_dir, "resource_usage.png")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def visualize_model_comparison(self, comparison: Dict[str, Any]) -> str:
        """Visualize model comparison with normalized metrics.
        
        Args:
            comparison: Results of model comparison
            
        Returns:
            Path to the generated visualization file
        """
        if not comparison or "metrics" not in comparison or not comparison["metrics"]:
            self.logger.warning("No comparison data available for visualization")
            return ""
            
        # Extract models and metrics
        models = comparison.get("models", [])
        metrics_data = comparison.get("metrics", {})
        
        if not models or not metrics_data:
            self.logger.warning("No valid model or metrics data for visualization")
            return ""
            
        # Prepare data for radar chart
        metrics = ["success_rate", "duration", "completed_steps_detected_ratio", 
                  "steps_skipped_ratio", "memory_usage", "cpu_usage"]
        
        # Filter to only include metrics that are present for all models
        metrics = [m for m in metrics if all(m in metrics_data[model] for model in models)]
        
        if not metrics:
            self.logger.warning("No common metrics found for all models")
            return ""
            
        # Normalize metrics to 0-1 scale (higher is better)
        normalized_data = {}
        for metric in metrics:
            values = [metrics_data[model][metric] for model in models]
            min_val = min(values)
            max_val = max(values)
            range_val = max_val - min_val if max_val > min_val else 1
            
            # For metrics where lower is better, invert the normalization
            if metric in ["duration", "memory_usage", "cpu_usage"]:
                normalized_data[metric] = [(max_val - v) / range_val for v in values]
            else:
                normalized_data[metric] = [(v - min_val) / range_val for v in values]
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        for i, model in enumerate(models):
            values = [normalized_data[metric][i] for metric in metrics]
            values += values[:1]  # Close the circle
            
            ax.plot(angles, values, linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.1)
        
        # Set labels and styling
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"])
        ax.set_ylim(0, 1)
        
        plt.title("Model Comparison (Normalized Metrics)")
        plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(self.output_dir, "model_comparison.png")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
