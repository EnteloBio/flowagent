#!/usr/bin/env python
"""Example script to run benchmarks using FlowAgent benchmarking system.

This script demonstrates how to use the FlowAgent benchmarking system
to compare the performance of different LLM models on a sample workflow.
"""

import os
import asyncio
import argparse
from dotenv import load_dotenv
from pathlib import Path

from flowagent.benchmarking import BenchmarkRunner
from flowagent.config.settings import Settings


async def run_benchmark(prompt, models, repetitions, output_dir, description):
    """Run a benchmark with the specified parameters."""
    # Create a benchmark runner
    benchmark_runner = BenchmarkRunner()
    
    # Run the benchmark
    results = await benchmark_runner.run_benchmark(
        prompt=prompt,
        models=models,
        repetitions=repetitions,
        output_base_dir=output_dir,
        description=description
    )
    
    # Print results summary
    print(f"Benchmark completed: {results['benchmark_id']}")
    print(f"Results saved to: {results['benchmark_dir']}")
    
    return results


def main():
    """Run the benchmark example with command line arguments."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Ensure the OPENAI_API_KEY is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set. Please set it in your .env file.")
        return 1
    
    # Define command line arguments
    parser = argparse.ArgumentParser(description="Run FlowAgent benchmarks")
    parser.add_argument(
        "--prompt", 
        type=str,
        default="Please analyze my RNA-seq data with Kallisto for transcript quantification. The data is in fastq format.",
        help="Workflow prompt to execute"
    )
    parser.add_argument(
        "--models", 
        type=str, 
        nargs="+",
        default=["gpt-3.5-turbo", "gpt-4-turbo"],
        help="List of models to benchmark"
    )
    parser.add_argument(
        "--repetitions", 
        type=int, 
        default=3,
        help="Number of times to repeat each workflow execution"
    )
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="benchmark_outputs",
        help="Base directory for workflow outputs"
    )
    parser.add_argument(
        "--description", 
        type=str,
        default="RNA-seq workflow benchmark",
        help="Description of the benchmark"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the benchmark
    try:
        asyncio.run(run_benchmark(
            prompt=args.prompt,
            models=args.models,
            repetitions=args.repetitions,
            output_dir=str(output_dir),
            description=args.description
        ))
        return 0
    except Exception as e:
        print(f"Error running benchmark: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
