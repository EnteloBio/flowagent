# FlowAgent Benchmarking System

This module provides tools for benchmarking different LLM models within FlowAgent to compare their performance when executing workflows.

## Overview

The benchmarking system allows you to:

1. Run the same workflow prompt across multiple LLM models (e.g., GPT-3.5, GPT-4, Claude)
2. Execute benchmarks multiple times for statistical reliability
3. Collect performance metrics such as:
   - Execution time
   - Success rates
   - Resource usage (memory, CPU)
   - Smart resume effectiveness
4. Generate visualizations to compare model performance
5. Produce detailed reports with analysis of results

## Components

- `BenchmarkRunner`: Orchestrates the benchmarking process
- `BenchmarkMetrics`: Extracts and analyzes metrics from workflow executions
- `BenchmarkVisualizer`: Generates visualizations of benchmark results

## Usage

### Basic Example

```python
import asyncio
from flowagent.benchmarking import BenchmarkRunner

async def run_example():
    # Create a benchmark runner
    benchmark_runner = BenchmarkRunner()
    
    # Define workflow prompt and models to benchmark
    prompt = "Please analyze my RNA-seq data with Kallisto for transcript quantification."
    models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
    
    # Run the benchmark
    results = await benchmark_runner.run_benchmark(
        prompt=prompt,
        models=models,
        repetitions=3,
        description="RNA-seq workflow benchmark"
    )
    
    print(f"Benchmark completed: {results['benchmark_id']}")
    print(f"Results saved to: {results['benchmark_dir']}")

# Run the example
asyncio.run(run_example())
```

### Using the Example Script

The package includes an example script at `examples/run_benchmark.py` that you can use to run benchmarks from the command line:

```
python examples/run_benchmark.py --models gpt-3.5-turbo gpt-4-turbo --repetitions 2
```

### Benchmark Results

Benchmark results are saved to the `benchmarks/` directory by default, with each run in a timestamped subdirectory. The results include:

- `metadata_*.json`: Details about the benchmark run
- `results_*.json`: Raw results from each model execution
- `summary_*.json`: Summary statistics and comparisons
- `visualizations_*.json`: Paths to generated visualizations
- Various visualization files (.png)
- `data_*.csv`: Raw data in CSV format for further analysis

## Metrics Collected

The benchmarking system collects the following metrics:

1. **Execution Metrics**
   - Total execution time
   - Success/failure status
   - Number of steps executed

2. **Smart Resume Metrics**
   - Completed steps detected ratio
   - Steps skipped ratio

3. **Resource Usage**
   - Memory usage (MB)
   - CPU usage (%)

4. **Command Quality Metrics**
   - Command count
   - Average command length
   - Tools and flags used
   - Error indicators

## Environment Variables

The benchmarking system requires the following environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key

## Requirements

- pandas
- matplotlib
- seaborn
- psutil
