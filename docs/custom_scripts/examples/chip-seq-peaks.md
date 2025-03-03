# ChIP-seq Peak Analysis Example

This example demonstrates how to create a custom ChIP-seq peak analysis script.

## Script Overview

The script implements custom peak detection and analysis for ChIP-seq data using signal processing techniques.

## Implementation

```python
# custom_peaks.py
import pandas as pd
import numpy as np
from scipy import signal
import json
import sys

def analyze_peaks(signal_data, params):
    """Analyze ChIP-seq peaks with custom parameters."""
    # Find peaks
    peaks, properties = signal.find_peaks(
        signal_data['intensity'],
        height=params['min_height'],
        distance=params['min_distance'],
        prominence=params['min_prominence']
    )
    
    # Calculate metrics
    peak_metrics = pd.DataFrame({
        'position': peaks,
        'height': properties['peak_heights'],
        'prominence': properties['prominences'],
        'width': properties['widths']
    })
    
    return peak_metrics

def main():
    # Parse input arguments
    args = json.loads(sys.argv[1])
    
    # Read signal data
    signal_data = pd.read_csv(args['signal_file'])
    
    # Set parameters
    params = {
        'min_height': 0.5,
        'min_distance': 50,
        'min_prominence': 0.2
    }
    
    # Analyze peaks
    results = analyze_peaks(signal_data, params)
    
    # Save results
    results.to_csv('peak_analysis.csv', index=False)
    
    # Output results location
    print(json.dumps({
        'peak_results': 'peak_analysis.csv'
    }))

if __name__ == '__main__':
    main()
```

## Metadata

```json
{
    "name": "custom_peak_analysis",
    "description": "Custom peak detection for ChIP-seq data",
    "script_file": "custom_peaks.py",
    "language": "python",
    "input_requirements": [
        {
            "name": "signal_file",
            "type": "file",
            "description": "CSV file containing ChIP-seq signal data"
        }
    ],
    "output_types": [
        {
            "name": "peak_results",
            "type": "file",
            "description": "CSV file containing peak analysis results"
        }
    ],
    "workflow_types": ["chip_seq"],
    "execution_order": {
        "after": ["alignment"],
        "before": ["motif_analysis"]
    },
    "requirements": {
        "python_packages": ["pandas", "numpy", "scipy"]
    }
}
```

## Usage

```python
from flowagent.core.workflow_executor import WorkflowExecutor

# Initialize workflow
executor = WorkflowExecutor(llm_interface)

# Execute workflow with custom peak analysis
results = await executor.execute_workflow(
    input_data={
        "signal_file": "chip_signal.csv"
    },
    workflow_type="chip_seq",
    custom_script_requests=["custom_peak_analysis"]
)

# Access peak results
peak_data = pd.read_csv(results["peak_results"])
```

## Output Format

The script produces a CSV file with columns:
- `position`: Genomic position of peak
- `height`: Peak height
- `prominence`: Peak prominence
- `width`: Peak width at half maximum

## Quality Metrics

The analysis tracks:
- Peak distribution
- Signal-to-noise ratio
- Peak shape characteristics
- Coverage statistics

## Best Practices

1. **Signal Processing**
   - Filter noise appropriately
   - Use robust peak detection
   - Consider local background

2. **Parameter Selection**
   - Optimize for data type
   - Validate on known regions
   - Consider replicates

3. **Quality Control**
   - Check peak distributions
   - Validate peak shapes
   - Monitor false positives
