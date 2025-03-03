#!/usr/bin/env python
"""Custom peak analysis for ChIP-seq data."""

import argparse
import json
import pandas as pd
import numpy as np
from scipy import signal
from pathlib import Path

def analyze_peaks(signal_file):
    """Analyze ChIP-seq peaks from signal data.
    
    Args:
        signal_file: Path to signal intensity data
        
    Returns:
        Dictionary with output file paths
    """
    # Read signal data
    signal_data = pd.read_csv(signal_file)
    
    # Find peaks
    peaks, properties = signal.find_peaks(
        signal_data['intensity'],
        height=0.5,
        distance=50,
        prominence=0.2
    )
    
    # Calculate peak metrics
    peak_metrics = pd.DataFrame({
        'position': peaks,
        'height': properties['peak_heights'],
        'prominence': properties['prominences'],
        'width': properties['widths']
    })
    
    # Save detailed results
    peaks_file = "peak_analysis.csv"
    peak_metrics.to_csv(peaks_file, index=False)
    
    # Generate summary statistics
    summary = {
        'total_peaks': len(peaks),
        'mean_height': float(np.mean(properties['peak_heights'])),
        'mean_width': float(np.mean(properties['widths']))
    }
    
    summary_file = "peak_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return {
        "peak_results": peaks_file,
        "peak_summary": summary_file
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze ChIP-seq peaks")
    parser.add_argument('--signal_data', required=True,
                      help="Path to signal intensity data file")
    args = parser.parse_args()
    
    try:
        # Run analysis
        results = analyze_peaks(args.signal_data)
        # Output results as JSON to stdout
        print(json.dumps(results))
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
