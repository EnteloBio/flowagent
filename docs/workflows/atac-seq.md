# ATAC-seq Workflow

The ATAC-seq workflow in FlowAgent implements comprehensive analysis of chromatin accessibility data.

## Workflow Steps

1. **Quality Control**
   - FastQC analysis
   - Fragment size distribution
   - Nucleosome positioning
   - Library complexity

2. **Alignment**
   - Read alignment
   - Duplicate removal
   - Mitochondrial filtering
   - Quality filtering

3. **Peak Calling**
   - Accessibility peaks
   - Signal normalization
   - IDR analysis
   - Peak annotation

4. **Footprinting**
   - TF footprint detection
   - Motif enrichment
   - Binding dynamics
   - Factor activity

5. **Integration**
   - ChIP-seq correlation
   - Gene expression
   - Chromatin state
   - Regulatory networks

## Custom Script Integration Points

The ATAC-seq workflow supports custom scripts at various stages:

### Pre-processing
- Custom filtering
- Quality metrics
- Fragment analysis

### Peak Analysis
- Custom peak calling
- Signal processing
- Feature detection

### Integration
- Multi-omics analysis
- Network inference
- Visualization tools

## Example: Custom Footprint Detector

```python
# custom_footprints.py
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

def detect_footprints(signal_data, motifs, params):
    """Detect transcription factor footprints."""
    window_size = params['window_size']
    min_depth = params['min_depth']
    
    footprints = []
    for motif in motifs:
        # Get signal around motif
        signal = get_signal_matrix(signal_data, motif, window_size)
        
        # Calculate protection score
        protection = calculate_protection(signal, window_size)
        
        # Call footprints
        if protection > min_depth:
            footprints.append({
                'motif': motif.name,
                'position': motif.position,
                'score': protection,
                'signal': signal.mean(axis=0)
            })
    
    return pd.DataFrame(footprints)

def calculate_protection(signal, window_size):
    """Calculate TF protection score."""
    flanks = np.concatenate([
        signal[:, :window_size//4],
        signal[:, -window_size//4:]
    ])
    center = signal[:, window_size//4:-window_size//4]
    
    flank_density = gaussian_kde(flanks.flatten())
    center_density = gaussian_kde(center.flatten())
    
    return flank_density.integrate_box_1d(0, np.inf) - \
           center_density.integrate_box_1d(0, np.inf)
```

## Usage

```python
from flowagent.core.workflow_executor import WorkflowExecutor

# Initialize workflow
executor = WorkflowExecutor(llm_interface)

# Execute ATAC-seq workflow with custom footprinting
results = await executor.execute_workflow(
    input_data={
        "fastq1": "read1.fastq",
        "fastq2": "read2.fastq",
        "genome": "reference.fa",
        "motifs": "motifs.txt"
    },
    workflow_type="atac_seq",
    custom_script_requests=["custom_footprints"]
)
```

## Output Structure

```
results/
├── qc/
│   ├── fastqc/
│   ├── fragment_sizes.pdf
│   └── library_complexity.txt
├── alignment/
│   ├── filtered.bam
│   └── metrics.txt
├── peaks/
│   ├── peaks.narrowPeak
│   └── annotated_peaks.txt
├── footprints/
│   ├── footprints.bed
│   └── motif_enrichment.txt
└── integration/
    ├── chip_correlation/
    └── regulatory_network/
```

## Quality Metrics

The workflow tracks various quality metrics:

1. **Library Quality**
   - Read quality scores
   - Fragment size distribution
   - Library complexity
   - Mitochondrial content

2. **Signal Quality**
   - Signal-to-noise ratio
   - Peak enrichment
   - Reproducibility
   - Coverage uniformity

3. **Analysis Quality**
   - Footprint depth
   - Motif enrichment
   - Integration scores
   - Regulatory potential

## Resource Requirements

Typical resource requirements for ATAC-seq analysis:

- **CPU**: 8-16 cores
- **Memory**: 32-64GB RAM
- **Storage**: 50-100GB per sample
- **Time**: 4-8 hours per sample

## Best Practices

1. **Quality Control**
   - Filter low-quality reads
   - Remove duplicates
   - Check fragment sizes
   - Monitor complexity

2. **Peak Calling**
   - Use appropriate parameters
   - Consider replicates
   - Validate peaks
   - Annotate features

3. **Footprinting**
   - Optimize window size
   - Use appropriate controls
   - Consider dynamics
   - Validate binding

4. **Integration**
   - Use matched samples
   - Consider time points
   - Validate networks
   - Compare conditions
