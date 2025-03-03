# RNA-seq Workflow

The RNA-seq workflow in FlowAgent follows best practices for RNA sequencing analysis.

## Workflow Steps

1. **Quality Control**
   - FastQC analysis of raw reads
   - Adapter identification
   - Quality score distribution
   - Sequence duplication levels

2. **Alignment**
   - STAR/Bowtie2 alignment
   - Mapping statistics
   - Duplicate marking
   - Quality filtering

3. **Feature Quantification**
   - Gene/transcript counting
   - UMI deduplication (if applicable)
   - Multi-mapping handling
   - Feature assignment stats

4. **Differential Expression**
   - Count normalization
   - Statistical testing
   - Multiple testing correction
   - Results visualization

## Custom Script Integration Points

The RNA-seq workflow supports custom scripts at various stages:

### Pre-alignment
- Quality filtering
- Adapter trimming
- Read preprocessing

### Post-alignment
- Alignment filtering
- BAM processing
- Quality metrics

### Analysis
- Custom normalization
- Alternative statistical tests
- Specialized visualizations

## Example: Custom Normalization

```R
# custom_normalize.R
library(DESeq2)
library(jsonlite)

# Read counts
counts <- read.csv(args_dict$counts_matrix, row.names=1)

# Normalize
dds <- DESeqDataSetFromMatrix(
    countData = counts,
    colData = data.frame(condition=factor(colnames(counts))),
    design = ~ 1
)
dds <- estimateSizeFactors(dds)
normalized_counts <- counts(dds, normalized=TRUE)

# Output results
write.csv(normalized_counts, "normalized_counts.csv")
cat(toJSON(list(normalized_counts = "normalized_counts.csv")))
```

## Usage

```python
from flowagent.core.workflow_executor import WorkflowExecutor

# Initialize workflow
executor = WorkflowExecutor(llm_interface)

# Execute RNA-seq workflow with custom normalization
results = await executor.execute_workflow(
    input_data={
        "fastq": "input.fastq",
        "annotation": "genes.gtf"
    },
    workflow_type="rna_seq",
    custom_script_requests=["deseq2_normalize"]
)
```

## Output Structure

```
results/
├── fastqc/
│   ├── fastqc_report.html
│   └── fastqc_data.txt
├── alignment/
│   ├── aligned.bam
│   └── alignment_stats.txt
├── counts/
│   ├── raw_counts.csv
│   └── normalized_counts.csv
└── differential_expression/
    ├── deseq2_results.csv
    └── ma_plot.pdf
```

## Quality Metrics

The workflow tracks various quality metrics:

1. **Raw Data Quality**
   - Base quality scores
   - GC content
   - Sequence complexity
   - Adapter content

2. **Alignment Quality**
   - Mapping rate
   - Unique vs. multi-mapped reads
   - Insert size distribution
   - Coverage uniformity

3. **Expression Quality**
   - Count distribution
   - Sample correlations
   - Batch effects
   - Technical artifacts

## Resource Requirements

Typical resource requirements for a standard RNA-seq analysis:

- **CPU**: 8-16 cores
- **Memory**: 32-64GB RAM
- **Storage**: 50-100GB per sample
- **Time**: 4-8 hours per sample

## Best Practices

1. **Quality Control**
   - Filter low-quality reads (Q < 20)
   - Remove adapter sequences
   - Check for sample contamination

2. **Alignment**
   - Use splice-aware aligners
   - Set appropriate multi-mapping parameters
   - Monitor alignment rates

3. **Quantification**
   - Consider gene vs. transcript level
   - Handle multi-mapped reads
   - Use appropriate normalization

4. **Analysis**
   - Account for batch effects
   - Use appropriate statistical models
   - Control for multiple testing
