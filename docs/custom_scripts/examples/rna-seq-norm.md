# RNA-seq Normalization Example

This example demonstrates how to create a custom RNA-seq normalization script using DESeq2.

## Script Overview

The script performs DESeq2 normalization on RNA-seq count data and outputs normalized counts.

## Implementation

```R
# custom_normalize.R
library(DESeq2)
library(jsonlite)

# Read input arguments
args_dict <- fromJSON(commandArgs(trailingOnly = TRUE)[1])

# Read counts matrix
counts <- read.csv(args_dict$counts_matrix, row.names=1)

# Create DESeq dataset
dds <- DESeqDataSetFromMatrix(
    countData = counts,
    colData = data.frame(condition=factor(colnames(counts))),
    design = ~ 1
)

# Perform normalization
dds <- estimateSizeFactors(dds)
normalized_counts <- counts(dds, normalized=TRUE)

# Write output
write.csv(normalized_counts, "normalized_counts.csv")
cat(toJSON(list(normalized_counts = "normalized_counts.csv")))
```

## Metadata

```json
{
    "name": "deseq2_normalize",
    "description": "Normalize RNA-seq counts using DESeq2",
    "script_file": "custom_normalize.R",
    "language": "R",
    "input_requirements": [
        {
            "name": "counts_matrix",
            "type": "file",
            "description": "CSV file containing raw counts matrix"
        }
    ],
    "output_types": [
        {
            "name": "normalized_counts",
            "type": "file",
            "description": "CSV file containing normalized counts"
        }
    ],
    "workflow_types": ["rna_seq"],
    "execution_order": {
        "after": ["feature_counts"],
        "before": ["differential_expression"]
    },
    "requirements": {
        "r_packages": ["DESeq2", "jsonlite"]
    }
}
```

## Usage

```python
from flowagent.core.workflow_executor import WorkflowExecutor

# Initialize workflow
executor = WorkflowExecutor(llm_interface)

# Execute workflow with custom normalization
results = await executor.execute_workflow(
    input_data={
        "counts_matrix": "raw_counts.csv"
    },
    workflow_type="rna_seq",
    custom_script_requests=["deseq2_normalize"]
)

# Access normalized counts
normalized_counts = pd.read_csv(results["normalized_counts"])
```

## Output

The script produces a CSV file containing the normalized counts matrix, where:
- Rows represent genes
- Columns represent samples
- Values are normalized counts

## Quality Metrics

The normalization process tracks:
- Size factors per sample
- Count distributions
- Normalization effectiveness
- Sample correlations

## Best Practices

1. **Input Data**
   - Use raw (unfiltered) counts
   - Include all samples
   - Verify gene names/IDs

2. **Quality Control**
   - Check for low counts
   - Verify sample grouping
   - Monitor outliers

3. **Output Handling**
   - Save normalized data
   - Document parameters
   - Track QC metrics
