# Example Workflow Prompts

This guide provides example prompts for different workflow types in FlowAgent. These prompts demonstrate how to effectively communicate your analysis requirements to the system.

## RNA-seq Analysis

### Basic Differential Expression
```
Analyze differential expression in RNA-seq data from breast cancer samples vs controls. I have:
- Paired-end FASTQ files in data/raw/
- Sample metadata in metadata.csv with condition labels
- Human genome reference (hg38)
Please include:
- Quality control with FastQC
- DESeq2 normalization
- Pathway analysis
- Volcano plot visualization
```

### Single-cell Analysis
```
Process single-cell RNA-seq data from 10x Genomics:
- Cell Ranger output in scRNA/counts/
- 8 samples (4 treated, 4 control)
- Human reference (GRCh38)
Analysis needed:
- Seurat preprocessing and QC
- Integration of all samples
- Differential expression by cluster
- Trajectory analysis with Monocle3
```

## ChIP-seq Analysis

### Histone Modification
```
Process ChIP-seq data for H3K27ac marks in neural progenitor cells. I have:
- ChIP FASTQ files in chip/
- Input control in input/
- mm10 genome
Requirements:
- Remove duplicates and low MAPQ reads
- Call peaks with MACS2
- Find enriched motifs
- Generate coverage plots around TSS regions
```

### Transcription Factor Binding
```
Analyze ChIP-seq for CTCF binding:
- Raw reads in tf_chip/fastq/
- Matched input controls
- hg38 genome
Include:
- IDR analysis between replicates
- Motif discovery with MEME
- Conservation analysis
- Integration with Hi-C data
```

## Hi-C Analysis

### Basic Interaction Analysis
```
Analyze chromosome interactions in Hi-C data from fibroblasts. Files:
- Raw reads in hic/reads/
- hg38 reference genome
- Restriction enzyme: MboI
Analysis needs:
- Generate contact matrices at multiple resolutions
- Call TADs and loop domains
- Find significant interactions
- Compare with published compartment annotations
```

### Multi-condition Comparison
```
Compare Hi-C data between wild-type and knockout:
- WT data in hic/wt/
- KO data in hic/ko/
- mm10 genome, DpnII digestion
Required:
- Matrix normalization
- Differential interaction analysis
- TAD boundary comparison
- Integration with RNA-seq changes
```

## ATAC-seq Analysis

### Basic Accessibility
```
Run ATAC-seq analysis on T cell activation data. Input:
- FASTQ files in atac/fastq/
- Sample groups: resting vs activated
- Human genome GRCh38
Required analysis:
- Fragment size distribution QC
- Call accessible regions
- Find differential accessibility
- Identify TF footprints
- Integrate with RNA-seq from matching samples
```

### Time Series Analysis
```
Process ATAC-seq time series during differentiation:
- Samples from 0h, 12h, 24h, 48h
- Technical replicates in atac/timeseries/
- Mouse genome mm10
Analysis:
- Quality metrics across time points
- Accessibility dynamics
- TF motif enrichment changes
- Pseudotime ordering of regions
```

## Best Practices for Prompts

When creating prompts for FlowAgent, consider including:

1. **Data Description**
   - File locations and formats
   - Sample organization
   - Reference genome
   - Experimental design

2. **Analysis Requirements**
   - Quality control steps
   - Core analysis methods
   - Statistical approaches
   - Integration needs

3. **Output Expectations**
   - Required visualizations
   - File formats
   - Statistical thresholds
   - Validation metrics

4. **Additional Context**
   - Biological background
   - Previous findings
   - Related datasets
   - Publication requirements

## Using Custom Scripts

You can request specific custom scripts in your prompts:

```
Please run RNA-seq analysis with:
- Custom normalization script (deseq2_normalize)
- Custom QC metrics (rna_qc_extended)
- Standard alignment and quantification
```

## Combining Workflows

For multi-omic analysis, you can combine workflows:

```
Integrate ATAC-seq and RNA-seq data:
- ATAC data in atac/fastq/
- RNA data in rna/fastq/
- Matching time points and conditions
Analysis:
- Process each assay independently
- Correlate accessibility with expression
- Find coordinated changes
- Generate integrated regulatory networks
```
