# Workflows

FlowAgent supports various bioinformatics workflows, each with a standardized structure and the ability to integrate custom scripts.

## Supported Workflows

### RNA-seq
- Quality control and preprocessing
- Alignment and quantification
- Differential expression analysis
- Custom normalization options
- Single-cell analysis support

### ChIP-seq
- Quality assessment
- Read alignment
- Peak calling
- Motif analysis
- Signal visualization

### Hi-C
- Quality control
- Contact matrix generation
- TAD calling
- Interaction analysis
- 3D structure prediction

### ATAC-seq
- Quality metrics
- Peak calling
- Accessibility analysis
- Footprinting
- Integration with ChIP-seq

### Bisulfite-seq
- Quality control
- Methylation calling
- Differential methylation
- Pattern analysis
- Integration with gene expression

### Single-cell Multi-omics
- RNA velocity
- CITE-seq analysis
- Multimodal integration
- Trajectory analysis
- Cell type annotation

## Workflow Structure

Each workflow follows a standardized structure:

1. **Data Quality & Preprocessing**
   - Raw data quality metrics (FastQC, MultiQC)
   - Data cleaning and filtering
   - Format validation

2. **Alignment & Quantification**
   - Mapping statistics
   - Tool-specific metrics
   - Quality filtering

3. **Analysis-Specific Steps**
   - Workflow-specific analyses
   - Custom script integration
   - Result generation

4. **Resource Management**
   - CPU/Memory monitoring
   - Runtime tracking
   - Performance optimization

5. **Results & Visualization**
   - Quality reports
   - Analysis summaries
   - Visualization outputs

## Custom Script Integration

Workflows can be extended with custom scripts:

```python
workflow = WorkflowExecutor(llm_interface)
results = await workflow.execute_workflow(
    input_data={"fastq": "input.fastq"},
    workflow_type="rna_seq",
    custom_script_requests=["deseq2_normalize"]
)
```

See the [Custom Scripts](../custom_scripts/index.md) section for more details.
