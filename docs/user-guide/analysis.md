# Analysis Reports

The FlowAgent analysis report functionality provides comprehensive insights into your workflow outputs. It analyzes quality metrics, alignment statistics, and expression data to generate actionable recommendations.

## Running Analysis Reports

```bash
# Basic analysis
flowagent "analyze workflow results" --analysis-dir=/path/to/workflow/output

# Focus on specific aspects
flowagent "analyze quality metrics" --analysis-dir=/path/to/workflow/output
flowagent "analyze alignment rates" --analysis-dir=/path/to/workflow/output
flowagent "analyze expression data" --analysis-dir=/path/to/workflow/output
```

The analyzer will recursively search for relevant files in your analysis directory, including:
- FastQC outputs
- MultiQC reports
- Kallisto results
- Log files

## Report Components

### Summary
- Number of files analyzed
- QC metrics processed
- Issues found
- Recommendations

### Quality Control Analysis
- FastQC metrics and potential issues
- Read quality distribution
- Adapter contamination levels
- Sequence duplication rates

### Alignment Analysis
- Overall alignment rates
- Unique vs multi-mapped reads
- Read distribution statistics

### Expression Analysis
- Gene expression levels
- TPM distributions
- Sample correlations

### Recommendations
- Quality improvement suggestions
- Parameter optimization tips
- Technical issue resolutions

## Report Output

By default, the analysis report is:
- Displayed in the console
- Saved as a markdown file (`analysis_report.md`) in your analysis directory

To only view the report without saving:
```bash
flowagent "analyze workflow results" --analysis-dir=results --no-save-report
```
