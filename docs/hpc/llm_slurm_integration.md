# LLM-Driven SLURM Workflow Generation

This guide explains how FlowAgent's LLM capabilities integrate with SLURM execution for bioinformatics workflows.

## Overview

FlowAgent uses a multi-agent approach where the LLM acts as the workflow designer and coordinator, automatically generating appropriate workflows based on user requirements and then executing them on SLURM clusters.

## LLM-Generated Workflow Architecture

Unlike traditional approaches that require users to write scripts, FlowAgent's LLM:

1. **Analyzes User Requirements**: Interprets natural language descriptions of analysis goals
2. **Designs Optimal Workflows**: Constructs appropriate workflow steps with dependencies
3. **Allocates Resources**: Determines appropriate computational resources for each step
4. **Generates Commands**: Creates optimized command lines for each tool
5. **Manages Execution**: Submits and monitors jobs on SLURM

## Example Interaction

Here's how a typical interaction with FlowAgent's LLM for SLURM execution works:

```
User: "I need to analyze RNA-seq data from 5 tumor samples and 5 matched controls to identify differentially expressed genes."

FlowAgent: "I'll help you analyze your RNA-seq data to find differentially expressed genes between tumor and control samples. I'll design a workflow that includes quality control, alignment, quantification, and differential expression analysis.

First, I need some information:
1. Where are your FASTQ files located?
2. Do you have a reference genome preference (e.g., hg38, GRCh38)?
3. Are your reads paired-end or single-end?
4. Do you have any specific tools you prefer for the analysis?"

User: "The FASTQ files are in /data/rnaseq/, I want to use GRCh38, they are paired-end reads, and I prefer STAR for alignment and DESeq2 for differential expression."

FlowAgent: "Thanks for the information. I'll create a workflow using STAR for alignment and DESeq2 for differential expression analysis with GRCh38 as the reference genome.

I'll submit this workflow to your SLURM cluster with appropriate resource allocations for each step. The workflow will include:
1. FastQC for quality control
2. STAR for alignment to GRCh38
3. featureCounts for gene quantification
4. DESeq2 for differential expression analysis
5. Pathway enrichment analysis
6. Results visualization

I'll monitor the execution and provide updates on progress."
```

## How It Works

### 1. Workflow Generation API

The LLM uses FlowAgent's API to generate workflows:

```python
# This is internal code - users don't need to write this
from flowagent.api.workflow_generator import generate_workflow
from flowagent.core.executors import CGATExecutor

# LLM analyzes user request and generates appropriate workflow
workflow_definition = generate_workflow(
    analysis_type="rna_seq",
    input_data="/data/rnaseq/*.fastq.gz",
    reference_genome="GRCh38",
    paired_end=True,
    tools={
        "alignment": "STAR",
        "quantification": "featureCounts",
        "differential_expression": "DESeq2"
    },
    output_dir="/results/rnaseq_analysis"
)

# LLM configures executor with appropriate settings
executor = CGATExecutor()

# LLM submits workflow to executor
workflow_id = executor.submit_workflow(workflow_definition)
```

### 2. Resource Optimization

The LLM automatically determines appropriate resources for each step based on:

- Tool requirements
- Data size
- Available SLURM resources
- Best practices for each analysis type

For example, it knows that STAR alignment typically requires more memory than FastQC, and will allocate resources accordingly.

### 3. Dynamic Workflow Adjustment

The LLM can dynamically adjust workflows based on:

- Intermediate results
- Resource availability
- Error conditions
- User feedback

If a job fails due to insufficient memory, the LLM can automatically resubmit it with increased memory allocation.

## API Endpoints for LLM Integration

FlowAgent provides these API endpoints for LLM integration with SLURM:

### Workflow Generation

```python
# Internal API used by the LLM
from flowagent.api import workflow

# Generate workflow based on analysis requirements
workflow_def = workflow.generate(
    analysis_type="rna_seq",
    input_data={
        "path": "/data/rnaseq/",
        "pattern": "*.fastq.gz",
        "paired": True
    },
    parameters={
        "reference_genome": "GRCh38",
        "alignment_tool": "STAR",
        "expression_tool": "DESeq2"
    }
)
```

### Executor Configuration

```python
# Internal API used by the LLM
from flowagent.api import executors

# Configure SLURM executor
executor = executors.configure(
    executor_type="cgat",
    settings={
        "queue": "medium",
        "account": "genomics",
        "default_memory": "8G",
        "default_cpus": 4
    }
)
```

### Workflow Submission

```python
# Internal API used by the LLM
from flowagent.api import jobs

# Submit workflow to SLURM
job = jobs.submit(
    workflow_id=workflow_def["id"],
    executor=executor,
    notify_on_completion=True
)

# Get job status
status = jobs.get_status(job["id"])
```

## Example: LLM-Generated RNA-seq Workflow

Here's an example of a workflow definition generated by the LLM for RNA-seq analysis:

```json
{
  "id": "rnaseq_analysis_20250302",
  "type": "rna_seq",
  "input": {
    "samples": [
      {"id": "tumor1", "fastq1": "/data/rnaseq/tumor1_R1.fastq.gz", "fastq2": "/data/rnaseq/tumor1_R2.fastq.gz", "group": "tumor"},
      {"id": "tumor2", "fastq1": "/data/rnaseq/tumor2_R1.fastq.gz", "fastq2": "/data/rnaseq/tumor2_R2.fastq.gz", "group": "tumor"},
      {"id": "control1", "fastq1": "/data/rnaseq/control1_R1.fastq.gz", "fastq2": "/data/rnaseq/control1_R2.fastq.gz", "group": "control"},
      {"id": "control2", "fastq1": "/data/rnaseq/control2_R1.fastq.gz", "fastq2": "/data/rnaseq/control2_R2.fastq.gz", "group": "control"}
    ],
    "reference": {
      "genome": "/references/GRCh38/genome.fa",
      "annotation": "/references/GRCh38/genes.gtf"
    }
  },
  "steps": [
    {
      "id": "quality_control",
      "tool": "fastqc",
      "params": {
        "output_dir": "${output_dir}/fastqc"
      },
      "resources": {
        "memory": "4G",
        "cpus": 2,
        "time_min": 30
      },
      "dependencies": []
    },
    {
      "id": "alignment",
      "tool": "star",
      "params": {
        "genome_dir": "${reference.genome_dir}",
        "output_prefix": "${output_dir}/star/${sample.id}/",
        "read_files": ["${sample.fastq1}", "${sample.fastq2}"],
        "read_files_command": "zcat"
      },
      "resources": {
        "memory": "32G",
        "cpus": 8,
        "time_min": 120
      },
      "dependencies": ["quality_control"],
      "iterate_over": "samples"
    },
    {
      "id": "quantification",
      "tool": "featurecounts",
      "params": {
        "annotation": "${reference.annotation}",
        "output": "${output_dir}/counts/${sample.id}.counts",
        "bam_file": "${output_dir}/star/${sample.id}/Aligned.sortedByCoord.out.bam",
        "paired": true,
        "strand": "unstranded"
      },
      "resources": {
        "memory": "8G",
        "cpus": 4,
        "time_min": 60
      },
      "dependencies": ["alignment"],
      "iterate_over": "samples"
    },
    {
      "id": "differential_expression",
      "tool": "deseq2",
      "params": {
        "count_files": "${output_dir}/counts/*.counts",
        "sample_sheet": "${output_dir}/sample_info.csv",
        "output_dir": "${output_dir}/deseq2",
        "comparisons": [
          {"name": "tumor_vs_control", "group1": "tumor", "group2": "control"}
        ]
      },
      "resources": {
        "memory": "16G",
        "cpus": 1,
        "time_min": 60
      },
      "dependencies": ["quantification"]
    }
  ],
  "output": {
    "dir": "/results/rnaseq_analysis",
    "results": [
      {"name": "quality_reports", "path": "${output_dir}/fastqc"},
      {"name": "aligned_bams", "path": "${output_dir}/star/*/Aligned.sortedByCoord.out.bam"},
      {"name": "count_files", "path": "${output_dir}/counts/*.counts"},
      {"name": "deseq2_results", "path": "${output_dir}/deseq2"}
    ]
  }
}
```

## Benefits of LLM-Driven SLURM Integration

1. **No Coding Required**: Users describe their analysis needs in natural language
2. **Optimal Resource Allocation**: LLM automatically determines appropriate SLURM resources
3. **Best Practices**: Workflows follow bioinformatics best practices
4. **Error Handling**: Intelligent error recovery and troubleshooting
5. **Reproducibility**: Workflows are fully documented and reproducible
6. **Adaptability**: LLM can adjust workflows based on intermediate results

## Configuration

The LLM uses the same SLURM configuration options described in the [SLURM Configuration](slurm_configuration.md) document, but applies them automatically without requiring users to write configuration files.

## Conclusion

FlowAgent's LLM-driven SLURM integration represents a paradigm shift in bioinformatics workflow management. Instead of requiring users to write scripts or configuration files, the LLM interprets user requirements in natural language and automatically generates, configures, and executes optimal workflows on SLURM clusters.

This approach makes high-performance bioinformatics analysis accessible to researchers without extensive programming or HPC knowledge, while still leveraging the full power of SLURM-based HPC environments.
