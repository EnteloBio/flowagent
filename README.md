# FlowAgent 1.0

An advanced multi-agent framework for automating complex bioinformatics workflows.

## Features

- **Workflow Automation**: Seamlessly automate RNA-seq, ChIP-seq, single-cell analysis, and Hi-C processing
- **Multi-Agent Architecture**: Distributed, fault-tolerant system with specialized agents
- **Dynamic Adaptation**: Real-time workflow optimization and error recovery
- **Enterprise-Grade Security**: Robust authentication, encryption, and audit logging
- **Advanced Monitoring**: Real-time metrics, alerts, and performance tracking
- **Scalable Performance**: Distributed processing and efficient resource management
- **Extensible Design**: Easy integration of new tools and workflows
- **Comprehensive Logging**: Detailed audit trails and debugging information

## Installation

```bash
# Clone the repository
git clone https://github.com/cribbslab/flowagent.git
cd flowagent

# Create and activate the conda environment:
conda env create -f conda/environment/environment.yml
conda activate flowagent

pip install .

# Verify installation of key components, e.g.:
kallisto version
fastqc --version
multiqc --version
flowagent --help
```

## Quick Start

1. Set up your environment:
```bash
# Copy the environment template
cp .env.example .env

# Edit .env with your settings
```

2. Run a CLI workflow:
```bash
# Basic workflow execution
flowagent prompt "run rna-seq analysis" --checkpoint-dir=workflow_state

# Resume a failed workflow
flowagent prompt "run rna-seq analysis" --checkpoint-dir=workflow_state --resume
```

3. Analyze workflow results:
```bash
# Generate analysis report
flowagent prompt "analyze workflow results" --analysis-dir=results

# Generate report without saving to file
flowagent prompt "analyze workflow results" --analysis-dir=results --no-save-report
```

4. Run a chatbot workflow on a local web server:
**Note that this is a work in progress, with significant further development planned for the future.**

```bash
# Start server.
flowagent serve --host 0.0.0.0 --port 8000
```


5. View chatbot in browser. Note the different port is correct:
```bash
open http://0.0.0.0:8080
```

## OpenAI Model Configuration

FlowAgent uses OpenAI's language models for workflow generation and analysis. Different operations have different model requirements:

1. **Workflow Generation** (`gpt-3.5-turbo` or better)
   - Basic workflow creation and execution can use `gpt-3.5-turbo`
   - Set in your `.env` file:
   ```bash
   OPENAI_MODEL=gpt-3.5-turbo
   ```

2. **Report Generation** (`gpt-4-turbo-preview` recommended)
   - For comprehensive analysis and insights, use `gpt-4-turbo-preview`
   - This model provides better reasoning and analysis capabilities
   - Set in your `.env` file:
   ```bash
   OPENAI_MODEL=gpt-4-turbo-preview
   ```

Example configurations:

1. For workflow execution:
```bash
# Set model in .env
OPENAI_MODEL=gpt-3.5-turbo

# Run workflow
flowagent "Analyze RNA-seq data in my fastq.gz files using Kallisto. The fastq files are in current directory and I want to use Homo_sapiens.GRCh38.cdna.all.fa as reference. The data is single ended. Generate QC reports and save everything in results/rna_seq_analysis."
```

2. For report generation:
```bash
# Set model in .env
OPENAI_MODEL=gpt-4-turbo-preview

# Generate comprehensive analysis
flowagent "analyze workflow results" --analysis-dir=results
```

You can also set the model temporarily using environment variables:
```bash
# For one-time report generation with gpt-4-turbo-preview
OPENAI_MODEL=gpt-4-turbo-preview flowagent "analyze workflow results" --analysis-dir=results
```

## HPC Configuration

FlowAgent supports High-Performance Computing (HPC) execution, with built-in support for SLURM, SGE, and TORQUE systems. The HPC settings can be configured through environment variables or in your `.env` file.

### Basic HPC Settings

```bash
# HPC Configuration
EXECUTOR_TYPE=hpc           # Use HPC executor instead of local
HPC_SYSTEM=slurm           # Options: slurm, sge, torque
HPC_QUEUE=all.q            # Your HPC queue name
HPC_DEFAULT_MEMORY=4G      # Default memory allocation
HPC_DEFAULT_CPUS=1         # Default CPU cores
HPC_DEFAULT_TIME=60        # Default time limit in minutes
```

### Resource Management

FlowAgent automatically manages HPC resources with sensible defaults that can be overridden:

1. **Memory Management**
   - Default: 4GB per job
   - Override with `HPC_DEFAULT_MEMORY`
   - Supports standard memory units (G, M, K)

2. **CPU Allocation**
   - Default: 1 CPU per job
   - Override with `HPC_DEFAULT_CPUS`
   - Automatically scales based on task requirements

3. **Queue Selection**
   - Default queue: "all.q"
   - Override with `HPC_QUEUE`
   - Queue-specific resource limits are respected

### Using HPC Execution

To run a workflow on your HPC system:

1. Basic execution:
```bash
flowagent "Your workflow description" --executor hpc
```

2. Specify custom resource requirements:
```bash
flowagent "Your workflow description" --executor hpc --memory 32G --threads 16
```

The system will automatically:
- Submit jobs to the appropriate queue
- Handle job dependencies
- Manage resource allocation
- Monitor job status
- Provide detailed logging

## Analysis Reports

The FlowAgent analysis report functionality provides comprehensive insights into your workflow outputs. It analyzes quality metrics, alignment statistics, and expression data to generate actionable recommendations.

### Running Analysis Reports

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

### Report Components

The analysis report includes:

1. **Summary**
   - Number of files analyzed
   - QC metrics processed
   - Issues found
   - Recommendations

2. **Quality Control Analysis**
   - FastQC metrics and potential issues
   - Read quality distribution
   - Adapter contamination levels
   - Sequence duplication rates

3. **Alignment Analysis**
   - Overall alignment rates
   - Unique vs multi-mapped reads
   - Read distribution statistics

4. **Expression Analysis**
   - Gene expression levels
   - TPM distributions
   - Sample correlations

5. **Recommendations**
   - Quality improvement suggestions
   - Parameter optimization tips
   - Technical issue resolutions

### Report Output

By default, the analysis report is:
1. Displayed in the console
2. Saved as a markdown file (`analysis_report.md`) in your analysis directory

To only view the report without saving:
```bash
flowagent "analyze workflow results" --analysis-dir=results --no-save-report
```

## Workflow State Management

FlowAgent includes a robust checkpointing system that helps manage long-running RNA-seq analysis workflows. This system allows you to resume interrupted workflows and avoid repeating expensive computations.

### Using Checkpoints

1. **Basic Usage**:
   ```bash
   # Run workflow with checkpointing
   flowagent prompt "Analyze RNA-seq data..." --checkpoint-dir workflow_state
   ```

2. **Resuming Interrupted Workflows**:
   ```bash
   # Resume from last successful checkpoint
   flowagent prompt "Analyze RNA-seq data..." --checkpoint-dir workflow_state --resume
   ```

### How It Works

The checkpoint directory (e.g., `workflow_state`) stores:
- Progress tracking for each workflow step
- Intermediate computation results
- Error logs and debugging information
- Workflow configuration and parameters

This allows FlowAgent to:
- Resume workflows from the last successful step
- Avoid recomputing expensive operations
- Maintain workflow state across system restarts
- Track errors and provide detailed debugging information

### Best Practices

1. **Choose Descriptive Directory Names**:
   ```bash
   # Use meaningful names for different analyses
   flowagent prompt "..." --checkpoint-dir rnaseq_liver_samples_20250225
   ```

2. **Backup Checkpoint Directories**:
   - Keep checkpoint directories for reproducibility
   - Back up important checkpoints before rerunning analyses
   - Use different checkpoint directories for different analyses

3. **Debugging Using Checkpoints**:
   - Examine checkpoint directory contents for troubleshooting
   - Use `--resume` to retry failed steps without restarting
   - Check error logs in checkpoint directory for detailed information

## Custom Scripts System

The FlowAgent custom scripts system allows you to integrate your own analysis scripts written in any programming language (R, Python, Bash, etc.) into the workflow. Scripts are automatically discovered and integrated into the workflow based on their metadata.

### Directory Structure
```
flowagent/
├── custom_scripts/
│   ├── rna_seq/           # RNA-seq specific scripts
│   │   └── normalization/
│   │       ├── custom_normalize.R
│   │       └── metadata.json
│   ├── chip_seq/          # ChIP-seq specific scripts
│   │   └── peak_analysis/
│   │       ├── custom_peaks.py
│   │       └── metadata.json
│   └── common/            # Scripts usable across workflows
       └── utils/
           ├── data_cleanup.sh
           └── metadata.json
```

### Adding Custom Scripts

1. **Create Script Directory**
   - Choose the appropriate workflow type directory (`rna_seq`, `chip_seq`, `common`)
   - Create a new directory for your script with a descriptive name

2. **Write Your Script**
   - Scripts can be written in any language (R, Python, Bash, etc.)
   - Must accept input parameters as command-line arguments
   - Must output results as JSON to stdout

3. **Create metadata.json**
   - Describes script purpose, inputs, outputs, and workflow integration
   - Specifies when the script should run in the workflow

### Metadata Structure

Each script requires a `metadata.json` file:

```json
{
    "name": "script_name",
    "description": "What the script does",
    "script_file": "script_name.ext",
    "language": "language_name",
    "input_requirements": [
        {
            "name": "input_name",
            "type": "file_type",
            "description": "Description of input"
        }
    ],
    "output_types": [
        {
            "name": "output_name",
            "type": "file_type",
            "description": "Description of output"
        }
    ],
    "workflow_types": ["workflow_type"],
    "execution_order": {
        "before": ["step_names"],
        "after": ["step_names"]
    },
    "requirements": {
        "r_packages": [],
        "python_packages": [],
        "system_dependencies": []
    }
}
```

### Example Scripts

1. **RNA-seq Normalization (R)**
```R
#!/usr/bin/env Rscript
# custom_normalize.R

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
args_dict <- list()
for (i in seq(1, length(args), 2)) {
    args_dict[[sub("^--", "", args[i])]] <- args[i + 1]
}

# Load required packages
library(DESeq2)
library(jsonlite)

# Read input data
counts <- read.csv(args_dict$counts_matrix, row.names=1)

# Perform normalization
dds <- DESeqDataSetFromMatrix(
    countData = counts,
    colData = data.frame(condition=factor(colnames(counts))),
    design = ~ 1
)
dds <- estimateSizeFactors(dds)
normalized_counts <- counts(dds, normalized=TRUE)

# Write output
output_file <- "normalized_counts.csv"
write.csv(normalized_counts, output_file)

# Return output paths as JSON
output <- list(
    normalized_counts = output_file
)
cat(toJSON(output))
```

2. **ChIP-seq Peak Analysis (Python)**
```python
#!/usr/bin/env python
# custom_peaks.py

import argparse
import json
import pandas as pd
from scipy import signal

def analyze_peaks(signal_file):
    # Read signal data
    signal_data = pd.read_csv(signal_file)
    
    # Find peaks
    peaks = signal.find_peaks(signal_data['intensity'])
    
    # Save results
    output_file = "peak_analysis.csv"
    pd.DataFrame({
        'position': peaks[0],
        'properties': peaks[1]
    }).to_csv(output_file)
    
    return {"peak_results": output_file}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--signal_data', required=True)
    args = parser.parse_args()
    
    # Run analysis and output results as JSON
    results = analyze_peaks(args.signal_data)
    print(json.dumps(results))
```

3. **Data Cleanup (Bash)**
```bash
#!/bin/bash
# data_cleanup.sh

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_file)
            INPUT_FILE="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Clean data
OUTPUT_FILE="cleaned_data.txt"
cat "$INPUT_FILE" | \
    grep -v '^#' | \
    awk 'NF > 0' | \
    sort | uniq > "$OUTPUT_FILE"

# Output results as JSON
echo "{\"cleaned_file\": \"$OUTPUT_FILE\"}"
```

### Integration with Workflows

The script manager automatically:
1. Discovers custom scripts by scanning the custom_scripts directory
2. Validates required packages and dependencies
3. Integrates scripts into the workflow based on execution order
4. Handles input/output management between workflow steps

### Script Requirements

1. **Input/Output**
   - Accept inputs as command-line arguments (e.g., `--input_name input_path`)
   - Output results as JSON to stdout
   - JSON should map output names to file paths

2. **Error Handling**
   - Exit with non-zero status on error
   - Write error messages to stderr

3. **Dependencies**
   - List required packages in metadata.json
   - System will validate requirements before execution

### Standard Workflow Steps

Scripts can be positioned relative to these standard steps:

**RNA-seq Workflow:**
- fastqc
- alignment
- feature_counts
- differential_expression

**ChIP-seq Workflow:**
- fastqc
- alignment
- peak_calling
- motif_analysis

## Architecture

FlowAgent 1.0 implements a modern, distributed architecture:

- **Core Engine**: Orchestrates workflow execution and agent coordination
- **Agent System**: Specialized agents for planning, execution, and monitoring
- **Knowledge Base**: Vector database for storing and retrieving domain knowledge
- **Security Layer**: Comprehensive security features and access control
- **API Layer**: RESTful and GraphQL APIs for integration
- **Monitoring System**: Real-time metrics and alerting

## Development

```bash
# Run tests
python -m pytest

# Run type checking
python -m mypy .

# Run linting
python -m ruff check .

# Format code
python -m black .
python -m isort .
```

## Building Documentation

FlowAgent uses MkDocs for documentation. To build the documentation locally, follow these steps:

1. Build the documentation:
```bash
mkdocs build
```

2. Serve the documentation locally to view it in your browser:
```bash
mkdocs serve
```

The documentation will be available at `http://127.0.0.1:8000`.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use FlowAgent in your research, please cite:

```bibtex
@software{flowagent2025,
  title={FlowAgent: An Advanced Multi-Agent Framework for Bioinformatics Workflows},
  author={Cribbs Lab},
  year={2025},
  url={https://github.com/cribbslab/flowagent}
}

```

## Version Compatibility

FlowAgent automatically handles version compatibility for Kallisto indices:

1. **Version Checking**
   - Checks Kallisto version before index creation
   - Validates index compatibility using `kallisto inspect`
   - Stores version information in workflow metadata

2. **Error Prevention**
   - Detects version mismatches before execution
   - Provides detailed error messages for incompatible indices
   - Suggests resolution steps for version conflicts

3. **Metadata Management**
   - Tracks index versions across workflows
   - Maintains compatibility information
   - Enables reproducible analyses

### Updating the Environment

To update your conda environment with new dependencies:

```bash
conda env update -f conda/environment/environment.yml
```

### Managing Multiple Environments

For development or testing, you can create a separate environment:

```bash
conda env create -f conda/environment/environment.yml -n flowagent-dev

```

### Basic Usage

```bash
# Local execution
flowagent "Analyze RNA-seq data in my fastq.gz files using Kallisto"

# SLURM cluster execution
flowagent --executor cgat "Analyze RNA-seq data in my fastq.gz files using Kallisto"
```
### Suggested prompts

1. Bulk paired-end RNA-seq workflow:

```bash
flowagent prompt "Analyze RNA-seq data I have paired ended data with the read 1 being named with the suffix .fastq.1.gz and the read2 being .fastq.2.gz using Kallisto. The fastq files are in current directory and I want to use Homo_sapiens.GRCh38.cdna.all.fa as reference. The data is pair ended. Generate QC reports and save everything in results/." --checkpoint-dir workflow_state
```
2. Bulk single-end RNA-seq workflow:

```bash
flowagent prompt "Analyze RNA-seq data in my fastq.gz files using Kallisto. The fastq files are in current directory and I want to use Homo_sapiens.GRCh38.cdna.all.fa as reference. The data is single ended. Generate QC reports and save everything in results/rna_seq_analysis." --checkpoint-dir workflow_state
```

3.single-nuclei workflow:

```bash
flowagent prompt "Analyze single-nuclei RNA-seq data. I have paired ended data with the read 1 being named with the suffix .fastq.1.gz and the read2 being .fastq.2.gz using kb_python. The fastq files are in current directory and I want to use Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz as genome fasta and Homo_sapiens.GRCh38.105.gtf.gz as the gtf file. I want the nuclei analysis. Generate QC reports and save everything in results/. The dependancy is kb-python and it can be installed using pip" --checkpoint-dir workflow_state
```

### Advanced Usage

1. Resume a failed workflow:
```bash
flowagent --resume --checkpoint-dir workflow_state "Your workflow prompt"
```

2. Specify custom resource requirements:
```bash
python -m flowagent.cli --executor cgat --memory 32G --threads 16 "Your workflow prompt"

```
