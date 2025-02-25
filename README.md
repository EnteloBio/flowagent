# Cognomic 1.0

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
git clone https://github.com/cribbslab/cognomic.git
cd cognomic

# Create and activate the conda environment:
conda env create -f conda/environment/environment.yml
conda activate cognomic

pip install .

# Verify installation of key components, e.g.:
kallisto version
fastqc --version
multiqc --version
cognomic --help
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
cognomic prompt "run rna-seq analysis" --checkpoint-dir=workflow_state

# Resume a failed workflow
cognomic prompt "run rna-seq analysis" --checkpoint-dir=workflow_state --resume
```

3. Analyze workflow results:
```bash
# Generate analysis report
cognomic prompt "analyze workflow results" --analysis-dir=results

# Generate report without saving to file
cognomic prompt "analyze workflow results" --analysis-dir=results --no-save-report
```

4. Run a chatbot workflow on a local web server:
```bash
# Start server.
cognomic serve --host 0.0.0.0 --port 8000
```


5. View chatbot in browser. Note the different port is correct:
```bash
open http://0.0.0.0:8080
```

## OpenAI Model Configuration

Cognomic uses OpenAI's language models for workflow generation and analysis. Different operations have different model requirements:

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
cognomic "Analyze RNA-seq data in my fastq.gz files using Kallisto. The fastq files are in current directory and I want to use Homo_sapiens.GRCh38.cdna.all.fa as reference. The data is single ended. Generate QC reports and save everything in results/rna_seq_analysis."
```

2. For report generation:
```bash
# Set model in .env
OPENAI_MODEL=gpt-4-turbo-preview

# Generate comprehensive analysis
cognomic "analyze workflow results" --analysis-dir=results
```

You can also set the model temporarily using environment variables:
```bash
# For one-time report generation with gpt-4-turbo-preview
OPENAI_MODEL=gpt-4-turbo-preview cognomic "analyze workflow results" --analysis-dir=results
```

## HPC Configuration

Cognomic supports High-Performance Computing (HPC) execution, with built-in support for SLURM, SGE, and TORQUE systems. The HPC settings can be configured through environment variables or in your `.env` file.

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

Cognomic automatically manages HPC resources with sensible defaults that can be overridden:

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
cognomic "Your workflow description" --executor hpc
```

2. Specify custom resource requirements:
```bash
cognomic "Your workflow description" --executor hpc --memory 32G --threads 16
```

The system will automatically:
- Submit jobs to the appropriate queue
- Handle job dependencies
- Manage resource allocation
- Monitor job status
- Provide detailed logging

## Analysis Reports

The Cognomic analysis report functionality provides comprehensive insights into your workflow outputs. It analyzes quality metrics, alignment statistics, and expression data to generate actionable recommendations.

### Running Analysis Reports

```bash
# Basic analysis
cognomic "analyze workflow results" --analysis-dir=/path/to/workflow/output

# Focus on specific aspects
cognomic "analyze quality metrics" --analysis-dir=/path/to/workflow/output
cognomic "analyze alignment rates" --analysis-dir=/path/to/workflow/output
cognomic "analyze expression data" --analysis-dir=/path/to/workflow/output
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
cognomic "analyze workflow results" --analysis-dir=results --no-save-report
```

## Workflow State Management

Cognomic includes a robust checkpointing system that helps manage long-running RNA-seq analysis workflows. This system allows you to resume interrupted workflows and avoid repeating expensive computations.

### Using Checkpoints

1. **Basic Usage**:
   ```bash
   # Run workflow with checkpointing
   cognomic prompt "Analyze RNA-seq data..." --checkpoint-dir workflow_state
   ```

2. **Resuming Interrupted Workflows**:
   ```bash
   # Resume from last successful checkpoint
   cognomic prompt "Analyze RNA-seq data..." --checkpoint-dir workflow_state --resume
   ```

### How It Works

The checkpoint directory (e.g., `workflow_state`) stores:
- Progress tracking for each workflow step
- Intermediate computation results
- Error logs and debugging information
- Workflow configuration and parameters

This allows Cognomic to:
- Resume workflows from the last successful step
- Avoid recomputing expensive operations
- Maintain workflow state across system restarts
- Track errors and provide detailed debugging information

### Best Practices

1. **Choose Descriptive Directory Names**:
   ```bash
   # Use meaningful names for different analyses
   cognomic prompt "..." --checkpoint-dir rnaseq_liver_samples_20250225
   ```

2. **Backup Checkpoint Directories**:
   - Keep checkpoint directories for reproducibility
   - Back up important checkpoints before rerunning analyses
   - Use different checkpoint directories for different analyses

3. **Debugging Using Checkpoints**:
   - Examine checkpoint directory contents for troubleshooting
   - Use `--resume` to retry failed steps without restarting
   - Check error logs in checkpoint directory for detailed information

## Architecture

Cognomic 1.0 implements a modern, distributed architecture:

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

Cognomic uses MkDocs for documentation. To build the documentation locally, follow these steps:

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

If you use Cognomic in your research, please cite:

```bibtex
@software{cognomic2025,
  title={Cognomic: An Advanced Multi-Agent Framework for Bioinformatics Workflows},
  author={Cribbs Lab},
  year={2025},
  url={https://github.com/cribbslab/cognomic}
}

```

## Version Compatibility

Cognomic automatically handles version compatibility for Kallisto indices:

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
conda env create -f conda/environment/environment.yml -n cognomic-dev

```

### Basic Usage

```bash
# Local execution
cognomic "Analyze RNA-seq data in my fastq.gz files using Kallisto"

# SLURM cluster execution
cognomic --executor cgat "Analyze RNA-seq data in my fastq.gz files using Kallisto"
```

### Advanced Usage

1. Resume a failed workflow:
```bash
cognomic --resume --checkpoint-dir workflow_state "Your workflow prompt"
```

2. Specify custom resource requirements:
```bash
python -m cognomic.cli --executor cgat --memory 32G --threads 16 "Your workflow prompt"

```
