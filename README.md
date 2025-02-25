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

# Verify installation of key components
kallisto version
fastqc --version
multiqc --version

# Add bioinformatics tools
mamba install -c bioconda fastqc=0.12.1
mamba install -c bioconda trim-galore=0.6.10
mamba install -c bioconda star=2.7.10b
mamba install -c bioconda subread=2.0.6
mamba install -c conda-forge r-base=4.2
mamba install -c bioconda bioconductor-deseq2
mamba install -c bioconda samtools=1.17
mamba install -c bioconda multiqc=1.14
```

## Quick Start

1. Set up your environment:
```bash
# Copy the environment template
cp .env.example .env

# Edit .env with your settings
# Required:
# - SECRET_KEY: Generate a secure random key (e.g., using: python -c "import secrets; print(secrets.token_hex(32))")
# - OPENAI_API_KEY: Your OpenAI API key (if using LLM features)
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

## API Key Configuration

Cognomic requires several API keys for full functionality. You can configure these using environment variables or a `.env` file in the project root directory.

### Required API Keys

1. **Secret Key** (for JWT token generation):
```bash
SECRET_KEY=your-secure-secret-key
```

2. **OpenAI API Key** (for LLM functionality):
```bash
OPENAI_API_KEY=your-openai-api-key
```

### Setting Up OpenAI API Keys

There are two ways to configure your API keys:

1. **Using Environment Variables**:
```bash
export SECRET_KEY=your-secure-secret-key
export OPENAI_API_KEY=your-openai-api-key
```

2. **Using a .env File**:
Create a `.env` file in the project root directory:
```bash
# .env
SECRET_KEY=your-secure-secret-key
OPENAI_API_KEY=your-openai-api-key

# Optional Settings
OPENAI_BASE_URL=https://api.openai.com/v1  # Default OpenAI API URL
OPENAI_MODEL=gpt-4                         # Default LLM model
```

### Security Best Practices

1. Never commit your `.env` file to version control
2. Use strong, unique keys for each environment (development, staging, production)
3. Regularly rotate your API keys
4. Keep your API keys secure and never share them in public repositories

The `.env` file is automatically loaded by the application when it starts. All sensitive information is handled securely using Pydantic's `SecretStr` type to prevent accidental exposure in logs or error messages.

## Security Configuration

### Setting up the Secret Key

The `SECRET_KEY` is a crucial security element in Cognomic used for:
- Generating and validating JSON Web Tokens (JWTs) for API authentication
- Securing session data
- Protecting against cross-site request forgery (CSRF) attacks

To generate a secure random key, run:
```bash
# Generate a secure random key using Python
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

Add the generated key to your `.env` file:
```bash
# Copy the example environment file
cp env.example /path/to/your/.env

# Edit .env and update the SECRET_KEY
SECRET_KEY=your-generated-key-here
```

### Security Best Practices

1. **Secret Key Management**:
   - Never commit your `.env` file to version control
   - Use different secret keys for development and production
   - Regenerate the secret key if it's ever compromised
   - Keep your secret key at least 32 characters long

2. **Token Configuration**:
   - `ACCESS_TOKEN_EXPIRE_MINUTES`: Controls how long API tokens remain valid
   - Default is 30 minutes
   - Shorter duration (15 mins) = More secure
   - Longer duration (60 mins) = More convenient
   - Adjust based on your security requirements

3. **API Key Header**:
   - `API_KEY_HEADER`: Default is `X-API-Key`
   - This header is used for API authentication
   - Keep the default unless you have specific requirements

Example security configuration in `.env`:
```bash
# Security Settings
SECRET_KEY=r39pR2XJXhRLEt8rb4GlkTA5snI971VO5c2vF2FSzL0  # Generated secure key
API_KEY_HEADER=X-API-Key                                  # Default header
ACCESS_TOKEN_EXPIRE_MINUTES=30                            # Token lifetime
```

## SLURM Configuration

Cognomic supports SLURM cluster execution. To configure SLURM, create a `.cgat.yml` file in the project root directory:

```yaml
cluster:
  queue_manager: slurm
  queue: your_queue
  parallel_environment: smp

slurm:
  account: your_account
  partition: your_partition
  mail_user: your.email@example.com

tools:
  kallisto_index:
    memory: 16G
    threads: 8
    queue: short
```

### SLURM Integration

Cognomic uses CGATCore for SLURM integration, which provides:

1. **Job Management**
   - Automatic job submission and dependency tracking
   - Resource allocation (memory, CPUs, time limits)
   - Queue selection and prioritization

2. **Resource Configuration**
   - Tool-specific resource requirements in `.cgat.yml`
   - Queue-specific limits and settings
   - Default resource allocations

3. **Error Handling**
   - Automatic job resubmission on failure
   - Detailed error logging
   - Email notifications for job completion/failure

### SLURM Usage

To execute a workflow on a SLURM cluster, use the `--executor cgat` option:

```bash
python -m cognomic.cli "Analyze RNA-seq data in my fastq.gz files using Kallisto. The fastq files are in current directory and I want to use Homo_sapiens.GRCh38.cdna.all.fa as reference. The data is single ended. Generate QC reports and save everything in results/rna_seq_analysis." --workflow rnaseq --input data/ --executor cgat
```

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
python -m cognomic.cli "Analyze RNA-seq data in my fastq.gz files using Kallisto"

# SLURM cluster execution
python -m cognomic.cli --executor cgat "Analyze RNA-seq data in my fastq.gz files using Kallisto"
```

### Advanced Usage

1. Resume a failed workflow:
```bash
python -m cognomic.cli --resume --checkpoint-dir workflow_state "Your workflow prompt"
```

2. Specify custom resource requirements:
```bash
python -m cognomic.cli --executor cgat --memory 32G --threads 16 "Your workflow prompt"
```
