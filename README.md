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

# Create and activate Conda environment
conda install mamba 
mamba env create -f environment.yml
mamba activate cognomic

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
cp env.example /path/to/your/.env
# Edit .env with your configuration
```

2. Run the server:
```bash
cognomic-server
```

3. Execute a workflow:
```bash
cognomic-run --workflow rnaseq --input data/
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

## Architecture

Cognomic 2.0 implements a modern, distributed architecture:

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
