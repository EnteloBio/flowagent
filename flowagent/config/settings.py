"""Settings module for FlowAgent."""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import SecretStr, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file
# First try the current directory
env_path = Path(".env")
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    logger = logging.getLogger(__name__)
    logger.info(f"Loaded .env file from current directory: {env_path.absolute()}")
# If USER_EXECUTION_DIR is set, also try that directory
elif "USER_EXECUTION_DIR" in os.environ:
    user_env_path = Path(os.environ["USER_EXECUTION_DIR"]) / ".env"
    if user_env_path.exists():
        load_dotenv(dotenv_path=user_env_path)
        logger = logging.getLogger(__name__)
        logger.info(f"Loaded .env file from USER_EXECUTION_DIR: {user_env_path.absolute()}")
else:
    # Default behavior if no specific path is found
    load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """Settings class for FlowAgent."""
    
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore', case_sensitive=False)
    
    # Application Settings
    APP_NAME: str = "FlowAgent"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    
    # Security
    SECRET_KEY: Optional[SecretStr] = None
    API_KEY_HEADER: str = "X-API-Key"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 3000
    
    # LLM Provider Settings (multi-provider)
    LLM_PROVIDER: str = Field("openai", description="LLM provider: openai | anthropic | google | ollama")
    LLM_MODEL: str = Field("gpt-4.1", description="Primary model (provider/model or bare name)")
    LLM_FALLBACK_MODEL: str = Field("gpt-4.1-mini", description="Fallback model if primary unavailable")
    LLM_BASE_URL: Optional[str] = Field(None, description="Custom endpoint URL (for Ollama/vLLM)")

    # Provider API keys
    OPENAI_API_KEY: Optional[str] = Field(None, description="OpenAI API Key")
    ANTHROPIC_API_KEY: Optional[str] = Field(None, description="Anthropic API Key")
    GOOGLE_API_KEY: Optional[str] = Field(None, description="Google Gemini API Key")

    # Legacy OpenAI settings (still read for backwards compatibility)
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = Field('gpt-4.1', description="OpenAI Model to use (legacy, prefer LLM_MODEL)")
    OPENAI_FALLBACK_MODEL: str = Field('gpt-4.1-mini', description="Fallback model (legacy, prefer LLM_FALLBACK_MODEL)")
    
    @field_validator('OPENAI_API_KEY')
    def validate_openai_key(cls, v):
        if not v:
            # Try to get from environment directly
            v = os.getenv('OPENAI_API_KEY')
            
        if not v:
            env_path = Path('.env')
            if not env_path.exists():
                logger.warning(
                    "\n⚠️  No .env file found in the current directory."
                    "\n   Please create a .env file with your OpenAI API key:"
                    "\n   OPENAI_API_KEY=your-api-key-here"
                    "\n   OPENAI_MODEL=gpt-4 (optional)"
                )
            else:
                logger.warning(
                    "\n⚠️  OPENAI_API_KEY not found in environment variables or .env file."
                    "\n   Please add your OpenAI API key to the .env file:"
                    "\n   OPENAI_API_KEY=your-api-key-here"
                )
        return v
    
    # Rate Limiting Settings
    MAX_RETRIES: int = 5
    RETRY_DELAY: float = 2.0
    TIMEOUT: float = 60.0
    REQUEST_INTERVAL: float = 1.0

    # Database Settings
    VECTOR_DB_PATH: Path = Path("data/vector_store")
    SQLITE_DB_PATH: Path = Path("data/flowagent.db")

    # Agent Settings
    MAX_CONCURRENT_WORKFLOWS: int = 5
    WORKFLOW_TIMEOUT: int = 3600
    AGENT_TIMEOUT: float = 30.0
    AGENT_MAX_RETRIES: int = 3
    AGENT_RETRY_DELAY: float = 1.0

    # Workflow Settings
    DEFAULT_WORKFLOW_PARAMS: Dict[str, Any] = {
        "threads": 4,
        "memory": "16G"
    }

    # Pipeline generation settings
    PIPELINE_FORMAT: str = Field("shell", description="Pipeline format: shell | nextflow | snakemake")
    PIPELINE_PROFILE: str = Field("local", description="Nextflow profile")
    CONTAINER_ENGINE: str = Field("docker", description="Container engine: docker | singularity | conda")
    AUTO_EXECUTE_PIPELINE: bool = Field(True, description="Auto-execute generated pipelines")

    # Executor settings
    EXECUTOR_TYPE: str = "local"
    HPC_SYSTEM: str = "slurm"
    HPC_QUEUE: str = "all.q"
    HPC_DEFAULT_MEMORY: str = "4G"
    HPC_DEFAULT_CPUS: int = 1
    HPC_DEFAULT_TIME: int = 60

    # Kubernetes Settings
    KUBERNETES_ENABLED: bool = False
    KUBERNETES_NAMESPACE: str = "default"
    KUBERNETES_SERVICE_ACCOUNT: str = "default"
    KUBERNETES_IMAGE: str = "python:3.11"
    KUBERNETES_CPU_REQUEST: str = "0.5"
    KUBERNETES_CPU_LIMIT: str = "1.0"
    KUBERNETES_MEMORY_REQUEST: str = "512Mi"
    KUBERNETES_MEMORY_LIMIT: str = "1Gi"
    KUBERNETES_JOB_TTL: int = 3600

    # Tool Settings
    REQUIRED_TOOLS: List[str] = [
        "kallisto",
        "fastqc",
        "multiqc",
        "cellranger",
        "seurat"
    ]
    TOOL_VERSIONS: Dict[str, str] = {
        "kallisto": "0.46.1",
        "fastqc": "0.11.9",
        "multiqc": "1.11",
        "cellranger": "7.0.0",
        "seurat": "4.3.0"
    }

    # Monitoring
    ENABLE_MONITORING: bool = True
    METRICS_PORT: int = 9090
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @property
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key."""
        return self.OPENAI_API_KEY

    @property
    def active_api_key(self) -> Optional[str]:
        """Get the API key for the currently-configured LLM provider."""
        key_map = {
            "openai": self.OPENAI_API_KEY,
            "anthropic": self.ANTHROPIC_API_KEY,
            "google": self.GOOGLE_API_KEY,
            "ollama": "ollama",
        }
        return key_map.get(self.LLM_PROVIDER.lower())
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT.lower() == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT.lower() == "production"
    
    @field_validator('LOG_LEVEL')
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v.upper()
    
    def get_tool_version(self, tool_name: str) -> str:
        """Get required version for a tool."""
        return self.TOOL_VERSIONS.get(tool_name, "latest")
    
    def is_tool_required(self, tool_name: str) -> bool:
        """Check if a tool is required."""
        return tool_name in self.REQUIRED_TOOLS
    
    def get_workflow_param(self, param_name: str, default: Any = None) -> Any:
        """Get default workflow parameter."""
        return self.DEFAULT_WORKFLOW_PARAMS.get(param_name, default)

# Create global settings instance
settings = Settings()
