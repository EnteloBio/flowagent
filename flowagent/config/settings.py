"""Settings module for FlowAgent."""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import SecretStr, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file
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
    
    # OpenAI Settings
    OPENAI_API_KEY: Optional[str] = Field(None, description="OpenAI API Key")
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = Field('gpt-3.5-turbo', description="OpenAI Model to use")
    OPENAI_FALLBACK_MODEL: str = Field('gpt-3.5-turbo', description="Fallback model if primary is unavailable")
    
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
    MAX_RETRIES: int = Field(5, env='MAX_RETRIES')
    RETRY_DELAY: float = Field(2.0, env='RETRY_DELAY')
    TIMEOUT: float = 60.0
    REQUEST_INTERVAL: float = Field(1.0, env='REQUEST_INTERVAL')
    
    # Database Settings
    VECTOR_DB_PATH: Path = Field(default=Path("data/vector_store"), env='VECTOR_DB_PATH')
    SQLITE_DB_PATH: Path = Field(default=Path("data/flowagent.db"), env='SQLITE_DB_PATH')
    
    # Agent Settings
    MAX_CONCURRENT_WORKFLOWS: int = Field(5, env='MAX_CONCURRENT_WORKFLOWS')
    WORKFLOW_TIMEOUT: int = Field(3600, env='WORKFLOW_TIMEOUT')
    AGENT_TIMEOUT: float = Field(30.0, env='AGENT_TIMEOUT')
    AGENT_MAX_RETRIES: int = Field(3, env='AGENT_MAX_RETRIES')
    AGENT_RETRY_DELAY: float = Field(1.0, env='AGENT_RETRY_DELAY')
    
    # Workflow Settings
    DEFAULT_WORKFLOW_PARAMS: Dict[str, Any] = {
        "threads": 4,
        "memory": "16G"
    }
    
    # Executor settings
    EXECUTOR_TYPE: str = Field("local", env='EXECUTOR_TYPE')  # Options: local, hpc
    HPC_SYSTEM: str = Field("slurm", env='HPC_SYSTEM')  # Options: slurm, sge, torque
    HPC_QUEUE: str = Field("all.q", env='HPC_QUEUE')
    HPC_DEFAULT_MEMORY: str = Field("4G", env='HPC_DEFAULT_MEMORY')
    HPC_DEFAULT_CPUS: int = Field(1, env='HPC_DEFAULT_CPUS')
    HPC_DEFAULT_TIME: int = Field(60, env='HPC_DEFAULT_TIME')
    
    # Kubernetes Settings
    KUBERNETES_ENABLED: bool = Field(default=False, env="KUBERNETES_ENABLED")
    KUBERNETES_NAMESPACE: str = Field(default="default", env="KUBERNETES_NAMESPACE")
    KUBERNETES_SERVICE_ACCOUNT: str = Field(default="default", env="KUBERNETES_SERVICE_ACCOUNT")
    KUBERNETES_IMAGE: str = Field(default="python:3.9", env="KUBERNETES_IMAGE")
    KUBERNETES_CPU_REQUEST: str = Field(default="0.5", env="KUBERNETES_CPU_REQUEST")
    KUBERNETES_CPU_LIMIT: str = Field(default="1.0", env="KUBERNETES_CPU_LIMIT")
    KUBERNETES_MEMORY_REQUEST: str = Field(default="512Mi", env="KUBERNETES_MEMORY_REQUEST")
    KUBERNETES_MEMORY_LIMIT: str = Field(default="1Gi", env="KUBERNETES_MEMORY_LIMIT")
    KUBERNETES_JOB_TTL: int = Field(default=3600, env="KUBERNETES_JOB_TTL")
    
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
    ENABLE_MONITORING: bool = Field(True, env='ENABLE_MONITORING')
    METRICS_PORT: int = Field(9090, env='METRICS_PORT')
    LOG_LEVEL: str = Field("INFO", env='LOG_LEVEL')
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @property
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key."""
        return self.OPENAI_API_KEY
    
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
