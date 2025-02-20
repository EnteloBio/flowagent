"""Settings module for Cognomic."""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import SecretStr, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Settings class for Cognomic."""
    
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')
    
    # Application Settings
    APP_NAME: str = "Cognomic"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    
    # Security
    SECRET_KEY: SecretStr
    API_KEY_HEADER: str = "X-API-Key"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 3000
    
    # OpenAI Settings
    OPENAI_API_KEY: Optional[str] = Field(None, env='OPENAI_API_KEY')
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = Field('gpt-4', env='OPENAI_MODEL')
    OPENAI_FALLBACK_MODEL: str = Field('gpt-3.5-turbo', env='OPENAI_FALLBACK_MODEL')
    
    # Rate Limiting Settings
    MAX_RETRIES: int = Field(5, env='MAX_RETRIES')
    RETRY_DELAY: float = Field(2.0, env='RETRY_DELAY')
    TIMEOUT: float = 60.0
    REQUEST_INTERVAL: float = Field(1.0, env='REQUEST_INTERVAL')
    
    # Database Settings
    VECTOR_DB_PATH: Path = Field(default=Path("data/vector_store"), env='VECTOR_DB_PATH')
    SQLITE_DB_PATH: Path = Field(default=Path("data/cognomic.db"), env='SQLITE_DB_PATH')
    
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
