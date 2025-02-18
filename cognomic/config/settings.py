"""Configuration management for Cognomic."""
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseSettings, SecretStr, validator


class Settings(BaseSettings):
    """Global settings for Cognomic."""
    
    # Application Settings
    APP_NAME: str = "Cognomic"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    
    # Security
    SECRET_KEY: SecretStr
    API_KEY_HEADER: str = "X-API-Key"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # OpenAI Settings
    OPENAI_API_KEY: SecretStr
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-4"
    
    # Database Settings
    VECTOR_DB_PATH: Path = Path("data/vector_store")
    SQLITE_DB_PATH: Path = Path("data/cognomic.db")
    
    # Agent Settings
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    TIMEOUT: float = 30.0
    
    # Workflow Settings
    MAX_CONCURRENT_WORKFLOWS: int = 5
    WORKFLOW_TIMEOUT: int = 3600  # 1 hour
    
    # Monitoring
    ENABLE_MONITORING: bool = True
    METRICS_PORT: int = 9090
    LOG_LEVEL: str = "INFO"
    
    @validator("VECTOR_DB_PATH", "SQLITE_DB_PATH")
    def create_directories(cls, v: Path) -> Path:
        """Ensure directories exist."""
        v.parent.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()

# Environment-specific settings
ENV_SETTINGS: Dict[str, Dict[str, Any]] = {
    "development": {
        "DEBUG": True,
        "LOG_LEVEL": "DEBUG",
    },
    "testing": {
        "DEBUG": True,
        "VECTOR_DB_PATH": Path("tests/data/vector_store"),
        "SQLITE_DB_PATH": Path("tests/data/cognomic.db"),
    },
    "production": {
        "DEBUG": False,
        "MAX_RETRIES": 5,
        "RETRY_DELAY": 2.0,
        "LOG_LEVEL": "WARNING",
    },
}
