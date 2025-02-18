"""Settings module for Cognomic."""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import SecretStr, Field
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
    VECTOR_DB_PATH: Path = Path("data/vector_store")
    SQLITE_DB_PATH: Path = Path("data/cognomic.db")
    
    # Agent Settings
    MAX_CONCURRENT_WORKFLOWS: int = 5
    WORKFLOW_TIMEOUT: int = 3600
    
    # Monitoring
    ENABLE_MONITORING: bool = True
    METRICS_PORT: int = 9090
    LOG_LEVEL: str = "INFO"
    
    @property
    def openai_api_key(self) -> Optional[str]:
        """Get the OpenAI API key."""
        return self.OPENAI_API_KEY
    
    @property
    def openai_model_to_use(self) -> str:
        """Get the OpenAI model to use, falling back to gpt-3.5-turbo if gpt-4 is not available."""
        return self.OPENAI_MODEL if self.OPENAI_MODEL != 'gpt-4' else self.OPENAI_FALLBACK_MODEL
        
    def __init__(self, **kwargs):
        """Initialize settings with environment variables."""
        super().__init__(**kwargs)
        # Load environment variables
        if os.path.exists('.env'):
            from dotenv import load_dotenv
            load_dotenv()
            
        # Update settings from environment
        for field in self.model_fields:
            env_val = os.getenv(field)
            if env_val is not None:
                setattr(self, field, env_val)

# Create global settings instance
settings = Settings()
