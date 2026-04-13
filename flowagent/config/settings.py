"""Settings module for FlowAgent."""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(".env")
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
elif "USER_EXECUTION_DIR" in os.environ:
    user_env_path = Path(os.environ["USER_EXECUTION_DIR"]) / ".env"
    if user_env_path.exists():
        load_dotenv(dotenv_path=user_env_path)
else:
    load_dotenv()

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Settings class for FlowAgent."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        extra="ignore", case_sensitive=False,
    )

    # ── LLM Provider ─────────────────────────────────────────
    LLM_PROVIDER: str = Field("openai", description="LLM provider: openai | anthropic | google | ollama")
    LLM_MODEL: str = Field("gpt-5.4-mini", description="Primary model")
    LLM_FALLBACK_MODEL: str = Field("gpt-4.1-mini", description="Fallback model if primary unavailable")
    LLM_BASE_URL: Optional[str] = Field(None, description="Custom endpoint URL (for Ollama/vLLM)")

    OPENAI_API_KEY: Optional[str] = Field(None, description="OpenAI API Key")
    ANTHROPIC_API_KEY: Optional[str] = Field(None, description="Anthropic API Key")
    GOOGLE_API_KEY: Optional[str] = Field(None, description="Google Gemini API Key")

    # Legacy OpenAI settings (backwards compatibility)
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = Field("gpt-5.4-mini", description="Legacy: prefer LLM_MODEL")
    OPENAI_FALLBACK_MODEL: str = Field("gpt-4.1-mini", description="Legacy: prefer LLM_FALLBACK_MODEL")

    @field_validator("OPENAI_API_KEY")
    def validate_openai_key(cls, v):
        if not v:
            v = os.getenv("OPENAI_API_KEY")
        if not v:
            logger.warning(
                "No OPENAI_API_KEY found. Set it in .env or environment variables."
            )
        return v

    # ── Rate Limiting ────────────────────────────────────────
    MAX_RETRIES: int = 5
    RETRY_DELAY: float = 2.0
    TIMEOUT: float = 60.0

    # ── Workflow ─────────────────────────────────────────────
    MAX_CONCURRENT_WORKFLOWS: int = 5
    WORKFLOW_TIMEOUT: int = 3600
    DEFAULT_WORKFLOW_PARAMS: Dict[str, Any] = {
        "threads": 4,
        "memory": "16G",
    }

    # ── Pipeline Generation ──────────────────────────────────
    PIPELINE_FORMAT: str = Field("shell", description="shell | nextflow | snakemake")
    PIPELINE_PROFILE: str = Field("local", description="Nextflow profile (local, docker, singularity, slurm)")

    # ── Executor ─────────────────────────────────────────────
    EXECUTOR_TYPE: str = "local"
    HPC_SYSTEM: str = "slurm"
    HPC_QUEUE: str = "all.q"
    HPC_DEFAULT_MEMORY: str = "4G"
    HPC_DEFAULT_CPUS: int = 1
    HPC_DEFAULT_TIME: int = 60

    # Aliases used by CGATExecutor
    SLURM_QUEUE: str = "all.q"
    SLURM_DEFAULT_MEMORY: str = "4G"
    SLURM_DEFAULT_CPUS: int = 1

    # ── Kubernetes ───────────────────────────────────────────
    KUBERNETES_ENABLED: bool = False
    KUBERNETES_NAMESPACE: str = "default"
    KUBERNETES_SERVICE_ACCOUNT: str = "default"
    KUBERNETES_IMAGE: str = "python:3.11"
    KUBERNETES_CPU_REQUEST: str = "0.5"
    KUBERNETES_CPU_LIMIT: str = "1.0"
    KUBERNETES_MEMORY_REQUEST: str = "512Mi"
    KUBERNETES_MEMORY_LIMIT: str = "1Gi"
    KUBERNETES_JOB_TTL: int = 3600

    # ── Monitoring ───────────────────────────────────────────
    ENABLE_MONITORING: bool = True
    METRICS_PORT: int = 9090
    LOG_LEVEL: str = "INFO"

    @field_validator("LOG_LEVEL")
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v.upper()

    # ── Properties ───────────────────────────────────────────

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

    def get_workflow_param(self, param_name: str, default: Any = None) -> Any:
        """Get default workflow parameter."""
        return self.DEFAULT_WORKFLOW_PARAMS.get(param_name, default)


# Create global settings instance
settings = Settings()
