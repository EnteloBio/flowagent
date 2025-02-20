"""Base agent implementation with configuration support."""

from typing import Dict, Any, Optional
import asyncio
from abc import ABC, abstractmethod

from ..utils.logging import get_logger
from ..config.settings import settings

class BaseAgent(ABC):
    """Base class for all agents."""
    
    def __init__(self, name: str):
        """Initialize base agent."""
        self.name = name
        self.logger = get_logger(
            f"{__name__}.{self.__class__.__name__}",
            level=settings.LOG_LEVEL
        )
        self.max_retries = settings.AGENT_MAX_RETRIES
        self.retry_delay = settings.AGENT_RETRY_DELAY
        self.timeout = settings.AGENT_TIMEOUT
    
    async def execute_with_retry(self, coro, operation: str):
        """Execute coroutine with retry logic."""
        for attempt in range(self.max_retries):
            try:
                return await asyncio.wait_for(coro, timeout=self.timeout)
            except asyncio.TimeoutError:
                if attempt < self.max_retries - 1:
                    self.logger.warning(
                        f"{operation} timed out after {self.timeout} seconds. "
                        f"Attempt {attempt + 1}/{self.max_retries}"
                    )
                else:
                    raise TimeoutError(
                        f"{operation} timed out after {self.timeout} seconds "
                        f"and {self.max_retries} attempts"
                    )
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (attempt + 1)
                    self.logger.warning(
                        f"{operation} failed (attempt {attempt + 1}/{self.max_retries}). "
                        f"Retrying in {delay} seconds. Error: {str(e)}"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise
    
    @abstractmethod
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent operation."""
        pass
    
    @abstractmethod
    async def validate(self, result: Dict[str, Any]) -> bool:
        """Validate operation result."""
        pass
    
    def _log_start(self, operation: str, params: Dict[str, Any]):
        """Log operation start."""
        if settings.DEBUG:
            self.logger.debug(
                f"Starting {operation} with parameters: {params}"
            )
        else:
            self.logger.info(f"Starting {operation}")
    
    def _log_complete(self, operation: str, result: Dict[str, Any]):
        """Log operation completion."""
        if settings.DEBUG:
            self.logger.debug(
                f"Completed {operation} with result: {result}"
            )
        else:
            self.logger.info(f"Completed {operation}")
    
    def _log_error(self, operation: str, error: Exception):
        """Log operation error."""
        self.logger.error(
            f"Error in {operation}: {str(error)}",
            exc_info=settings.DEBUG
        )
