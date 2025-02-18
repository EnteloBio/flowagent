"""Metrics collection and monitoring for Cognomic."""
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

from prometheus_client import Counter, Gauge, Histogram, start_http_server

from ..config.settings import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """Collects and exposes metrics for monitoring."""

    def __init__(self) -> None:
        """Initialize metrics collector."""
        # Workflow metrics
        self.workflow_duration = Histogram(
            "workflow_duration_seconds",
            "Time spent executing workflows",
            ["workflow_type"]
        )
        self.workflow_status = Counter(
            "workflow_status_total",
            "Workflow execution status",
            ["workflow_type", "status"]
        )
        
        # Agent metrics
        self.agent_execution_duration = Histogram(
            "agent_execution_duration_seconds",
            "Time spent in agent execution",
            ["agent_name", "operation"]
        )
        self.agent_errors = Counter(
            "agent_errors_total",
            "Number of agent errors",
            ["agent_name", "error_type"]
        )
        
        # Resource metrics
        self.active_workflows = Gauge(
            "active_workflows",
            "Number of currently active workflows"
        )
        self.memory_usage = Gauge(
            "memory_usage_bytes",
            "Current memory usage"
        )
        
        # API metrics
        self.api_requests = Counter(
            "api_requests_total",
            "Number of API requests",
            ["endpoint", "method", "status"]
        )
        self.api_latency = Histogram(
            "api_latency_seconds",
            "API request latency",
            ["endpoint"]
        )

    def start_metrics_server(self) -> None:
        """Start the Prometheus metrics server."""
        if settings.ENABLE_MONITORING:
            try:
                start_http_server(settings.METRICS_PORT)
                logger.info(f"Metrics server started on port {settings.METRICS_PORT}")
            except Exception as e:
                logger.error(f"Failed to start metrics server: {str(e)}")

    @contextmanager
    def measure_workflow_duration(
        self, workflow_type: str
    ) -> Generator[None, None, None]:
        """Measure workflow execution duration."""
        start_time = time.time()
        self.active_workflows.inc()
        
        try:
            yield
            self.workflow_status.labels(
                workflow_type=workflow_type, status="success"
            ).inc()
        except Exception:
            self.workflow_status.labels(
                workflow_type=workflow_type, status="error"
            ).inc()
            raise
        finally:
            duration = time.time() - start_time
            self.workflow_duration.labels(workflow_type=workflow_type).observe(duration)
            self.active_workflows.dec()

    @contextmanager
    def measure_agent_duration(
        self, agent_name: str, operation: str
    ) -> Generator[None, None, None]:
        """Measure agent operation duration."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.agent_execution_duration.labels(
                agent_name=agent_name, operation=operation
            ).observe(duration)

    def record_agent_error(self, agent_name: str, error_type: str) -> None:
        """Record an agent error."""
        self.agent_errors.labels(
            agent_name=agent_name, error_type=error_type
        ).inc()

    def record_api_request(
        self, endpoint: str, method: str, status: int
    ) -> None:
        """Record an API request."""
        self.api_requests.labels(
            endpoint=endpoint, method=method, status=status
        ).inc()

    @contextmanager
    def measure_api_latency(self, endpoint: str) -> Generator[None, None, None]:
        """Measure API request latency."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.api_latency.labels(endpoint=endpoint).observe(duration)

    def update_memory_usage(self, usage_bytes: int) -> None:
        """Update memory usage metric."""
        self.memory_usage.set(usage_bytes)


# Global metrics collector instance
metrics = MetricsCollector()
