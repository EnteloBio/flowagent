"""Workflow executors for different execution environments."""

from typing import Dict, Any, List, Optional
import asyncio
import logging
from pathlib import Path
import subprocess
from abc import ABC, abstractmethod

from cgatcore import pipeline as P
from ..utils.logging import get_logger

logger = get_logger(__name__)

class BaseExecutor(ABC):
    """Base class for workflow executors."""
    
    @abstractmethod
    async def execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step."""
        pass
        
    @abstractmethod
    async def wait_for_completion(self, jobs: Dict[str, Any]) -> Dict[str, Any]:
        """Wait for all jobs to complete and return results."""
        pass

class LocalExecutor(BaseExecutor):
    """Execute workflow steps locally using subprocess."""
    
    async def execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow step locally."""
        try:
            process = await asyncio.create_subprocess_shell(
                step["command"],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            return {
                "step_id": step["name"],
                "status": "completed" if process.returncode == 0 else "failed",
                "returncode": process.returncode,
                "stdout": stdout.decode(),
                "stderr": stderr.decode()
            }
        except Exception as e:
            logger.error(f"Error executing step {step['name']}: {str(e)}")
            raise
    
    async def wait_for_completion(self, jobs: Dict[str, Any]) -> Dict[str, Any]:
        """Local execution is synchronous, so just return results."""
        return jobs

class CGATExecutor(BaseExecutor):
    """Execute workflow steps using CGATCore pipeline."""
    
    def __init__(self):
        """Initialize CGATCore pipeline."""
        self.pipeline = P
        self.pipeline.start_pipeline()
        logger.info("Initialized CGATCore pipeline")
    
    def _prepare_job_options(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare job options for CGATCore."""
        params = step.get("parameters", {})
        
        # Default resource requirements
        memory = params.get("memory", "4G")
        if isinstance(memory, int):
            memory = f"{memory}G"
            
        options = {
            "job_name": step["name"],
            "job_memory": memory,
            "job_threads": params.get("threads", 1),
            "job_queue": params.get("queue", "all.q"),
        }
        
        # Add dependencies if present
        if deps := step.get("dependencies", []):
            options["job_depends"] = deps
            
        return options
    
    async def execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow step using CGATCore."""
        try:
            # Prepare job options
            job_options = self._prepare_job_options(step)
            logger.info(f"Submitting step {step['name']} with options: {job_options}")
            
            # Submit job via CGATCore
            statement = step["command"]
            job = self.pipeline.submit(
                statement,
                job_options=job_options
            )
            
            return {
                "step_id": step["name"],
                "job_id": job.jobid,
                "status": "submitted",
                "command": statement,
                "job_options": job_options
            }
            
        except Exception as e:
            logger.error(f"Error submitting step {step['name']}: {str(e)}")
            raise
    
    async def wait_for_completion(self, jobs: Dict[str, Any]) -> Dict[str, Any]:
        """Wait for all CGATCore jobs to complete."""
        try:
            # Wait for all jobs to complete
            self.pipeline.run()
            
            # Update job statuses
            results = {}
            for step_id, job_info in jobs.items():
                job_id = job_info["job_id"]
                status = self.pipeline.get_job_status(job_id)
                
                results[step_id] = {
                    **job_info,
                    "status": "completed" if status == "completed" else "failed",
                    "completion_status": status
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error waiting for jobs to complete: {str(e)}")
            raise
        finally:
            # Clean up pipeline
            self.pipeline.close_pipeline()
