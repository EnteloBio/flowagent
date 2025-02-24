"""Workflow executors for different execution environments."""

from typing import Dict, Any, List, Optional
import asyncio
import logging
import time
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
    
    def _check_resources(self, step: Dict[str, Any]) -> None:
        """Check if local resources are sufficient for the step."""
        import psutil
        
        resources = step.get("resources", {})
        required_memory_mb = resources.get("memory_mb", 4000)
        required_cpus = resources.get("cpus", 1)
        
        # Get system resources
        system_memory_mb = psutil.virtual_memory().total / (1024 * 1024)  # Convert bytes to MB
        system_cpus = psutil.cpu_count()
        
        # Log resource requirements and availability
        logger.info(
            f"Step {step['name']} resource requirements:\n"
            f"  - Memory: {required_memory_mb:.0f}MB / {system_memory_mb:.0f}MB available\n"
            f"  - CPUs: {required_cpus} / {system_cpus} available"
        )
        
        # Warn if resources might be insufficient
        if required_memory_mb > system_memory_mb * 0.9:  # 90% of total memory
            logger.warning(
                f"Step {step['name']} requires {required_memory_mb:.0f}MB memory, "
                f"which is close to or exceeds system memory ({system_memory_mb:.0f}MB)"
            )
        
        if required_cpus > system_cpus:
            logger.warning(
                f"Step {step['name']} requests {required_cpus} CPUs, "
                f"but only {system_cpus} are available"
            )
    
    async def execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow step locally."""
        try:
            # Check resource requirements
            self._check_resources(step)
            
            # Log step execution
            logger.info(f"Executing step: {step['name']}")
            logger.debug(f"Command: {step['command']}")
            
            start_time = time.time()
            process = await asyncio.create_subprocess_shell(
                step["command"],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            execution_time = time.time() - start_time
            
            # Convert output to string and truncate if too long
            stdout_str = stdout.decode()
            stderr_str = stderr.decode()
            if len(stdout_str) > 1000:
                stdout_str = stdout_str[:500] + "\n...[truncated]...\n" + stdout_str[-500:]
            if len(stderr_str) > 1000:
                stderr_str = stderr_str[:500] + "\n...[truncated]...\n" + stderr_str[-500:]
            
            status = "completed" if process.returncode == 0 else "failed"
            
            # Log completion status and time
            logger.info(
                f"Step {step['name']} {status} in {execution_time:.1f}s "
                f"(return code: {process.returncode})"
            )
            if status == "failed":
                logger.error(f"Step {step['name']} failed with error:\n{stderr_str}")
            
            return {
                "step_id": step["name"],
                "status": status,
                "returncode": process.returncode,
                "stdout": stdout_str,
                "stderr": stderr_str,
                "execution_time": execution_time,
                "resources": step.get("resources", {}),  # Include resource info in result
                "command": step["command"]
            }
        except Exception as e:
            error_msg = f"Error executing step {step['name']}: {str(e)}"
            logger.error(error_msg)
            return {
                "step_id": step["name"],
                "status": "failed",
                "error": error_msg,
                "resources": step.get("resources", {})
            }
    
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
        # Get resource requirements from step
        resources = step.get("resources", {})
        
        # Convert memory from MB to GB and format as string
        memory_gb = resources.get("memory_mb", 4000) / 1024
        memory = f"{int(memory_gb)}G"
        
        options = {
            "job_name": step["name"],
            "job_memory": memory,
            "job_threads": resources.get("cpus", 1),
            "job_time": resources.get("time_min", 60),  # Time limit in minutes
            "job_queue": step.get("queue", "all.q"),
        }
        
        # Add dependencies if present
        if deps := step.get("dependencies", []):
            options["job_depends"] = deps
            
        # Log resource allocation
        logger.info(f"Resource allocation for {step['name']}: {memory} RAM, {options['job_threads']} cores")
            
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
                job_options=job_options,
                to_cluster=True  # Ensure job goes to SLURM
            )
            
            return {
                "step_id": step["name"],
                "job_id": job.jobid,
                "status": "submitted",
                "command": statement,
                "job_options": job_options,
                "resources": step.get("resources", {})  # Include resource allocation in job info
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
