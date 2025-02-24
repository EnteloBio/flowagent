"""Workflow executors for different execution environments."""

from typing import Dict, Any, List, Optional
import asyncio
import logging
from pathlib import Path
import subprocess
from abc import ABC, abstractmethod

from cgatcore import pipeline as P
from ..utils.logging import get_logger
from cognomic.config.settings import Settings

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
        self.settings = Settings()
        logger.info(
            f"Initialized CGATCore pipeline with settings:\n"
            f"  Queue: {self.settings.SLURM_QUEUE}\n"
            f"  Default Memory: {self.settings.SLURM_DEFAULT_MEMORY}\n"
            f"  Default CPUs: {self.settings.SLURM_DEFAULT_CPUS}"
        )
    
    def _prepare_job_options(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare job options for CGATCore."""
        # Get resource requirements from step
        resources = step.get("resources", {})
        
        # Use settings as defaults if not specified in step
        memory = resources.get("memory", self.settings.SLURM_DEFAULT_MEMORY)
        cpus = resources.get("cpus", self.settings.SLURM_DEFAULT_CPUS)
        queue = resources.get("queue", self.settings.SLURM_QUEUE)
        
        options = {
            "job_name": step["name"],
            "job_memory": memory,
            "job_threads": cpus,
            "job_time": resources.get("time_min", 60),  # Time limit in minutes
            "job_queue": queue,
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

class HPCExecutor(BaseExecutor):
    """Execute workflow steps using various HPC systems (SLURM, SGE, TORQUE)."""
    
    def __init__(self):
        """Initialize HPC executor."""
        self.settings = Settings()
        self.hpc_system = self.settings.HPC_SYSTEM.lower()
        
        # Initialize appropriate backend
        if self.hpc_system == "slurm":
            from cgatcore import pipeline as P
            self.backend = P
            self.backend.start_pipeline()
        elif self.hpc_system == "sge":
            import drmaa
            self.backend = drmaa.Session()
            self.backend.initialize()
        elif self.hpc_system == "torque":
            import drmaa
            self.backend = drmaa.Session()
            self.backend.initialize()
        else:
            raise ValueError(f"Unsupported HPC system: {self.hpc_system}")
            
        logger.info(
            f"Initialized {self.hpc_system.upper()} executor with settings:\n"
            f"  Queue: {self.settings.HPC_QUEUE}\n"
            f"  Default Memory: {self.settings.HPC_DEFAULT_MEMORY}\n"
            f"  Default CPUs: {self.settings.HPC_DEFAULT_CPUS}\n"
            f"  Default Time: {self.settings.HPC_DEFAULT_TIME} minutes"
        )
    
    def _prepare_job_options(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare job options for the HPC system."""
        # Get resource requirements from step
        resources = step.get("resources", {})
        
        # Use settings as defaults if not specified in step
        memory = resources.get("memory", self.settings.HPC_DEFAULT_MEMORY)
        cpus = resources.get("cpus", self.settings.HPC_DEFAULT_CPUS)
        queue = resources.get("queue", self.settings.HPC_QUEUE)
        time = resources.get("time_min", self.settings.HPC_DEFAULT_TIME)
        
        if self.hpc_system == "slurm":
            return {
                "job_name": step["name"],
                "job_memory": memory,
                "job_threads": cpus,
                "job_time": time,
                "job_queue": queue,
            }
        elif self.hpc_system == "sge":
            return {
                "-N": step["name"],
                "-l": f"h_vmem={memory},cpu={cpus}",
                "-q": queue,
                "-l": f"h_rt={time}:00",
            }
        elif self.hpc_system == "torque":
            return {
                "-N": step["name"],
                "-l": f"mem={memory},nodes=1:ppn={cpus}",
                "-q": queue,
                "-l": f"walltime={time}:00",
            }
    
    async def execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow step on the HPC system."""
        try:
            job_options = self._prepare_job_options(step)
            logger.info(f"Submitting step {step['name']} with options: {job_options}")
            
            if self.hpc_system == "slurm":
                # Use CGAT pipeline for SLURM
                job = self.backend.submit(
                    step["command"],
                    job_options=job_options,
                    to_cluster=True
                )
                job_id = job.jobid
            else:
                # Use DRMAA for SGE and TORQUE
                jt = self.backend.createJobTemplate()
                jt.remoteCommand = "/bin/bash"
                jt.args = ["-c", step["command"]]
                jt.nativeSpecification = " ".join([f"{k} {v}" for k, v in job_options.items()])
                job_id = self.backend.runJob(jt)
                self.backend.deleteJobTemplate(jt)
            
            return {
                "step_id": step["name"],
                "job_id": job_id,
                "status": "submitted",
                "command": step["command"],
                "job_options": job_options,
                "hpc_system": self.hpc_system
            }
            
        except Exception as e:
            error_msg = f"Error submitting job for step {step['name']}: {str(e)}"
            logger.error(error_msg)
            return {
                "step_id": step["name"],
                "status": "failed",
                "error": error_msg
            }
    
    async def wait_for_completion(self, jobs: Dict[str, Any]) -> Dict[str, Any]:
        """Wait for all HPC jobs to complete."""
        try:
            results = {}
            
            if self.hpc_system == "slurm":
                # Use CGAT pipeline for SLURM
                self.backend.run()
                for step_id, job_info in jobs.items():
                    results[step_id] = {
                        **job_info,
                        "status": "completed"
                    }
            else:
                # Use DRMAA for SGE and TORQUE
                for step_id, job_info in jobs.items():
                    if job_info["status"] == "failed":
                        results[step_id] = job_info
                        continue
                    
                    job_id = job_info["job_id"]
                    retval = self.backend.wait(job_id, drmaa.Session.TIMEOUT_WAIT_FOREVER)
                    
                    results[step_id] = {
                        **job_info,
                        "status": "completed" if retval.hasExited and retval.exitStatus == 0 else "failed",
                        "exit_status": retval.exitStatus
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Error waiting for jobs to complete: {str(e)}")
            return jobs
        finally:
            if self.hpc_system != "slurm":
                self.backend.exit()
