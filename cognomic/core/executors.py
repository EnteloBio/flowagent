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

class KubernetesExecutor(BaseExecutor):
    """Execute workflow steps using Kubernetes Jobs."""
    
    def __init__(self):
        """Initialize Kubernetes executor."""
        self.settings = Settings()
        
        # Load Kubernetes configuration
        try:
            from kubernetes import client, config
            # Try to load from kube config file first
            try:
                config.load_kube_config()
            except:
                # If running inside cluster, load service account config
                config.load_incluster_config()
            
            self.k8s_batch = client.BatchV1Api()
            self.k8s_core = client.CoreV1Api()
            
            logger.info(
                f"Initialized Kubernetes executor with settings:\n"
                f"  Namespace: {self.settings.KUBERNETES_NAMESPACE}\n"
                f"  Service Account: {self.settings.KUBERNETES_SERVICE_ACCOUNT}\n"
                f"  Default Image: {self.settings.KUBERNETES_IMAGE}\n"
                f"  CPU Request/Limit: {self.settings.KUBERNETES_CPU_REQUEST}/{self.settings.KUBERNETES_CPU_LIMIT}\n"
                f"  Memory Request/Limit: {self.settings.KUBERNETES_MEMORY_REQUEST}/{self.settings.KUBERNETES_MEMORY_LIMIT}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {str(e)}")
            raise
    
    def _prepare_job_spec(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare Kubernetes Job specification."""
        from kubernetes import client
        
        # Get resource requirements from step
        resources = step.get("resources", {})
        
        # Use settings as defaults if not specified in step
        container = {
            "name": step["name"],
            "image": resources.get("image", self.settings.KUBERNETES_IMAGE),
            "command": ["sh", "-c", step["command"]],
            "resources": {
                "requests": {
                    "cpu": resources.get("cpu_request", self.settings.KUBERNETES_CPU_REQUEST),
                    "memory": resources.get("memory_request", self.settings.KUBERNETES_MEMORY_REQUEST)
                },
                "limits": {
                    "cpu": resources.get("cpu_limit", self.settings.KUBERNETES_CPU_LIMIT),
                    "memory": resources.get("memory_limit", self.settings.KUBERNETES_MEMORY_LIMIT)
                }
            }
        }
        
        # Add volume mounts if specified
        if volumes := step.get("volumes", []):
            container["volumeMounts"] = volumes
        
        # Add environment variables if specified
        if env := step.get("env", {}):
            container["env"] = [
                {"name": k, "value": v} for k, v in env.items()
            ]
        
        # Prepare job spec
        job_spec = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": f"flowagent-{step['name'].lower()}",
                "namespace": self.settings.KUBERNETES_NAMESPACE
            },
            "spec": {
                "template": {
                    "spec": {
                        "containers": [container],
                        "restartPolicy": "Never",
                        "serviceAccountName": self.settings.KUBERNETES_SERVICE_ACCOUNT
                    }
                },
                "backoffLimit": 0,  # Don't retry on failure
                "ttlSecondsAfterFinished": self.settings.KUBERNETES_JOB_TTL
            }
        }
        
        return job_spec
    
    async def execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow step using Kubernetes."""
        try:
            # Prepare job specification
            job_spec = self._prepare_job_spec(step)
            logger.info(f"Submitting step {step['name']} as Kubernetes Job")
            
            # Create the job
            job = self.k8s_batch.create_namespaced_job(
                namespace=self.settings.KUBERNETES_NAMESPACE,
                body=job_spec
            )
            
            return {
                "step_id": step["name"],
                "job_name": job.metadata.name,
                "status": "submitted",
                "command": step["command"],
                "job_spec": job_spec
            }
            
        except Exception as e:
            logger.error(f"Error submitting step {step['name']}: {str(e)}")
            raise
    
    async def wait_for_completion(self, jobs: Dict[str, Any]) -> Dict[str, Any]:
        """Wait for all Kubernetes jobs to complete."""
        try:
            results = {}
            
            for step_id, job_info in jobs.items():
                job_name = job_info["job_name"]
                
                # Wait for job completion
                while True:
                    job = self.k8s_batch.read_namespaced_job_status(
                        name=job_name,
                        namespace=self.settings.KUBERNETES_NAMESPACE
                    )
                    
                    if job.status.succeeded is not None or job.status.failed is not None:
                        break
                        
                    await asyncio.sleep(5)  # Check every 5 seconds
                
                # Get job logs
                pod = self.k8s_core.list_namespaced_pod(
                    namespace=self.settings.KUBERNETES_NAMESPACE,
                    label_selector=f"job-name={job_name}"
                ).items[0]
                
                logs = self.k8s_core.read_namespaced_pod_log(
                    name=pod.metadata.name,
                    namespace=self.settings.KUBERNETES_NAMESPACE
                )
                
                # Update job status
                results[step_id] = {
                    **job_info,
                    "status": "completed" if job.status.succeeded else "failed",
                    "completion_status": "succeeded" if job.status.succeeded else "failed",
                    "logs": logs
                }
                
                # Clean up the job if it's completed
                try:
                    self.k8s_batch.delete_namespaced_job(
                        name=job_name,
                        namespace=self.settings.KUBERNETES_NAMESPACE,
                        body=client.V1DeleteOptions(
                            propagation_policy='Background'
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to delete job {job_name}: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error waiting for jobs to complete: {str(e)}")
            raise
