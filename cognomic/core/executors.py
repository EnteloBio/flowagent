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
        
        # Get resource requirements, ensuring numeric values
        resources = step.get("resources", {})
        required_memory_mb = int(resources.get("memory_mb", 4000))
        required_cpus = int(resources.get("cpus", 1))
        
        # Get system resources
        system_memory_mb = int(psutil.virtual_memory().total / (1024 * 1024))  # Convert bytes to MB
        system_cpus = psutil.cpu_count()
        
        # Log resource requirements and availability
        logger.info(
            f"Step {step['name']} resource requirements:\n"
            f"  - Memory: {required_memory_mb:,}MB / {system_memory_mb:,}MB available\n"
            f"  - CPUs: {required_cpus} / {system_cpus} available\n"
            f"  - Profile: {resources.get('profile', 'default')}"
        )
        
        # Warn if resources might be insufficient
        if required_memory_mb >= int(system_memory_mb * 0.9):  # 90% of total memory
            logger.warning(
                f"Step {step['name']} requires {required_memory_mb:,}MB memory, "
                f"which is close to or exceeds system memory ({system_memory_mb:,}MB)"
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

class KubernetesExecutor(BaseExecutor):
    """Execute workflow steps using Kubernetes Jobs."""
    
    def __init__(self, 
                 namespace: str = "default",
                 image: str = "ubuntu:latest",
                 pull_policy: str = "IfNotPresent",
                 service_account: Optional[str] = None):
        """Initialize Kubernetes executor.
        
        Args:
            namespace: Kubernetes namespace to run jobs in
            image: Default container image to use
            pull_policy: Image pull policy
            service_account: Optional service account for jobs
        """
        try:
            from kubernetes import client, config
            
            # Load kube config
            try:
                config.load_incluster_config()
                logger.info("Using in-cluster Kubernetes configuration")
            except config.ConfigException:
                config.load_kube_config()
                logger.info("Using local Kubernetes configuration")
            
            self.k8s_batch = client.BatchV1Api()
            self.k8s_core = client.CoreV1Api()
            
        except ImportError:
            raise ImportError(
                "kubernetes package not found. Install with: pip install kubernetes"
            )
        
        self.namespace = namespace
        self.default_image = image
        self.pull_policy = pull_policy
        self.service_account = service_account
        
        logger.info(
            f"Initialized KubernetesExecutor:\n"
            f"  Namespace: {namespace}\n"
            f"  Default image: {image}\n"
            f"  Pull policy: {pull_policy}\n"
            f"  Service account: {service_account or 'default'}"
        )
    
    def _prepare_job_spec(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare Kubernetes job specification."""
        from kubernetes import client
        
        # Get resource requirements
        resources = step.get("resources", {})
        memory_mb = resources.get("memory_mb", 4000)
        cpus = resources.get("cpus", 1)
        
        # Convert memory to Kubernetes format (Mi)
        memory = f"{memory_mb}Mi"
        
        # Get container configuration
        container_config = step.get("container", {})
        image = container_config.get("image", self.default_image)
        
        # Create container spec
        container = client.V1Container(
            name=step["name"].lower().replace("_", "-"),
            image=image,
            image_pull_policy=self.pull_policy,
            command=["/bin/sh", "-c", step["command"]],
            resources=client.V1ResourceRequirements(
                requests={
                    "memory": memory,
                    "cpu": str(cpus)
                },
                limits={
                    "memory": memory,
                    "cpu": str(cpus)
                }
            )
        )
        
        # Create pod spec
        pod_spec = client.V1PodSpec(
            restart_policy="Never",
            containers=[container],
            service_account_name=self.service_account
        )
        
        # Create job spec
        job_spec = client.V1JobSpec(
            template=client.V1PodTemplateSpec(
                spec=pod_spec
            ),
            backoff_limit=0  # Don't retry on failure
        )
        
        # Add job metadata
        metadata = client.V1ObjectMeta(
            name=f"{step['name'].lower().replace('_', '-')}-{int(time.time())}",
            labels={
                "app": "cognomic",
                "workflow-step": step["name"]
            }
        )
        
        return client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=metadata,
            spec=job_spec
        )
    
    async def execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow step as a Kubernetes job."""
        try:
            # Prepare job specification
            job = self._prepare_job_spec(step)
            
            # Log job creation
            logger.info(
                f"Creating Kubernetes job for step {step['name']}:\n"
                f"  Image: {job.spec.template.spec.containers[0].image}\n"
                f"  Resources: {job.spec.template.spec.containers[0].resources.requests}"
            )
            
            # Create job
            api_response = self.k8s_batch.create_namespaced_job(
                namespace=self.namespace,
                body=job
            )
            
            return {
                "step_id": step["name"],
                "status": "submitted",
                "job_name": api_response.metadata.name,
                "resources": step.get("resources", {}),
                "command": step["command"],
                "container": step.get("container", {})
            }
            
        except Exception as e:
            error_msg = f"Error creating Kubernetes job for step {step['name']}: {str(e)}"
            logger.error(error_msg)
            return {
                "step_id": step["name"],
                "status": "failed",
                "error": error_msg
            }
    
    async def wait_for_completion(self, jobs: Dict[str, Any]) -> Dict[str, Any]:
        """Wait for all Kubernetes jobs to complete."""
        try:
            results = {}
            
            for step_id, job_info in jobs.items():
                if job_info["status"] == "failed":
                    results[step_id] = job_info
                    continue
                
                job_name = job_info["job_name"]
                logger.info(f"Waiting for job {job_name} to complete...")
                
                # Wait for job completion
                while True:
                    job = self.k8s_batch.read_namespaced_job_status(
                        name=job_name,
                        namespace=self.namespace
                    )
                    
                    if job.status.succeeded:
                        status = "completed"
                        break
                    elif job.status.failed:
                        status = "failed"
                        break
                    
                    await asyncio.sleep(10)  # Check every 10 seconds
                
                # Get pod logs if available
                selector = f"job-name={job_name}"
                pods = self.k8s_core.list_namespaced_pod(
                    namespace=self.namespace,
                    label_selector=selector
                )
                
                logs = ""
                if pods.items:
                    try:
                        logs = self.k8s_core.read_namespaced_pod_log(
                            name=pods.items[0].metadata.name,
                            namespace=self.namespace
                        )
                    except Exception as e:
                        logger.warning(f"Could not retrieve logs for job {job_name}: {e}")
                
                # Store results
                results[step_id] = {
                    **job_info,
                    "status": status,
                    "completion_time": job.status.completion_time,
                    "logs": logs
                }
                
                # Clean up job
                try:
                    self.k8s_batch.delete_namespaced_job(
                        name=job_name,
                        namespace=self.namespace,
                        body=client.V1DeleteOptions(
                            propagation_policy="Background"
                        )
                    )
                except Exception as e:
                    logger.warning(f"Error cleaning up job {job_name}: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error waiting for jobs to complete: {str(e)}")
            return jobs
