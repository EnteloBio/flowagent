"""
Executor module for FlowAgent.

This module contains the Executor class that is responsible for executing
individual workflow steps.
"""

import os
import sys
import json
import logging
import asyncio
import subprocess
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

class Executor:
    """Executor for workflow steps."""
    
    def __init__(self, executor_type: str = "local"):
        """Initialize the executor.
        
        Args:
            executor_type: Type of executor to use (local, slurm, etc.)
        """
        self.executor_type = executor_type
        self.logger = logging.getLogger(__name__)
    
    async def execute_step(self, step: Dict[str, Any], output_dir: str, cwd: Optional[str] = None) -> Dict[str, Any]:
        """Execute a single workflow step.
        
        Args:
            step: Step to execute
            output_dir: Output directory for the workflow
            cwd: Working directory for command execution (defaults to output_dir)
            
        Returns:
            Dictionary with execution results
        """
        step_name = step.get("name", "Unnamed step")
        command = step.get("command", "")
        
        if not command:
            self.logger.warning(f"Step {step_name} has no command, skipping")
            return {
                "status": "skipped",
                "step": step,
                "step_name": step_name,
                "reason": "No command specified"
            }
        
        self.logger.info(f"Executing step: {step_name}")
        self.logger.info(f"Command: {command}")
        
        # Create working directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Use provided working directory or default to output_dir
        working_dir = cwd if cwd else output_dir
        
        # Execute the command
        if self.executor_type == "local":
            return await self._execute_local(step, command, output_dir, working_dir)
        elif self.executor_type == "slurm":
            return await self._execute_slurm(step, command, output_dir, working_dir)
        else:
            raise ValueError(f"Unknown executor type: {self.executor_type}")
    
    async def _execute_local(self, step: Dict[str, Any], command: str, output_dir: str, cwd: str) -> Dict[str, Any]:
        """Execute a command locally.
        
        Args:
            step: Step to execute
            command: Command to execute
            output_dir: Output directory for the workflow
            cwd: Working directory for command execution
            
        Returns:
            Dictionary with execution results
        """
        step_name = step.get("name", "Unnamed step")
        
        # Create a log file for this step
        log_dir = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{step_name.replace(' ', '_')}.log")
        
        try:
            # Execute the command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd
            )
            
            # Wait for the command to complete
            stdout, stderr = await process.communicate()
            
            # Decode stdout and stderr
            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")
            
            # Write stdout and stderr to log file
            with open(log_file, "w") as f:
                f.write(f"COMMAND: {command}\n\n")
                f.write(f"STDOUT:\n{stdout_str}\n\n")
                f.write(f"STDERR:\n{stderr_str}\n\n")
                f.write(f"EXIT CODE: {process.returncode}\n")
            
            # Check if the command was successful
            if process.returncode == 0:
                self.logger.info(f"Step {step_name} completed successfully")
                return {
                    "status": "success",
                    "step": step,
                    "step_name": step_name,
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "exit_code": process.returncode,
                    "log_file": log_file
                }
            else:
                self.logger.error(f"Step {step_name} failed with exit code {process.returncode}")
                self.logger.error(f"STDERR: {stderr_str}")
                return {
                    "status": "error",
                    "step": step,
                    "step_name": step_name,
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "exit_code": process.returncode,
                    "log_file": log_file,
                    "error": f"Command failed with exit code {process.returncode}"
                }
        
        except Exception as e:
            self.logger.error(f"Error executing step {step_name}: {str(e)}")
            
            # Write error to log file
            with open(log_file, "w") as f:
                f.write(f"COMMAND: {command}\n\n")
                f.write(f"ERROR: {str(e)}\n")
            
            return {
                "status": "error",
                "step": step,
                "step_name": step_name,
                "error": str(e),
                "log_file": log_file
            }
    
    async def _execute_slurm(self, step: Dict[str, Any], command: str, output_dir: str, cwd: str) -> Dict[str, Any]:
        """Execute a command using SLURM.
        
        Args:
            step: Step to execute
            command: Command to execute
            output_dir: Output directory for the workflow
            cwd: Working directory for command execution
            
        Returns:
            Dictionary with execution results
        """
        step_name = step.get("name", "Unnamed step")
        
        # Create a log file for this step
        log_dir = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{step_name.replace(' ', '_')}.log")
        
        # Create a SLURM script for this step
        slurm_dir = os.path.join(output_dir, "slurm")
        os.makedirs(slurm_dir, exist_ok=True)
        slurm_script = os.path.join(slurm_dir, f"{step_name.replace(' ', '_')}.sh")
        
        # Get SLURM parameters
        slurm_params = step.get("slurm", {})
        partition = slurm_params.get("partition", "normal")
        nodes = slurm_params.get("nodes", 1)
        ntasks = slurm_params.get("ntasks", 1)
        cpus_per_task = slurm_params.get("cpus_per_task", 1)
        mem = slurm_params.get("mem", "4G")
        time = slurm_params.get("time", "01:00:00")
        
        # Write SLURM script
        with open(slurm_script, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"#SBATCH --job-name={step_name}\n")
            f.write(f"#SBATCH --output={log_file}\n")
            f.write(f"#SBATCH --partition={partition}\n")
            f.write(f"#SBATCH --nodes={nodes}\n")
            f.write(f"#SBATCH --ntasks={ntasks}\n")
            f.write(f"#SBATCH --cpus-per-task={cpus_per_task}\n")
            f.write(f"#SBATCH --mem={mem}\n")
            f.write(f"#SBATCH --time={time}\n")
            f.write("\n")
            f.write(f"cd {cwd}\n")
            f.write(f"{command}\n")
        
        try:
            # Submit the SLURM job
            process = await asyncio.create_subprocess_exec(
                "sbatch", slurm_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=output_dir
            )
            
            # Wait for the submission to complete
            stdout, stderr = await process.communicate()
            
            # Decode stdout and stderr
            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")
            
            # Check if the submission was successful
            if process.returncode == 0:
                # Extract job ID from stdout
                import re
                job_id_match = re.search(r"Submitted batch job (\d+)", stdout_str)
                if job_id_match:
                    job_id = job_id_match.group(1)
                    self.logger.info(f"Step {step_name} submitted as SLURM job {job_id}")
                    
                    # Wait for the job to complete
                    job_completed = False
                    while not job_completed:
                        # Check job status
                        status_process = await asyncio.create_subprocess_exec(
                            "squeue", "-j", job_id, "-h",
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        
                        status_stdout, status_stderr = await status_process.communicate()
                        status_stdout_str = status_stdout.decode("utf-8", errors="replace")
                        
                        if not status_stdout_str.strip():
                            # Job has completed
                            job_completed = True
                        else:
                            # Job is still running, wait for a bit
                            await asyncio.sleep(10)
                    
                    # Job has completed, check if it was successful
                    if os.path.exists(log_file):
                        with open(log_file, "r") as f:
                            log_content = f.read()
                        
                        # Check for errors in the log file
                        if "ERROR" in log_content or "Error" in log_content or "error" in log_content:
                            self.logger.error(f"SLURM job {job_id} (step {step_name}) failed")
                            return {
                                "status": "error",
                                "step": step,
                                "job_id": job_id,
                                "log_file": log_file,
                                "error": "SLURM job failed, see log file for details"
                            }
                        else:
                            self.logger.info(f"SLURM job {job_id} (step {step_name}) completed successfully")
                            return {
                                "status": "success",
                                "step": step,
                                "job_id": job_id,
                                "log_file": log_file
                            }
                    else:
                        self.logger.error(f"SLURM job {job_id} (step {step_name}) log file not found")
                        return {
                            "status": "error",
                            "step": step,
                            "job_id": job_id,
                            "error": "SLURM job log file not found"
                        }
                else:
                    self.logger.error(f"Failed to extract SLURM job ID for step {step_name}")
                    return {
                        "status": "error",
                        "step": step,
                        "stdout": stdout_str,
                        "stderr": stderr_str,
                        "error": "Failed to extract SLURM job ID"
                    }
            else:
                self.logger.error(f"Failed to submit SLURM job for step {step_name}")
                self.logger.error(f"STDERR: {stderr_str}")
                return {
                    "status": "error",
                    "step": step,
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "exit_code": process.returncode,
                    "error": f"Failed to submit SLURM job: {stderr_str}"
                }
        
        except Exception as e:
            self.logger.error(f"Error executing step {step_name} with SLURM: {str(e)}")
            return {
                "status": "error",
                "step": step,
                "error": str(e)
            }
