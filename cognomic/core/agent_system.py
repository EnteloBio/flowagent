"""Agent system for workflow execution."""

import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import logging
import shutil
import os
from datetime import datetime
import chromadb
from pydantic import BaseModel
from dataclasses import dataclass

from ..utils.logging import get_logger
from .agent_types import TASK_agent, PLAN_agent, DEBUG_agent
from .llm import LLMInterface
from ..config.settings import settings

logger = get_logger(__name__)

class DiskSpaceError(Exception):
    """Raised when there isn't enough disk space"""
    pass

class PermissionError(Exception):
    """Raised when there are permission issues"""
    pass

@dataclass
class WorkflowStep:
    """Represents a single step in a workflow."""
    name: str
    tool: str
    action: str
    parameters: Dict[str, Any]
    type: str = "command"  # Default to command type
    description: str = ""  # Optional description of what the step does
    
    def __post_init__(self):
        """Validate and process step parameters after initialization."""
        # Ensure parameters is a dict
        if not isinstance(self.parameters, dict):
            self.parameters = {} if self.parameters is None else dict(self.parameters)
            
        # Ensure required fields are present
        if not all([self.name, self.tool, self.action]):
            raise ValueError("Name, tool, and action are required")
            
        # Validate type
        if self.type not in ["command", "function", "script"]:
            raise ValueError("Type must be one of: command, function, script")
            
        # Convert parameters to proper types
        if "threads" in self.parameters:
            self.parameters["threads"] = int(self.parameters["threads"])
            
        if "single" in self.parameters:
            self.parameters["single"] = bool(self.parameters["single"])
            
        if "fragment_length" in self.parameters:
            self.parameters["fragment_length"] = int(self.parameters["fragment_length"])
            
        if "sd" in self.parameters:
            self.parameters["sd"] = float(self.parameters["sd"])
            
        # Create output directories if needed
        if "output_dir" in self.parameters:
            os.makedirs(self.parameters["output_dir"], exist_ok=True)
            
        # Handle input files
        if "input_file" in self.parameters and isinstance(self.parameters["input_file"], str) and "*" in self.parameters["input_file"]:
            # Expand glob patterns
            import glob
            files = glob.glob(self.parameters["input_file"])
            if files:
                self.parameters["input_files"] = files
                del self.parameters["input_file"]
            else:
                raise ValueError(f"No files found matching pattern: {self.parameters['input_file']}")

class WorkflowStateManager:
    """Manages workflow state and results."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.state = {
            'steps': {},
            'artifacts': {},
            'start_time': str(datetime.now()),
            'status': 'initialized'
        }
        
    def archive_results(self) -> Dict[str, Any]:
        """Archive workflow results"""
        self.state['end_time'] = str(datetime.now())
        self.state['status'] = 'completed'
        
        # Save state to file
        state_file = self.output_dir / 'workflow_state.json'
        with open(state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
            
        return self.state

class BaseAgent:
    """Base class for all agents."""
    
    def __init__(self, llm=None, logger=None):
        """Initialize base agent."""
        self.llm = llm
        self.has_llm = llm is not None
        self.llm_agent = llm if self.has_llm else None
        self.logger = logger or logging.getLogger(__name__)
        self.cwd = os.getcwd()
        
    async def execute_command(self, command: str) -> Dict[str, Any]:
        """Execute a shell command."""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd
            )
            stdout, stderr = await process.communicate()
            return {
                "returncode": process.returncode,
                "stdout": stdout.decode() if stdout else "",
                "stderr": stderr.decode() if stderr else ""
            }
        except Exception as e:
            self.logger.error(f"Command execution failed: {str(e)}")
            raise

    def _check_disk_space(self, output_dir: str) -> None:
        """Check disk space and permissions for output directory."""
        self.logger.info(f"Checking disk space in {output_dir}")
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Check write permissions
            test_file = os.path.join(output_dir, ".test_write")
            try:
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
            except Exception as e:
                raise PermissionError(f"Cannot write to output directory {output_dir}: {str(e)}")
            
            # Check available disk space (require at least 1GB)
            if os.name == 'posix':  # Unix/Linux/MacOS
                st = os.statvfs(output_dir)
                free_space = st.f_bavail * st.f_frsize
                if free_space < 1024 * 1024 * 1024:  # 1GB
                    raise RuntimeError(f"Insufficient disk space in {output_dir}. At least 1GB required.")
                    
        except Exception as e:
            self.logger.error(f"Disk space check failed: {str(e)}")
            raise

    def _check_input_files(self, input_files: List[str]) -> None:
        """Check that input files exist and are readable."""
        if not input_files:
            raise ValueError("No input files provided")
            
        for file in input_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Input file not found: {file}")
            if not os.access(file, os.R_OK):
                raise PermissionError(f"Cannot read input file: {file}")

class PLAN_agent(BaseAgent):
    """Agent for planning workflow steps."""
    
    async def decompose_workflow(self, prompt: str) -> List[WorkflowStep]:
        """Decompose workflow from prompt into executable steps."""
        self.logger.info(f"Decomposing workflow from prompt: {prompt}")
        
        # Create basic workflow steps for RNA-seq analysis
        steps = []
        
        # Find input files
        input_files = await self._find_fastq_files()
        
        # 1. Quality Control
        steps.append(WorkflowStep(
            name="quality_control",
            tool="FastQC",
            action="analyze",
            type="command",
            parameters={
                "input_files": input_files,
                "output_dir": "results/rna_seq_analysis/fastqc_reports"
            }
        ))
        
        # 2. Build Kallisto Index
        steps.append(WorkflowStep(
            name="build_index",
            tool="kallisto",
            action="index",
            type="command",
            parameters={
                "reference": "Homo_sapiens.GRCh38.cdna.all.fa",
                "output": "results/rna_seq_analysis/kallisto_index/transcriptome.idx"
            }
        ))
        
        # 3. Kallisto Quantification
        steps.append(WorkflowStep(
            name="quantify",
            tool="kallisto",
            action="quant",
            type="command",
            parameters={
                "index": "results/rna_seq_analysis/kallisto_index/transcriptome.idx",
                "input_files": input_files,
                "output_dir": "results/rna_seq_analysis/kallisto_output",
                "single": True,
                "fragment_length": 200,
                "sd": 20
            }
        ))
        
        # 4. MultiQC Report
        steps.append(WorkflowStep(
            name="generate_report",
            tool="MultiQC",
            action="report",
            type="command",
            parameters={
                "input_dir": "results/rna_seq_analysis",
                "output_dir": "results/rna_seq_analysis/multiqc_report"
            }
        ))
        
        return steps
        
    async def _find_fastq_files(self) -> List[str]:
        """Find all FASTQ files in the current directory."""
        try:
            result = await self.execute_command(
                "find . -maxdepth 1 -type f -name '*.fastq.gz' -o -name '*.fastq' -o -name '*.fq.gz' -o -name '*.fq'"
            )
            if result["returncode"] == 0:
                files = [f.strip() for f in result["stdout"].split("\n") if f.strip()]
                if not files:
                    raise RuntimeError("No FASTQ files found in current directory")
                return files
        except Exception as e:
            self.logger.error(f"Error finding FASTQ files: {str(e)}")
            raise

class TASK_agent(BaseAgent):
    """Agent for executing workflow steps."""
    
    async def execute_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Execute a workflow step."""
        self.logger.info(f"Generating command for step: {step.name}")
        self.logger.info(f"Tool: {step.tool}, Action: {step.action}, Type: {step.type}")
        self.logger.info(f"Parameters: {json.dumps(step.parameters, indent=2)}")
        
        try:
            # Create output directories
            if "output_dir" in step.parameters:
                os.makedirs(step.parameters["output_dir"], exist_ok=True)
                self._check_disk_space(step.parameters["output_dir"])
            
            # For kallisto index, also create index directory
            if step.tool == "kallisto" and step.action == "index":
                index_dir = os.path.dirname(step.parameters["output"])
                os.makedirs(index_dir, exist_ok=True)
                self._check_disk_space(index_dir)
                
                # Check kallisto version first
                version_result = await self.execute_command("kallisto version")
                if version_result["returncode"] != 0:
                    raise RuntimeError("Failed to get Kallisto version")
                kallisto_version = version_result["stdout"].strip()
                self.logger.info(f"Kallisto version: {kallisto_version}")
            
            # Check input files
            if "input_files" in step.parameters:
                self._check_input_files(step.parameters["input_files"])
            elif "reference" in step.parameters:
                if not os.path.exists(step.parameters["reference"]):
                    raise FileNotFoundError(f"Reference file not found: {step.parameters['reference']}")
                if not os.access(step.parameters["reference"], os.R_OK):
                    raise PermissionError(f"Cannot read reference file: {step.parameters['reference']}")
            
            # Generate and execute command based on step type
            if step.type == "command":
                # Get command from LLM
                command = await self._get_llm_command(step)
                if not command:
                    raise ValueError(f"Failed to generate command for step: {step.name}")
                    
                self.logger.info(f"Executing command: {command}")
                result = await self.execute_command(command)
                
                if result["returncode"] != 0:
                    error_msg = result["stderr"] or result["stdout"]
                    if step.tool == "kallisto" and step.action == "index":
                        # Check if error is due to disk space
                        if "could not write" in error_msg:
                            df_result = await self.execute_command(f"df -h {index_dir}")
                            self.logger.error(f"Disk space information:\n{df_result['stdout']}")
                            raise RuntimeError(f"Failed to write Kallisto index. Check disk space and permissions in {index_dir}")
                    raise RuntimeError(f"Command failed: {error_msg}")
                
                # For kallisto index, verify the index after creation
                if step.tool == "kallisto" and step.action == "index" and os.path.exists(step.parameters["output"]):
                    inspect_result = await self.execute_command(f"kallisto inspect {step.parameters['output']}")
                    if inspect_result["returncode"] != 0:
                        raise RuntimeError(f"Failed to verify Kallisto index: {inspect_result['stderr']}")
                    
                    # Parse version from inspect output more safely
                    version_lines = [line for line in inspect_result["stdout"].split("\n") 
                                if "version" in line.lower()]
                    if version_lines:
                        try:
                            index_version = version_lines[0].split(":")[1].strip()
                            self.logger.info(f"Created Kallisto index version: {index_version}")
                            if index_version != kallisto_version:
                                self.logger.warning(f"Index version ({index_version}) does not match Kallisto version ({kallisto_version})")
                        except (IndexError, KeyError) as e:
                            self.logger.warning(f"Could not parse index version from output: {version_lines[0]}")
                    else:
                        self.logger.warning("No version information found in index inspection output")
                
                return {
                    "command": command,
                    "output": result["stdout"],
                    "error": result["stderr"],
                    "returncode": result["returncode"]
                }
            else:
                raise ValueError(f"Unsupported step type: {step.type}")
                
        except Exception as e:
            self.logger.error(f"Error executing step {step.name}: {str(e)}")
            raise
            
    async def _get_llm_command(self, step: WorkflowStep) -> str:
        """Get command from LLM based on step parameters."""
        # Convert step to dictionary format expected by LLM
        step_dict = {
            "tool": step.tool,
            "action": step.action,
            "parameters": step.parameters
        }
        
        # For multiple files in kallisto quant, generate separate commands
        if step.tool == "kallisto" and step.action == "quant" and len(step.parameters['input_files']) > 1:
            commands = []
            for input_file in step.parameters['input_files']:
                # Create sample-specific output directory using full filename
                sample_name = os.path.basename(input_file)  # Keep full filename including extension
                if sample_name.endswith('.gz'):  # Remove .gz if present
                    sample_name = sample_name[:-3]
                sample_output_dir = os.path.join(step.parameters['output_dir'], sample_name)
                os.makedirs(sample_output_dir, exist_ok=True)
                
                # Create step dict for this file
                file_step = {
                    "tool": step.tool,
                    "action": step.action,
                    "parameters": {
                        **step.parameters,
                        "input_files": [input_file],
                        "output_dir": sample_output_dir
                    }
                }
                command = await self.llm.generate_command(file_step)
                if command:
                    commands.append(command)
                    
            return " && ".join(commands) if commands else None
        else:
            return await self.llm.generate_command(step_dict)

class DEBUG_agent(BaseAgent):
    """Agent responsible for error diagnosis and recovery."""
    
    async def diagnose_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnose workflow errors and suggest recovery steps."""
        try:
            self.logger.info(f"Diagnosing error: {str(error)}")
            self.logger.info(f"Error context: {json.dumps(context, indent=2)}")
            
            # Basic error diagnosis
            diagnosis = {
                "error_type": error.__class__.__name__,
                "error_message": str(error),
                "diagnosis": "Unknown error occurred",
                "recovery_steps": [],
                "suggestions": []
            }
            
            if "No FASTQ files found" in str(error):
                diagnosis.update({
                    "diagnosis": "No FASTQ files found in working directory",
                    "recovery_steps": [
                        "Check that FASTQ files exist in the working directory",
                        "Verify file extensions (.fastq, .fastq.gz, .fq, .fq.gz)",
                        "Check file permissions"
                    ],
                    "suggestions": [
                        "Place FASTQ files in the working directory",
                        "Use correct file extensions",
                        "Check file access permissions"
                    ]
                })
            elif "disk space" in str(error).lower():
                diagnosis.update({
                    "diagnosis": "Insufficient disk space",
                    "recovery_steps": [
                        "Free up disk space",
                        "Choose a different output directory",
                        "Remove temporary files"
                    ],
                    "suggestions": [
                        "Clean up unnecessary files",
                        "Use a disk with more free space",
                        "Compress or archive old files"
                    ]
                })
            elif "permission" in str(error).lower():
                diagnosis.update({
                    "diagnosis": "Permission error",
                    "recovery_steps": [
                        "Check file/directory permissions",
                        "Run with appropriate permissions",
                        "Use a different output directory"
                    ],
                    "suggestions": [
                        "Check ownership of files/directories",
                        "Run with necessary permissions",
                        "Choose a writable output location"
                    ]
                })
                
            return diagnosis
            
        except Exception as e:
            self.logger.error(f"Failed to diagnose error: {str(e)}")
            raise
