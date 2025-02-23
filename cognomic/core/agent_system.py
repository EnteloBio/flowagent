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
    dependencies: List[str] = None  # List of step names this step depends on
    type: str = "command"  # Default to command type
    description: str = ""  # Optional description of what the step does
    
    def __post_init__(self):
        """Validate and process step parameters after initialization."""
        # Ensure parameters is a dict
        if not isinstance(self.parameters, dict):
            self.parameters = {} if self.parameters is None else dict(self.parameters)
            
        # Initialize dependencies as empty list if None
        if self.dependencies is None:
            self.dependencies = []
            
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'tool': self.tool,
            'action': self.action,
            'parameters': self.parameters,
            'dependencies': self.dependencies,
            'type': self.type,
            'description': self.description
        }

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
    
    def __init__(self, llm=None, logger=None):
        """Initialize plan agent."""
        self.llm = llm
        self.has_llm = llm is not None
        self.llm_agent = llm if self.has_llm else None
        self.logger = logger or logging.getLogger(__name__)
        
    async def decompose_workflow(self, prompt: str) -> List[WorkflowStep]:
        """Decompose workflow from prompt into executable steps."""
        self.logger.info(f"Decomposing workflow from prompt: {prompt}")
        
        # Get workflow plan from LLM
        if self.has_llm:
            workflow_plan = await self.llm.generate_workflow_plan_from_context({"prompt": prompt})
        else:
            # Default workflow plan for RNA-seq analysis
            workflow_plan = {
                "workflow_type": "RNA-seq analysis",
                "steps": [
                    {
                        "name": "find_fastq",
                        "tool": "find_by_name",
                        "action": "SearchDirectory",
                        "parameters": {
                            "SearchDirectory": ".",
                            "Pattern": "*.fastq.gz",
                            "Type": "file"
                        },
                        "dependencies": []
                    },
                    {
                        "name": "kallisto_index",
                        "tool": "kallisto",
                        "action": "index",
                        "parameters": {
                            "reference": "Homo_sapiens.GRCh38.cdna.all.fa",
                            "output": "index.idx"
                        },
                        "dependencies": []
                    },
                    {
                        "name": "kallisto_quant",
                        "tool": "kallisto",
                        "action": "quant",
                        "parameters": {
                            "index": "index.idx",
                            "output_dir": "results/rna_seq_analysis",
                            "single": True,
                            "fragment_length": 200,
                            "fragment_sd": 20
                        },
                        "dependencies": ["find_fastq", "kallisto_index"]
                    },
                    {
                        "name": "fastqc",
                        "tool": "fastqc",
                        "action": "analyze",
                        "parameters": {
                            "output_dir": "results/rna_seq_analysis/fastqc_reports"
                        },
                        "dependencies": ["find_fastq"]
                    },
                    {
                        "name": "multiqc",
                        "tool": "multiqc",
                        "action": "aggregate",
                        "parameters": {
                            "input_dir": "results/rna_seq_analysis/fastqc_reports",
                            "output_dir": "results/rna_seq_analysis/multiqc_report"
                        },
                        "dependencies": ["fastqc"]
                    }
                ]
            }
            
        # Convert workflow plan to WorkflowStep objects
        workflow_steps = []
        for step_dict in workflow_plan['steps']:
            try:
                # Ensure all required fields are present
                required_fields = ['name', 'tool', 'action', 'parameters']
                for field in required_fields:
                    if field not in step_dict:
                        raise ValueError(f"Missing required field '{field}' in workflow step")
                
                # Create WorkflowStep object with proper defaults
                step = WorkflowStep(
                    name=step_dict['name'],
                    tool=step_dict['tool'],
                    action=step_dict['action'],
                    parameters=step_dict.get('parameters', {}),
                    dependencies=step_dict.get('dependencies', []),
                    type=step_dict.get('type', 'command'),
                    description=step_dict.get('description', '')
                )
                workflow_steps.append(step)
                self.logger.debug(f"Created workflow step: {step.name}")
            except Exception as e:
                self.logger.error(f"Error creating workflow step: {str(e)}")
                raise
            
        return workflow_steps

class TASK_agent(BaseAgent):
    """Agent for executing workflow steps."""
    
    async def execute_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Execute a workflow step."""
        try:
            # Create output directory if specified in parameters
            if 'output_dir' in step.parameters:
                os.makedirs(step.parameters['output_dir'], exist_ok=True)
                self.logger.info(f"Created output directory: {step.parameters['output_dir']}")

            # Handle find_by_name step
            if step.tool == "find_by_name":
                import glob
                files = glob.glob(os.path.join(step.parameters["SearchDirectory"], step.parameters["Pattern"]))
                return {"files": files}
            
            # Handle kallisto quant step
            if step.tool == "kallisto" and step.action == "quant":
                if "input_files" not in step.parameters:
                    raise ValueError("No input files specified for kallisto quant")
                
                # Ensure input_files is a list
                if isinstance(step.parameters["input_files"], str):
                    step.parameters["input_files"] = [step.parameters["input_files"]]
                
                # Generate command using LLM
                command = await self._get_llm_command(step)
                self.logger.info(f"Generated command for kallisto quant: {command}")
                
                # Execute command
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.cwd
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    raise RuntimeError(f"Kallisto quant failed: {stderr.decode()}")
                
                return {
                    "stdout": stdout.decode(),
                    "stderr": stderr.decode()
                }
            
            # Handle fastqc step
            if step.tool == "fastqc":
                if "input_files" not in step.parameters:
                    raise ValueError("No input files specified for fastqc")
                
                # Ensure input_files is a list
                if isinstance(step.parameters["input_files"], str):
                    step.parameters["input_files"] = [step.parameters["input_files"]]
                
                # Generate command using LLM
                command = await self._get_llm_command(step)
                self.logger.info(f"Generated command for fastqc: {command}")
                
                # Execute command
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.cwd
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    raise RuntimeError(f"FastQC failed: {stderr.decode()}")
                
                return {
                    "stdout": stdout.decode(),
                    "stderr": stderr.decode()
                }
            
            # For other steps, use LLM to generate command
            if self.has_llm:
                command = await self._get_llm_command(step)
            else:
                command = self._get_default_command(step)
                
            # Execute command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"Command failed: {stderr.decode()}")
            
            return {
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
                "returncode": process.returncode
            }
            
        except Exception as e:
            self.logger.error(f"Step execution failed: {str(e)}")
            raise

    async def _get_llm_command(self, step: WorkflowStep) -> str:
        """Get command from LLM based on step parameters."""
        # For multiple files in kallisto quant, generate separate commands
        if step.tool == "kallisto" and step.action == "quant" and len(step.parameters['input_files']) > 1:
            commands = []
            for input_file in step.parameters['input_files']:
                # Create sample-specific output directory
                sample_name = os.path.basename(input_file).replace('.fastq.gz', '')
                sample_output_dir = os.path.join(step.parameters['output_dir'], sample_name)
                os.makedirs(sample_output_dir, exist_ok=True)
                
                # Create new WorkflowStep for this file
                file_step = WorkflowStep(
                    name=f"{step.name}_{sample_name}",
                    tool=step.tool,
                    action=step.action,
                    parameters={
                        **step.parameters,
                        "input_files": [f"'{input_file}'"],  # Quote the path
                        "output_dir": sample_output_dir
                    },
                    dependencies=step.dependencies
                )
                command = await self.llm.generate_command(file_step.to_dict())
                if command:
                    commands.append(command)
                    
            return " && ".join(commands) if commands else None
        else:
            # Quote any file paths in parameters
            if 'input_files' in step.parameters:
                step.parameters['input_files'] = [f"'{f}'" for f in step.parameters['input_files']]
            return await self.llm.generate_command(step.to_dict())

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
