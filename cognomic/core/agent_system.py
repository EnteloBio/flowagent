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

class WorkflowStep(BaseModel):
    """Model for a workflow step."""
    name: str
    tool: str
    action: str
    parameters: Dict[str, Any]
    type: str

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
    
    def __init__(self, knowledge_db: chromadb.Client):
        """Initialize base agent."""
        self.knowledge_db = knowledge_db
        self.llm = LLMInterface()
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

class PLAN_agent(BaseAgent):
    """Agent responsible for workflow planning."""
    
    async def plan_from_prompt(self, prompt: str) -> Dict[str, Any]:
        """Parse natural language prompt and create workflow plan."""
        try:
            workflow_plan = await self.llm.generate_workflow_plan(prompt)
            if not self._validate_workflow_plan(workflow_plan):
                raise ValueError("Invalid workflow plan structure")
            return workflow_plan
        except Exception as e:
            self.logger.error(f"Failed to generate workflow plan: {str(e)}")
            raise
    
    def _validate_workflow_plan(self, plan: Dict[str, Any]) -> bool:
        """Validate workflow plan structure."""
        required_keys = ['workflow_type', 'steps']
        if not all(key in plan for key in required_keys):
            return False
        
        if not isinstance(plan['steps'], list):
            return False
            
        for step in plan['steps']:
            if not all(key in step for key in ['name', 'tool', 'action', 'parameters']):
                return False
        
        return True
    
    async def decompose_workflow(self, prompt: str) -> Dict[str, Any]:
        """Convert prompt into workflow steps."""
        try:
            return await self.llm.decompose_workflow(prompt)
        except Exception as e:
            self.logger.error(f"Failed to decompose workflow: {str(e)}")
            raise

class TASK_agent(BaseAgent):
    """Agent responsible for task execution"""
    def __init__(self, knowledge_db: chromadb.Client):
        super().__init__(knowledge_db)
        self.cwd = os.getcwd()  # Store the current working directory
        logger.info(f"Working directory: {self.cwd}")
        
    def _check_disk_space(self, path: str, required_gb: float = 10.0) -> None:
        """Check if there's enough disk space in the target directory"""
        try:
            path_obj = Path(path)
            if not path_obj.is_absolute():
                path_obj = path_obj.resolve()
                
            total, used, free = shutil.disk_usage(str(path_obj.parent))
            free_gb = free / (2**30)  # Convert to GB
            
            logger.info(f"Checking disk space in {path_obj.parent}")
            logger.info(f"Free space: {free_gb:.2f} GB")
            
            if free_gb < required_gb:
                raise DiskSpaceError(
                    f"Not enough disk space. Required: {required_gb} GB, Available: {free_gb:.2f} GB"
                )
                
        except Exception as e:
            logger.error(f"Error checking disk space: {str(e)}")
            raise
            
    def _check_permissions(self, path: str) -> None:
        """Check if we have write permissions to the target directory"""
        try:
            path_obj = Path(path)
            if not path_obj.is_absolute():
                path_obj = path_obj.resolve()
                
            parent_dir = path_obj.parent
            
            # Create parent directory if it doesn't exist
            if not parent_dir.exists():
                logger.info(f"Creating directory: {parent_dir}")
                parent_dir.mkdir(parents=True, exist_ok=True)
            
            # Check write permissions
            if not os.access(str(parent_dir), os.W_OK):
                raise PermissionError(
                    f"No write permission for directory: {parent_dir}"
                )
                
            logger.info(f"Write permissions verified for: {parent_dir}")
            
        except Exception as e:
            logger.error(f"Error checking permissions: {str(e)}")
            raise

    def _cleanup_kallisto_index(self, index_path: str) -> None:
        """Clean up any existing Kallisto index files"""
        try:
            index_path = Path(index_path)
            
            # If it's a directory, remove it and its contents
            if index_path.is_dir():
                logger.info(f"Removing existing index directory: {index_path}")
                try:
                    import shutil
                    shutil.rmtree(index_path)
                except Exception as e:
                    logger.warning(f"Could not remove index directory: {e}")
                    
            # Ensure we use a file path, not a directory
            if not index_path.suffix:
                index_path = index_path.with_suffix('.idx')
                
            if index_path.exists():
                if not os.access(str(index_path), os.W_OK):
                    logger.warning(f"No write permission for index file: {index_path}")
                    return
                    
                logger.info(f"Removing existing index file: {index_path}")
                try:
                    index_path.unlink()
                except OSError as e:
                    logger.warning(f"Could not remove index file: {e}")
                    
        except Exception as e:
            logger.error(f"Error during Kallisto cleanup: {str(e)}")
            
    def _validate_kallisto_params(self, parameters: Dict[str, Any]) -> None:
        """Validate Kallisto parameters"""
        # Check reference file exists
        ref_file = Path(parameters.get('reference', ''))
        if not ref_file.exists():
            raise ValueError(f"Reference file not found: {ref_file}")
            
        # Ensure index path has proper extension
        index_path = Path(parameters.get('output', ''))
        if not index_path.suffix:
            index_path = index_path.with_suffix('.idx')
            parameters['output'] = str(index_path)
            
        # Check parent directory is writable
        parent_dir = index_path.parent
        if not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)
        if not os.access(str(parent_dir), os.W_OK):
            raise PermissionError(f"No write permission for index directory: {parent_dir}")
                
    async def get_tool_command(self, tool: str, action: str, parameters: Dict[str, Any]) -> str:
        """Generate command for tool execution"""
        if tool.lower() == 'kallisto':
            return self._generate_kallisto_command(action, parameters)
        else:
            command = await self.llm.generate_command(tool, action, parameters)
            command = self._validate_command(command)
            logger.info(f"Generated command: {command}")
            return command
    
    def _generate_kallisto_command(self, action: str, parameters: Dict[str, Any]) -> str:
        """Generate Kallisto command."""
        if action == "index":
            reference = parameters.get('reference')
            output = parameters.get('output')
            return f"kallisto index -i {output} {reference}"
            
        elif action == "quant":
            index = parameters.get('index')
            output_dir = parameters.get('output_dir')
            fragment_length = parameters.get('fragment_length', 200)
            sd = parameters.get('sd', 20)
            
            # Return a function that generates command for a single file
            def make_command(input_file: str) -> str:
                return (f"kallisto quant -i {index} -o {output_dir} "
                       f"--single -l {fragment_length} -s {sd} {input_file}")
            return make_command
            
        else:
            raise ValueError(f"Unknown Kallisto action: {action}")

    def _validate_command(self, command: str) -> str:
        """Validate generated command"""
        dangerous_ops = ['|', '>', '<', ';', '&&', '||', '`', '$']
        for op in dangerous_ops:
            if op in command:
                logger.error(f"Command validation failed: contains dangerous operator '{op}'")
                raise ValueError(f"Generated command contains dangerous operator: {op}")
        return command.strip()
    
    def _find_input_files(self, pattern: str) -> List[str]:
        """Find input files matching the given pattern."""
        # Define valid FASTQ extensions
        fastq_extensions = ('.fastq', '.fastq.gz', '.fq', '.fq.gz')
        
        # Get all files in current directory
        files = [f for f in os.listdir(self.cwd) 
                if os.path.isfile(os.path.join(self.cwd, f))]
        
        # Filter for FASTQ files
        fastq_files = [f for f in files if any(f.endswith(ext) for ext in fastq_extensions)]
        
        if not fastq_files:
            self.logger.warning(f"No FASTQ files found in {self.cwd}")
            return []
            
        self.logger.info(f"Found {len(fastq_files)} FASTQ files: {fastq_files}")
        return fastq_files

    def _generate_fastqc_command(self, input_file: str, output_dir: str) -> str:
        """Generate FastQC command for a single file."""
        return f"fastqc {input_file} -o {output_dir}"

    async def execute_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Execute a workflow step"""
        logger.info(f"Generating command for step: {step.name}")
        logger.info(f"Tool: {step.tool}, Action: {step.action}, Type: {step.type}")
        logger.info(f"Parameters: {json.dumps(step.parameters, indent=2)}")
        
        # Check disk space and permissions for output directory
        if 'output_dir' in step.parameters:
            output_dir = step.parameters['output_dir']
            self._check_disk_space(output_dir)
            self._check_permissions(output_dir)
            
        if step.tool.lower() == 'fastqc':
            # For FastQC, process each file individually
            fastq_files = self._find_input_files("*")
            results = []
            
            for input_file in fastq_files:
                command = self._generate_fastqc_command(input_file, step.parameters['output_dir'])
                command = self._validate_command(command)
                logger.info(f"Generated command: {command}")
                
                result = await self.execute_command(command)
                results.append(result)
                
            # Return combined results
            return {
                'status': 'success' if all(r['returncode'] == 0 for r in results) else 'error',
                'output': '\n'.join(r.get('stdout', '') for r in results),
                'error': '\n'.join(r.get('stderr', '') for r in results if r.get('stderr'))
            }
            
        elif step.tool.lower() == 'kallisto':
            if step.action == "index":
                # For index, just run single command
                command = self._generate_kallisto_command(step.action, step.parameters)
                command = self._validate_command(command)
                logger.info(f"Generated command: {command}")
                return await self.execute_command(command)
            elif step.action == "quant":
                # For quant, process each file individually
                fastq_files = self._find_input_files("*")
                results = []
                
                # Get command generator function
                make_command = self._generate_kallisto_command(step.action, step.parameters)
                
                for input_file in fastq_files:
                    command = make_command(input_file)
                    command = self._validate_command(command)
                    logger.info(f"Generated command: {command}")
                    
                    result = await self.execute_command(command)
                    results.append(result)
                
                # Return combined results
                return {
                    'status': 'success' if all(r['returncode'] == 0 for r in results) else 'error',
                    'output': '\n'.join(r.get('stdout', '') for r in results),
                    'error': '\n'.join(r.get('stderr', '') for r in results if r.get('stderr'))
                }
            
        else:
            # For other tools, use LLM command generation
            command = await self.get_tool_command(step.tool, step.action, step.parameters)
            command = self._validate_command(command)
            logger.info(f"Generated command: {command}")
            return await self.execute_command(command)
    
    async def execute_command(self, command: str) -> Dict[str, Any]:
        """Execute a shell command"""
        logger.info(f"Executing command: {command}")
        logger.info(f"Working directory: {self.cwd}")
        
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,  # Set the working directory
                env=os.environ.copy()  # Use current environment
            )
            
            stdout, stderr = await process.communicate()
            stdout_str = stdout.decode() if stdout else ""
            stderr_str = stderr.decode() if stderr else ""
            
            result = {
                "returncode": process.returncode,
                "stdout": stdout_str,
                "stderr": stderr_str,
                "command": command,
                "cwd": self.cwd
            }
            
            if process.returncode != 0:
                logger.error(f"Command failed with return code {process.returncode}")
                if stderr_str:
                    logger.error(f"Error output:\n{stderr_str}")
            else:
                logger.info(f"Command completed successfully")
                if stdout_str:
                    logger.debug(f"Command output:\n{stdout_str}")
                    
            return result
            
        except Exception as e:
            logger.error(f"Error executing command: {str(e)}")
            raise

class DEBUG_agent(BaseAgent):
    """Agent responsible for error diagnosis and recovery."""
    
    async def diagnose_failure(self, error_context: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnose workflow failure and suggest recovery steps."""
        try:
            # Get error details
            error_type = error_context.get('error_type', 'Unknown')
            error_message = error_context.get('error_message', '')
            step_name = error_context.get('step_name', '')
            
            # Log error for diagnosis
            self.logger.error(
                f"Diagnosing failure in step '{step_name}': "
                f"{error_type} - {error_message}"
            )
            
            # Generate diagnosis and recovery plan
            diagnosis = await self.llm.diagnose_error(
                error_type=error_type,
                error_message=error_message,
                step_name=step_name,
                context=error_context
            )
            
            return {
                'error_type': error_type,
                'error_message': error_message,
                'step_name': step_name,
                'diagnosis': diagnosis.get('diagnosis', ''),
                'recovery_steps': diagnosis.get('recovery_steps', []),
                'suggestions': diagnosis.get('suggestions', [])
            }
            
        except Exception as e:
            self.logger.error(f"Failed to diagnose error: {str(e)}")
            return {
                'error_type': 'DiagnosisError',
                'error_message': str(e),
                'diagnosis': 'Failed to diagnose error',
                'recovery_steps': [],
                'suggestions': ['Contact support for assistance']
            }
