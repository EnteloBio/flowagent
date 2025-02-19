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
from .llm import LLMInterface

logger = logging.getLogger(__name__)

class DiskSpaceError(Exception):
    """Raised when there isn't enough disk space"""
    pass

class PermissionError(Exception):
    """Raised when there are permission issues"""
    pass

class WorkflowStep(BaseModel):
    """Model for a workflow step"""
    name: str
    tool: str
    action: str
    parameters: Dict[str, Any]
    type: str

class WorkflowStateManager:
    """Manages workflow state and results"""
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
    """Base class for all agents"""
    def __init__(self, knowledge_db: chromadb.Client):
        self.knowledge_db = knowledge_db
        self.llm = LLMInterface()

class PLAN_agent(BaseAgent):
    """Agent responsible for workflow planning"""
    async def plan_from_prompt(self, prompt: str) -> Dict[str, Any]:
        """Parse natural language prompt and create workflow plan"""
        return await self.llm.generate_workflow_plan(prompt)
    
    def _validate_workflow_plan(self, plan: Dict) -> Dict:
        """Validate workflow plan structure"""
        required_keys = ['inputs', 'steps', 'outputs', 'validation']
        if not all(key in plan for key in required_keys):
            raise ValueError(f"Workflow plan missing required keys: {required_keys}")
        if not isinstance(plan['steps'], list) or not plan['steps']:
            raise ValueError("Workflow plan must contain non-empty steps list")
        return plan
    
    async def decompose_workflow(self, prompt: str) -> List[WorkflowStep]:
        """Convert prompt into workflow steps"""
        plan = await self.plan_from_prompt(prompt)
        plan = self._validate_workflow_plan(plan)
        return [WorkflowStep(**step) for step in plan['steps']]

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
        command = await self.llm.generate_command(tool, action, parameters)
        command = self._validate_command(command)
        logger.info(f"Generated command: {command}")
        return command
    
    def _validate_command(self, command: str) -> str:
        """Validate generated command"""
        dangerous_ops = ['|', '>', '<', ';', '&&', '||', '`', '$']
        for op in dangerous_ops:
            if op in command:
                logger.error(f"Command validation failed: contains dangerous operator '{op}'")
                raise ValueError(f"Generated command contains dangerous operator: {op}")
        return command.strip()
    
    async def execute_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Execute a workflow step"""
        logger.info(f"Generating command for step: {step.name}")
        logger.info(f"Tool: {step.tool}, Action: {step.action}, Type: {step.type}")
        logger.info(f"Parameters: {json.dumps(step.parameters, indent=2)}")
        
        # Special handling for Kallisto index
        if step.tool.lower() == 'kallisto' and step.action.lower() == 'index':
            self._validate_kallisto_params(step.parameters)
            self._cleanup_kallisto_index(step.parameters.get('output', ''))
        
        # Check disk space and permissions for output paths
        for param_name, param_value in step.parameters.items():
            if any(key in param_name.lower() for key in ['output', 'outdir', 'out_dir', 'index']):
                logger.info(f"Checking disk space and permissions for: {param_value}")
                self._check_disk_space(param_value)
                self._check_permissions(param_value)
        
        command = await self.get_tool_command(step.tool, step.action, step.parameters)
        result = await self.execute_command(command)
        
        if result['returncode'] == 0:
            logger.info(f"Step {step.name} completed successfully")
            if result['stdout']:
                logger.debug(f"Command output:\n{result['stdout']}")
        else:
            logger.error(f"Step {step.name} failed")
            logger.error(f"Command error output:\n{result['stderr']}")
            
        return result
    
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
    """Agent responsible for error diagnosis and recovery"""
    async def diagnose_failure(self, error_context: Dict) -> Dict[str, Any]:
        """Diagnose workflow failure and suggest recovery steps"""
        return await self.llm.diagnose_error(error_context)
