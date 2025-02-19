import json
import chromadb
from chromadb.config import Settings
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Type, Optional
from pathlib import Path
import hashlib
import logging
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class WorkflowStep(BaseModel):
    name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    validation_schema: Optional[Type[BaseModel]] = None

class BaseAgent:
    def __init__(self, knowledge_db: chromadb.Client):
        self.knowledge_db = knowledge_db
        self.workflow_state = {}
        self._init_collections()
        
    def _init_collections(self):
        self.collections = {
            'workflow_patterns': self.knowledge_db.get_collection("workflow_patterns"),
            'tool_docs': self.knowledge_db.get_collection("tool_documentation"),
            'error_patterns': self.knowledge_db.get_collection("error_solutions")
        }

class PLAN_agent(BaseAgent):
    async def decompose_workflow(self, objective: str) -> List[WorkflowStep]:
        try:
            results = self.collections['workflow_patterns'].query(
                query_texts=[objective],
                n_results=3
            )
            return self._parse_results(results)
        except Exception as e:
            logger.error(f"Workflow decomposition failed: {str(e)}")
            raise
            
    def _parse_results(self, results: Dict) -> List[WorkflowStep]:
        steps = []
        for doc, metadata in zip(results['documents'], results['metadatas']):
            steps.append(WorkflowStep(
                name=metadata.get('tool_name'),
                parameters=metadata.get('default_params', {}),
                dependencies=metadata.get('dependencies', [])
            ))
        return steps

class TASK_agent(BaseAgent):
    def __init__(self, knowledge_db: chromadb.Client, max_retries: int = 5):
        super().__init__(knowledge_db)
        self.max_retries = max_retries
        self.llm = None  # Initialize LLM connection here
        
    async def get_tool_command(self, tool_name: str, action: str, parameters: Dict[str, Any]) -> str:
        """Generate command using LLM based on tool documentation"""
        # Get tool documentation from knowledge base
        tool_doc = await self.knowledge_db.get_tool_doc(tool_name)
        
        # Format prompt for LLM
        prompt = self._format_command_prompt(tool_name, action, parameters, tool_doc)
        
        # Get command from LLM
        response = await self.llm.generate(prompt)
        
        # Validate and return command
        command = self._validate_command(response)
        return command
        
    def _format_command_prompt(self, tool_name: str, action: str, parameters: Dict[str, Any], tool_doc: str) -> str:
        """Format prompt for command generation"""
        return f"""
        Tool: {tool_name}
        Action: {action}
        Parameters: {json.dumps(parameters, indent=2)}
        
        Documentation:
        {tool_doc}
        
        Generate the exact command line to execute this tool with the given parameters.
        The command should follow best practices and be safe to execute.
        Only return the command itself, no explanation or additional text.
        """
        
    def _validate_command(self, command: str) -> str:
        """Validate generated command for safety and correctness"""
        # Remove any dangerous shell operators
        dangerous_ops = ['|', '>', '<', ';', '&&', '||', '`', '$']
        for op in dangerous_ops:
            if op in command:
                raise ValueError(f"Generated command contains dangerous operator: {op}")
        
        # Validate command structure
        if not command.strip():
            raise ValueError("Empty command generated")
            
        return command.strip()
        
    async def execute_command(self, command: str) -> Dict[str, Any]:
        """Execute a shell command and return results"""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                "returncode": process.returncode,
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
                "command": command
            }
            
        except Exception as e:
            raise RuntimeError(f"Command execution failed: {str(e)}")
            
    async def execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow step"""
        try:
            # Get command for step
            command = await self.get_tool_command(
                step["tool"],
                step["action"],
                step["parameters"]
            )
            
            # Execute command
            result = await self.execute_command(command)
            
            if result["returncode"] != 0:
                raise RuntimeError(f"Step failed: {result['stderr']}")
                
            return result
            
        except Exception as e:
            raise RuntimeError(f"Step execution failed: {str(e)}")

class DEBUG_agent(BaseAgent):
    async def diagnose_failure(self, error_context: Dict) -> Dict:
        try:
            results = self.collections['error_patterns'].query(
                query_texts=[error_context['error_log']],
                n_results=3
            )
            return self._generate_recovery_plan(results)
        except Exception as e:
            logger.error(f"Diagnosis failed: {str(e)}")
            raise
            
    def _generate_recovery_plan(self, results: Dict) -> Dict:
        if not results['documents']:
            return {"action": "retry", "reason": "No known error pattern matched"}
        
        return {
            "action": "fix",
            "solution": results['documents'][0],
            "confidence": results['distances'][0]
        }

class WorkflowStateManager:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.state = {
            'workflow_id': self._generate_id(),
            'steps': {},
            'artifacts': {}
        }
        
    def _generate_id(self) -> str:
        return hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:12]
    
    def archive_results(self):
        archive_path = self.output_dir / self.state['workflow_id']
        archive_path.mkdir(exist_ok=True)
        (archive_path / "state.json").write_text(json.dumps(self.state))
        return archive_path
