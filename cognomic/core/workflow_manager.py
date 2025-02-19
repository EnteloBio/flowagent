import asyncio
from pathlib import Path
from typing import Dict, Any, List
import logging
import os
from .agent_system import PLAN_agent, TASK_agent, DEBUG_agent, WorkflowStateManager, WorkflowStep
from ..validation.validator import ValidationService, BioValidationSchemas

logger = logging.getLogger(__name__)

class WorkflowManager:
    def __init__(self, knowledge_db, output_dir: Path):
        self.plan_agent = PLAN_agent(knowledge_db)
        self.task_agent = TASK_agent(knowledge_db)
        self.debug_agent = DEBUG_agent(knowledge_db)
        self.state_manager = WorkflowStateManager(output_dir)
        self.validation = ValidationService({
            'kallisto_quant': BioValidationSchemas.KallistoQuantInput,
            'fastqc': BioValidationSchemas.FastQCInput,
            'multiqc': BioValidationSchemas.MultiQCInput,
            'deseq2': BioValidationSchemas.DESeq2Input
        })
        
        # Store the initial working directory
        self.initial_cwd = os.getcwd()
        logger.info(f"Initial working directory: {self.initial_cwd}")

    async def execute_from_prompt(self, prompt: str) -> Dict[str, Any]:
        """Execute workflow based on natural language prompt"""
        try:
            # Plan workflow steps from prompt
            logger.info("Planning workflow steps...")
            steps = await self.plan_agent.decompose_workflow(prompt)
            self._initialize_state(steps)
            
            # Create output directories
            self._create_output_directories(steps)
            
            # Update task agent's working directory
            self.task_agent.cwd = self.initial_cwd
            logger.info(f"Set task agent working directory to: {self.task_agent.cwd}")
            
            # Execute steps in order
            for step in steps:
                logger.info(f"Executing step: {step.name}")
                await self._execute_with_retry(step)
                
            # Archive results
            logger.info("Archiving workflow results...")
            return self.state_manager.archive_results()
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            diagnosis = await self.debug_agent.diagnose_failure({
                'error_log': str(e),
                'workflow_state': self.state_manager.state,
                'user_prompt': prompt
            })
            await self._handle_recovery(diagnosis)
            raise

    def _create_output_directories(self, steps: List[WorkflowStep]):
        """Create output directories for all steps"""
        for step in steps:
            # Check parameters for output directories
            for param_name, param_value in step.parameters.items():
                if any(key in param_name.lower() for key in ['output', 'outdir', 'out_dir']):
                    output_dir = Path(param_value)
                    if not output_dir.is_absolute():
                        output_dir.mkdir(parents=True, exist_ok=True)
                        logger.info(f"Created output directory: {output_dir}")

    def _initialize_state(self, steps: List[WorkflowStep]):
        """Initialize workflow state with planned steps"""
        self.state_manager.state['steps'] = {
            step.name: {
                'status': 'pending',
                'attempts': 0,
                'parameters': step.parameters
            } for step in steps
        }

    async def _execute_with_retry(self, step: WorkflowStep, max_retries: int = 5):
        """Execute a workflow step with retries"""
        step_state = self.state_manager.state['steps'][step.name]
        
        for attempt in range(max_retries):
            try:
                step_state['status'] = 'running'
                step_state['attempts'] += 1
                
                # Execute step
                result = await self.task_agent.execute_step(step)
                
                if result['returncode'] == 0:
                    self._update_state(step, result)
                    step_state['status'] = 'completed'
                    return
                else:
                    raise RuntimeError(f"Step failed: {result['stderr']}")
                    
            except Exception as e:
                logger.warning(f"Step {step.name} attempt {attempt+1} failed: {str(e)}")
                step_state['status'] = 'failed'
                step_state['last_error'] = str(e)
                
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Step {step.name} failed after {max_retries} attempts: {str(e)}")
                
                # Get recovery plan before retry
                diagnosis = await self.debug_agent.diagnose_failure({
                    'error_log': str(e),
                    'step_name': step.name,
                    'attempt': attempt + 1
                })
                
                if diagnosis['action'] == 'abort':
                    raise RuntimeError(f"Step {step.name} aborted: {diagnosis['diagnosis']}")
                
                # Apply fixes if suggested
                if diagnosis['action'] == 'fix':
                    await self._apply_fix(step, diagnosis['solution'])

    def _update_state(self, step: WorkflowStep, result: Dict[str, Any]):
        """Update workflow state with step results"""
        step_state = self.state_manager.state['steps'][step.name]
        step_state['result'] = result
        
        # Archive step artifacts
        if 'output_files' in result:
            self.state_manager.state['artifacts'].update(result['output_files'])

    async def _apply_fix(self, step: WorkflowStep, solution: str):
        """Apply fix suggested by DEBUG_agent"""
        logger.info(f"Applying fix for step {step.name}: {solution}")
        # Implement fix application logic here
        pass

    async def _handle_recovery(self, diagnosis: Dict[str, Any]):
        """Handle workflow recovery based on diagnosis"""
        if diagnosis['action'] == 'retry':
            logger.info("Retrying workflow with adjusted parameters")
            # Implement retry logic
        elif diagnosis['action'] == 'fix':
            logger.info(f"Applying fix: {diagnosis['solution']}")
            # Implement fix logic
        else:
            logger.error("No recovery action available")
