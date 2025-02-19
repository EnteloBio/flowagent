"""
Workflow Manager for Cognomic
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, List
import logging
from .agent_system import PLAN_agent, TASK_agent, DEBUG_agent, WorkflowStateManager, WorkflowStep
from ..validation.validator import ValidationService, BioValidationSchemas
from datetime import datetime

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

    async def execute_workflow(self, protocol: str):
        """Execute a workflow based on the provided protocol"""
        try:
            # Plan workflow steps
            steps = await self.plan_agent.decompose_workflow(protocol)
            self._initialize_state(steps)
            
            # Execute steps in order
            for step in steps:
                await self._execute_with_retry(step)
                
            # Archive results
            return self.state_manager.archive_results()
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            diagnosis = await self.debug_agent.diagnose_failure({
                'error_log': str(e),
                'workflow_state': self.state_manager.state
            })
            await self._handle_recovery(diagnosis)
            raise

    def _initialize_state(self, steps: List[WorkflowStep]):
        """Initialize workflow state with planned steps"""
        self.state_manager.state['steps'] = {
            step.name: {
                'status': 'pending',
                'attempts': 0,
                'parameters': step.parameters
            } for step in steps
        }

    async def _execute_with_retry(self, step: WorkflowStep):
        """Execute a workflow step with retries"""
        step_state = self.state_manager.state['steps'][step.name]
        
        for attempt in range(5):
            try:
                step_state['status'] = 'running'
                step_state['attempts'] += 1
                
                result = await self.task_agent.execute_step(step)
                
                if self.validation.validate_inputs(step.name, result):
                    self._update_state(step, result)
                    step_state['status'] = 'completed'
                    break
                    
            except Exception as e:
                logger.warning(f"Step {step.name} attempt {attempt+1} failed: {str(e)}")
                step_state['status'] = 'failed'
                step_state['last_error'] = str(e)
                
                if attempt == 4:  # Last attempt
                    raise RuntimeError(f"Step {step.name} failed after 5 attempts: {str(e)}")

    def _update_state(self, step: WorkflowStep, result: Dict[str, Any]):
        """Update workflow state with step results"""
        step_state = self.state_manager.state['steps'][step.name]
        step_state['result'] = result
        step_state['completed_at'] = str(datetime.now())
        
        # Archive step artifacts
        if 'output_files' in result:
            self.state_manager.state['artifacts'].update(result['output_files'])

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
