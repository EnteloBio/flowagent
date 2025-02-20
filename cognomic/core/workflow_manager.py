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

    def _workflow_plan_to_steps(self, plan: Dict[str, Any]) -> List[WorkflowStep]:
        """Convert workflow plan to list of WorkflowStep objects."""
        steps = []
        for step_data in plan['steps']:
            step = WorkflowStep(
                name=step_data['name'],
                tool=step_data['tool'],
                action=step_data['action'],
                parameters=step_data['parameters'],
                type=step_data.get('type', 'command')  # Default to command type
            )
            steps.append(step)
        return steps

    def _create_output_directories(self, steps: List[WorkflowStep]) -> None:
        """Create output directories specified in workflow steps."""
        for step in steps:
            # Create output directory if specified in parameters
            output_dir = step.parameters.get('output_dir')
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Created output directory: {output_dir}")

    def _initialize_state(self, steps: List[WorkflowStep]) -> None:
        """Initialize workflow state with steps."""
        self.state_manager.state['steps'] = {
            step.name: {
                'status': 'pending',
                'tool': step.tool,
                'action': step.action,
                'parameters': step.parameters
            }
            for step in steps
        }

    async def execute_from_prompt(self, prompt: str) -> Dict[str, Any]:
        """Execute workflow based on natural language prompt"""
        try:
            # Plan workflow steps from prompt
            logger.info("Planning workflow steps...")
            workflow_plan = await self.plan_agent.decompose_workflow(prompt)
            
            # Convert plan to steps
            steps = self._workflow_plan_to_steps(workflow_plan)
            
            # Initialize state and create directories
            self._initialize_state(steps)
            self._create_output_directories(steps)
            
            # Update task agent's working directory
            self.task_agent.cwd = self.initial_cwd
            logger.info(f"Set task agent working directory to: {self.task_agent.cwd}")
            
            # Execute steps in order
            for step in steps:
                logger.info(f"Executing step: {step.name}")
                try:
                    await self._execute_with_retry(step)
                except Exception as e:
                    logger.error(f"Step failed: {step.name} - {str(e)}")
                    raise
                
            # Archive results
            logger.info("Archiving workflow results...")
            return self.state_manager.archive_results()
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            try:
                if self.debug_agent:
                    error_context = {
                        'prompt': prompt,
                        'error': str(e),
                        'state': self.state_manager.state
                    }
                    diagnosis = await self.debug_agent.diagnose_failure(error_context)
                    logger.info(f"Error diagnosis: {diagnosis}")
            except Exception as debug_error:
                logger.error(f"Error diagnosis failed: {str(debug_error)}")
            raise

    async def _execute_with_retry(self, step: WorkflowStep) -> None:
        """Execute a workflow step with retry logic."""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Special handling for Kallisto quantification
                if step.tool.lower() == 'kallisto' and step.action.lower() == 'quant':
                    # Get input files
                    input_pattern = step.parameters.get('input')
                    if input_pattern and '*' in input_pattern:
                        # Get list of matching files
                        import glob
                        input_files = sorted(glob.glob(input_pattern))
                        logger.info(f"Found input files: {input_files}")
                        
                        # Process each file individually
                        for input_file in input_files:
                            file_params = step.parameters.copy()
                            file_params['input'] = input_file
                            
                            # Create file-specific step
                            file_step = WorkflowStep(
                                name=f"{step.name}_{os.path.basename(input_file)}",
                                tool=step.tool,
                                action=step.action,
                                parameters=file_params,
                                type=step.type
                            )
                            
                            # Execute for individual file
                            await self.task_agent.execute_step(file_step)
                            self.state_manager.state['steps'][file_step.name] = {
                                'status': 'completed',
                                'tool': file_step.tool,
                                'action': file_step.action,
                                'parameters': file_step.parameters
                            }
                    else:
                        # Single file case
                        await self.task_agent.execute_step(step)
                        self.state_manager.state['steps'][step.name]['status'] = 'completed'
                else:
                    # Normal execution for other steps
                    await self.task_agent.execute_step(step)
                    self.state_manager.state['steps'][step.name]['status'] = 'completed'
                break
                
            except Exception as e:
                retry_count += 1
                logger.warning(f"Step failed (attempt {retry_count}/{max_retries}): {str(e)}")
                if retry_count >= max_retries:
                    logger.error(f"Step failed after {max_retries} attempts")
                    self.state_manager.state['steps'][step.name]['status'] = 'failed'
                    self.state_manager.state['steps'][step.name]['error'] = str(e)
                    raise
