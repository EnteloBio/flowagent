"""Workflow execution engine with custom script integration."""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from .script_manager import ScriptManager, ScriptInfo
from .llm import LLMInterface
from ..utils.logging import get_logger

@dataclass
class WorkflowStep:
    """Represents a step in the workflow."""
    name: str
    type: str  # 'standard' or 'custom'
    inputs: Dict[str, str]
    outputs: Dict[str, str]
    script_info: Optional[ScriptInfo] = None

class WorkflowExecutor:
    """Executes workflows with integrated custom scripts."""
    
    def __init__(self, llm_interface: LLMInterface):
        """Initialize workflow executor.
        
        Args:
            llm_interface: LLM interface for workflow customization
        """
        self.llm = llm_interface
        self.logger = get_logger(__name__)
        self.script_manager = ScriptManager(
            os.path.join(os.path.dirname(__file__), "..", "custom_scripts")
        )
        
    async def execute_workflow(self, 
                             input_data: Dict[str, str],
                             workflow_type: str,
                             custom_script_requests: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute workflow with custom scripts.
        
        Args:
            input_data: Dictionary of input file paths
            workflow_type: Type of workflow (e.g., "rna_seq", "chip_seq")
            custom_script_requests: Optional list of custom scripts to include
            
        Returns:
            Dictionary containing workflow results
        """
        try:
            # 1. Build workflow DAG
            workflow_steps = await self._build_workflow(
                workflow_type, 
                custom_script_requests
            )
            
            # 2. Initialize results dictionary with input data
            results = input_data.copy()
            
            # 3. Execute each step in order
            for step in workflow_steps:
                self.logger.info(f"Executing step: {step.name}")
                
                try:
                    # Execute custom script
                    if step.type == 'custom':
                        if not step.script_info:
                            raise ValueError(f"Missing script info for custom step {step.name}")
                            
                        # Validate script requirements
                        if not self.script_manager.validate_script_requirements(step.script_info):
                            raise RuntimeError(f"Script requirements not met for {step.name}")
                        
                        # Execute script with proper inputs
                        step_inputs = {
                            name: results[path]
                            for name, path in step.inputs.items()
                        }
                        
                        step_results = await self.script_manager.execute_script(
                            step.script_info,
                            step_inputs
                        )
                        
                    # Execute standard workflow step
                    else:
                        step_results = await self._execute_standard_step(
                            step.name,
                            step.inputs,
                            workflow_type
                        )
                    
                    # Update results with step outputs
                    results.update(step_results)
                    
                except Exception as e:
                    self.logger.error(f"Error in step {step.name}: {str(e)}")
                    raise
                    
            return results
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            raise
            
    async def _build_workflow(self, 
                            workflow_type: str,
                            custom_script_requests: Optional[List[str]] = None) -> List[WorkflowStep]:
        """Build workflow DAG incorporating custom scripts.
        
        Args:
            workflow_type: Type of workflow
            custom_script_requests: Optional list of custom scripts to include
            
        Returns:
            List of WorkflowStep objects in execution order
        """
        # 1. Get standard workflow steps
        standard_steps = self._get_standard_workflow_steps(workflow_type)
        
        # 2. Get requested custom scripts
        custom_scripts = []
        if custom_script_requests:
            for script_name in custom_script_requests:
                script = self.script_manager.get_script(script_name)
                if not script:
                    raise ValueError(f"Custom script not found: {script_name}")
                custom_scripts.append(script)
        
        # 3. Get all compatible scripts if no specific requests
        else:
            custom_scripts = self.script_manager.get_scripts_for_workflow(workflow_type)
        
        # 4. Build DAG
        workflow_steps = []
        
        # Add standard steps first
        for step_name in standard_steps:
            workflow_steps.append(WorkflowStep(
                name=step_name,
                type='standard',
                inputs=self._get_standard_step_inputs(step_name),
                outputs=self._get_standard_step_outputs(step_name)
            ))
        
        # Insert custom scripts based on execution order
        for script in custom_scripts:
            # Find position based on before/after requirements
            pos = 0
            for after_step in script.execution_order['after']:
                if after_step in [s.name for s in workflow_steps]:
                    pos = max(pos, next(
                        i for i, s in enumerate(workflow_steps)
                        if s.name == after_step
                    ) + 1)
            
            for before_step in script.execution_order['before']:
                if before_step in [s.name for s in workflow_steps]:
                    pos = min(pos, next(
                        i for i, s in enumerate(workflow_steps)
                        if s.name == before_step
                    ))
            
            # Create input/output mappings
            script_inputs = {
                req['name']: f"{script.name}_{req['name']}"
                for req in script.input_requirements
            }
            
            script_outputs = {
                out['name']: f"{script.name}_{out['name']}"
                for out in script.output_types
            }
            
            # Insert custom step
            workflow_steps.insert(pos, WorkflowStep(
                name=script.name,
                type='custom',
                inputs=script_inputs,
                outputs=script_outputs,
                script_info=script
            ))
        
        return workflow_steps
    
    def _get_standard_workflow_steps(self, workflow_type: str) -> List[str]:
        """Get standard steps for a workflow type."""
        if workflow_type == "rna_seq":
            return ["fastqc", "alignment", "feature_counts", "differential_expression"]
        elif workflow_type == "chip_seq":
            return ["fastqc", "alignment", "peak_calling", "motif_analysis"]
        else:
            raise ValueError(f"Unsupported workflow type: {workflow_type}")
    
    def _get_standard_step_inputs(self, step_name: str) -> Dict[str, str]:
        """Get input requirements for a standard step."""
        # This would be replaced with actual input requirements
        return {"input": f"{step_name}_input"}
    
    def _get_standard_step_outputs(self, step_name: str) -> Dict[str, str]:
        """Get output types for a standard step."""
        # This would be replaced with actual output types
        return {"output": f"{step_name}_output"}
    
    async def _execute_standard_step(self,
                                   step_name: str,
                                   inputs: Dict[str, str],
                                   workflow_type: str) -> Dict[str, Any]:
        """Execute a standard workflow step."""
        # This would be replaced with actual step execution
        return {f"{step_name}_output": f"path/to/{step_name}_result"}
