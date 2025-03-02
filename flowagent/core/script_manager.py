"""Manager for custom workflow scripts."""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import subprocess
from ..utils.logging import get_logger

@dataclass
class ScriptInfo:
    """Information about a custom script."""
    name: str
    description: str
    script_file: str
    language: str
    input_requirements: List[Dict[str, str]]
    output_types: List[Dict[str, str]]
    workflow_types: List[str]
    execution_order: Dict[str, List[str]]
    requirements: Dict[str, List[str]]
    path: Path

class ScriptManager:
    """Manages custom workflow scripts."""
    
    def __init__(self, custom_scripts_dir: str):
        """Initialize script manager.
        
        Args:
            custom_scripts_dir: Path to custom scripts directory
        """
        self.scripts_dir = Path(custom_scripts_dir)
        self.logger = get_logger(__name__)
        self.scripts: Dict[str, ScriptInfo] = {}
        self._load_scripts()
    
    def _load_scripts(self):
        """Load all custom scripts from the scripts directory."""
        for workflow_type in self.scripts_dir.iterdir():
            if not workflow_type.is_dir() or workflow_type.name == 'templates':
                continue
                
            self.logger.info(f"Loading scripts from {workflow_type.name}")
            for script_dir in workflow_type.rglob("**/metadata.json"):
                try:
                    with open(script_dir, 'r') as f:
                        metadata = json.load(f)
                    
                    script_path = script_dir.parent / metadata['script_file']
                    if not script_path.exists():
                        self.logger.warning(f"Script file {metadata['script_file']} not found in {script_dir.parent}")
                        continue
                    
                    script_info = ScriptInfo(
                        name=metadata['name'],
                        description=metadata['description'],
                        script_file=metadata['script_file'],
                        language=metadata['language'],
                        input_requirements=metadata['input_requirements'],
                        output_types=metadata['output_types'],
                        workflow_types=metadata['workflow_types'],
                        execution_order=metadata['execution_order'],
                        requirements=metadata['requirements'],
                        path=script_dir.parent
                    )
                    
                    self.scripts[metadata['name']] = script_info
                    self.logger.info(f"Loaded script {metadata['name']}")
                    
                except Exception as e:
                    self.logger.error(f"Error loading script from {script_dir}: {str(e)}")
    
    def get_script(self, name: str) -> Optional[ScriptInfo]:
        """Get script info by name."""
        return self.scripts.get(name)
    
    def get_scripts_for_workflow(self, workflow_type: str) -> List[ScriptInfo]:
        """Get all scripts compatible with a workflow type."""
        return [
            script for script in self.scripts.values()
            if workflow_type in script.workflow_types
        ]
    
    def validate_script_requirements(self, script: ScriptInfo) -> bool:
        """Validate that all script requirements are met."""
        try:
            if script.language.lower() == 'r':
                return self._validate_r_packages(script.requirements.get('r_packages', []))
            elif script.language.lower() == 'python':
                return self._validate_python_packages(script.requirements.get('python_packages', []))
            return True
        except Exception as e:
            self.logger.error(f"Error validating requirements for {script.name}: {str(e)}")
            return False
    
    def _validate_r_packages(self, packages: List[str]) -> bool:
        """Validate R package requirements."""
        if not packages:
            return True
            
        try:
            check_cmd = ['Rscript', '-e', 
                        'installed <- installed.packages(); ' + 
                        'pkg_check <- function(pkg) {pkg %in% installed[,1]}; ' +
                        f'result <- all(sapply(c({",".join(repr(p) for p in packages)}), pkg_check)); ' +
                        'quit(status=if(result) 0 else 1)']
            result = subprocess.run(check_cmd, capture_output=True)
            return result.returncode == 0
        except Exception as e:
            self.logger.error(f"Error checking R packages: {str(e)}")
            return False
    
    def _validate_python_packages(self, packages: List[str]) -> bool:
        """Validate Python package requirements."""
        if not packages:
            return True
            
        try:
            import importlib
            return all(importlib.util.find_spec(pkg) is not None for pkg in packages)
        except Exception as e:
            self.logger.error(f"Error checking Python packages: {str(e)}")
            return False
    
    async def execute_script(self, script: ScriptInfo, inputs: Dict[str, str]) -> Dict[str, Any]:
        """Execute a custom script.
        
        Args:
            script: Script information
            inputs: Dictionary mapping input names to file paths
            
        Returns:
            Dictionary containing output file paths
        """
        try:
            # Validate inputs
            for req in script.input_requirements:
                if req['name'] not in inputs:
                    raise ValueError(f"Missing required input: {req['name']}")
            
            # Prepare command based on language
            if script.language.lower() == 'r':
                cmd = ['Rscript', script.path / script.script_file]
            elif script.language.lower() == 'python':
                cmd = ['python', script.path / script.script_file]
            elif script.language.lower() == 'bash':
                cmd = ['bash', script.path / script.script_file]
            else:
                raise ValueError(f"Unsupported language: {script.language}")
            
            # Add input arguments
            for name, path in inputs.items():
                cmd.extend(['--' + name, path])
            
            # Execute script
            self.logger.info(f"Executing script {script.name}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Script execution failed: {result.stderr}")
            
            # Parse outputs from script output
            outputs = json.loads(result.stdout)
            return outputs
            
        except Exception as e:
            self.logger.error(f"Error executing script {script.name}: {str(e)}")
            raise
