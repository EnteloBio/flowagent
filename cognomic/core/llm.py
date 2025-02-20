"""LLM interface for workflow planning and execution."""

from typing import Dict, Any, List
import json
import os
from pathlib import Path
import openai
from openai import AsyncOpenAI

from ..utils.logging import get_logger
from ..config.settings import settings

logger = get_logger(__name__)

class LLMInterface:
    """Interface for LLM-based workflow planning and execution."""
    
    def __init__(self, api_key: str = None):
        """Initialize LLM interface."""
        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise ValueError("OpenAI API key not found")
            
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=settings.OPENAI_BASE_URL
        )
        self.model = settings.OPENAI_MODEL
        self.logger = get_logger(__name__)
        
        self.logger.info(f"Initialized LLM interface with model: {self.model}")

    def _get_workflow_planning_prompt(self, request: str) -> str:
        """Get the prompt for workflow planning."""
        return f"""Plan a bioinformatics workflow based on the following request:
{request}

Follow these rules when planning:
1. Break down the workflow into logical steps
2. For each step specify:
   - Tool name
   - Action
   - Required parameters
   - Expected outputs
3. For file patterns:
   - For input files, just use '*' to match all files of appropriate type
   - The system will automatically handle file extensions (.fastq, .fastq.gz, etc.)
   - For output files and directories, use full paths with appropriate extensions
4. Always include quality control steps
5. Always specify output directories for each step
6. Consider error handling and data validation

Example workflow plan:
{{
    "workflow_type": "rna_seq",
    "steps": [
        {{
            "name": "quality_control",
            "tool": "FastQC",
            "action": "Perform quality control on the input fastq files",
            "type": "command",
            "parameters": {{
                "input": "*",  # Will match .fastq, .fastq.gz, .fq, .fq.gz
                "output_dir": "results/rna_seq_analysis/fastqc_reports"
            }}
        }},
        {{
            "name": "build_index",
            "tool": "kallisto",
            "action": "index",
            "type": "command",
            "parameters": {{
                "reference": "Homo_sapiens.GRCh38.cdna.all.fa",
                "output": "results/rna_seq_analysis/kallisto_index"
            }}
        }},
        {{
            "name": "quantify",
            "tool": "kallisto",
            "action": "quant",
            "type": "command",
            "parameters": {{
                "index": "results/rna_seq_analysis/kallisto_index",
                "input": "*",  # Will match .fastq, .fastq.gz, .fq, .fq.gz
                "output_dir": "results/rna_seq_analysis/kallisto_output",
                "single": true,
                "fragment_length": 200,
                "sd": 20
            }}
        }},
        {{
            "name": "generate_report",
            "tool": "MultiQC",
            "action": "Generate quality control report",
            "type": "command",
            "parameters": {{
                "input": "results/rna_seq_analysis",
                "output_dir": "results/rna_seq_analysis/multiqc_report"
            }}
        }}
    ]
}}

Generate a workflow plan in JSON format:"""

    def _get_error_diagnosis_prompt(self) -> str:
        """Get the prompt for error diagnosis."""
        return """You are an expert at diagnosing and fixing bioinformatics workflow errors.

Return a JSON object with the following structure:
{
    "diagnosis": "Detailed description of the error",
    "error_type": "Specific error category",
    "recovery_steps": [
        "Step 1 description",
        "Step 2 description"
    ],
    "suggestions": [
        "Suggestion 1",
        "Suggestion 2"
    ]
}"""

    def _get_command_generation_prompt(self) -> str:
        """Get the prompt for command generation."""
        return """You are an expert at generating shell commands for bioinformatics tools.
Your task is to generate the exact command line string for a given tool and action.

Common tools and their command formats:

1. Filesystem:
   - mkdir: mkdir -p <directory>
   - rm: rm [-r] <path>
   - cp: cp [-r] <source> <destination>

2. FastQC:
   - analyze: fastqc <input_files> -o <output_dir> [options]
   Note: FastQC can accept file globs (e.g., *.fastq.gz) directly

3. Kallisto:
   - index: kallisto index -i <index_file> <reference_fasta>
   - quant (single-end): kallisto quant -i <index> -o <output_dir> --single -l 200 -s 20 <single_input_file>
   - quant (paired-end): kallisto quant -i <index> -o <output_dir> <reads1> <reads2>
   Note: Kallisto quantification MUST process each file separately - never use globs

4. MultiQC:
   - report: multiqc <input_dir> -o <output_dir> [options]

Important Rules:
1. NEVER use shell operators like |, >, <, ;, &&, ||
2. For Kallisto quantification, if a glob pattern is detected, return the error message: 'ERROR: Kallisto quantification requires individual file processing'
3. For single-end RNA-seq, always include --single -l 200 -s 20 parameters
4. Ensure all paths are properly specified
5. FastQC and MultiQC can accept glob patterns directly, but Kallisto quantification cannot

Return ONLY the command string, with no additional text or explanation."""

    async def generate_workflow_plan(self, prompt: str) -> Dict[str, Any]:
        """Generate a workflow plan from a natural language prompt."""
        try:
            messages = [
                {"role": "system", "content": self._get_workflow_planning_prompt(prompt)},
                {"role": "user", "content": f"Create a detailed workflow plan for the following task: {prompt}"}
            ]
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            # Parse the response into a workflow plan
            workflow_text = response.choices[0].message.content
            try:
                workflow_plan = json.loads(workflow_text)
                
                # Validate workflow structure
                if not isinstance(workflow_plan, dict):
                    raise ValueError("Workflow plan must be a dictionary")
                if 'workflow_type' not in workflow_plan:
                    raise ValueError("Workflow plan must specify workflow_type")
                if 'steps' not in workflow_plan:
                    raise ValueError("Workflow plan must contain steps")
                if not isinstance(workflow_plan['steps'], list):
                    raise ValueError("Steps must be a list")
                
                # Validate each step
                seen_names = set()
                for step in workflow_plan['steps']:
                    # Check required fields
                    required = ['name', 'tool', 'action', 'parameters']
                    missing = [key for key in required if key not in step]
                    if missing:
                        raise ValueError(f"Step missing required keys: {missing}")
                    
                    # Validate name uniqueness
                    if step['name'] in seen_names:
                        raise ValueError(f"Duplicate step name: {step['name']}")
                    seen_names.add(step['name'])
                    
                    # Ensure parameters is a dictionary
                    if not isinstance(step['parameters'], dict):
                        raise ValueError(f"Parameters for step {step['name']} must be a dictionary")
                    
                    # Add optional fields if missing
                    if 'type' not in step:
                        step['type'] = 'command'
                    if 'description' not in step:
                        step['description'] = None
                
                return workflow_plan
                
            except json.JSONDecodeError:
                self.logger.error("Failed to parse workflow plan as JSON")
                raise ValueError("Invalid workflow plan format")
            
        except Exception as e:
            self.logger.error(f"Error generating workflow plan: {str(e)}")
            raise

    async def generate_command(self, tool: str, action: str, parameters: Dict[str, Any]) -> str:
        """Generate a command string for a given tool and action."""
        try:
            # Convert parameters to a formatted string
            params_str = json.dumps(parameters, indent=2)
            
            messages = [
                {"role": "system", "content": self._get_command_generation_prompt()},
                {"role": "user", "content": f"Generate the command for:\nTool: {tool}\nAction: {action}\nParameters: {params_str}"}
            ]
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,  # Lower temperature for more consistent outputs
                max_tokens=200  # Commands should be relatively short
            )
            
            command = response.choices[0].message.content.strip()
            
            # Basic validation
            if not command:
                raise ValueError("Generated empty command")
            
            return command
            
        except Exception as e:
            self.logger.error(f"Error generating command: {str(e)}")
            raise

    async def decompose_workflow(self, prompt: str) -> Dict[str, Any]:
        """Convert prompt into workflow steps."""
        return await self.generate_workflow_plan(prompt)

    async def diagnose_error(
        self,
        error_type: str,
        error_message: str,
        step_name: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Diagnose workflow errors and suggest recovery steps."""
        try:
            messages = [
                {"role": "system", "content": self._get_error_diagnosis_prompt()},
                {"role": "user", "content": f"""
Diagnose this workflow error and suggest recovery steps:
Error Type: {error_type}
Error Message: {error_message}
Step: {step_name}
Context: {json.dumps(context, indent=2)}
"""}
            ]
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            # Parse the response into diagnosis
            diagnosis_text = response.choices[0].message.content
            try:
                diagnosis = json.loads(diagnosis_text)
                required = ['diagnosis', 'error_type', 'recovery_steps', 'suggestions']
                if not all(key in diagnosis for key in required):
                    raise ValueError(f"Diagnosis missing required keys: {required}")
            except json.JSONDecodeError:
                self.logger.error("Failed to parse error diagnosis as JSON")
                raise ValueError("Invalid diagnosis format")
            
            return diagnosis
            
        except Exception as e:
            self.logger.error(f"Failed to diagnose error: {str(e)}")
            raise
