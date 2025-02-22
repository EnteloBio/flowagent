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

    async def _get_workflow_planning_prompt(self, request: str) -> str:
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
3. For file handling:
   - Never use glob patterns (*)
   - Each file must be processed individually
   - For input files, specify "input_file" and the system will handle each file
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
            "action": "analyze",
            "type": "command",
            "parameters": {{
                "input_file": "test1.fastq.gz",  # System will handle each file
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
                "input_file": "test1.fastq.gz",  # System will handle each file
                "output_dir": "results/rna_seq_analysis/kallisto_output",
                "single": true,
                "fragment_length": 200,
                "sd": 20
            }}
        }},
        {{
            "name": "generate_report",
            "tool": "MultiQC",
            "action": "report",
            "type": "command",
            "parameters": {{
                "input_dir": "results/rna_seq_analysis",
                "output_dir": "results/rna_seq_analysis/multiqc_report"
            }}
        }}
    ]
}}

Generate a workflow plan in JSON format:"""

    async def _get_error_diagnosis_prompt(self) -> str:
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

    async def _get_command_generation_prompt(self) -> str:
        """Get the prompt for command generation."""
        return """You are an expert at generating shell commands for bioinformatics tools.
Your task is to generate the exact command line string for a given tool and action.

Common tools and their command formats:

1. FastQC:
   - analyze: fastqc <input_file> -o <output_dir>
   Note: Process one file at a time, never use globs

2. Kallisto:
   - index: kallisto index -i <index_file> <reference_fasta>
   - quant (single-end): kallisto quant -i <index> -o <output_dir> --single -l 200 -s 20 <input_file>
   - quant (paired-end): kallisto quant -i <index> -o <output_dir> <reads1> <reads2>
   Note: Process one file at a time, never use globs

3. MultiQC:
   - report: multiqc <input_dir> -o <output_dir>

Important Rules:
1. NEVER use shell operators like |, >, <, ;, &&, ||
2. NEVER use glob patterns (*) in file paths
3. For tools that process individual files (like Kallisto, FastQC):
   - Create a separate output directory for each input file
   - Use the input filename (without extension) as part of the output directory
   Example: For input file 'sample1.fastq.gz', use 'output_dir/sample1' as output
4. For tools that aggregate results (like MultiQC):
   - Use the specified output directory as is
5. Always use explicit paths, never relative paths with ..
6. Never include explanatory text, only output the exact command to run
7. For Kallisto quantification:
   - CHECK THE PARAMETERS CAREFULLY - if "single": true exists, this is SINGLE-END data
   - For single-end data, you MUST add --single -l <fragment_length> -s <sd>
   - For paired-end data, DO NOT add the --single flag
   - Process each file separately with its own output directory
   - Fragment length and SD are REQUIRED for single-end data
   - Default fragment_length=200 and sd=20 if not specified

Example formats:
- Kallisto (single-end): kallisto quant -i index.idx -o output_dir/sample1 --single -l 200 -s 20 sample1.fastq.gz
- Kallisto (paired-end): kallisto quant -i index.idx -o output_dir/sample1 sample1_1.fastq.gz sample1_2.fastq.gz
- FastQC: fastqc -o output_dir/sample1 sample1.fastq.gz
- MultiQC: multiqc input_dir -o output_dir

IMPORTANT: Double check if "single": true exists in the parameters. If it does, you MUST use --single flag with -l and -s options.
"""

    async def generate_workflow_plan(self, prompt: str) -> Dict[str, Any]:
        """Generate a workflow plan from a natural language prompt."""
        try:
            messages = [
                {"role": "system", "content": await self._get_workflow_planning_prompt(prompt)},
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

    async def generate_command(self, step: Dict[str, Any]) -> str:
        """Generate a command for a workflow step."""
        try:
            # Create prompt
            prompt = f"""
            Generate a command for the following workflow step:
            Tool: {step['tool']}
            Action: {step['action']}
            Parameters: {json.dumps(step['parameters'], indent=2)}
            
            Important rules:
            1. NEVER use glob patterns (*) in file paths
            2. For tools that process individual files (like Kallisto, FastQC):
               - Create a separate output directory for each input file
               - Use the input filename (without extension) as part of the output directory
               Example: For input file 'sample1.fastq.gz', use 'output_dir/sample1' as output
            3. For tools that aggregate results (like MultiQC):
               - Use the specified output directory as is
            4. Always use explicit paths, never relative paths with ..
            5. Never include explanatory text, only output the exact command to run
            6. For Kallisto quantification:
               - CHECK THE PARAMETERS CAREFULLY - if "single": true exists, this is SINGLE-END data
               - For single-end data, you MUST add --single -l <fragment_length> -s <sd>
               - For paired-end data, DO NOT add the --single flag
               - Process each file separately with its own output directory
               - Fragment length and SD are REQUIRED for single-end data
               - Default fragment_length=200 and sd=20 if not specified
            
            Example formats:
            - Kallisto (single-end): kallisto quant -i index.idx -o output_dir/sample1 --single -l 200 -s 20 sample1.fastq.gz
            - Kallisto (paired-end): kallisto quant -i index.idx -o output_dir/sample1 sample1_1.fastq.gz sample1_2.fastq.gz
            - FastQC: fastqc -o output_dir/sample1 sample1.fastq.gz
            - MultiQC: multiqc input_dir -o output_dir
            
            IMPORTANT: Double check if "single": true exists in the parameters. If it does, you MUST use --single flag with -l and -s options.
            """
            
            # Get command from LLM
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a bioinformatics command generator. Only output the exact command to run, no other text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Lower temperature for more consistent outputs
                max_tokens=500,
                response_format={"type": "text"}
            )
            
            command = response.choices[0].message.content.strip()
            if not command:
                raise ValueError("Empty command generated")
                
            return command
            
        except Exception as e:
            self.logger.error(f"Failed to generate command: {str(e)}")
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
                {"role": "system", "content": await self._get_error_diagnosis_prompt()},
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

    async def generate_analysis(self, prompt: str) -> Dict[str, Any]:
        """Generate analysis from workflow outputs."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": """You are an expert bioinformatician analyzing workflow outputs.
                    For any workflow type, focus on providing clear, actionable insights about:
                    1. Quality metrics and their interpretation
                    2. Tool-specific outputs and their meaning
                    3. Any technical issues that need attention
                    4. Workflow-specific recommendations
                    
                    Format your response as a JSON object with these keys:
                    {
                        "overall_quality": "good|warning|poor",
                        "key_metrics": {
                            "tool_name": {
                                "metric_name": "value",
                                "interpretation": "what this value means"
                            }
                        },
                        "issues": [
                            {
                                "severity": "high|medium|low",
                                "description": "issue description",
                                "impact": "how this affects results",
                                "solution": "how to fix it"
                            }
                        ],
                        "warnings": [...],
                        "recommendations": [
                            {
                                "type": "quality|performance|analysis",
                                "description": "what to do",
                                "reason": "why this is recommended"
                            }
                        ]
                    }
                    """},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            # Parse JSON response
            analysis = json.loads(response.choices[0].message.content)
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error generating analysis: {str(e)}")
            return {
                "overall_quality": "unknown",
                "error": str(e),
                "issues": [{
                    "severity": "high",
                    "description": "Failed to analyze tool outputs",
                    "impact": "Cannot provide quality assessment",
                    "solution": "Check logs for details"
                }]
            }
