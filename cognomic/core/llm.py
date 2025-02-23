"""LLM interface for workflow planning and execution."""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from pathlib import Path

from ..config.settings import settings
from ..utils.logging import get_logger
from ..utils import file_utils
from .agent_types import WorkflowStep

logger = get_logger(__name__)

class LLMInterface:
    """Interface for LLM-based workflow planning and execution."""
    
    def __init__(self):
        """Initialize LLM interface."""
        self.logger = get_logger(__name__)
        self.model = settings.OPENAI_MODEL
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
    async def generate_workflow_plan(self, prompt: str) -> Dict[str, Any]:
        """Generate workflow steps from user prompt using LLM."""
        try:
            # Get current working directory
            cwd = os.getcwd()
            
            # Find FASTQ files in current directory (now returns relative paths)
            fastq_files = file_utils.find_fastq_files(cwd)
            self.logger.info(f"Found FASTQ files: {fastq_files}")
            
            # Create system prompt
            system_prompt = """You are an expert bioinformatics workflow planner. 
            Generate a detailed workflow plan for RNA-seq analysis using Kallisto.
            The workflow should include:
            1. FastQC for quality control
            2. Kallisto index for reference transcriptome
            3. Kallisto quant for quantification
            4. MultiQC to aggregate QC reports
            
            Important rules for file paths:
            1. ALWAYS use relative paths for all files
            2. Input files should be referenced relative to the current working directory
            3. Output files should be created under the specified output directory
            4. Never use absolute paths in any command
            5. For reference files, assume they are in the current working directory
            
            Return the plan as a JSON object with the following structure:
            {
                "steps": [
                    {
                        "name": "step_name",
                        "tool": "tool_name",
                        "action": "action_name",
                        "parameters": {
                            "param1": "value1",
                            ...
                        },
                        "dependencies": ["step_name1", "step_name2"]
                    }
                ]
            }"""
            
            # Create messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate a workflow plan for: {prompt}\nAvailable FASTQ files (relative paths): {fastq_files}"}
            ]
            
            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=2000
            )
            
            # Extract workflow plan from response
            workflow_plan = json.loads(response.choices[0].message.content)
            
            # Add MultiQC step if not present
            has_multiqc = any(step["tool"] == "MultiQC" for step in workflow_plan["steps"])
            if not has_multiqc:
                workflow_plan["steps"].append({
                    "name": "multiqc",
                    "tool": "MultiQC",
                    "action": "run",
                    "parameters": {
                        "input_dirs": ["results/rna_seq_analysis/fastqc_reports"],
                        "output_dir": "results/rna_seq_analysis/multiqc_report"
                    },
                    "dependencies": ["fastqc"]
                })
            
            self.logger.info(f"Generated workflow plan: {json.dumps(workflow_plan, indent=4)}")
            return workflow_plan
            
        except Exception as e:
            self.logger.error(f"Error generating workflow plan: {str(e)}")
            raise
            
    async def generate_command(self, step: Dict[str, Any]) -> str:
        """Generate shell command for a workflow step."""
        try:
            # Create system prompt
            system_prompt = """You are an expert bioinformatics command generator.
            Generate the exact shell command to execute a workflow step.
            The command should be runnable in a Unix/Linux environment.
            
            Important rules for file paths:
            1. ALWAYS use relative paths for all files
            2. Input files should be referenced relative to the current directory
            3. Output files should be created under the specified output directory
            4. Never use absolute paths in any command
            5. For reference files, assume they are in the current directory
            
            Important rules for Kallisto:
            1. For single-end RNA-seq data, process each input file separately
            2. Create a separate output directory for each input file
            3. Use --single -l 200 -s 20 for single-end reads
            
            Available tools:
            - FastQC: Quality control for sequencing data
              Command format: fastqc [options] <input_files> -o <output_dir>
            
            - Kallisto: RNA-seq quantification
              - index: Create transcriptome index
                Command format: kallisto index -i <output_index> <reference_transcriptome>
              - quant (single-end): Process each file separately
                Command format: kallisto quant -i <index> -o <output_dir>/<sample_name> --single -l 200 -s 20 <input_file>
              - quant (paired-end):
                Command format: kallisto quant -i <index> -o <output_dir> <input_files>
            
            - MultiQC: Aggregate QC reports
              Command format: multiqc <input_dirs> -o <output_dir>
            
            Return ONLY the command string, no explanation or markdown. For multiple commands, join them with ' && '."""
            
            # Create step description
            step_desc = f"""Tool: {step['tool']}
            Action: {step.get('action', 'run')}
            Parameters: {json.dumps(step['parameters'], indent=2)}"""
            
            # Create messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate command for:\n{step_desc}"}
            ]
            
            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=1000
            )
            
            # Extract command from response
            command = response.choices[0].message.content.strip()
            self.logger.info(f"Generated command: {command}")
            
            return command
            
        except Exception as e:
            self.logger.error(f"Error generating command: {str(e)}")
            raise

    async def analyze_error(self, error: str, context: Dict[str, Any]) -> str:
        """Analyze an error and provide debugging suggestions."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a bioinformatics workflow debugger. Analyze the error and provide debugging suggestions."
                },
                {
                    "role": "user",
                    "content": f"Error: {error}\nContext: {json.dumps(context, indent=2)}"
                }
            ]
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error analyzing error: {str(e)}")
            raise
