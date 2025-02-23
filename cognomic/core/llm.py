"""LLM interface for workflow generation and command creation."""

import json
import logging
from typing import Dict, Any, List
import openai
from openai import AsyncOpenAI
import glob
import networkx as nx

from ..utils import file_utils
from ..utils.logging import get_logger

logger = get_logger(__name__)

class LLMInterface:
    """Interface for LLM-based workflow generation."""
    
    def __init__(self):
        """Initialize LLM interface."""
        self.logger = get_logger(__name__)
        self.client = AsyncOpenAI()
    
    async def _call_openai(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Call OpenAI API with retry logic."""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {str(e)}")
            raise

    async def generate_workflow_plan(self, prompt: str) -> Dict[str, Any]:
        """Generate a workflow plan from a prompt."""
        try:
            # Get available input files
            fastq_files = glob.glob("*.fastq.gz")
            if not fastq_files:
                raise ValueError("No .fastq.gz files found in current directory")
            
            self.logger.info(f"Found input files: {fastq_files}")
            
            # Add specific instructions for dependency specification
            enhanced_prompt = f"""
You are a bioinformatics workflow expert. Generate a workflow plan as a JSON object with the following structure:
{{
    "steps": [
        {{
            "name": "step_name",
            "command": "command_to_execute",
            "parameters": {{"param1": "value1"}},
            "dependencies": ["dependent_step_name1"],
            "outputs": ["expected_output1"]
        }}
    ]
}}

Available input files: {fastq_files}
Task: {prompt}

Rules:
1. First step MUST be directory creation with this EXACT command:
   "mkdir -p results/rna_seq_analysis/fastqc results/rna_seq_analysis/kallisto_index results/rna_seq_analysis/kallisto_quant results/rna_seq_analysis/qc"

2. Dependencies must form a valid DAG (no cycles)
3. Each step needs a unique name
4. Process each file individually, no wildcards:
   - FastQC: fastqc file.fastq.gz -o results/rna_seq_analysis/fastqc
   - Kallisto index: kallisto index -i results/rna_seq_analysis/kallisto_index/transcripts.idx reference.fa
   - Kallisto quant: kallisto quant -o results/rna_seq_analysis/kallisto_quant/sample_name --single -l 200 -s 20 -i results/rna_seq_analysis/kallisto_index/transcripts.idx file.fastq.gz
   - MultiQC: multiqc results/rna_seq_analysis/fastqc results/rna_seq_analysis/kallisto_quant -o results/rna_seq_analysis/qc

5. Return ONLY the JSON object, no markdown formatting or other text
"""
            messages = [
                {"role": "system", "content": "You are a bioinformatics workflow expert. Return only valid JSON."},
                {"role": "user", "content": enhanced_prompt}
            ]
            
            response = await self._call_openai(messages)
            
            try:
                workflow_plan = json.loads(response)
                # Log workflow plan
                self.logger.info("Generated workflow plan:")
                for step in workflow_plan["steps"]:
                    self.logger.info(f"  Step: {step['name']}")
                    self.logger.info(f"    Command: {step['command']}")
                    self.logger.info(f"    Dependencies: {step['dependencies']}")
                
                return workflow_plan
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse workflow plan: {str(e)}")
                self.logger.error(f"Raw response: {response}")
                raise ValueError("Generated workflow plan is not valid JSON")
            
        except Exception as e:
            self.logger.error(f"Failed to generate workflow plan: {str(e)}")
            raise

    async def generate_command(self, step: Dict[str, Any]) -> str:
        """Generate a command for a workflow step."""
        try:
            # If command is already provided, use it
            if "command" in step and step["command"]:
                return step["command"]
            
            messages = [
                {"role": "system", "content": "You are a bioinformatics workflow expert. Generate precise shell commands."},
                {"role": "user", "content": f"Generate a shell command for the following workflow step:\n{json.dumps(step, indent=2)}"}
            ]
            
            response = await self._call_openai(messages)
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Failed to generate command: {str(e)}")
            raise
