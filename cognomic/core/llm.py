import os
import json
import logging
from typing import Any, Dict, Optional
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class LLMInterface:
    """Interface for LLM interactions"""
    def __init__(self):
        self.client = AsyncOpenAI()
        self.model = "gpt-3.5-turbo"
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client.api_key = self.api_key
        logger.info(f"Initialized LLM interface with model: {self.model}")
        
    async def generate(self, prompt: str) -> str:
        """Generate response from LLM"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a bioinformatics workflow assistant. Generate precise, safe commands for bioinformatics tools."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for more deterministic outputs
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            raise

    async def generate_workflow_plan(self, prompt: str) -> Dict[str, Any]:
        """Generate workflow plan from natural language prompt"""
        system_prompt = self._get_workflow_planning_prompt()
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            plan = json.loads(response.choices[0].message.content.strip())
            
            # Validate plan structure
            required_keys = ['inputs', 'steps', 'outputs', 'validation']
            if not all(key in plan for key in required_keys):
                raise ValueError(f"Workflow plan missing required keys: {required_keys}")
            if not isinstance(plan['steps'], list) or not plan['steps']:
                raise ValueError("Workflow plan must contain non-empty steps list")
                
            return plan
            
        except Exception as e:
            logger.error(f"Workflow planning failed: {str(e)}")
            raise

    def _get_workflow_planning_prompt(self) -> str:
        return """Plan a bioinformatics workflow based on the user's request. Follow these guidelines:

1. Determine input data type:
   - Check if files are single-end or paired-end reads
   - For paired-end, files should have _1/_2 or R1/R2 in their names
   - Process each file independently unless explicitly paired

2. Quality Control:
   - Run FastQC on each input file
   - Generate MultiQC report for all FastQC results

3. Kallisto Analysis:
   - Create Kallisto index from reference transcriptome
   - Run Kallisto quantification:
     * For single-end reads: process each file independently with --single -l 200 -s 20
     * For paired-end reads: process read pairs together
   - Verify index compatibility before quantification

4. Output Organization:
   - Create separate output directories for each sample
   - Use consistent naming conventions
   - Save all reports and logs

Return a JSON object with these fields:
{
    "steps": [
        {
            "name": "step name",
            "tool": "tool name",
            "action": "action type",
            "type": "step type",
            "parameters": {}
        }
    ]
}"""

    async def generate_command(self, tool: str, action: str, parameters: Dict[str, Any]) -> str:
        """Generate command for tool execution"""
        prompt = f"""Generate the exact command line for:
        Tool: {tool}
        Action: {action}
        Parameters: {json.dumps(parameters, indent=2)}
        
        Return only the command, no explanation.
        Ensure the command follows best practices and is safe to execute."""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a bioinformatics command generator. Generate precise, safe commands."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Command generation failed: {str(e)}")
            raise

    async def diagnose_error(self, error_context: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnose workflow error and suggest recovery"""
        prompt = f"""Analyze this workflow error and suggest recovery steps:
        Error: {error_context.get('error_log')}
        State: {json.dumps(error_context.get('workflow_state'), indent=2)}
        
        Return a JSON object with:
        1. diagnosis: Brief error analysis
        2. action: One of ['retry', 'fix', 'abort']
        3. solution: Specific steps to resolve the issue"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a bioinformatics error diagnostician. Analyze errors and suggest fixes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            return json.loads(response.choices[0].message.content.strip())
            
        except Exception as e:
            logger.error(f"Error diagnosis failed: {str(e)}")
            raise
