"""LLM interface for workflow planning and command generation."""

import os
import json
from typing import Dict, Any, List, Optional
import openai
from openai import AsyncOpenAI
import glob

from ..utils import file_utils
from ..utils.logging import get_logger

class LLMInterface:
    """Interface for interacting with LLM."""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """Initialize LLM interface."""
        self.logger = get_logger(__name__)
        self.model = model
        self.client = AsyncOpenAI()
        
    async def _validate_json(self, content: str) -> Dict[str, Any]:
        """Validate and parse JSON content."""
        try:
            # Remove any markdown formatting
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            # Parse JSON
            return json.loads(content)
        except Exception as e:
            self.logger.error(f"Invalid JSON response: {content}")
            self.logger.error(f"JSON parsing error: {str(e)}")
            raise ValueError(f"Invalid JSON response from LLM: {str(e)}")
        
    async def _call_openai(self, messages: List[Dict[str, Any]], response_format: Optional[Dict[str, str]] = None) -> str:
        """Call OpenAI API."""
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 2000
            }
            
            if response_format:
                kwargs["response_format"] = response_format
                
            response = await self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error calling OpenAI API: {str(e)}")
            raise

    async def generate_workflow_plan(self, prompt: str) -> Dict[str, Any]:
        """Generate workflow plan from user prompt."""
        try:
            # Find input files
            fastq_files = glob.glob("*.fastq.gz")
            if not fastq_files:
                raise ValueError("No .fastq.gz files found in current directory")
            
            self.logger.info(f"Found input files: {fastq_files}")
                
            messages = [
                {"role": "system", "content": """You are a bioinformatics workflow planner. Generate a workflow plan in JSON format.
                You MUST return a JSON object with EXACTLY this structure:
                {
                    "steps": [
                        {
                            "name": "step_name",
                            "tool": "tool_name",
                            "action": "action_name",
                            "parameters": {
                                "param1": "value1",
                                "param2": "value2"
                            },
                            "dependencies": ["step_name_1", "step_name_2"]
                        }
                    ]
                }
                
                Important rules for RNA-seq workflows:
                1. For Kallisto steps:
                   - Add a mkdir step before index creation and quantification
                   - Index creation must happen before quantification
                   - Index path must be consistent between steps
                   - Add dependencies between steps
                   - For multiple input files, create separate quantification steps with unique names
                2. For FastQC:
                   - Run FastQC on every input FASTQ file
                   - Create separate FastQC steps with unique names
                   - Output to results/rna_seq_analysis/fastqc directory
                3. For MultiQC:
                   - Add a mkdir step before running MultiQC
                   - Add dependencies on ALL quantification AND FastQC steps
                   - Run from root directory to capture all reports
                4. All output paths should be relative to working directory
                5. Each step must have a unique name and clear dependencies
                6. Use EXACT input file names, no wildcards
                
                Example workflow:
                {
                    "steps": [
                        {
                            "name": "mkdir_fastqc",
                            "tool": "Command",
                            "action": "run",
                            "parameters": {
                                "command": "mkdir -p results/rna_seq_analysis/fastqc"
                            },
                            "dependencies": []
                        },
                        {
                            "name": "fastqc_1",
                            "tool": "FastQC",
                            "action": "run",
                            "parameters": {
                                "input": "sample1.fastq.gz",
                                "outdir": "results/rna_seq_analysis/fastqc"
                            },
                            "dependencies": ["mkdir_fastqc"]
                        },
                        {
                            "name": "fastqc_2",
                            "tool": "FastQC",
                            "action": "run",
                            "parameters": {
                                "input": "sample2.fastq.gz",
                                "outdir": "results/rna_seq_analysis/fastqc"
                            },
                            "dependencies": ["mkdir_fastqc"]
                        },
                        {
                            "name": "mkdir_kallisto_index",
                            "tool": "Command",
                            "action": "run",
                            "parameters": {
                                "command": "mkdir -p results/rna_seq_analysis/kallisto_index"
                            },
                            "dependencies": []
                        },
                        {
                            "name": "kallisto_index",
                            "tool": "Kallisto",
                            "action": "index",
                            "parameters": {
                                "index": "results/rna_seq_analysis/kallisto_index/transcripts.idx",
                                "fasta": "reference.fa"
                            },
                            "dependencies": ["mkdir_kallisto_index"]
                        },
                        {
                            "name": "mkdir_kallisto_quant",
                            "tool": "Command",
                            "action": "run",
                            "parameters": {
                                "command": "mkdir -p results/rna_seq_analysis/kallisto_quant"
                            },
                            "dependencies": ["kallisto_index"]
                        },
                        {
                            "name": "kallisto_quant_1",
                            "tool": "Kallisto",
                            "action": "quant",
                            "parameters": {
                                "index": "results/rna_seq_analysis/kallisto_index/transcripts.idx",
                                "output_dir": "results/rna_seq_analysis/kallisto_quant/sample1",
                                "fastq": "sample1.fastq.gz"
                            },
                            "dependencies": ["mkdir_kallisto_quant"]
                        },
                        {
                            "name": "kallisto_quant_2",
                            "tool": "Kallisto",
                            "action": "quant",
                            "parameters": {
                                "index": "results/rna_seq_analysis/kallisto_index/transcripts.idx",
                                "output_dir": "results/rna_seq_analysis/kallisto_quant/sample2",
                                "fastq": "sample2.fastq.gz"
                            },
                            "dependencies": ["mkdir_kallisto_quant"]
                        },
                        {
                            "name": "mkdir_multiqc",
                            "tool": "Command",
                            "action": "run",
                            "parameters": {
                                "command": "mkdir -p results/rna_seq_analysis/qc"
                            },
                            "dependencies": []
                        },
                        {
                            "name": "multiqc",
                            "tool": "MultiQC",
                            "action": "run",
                            "parameters": {
                                "input_dir": ".",
                                "output_dir": "results/rna_seq_analysis/qc"
                            },
                            "dependencies": [
                                "kallisto_quant_1",
                                "kallisto_quant_2",
                                "fastqc_1",
                                "fastqc_2",
                                "mkdir_multiqc"
                            ]
                        }
                    ]
                }"""},
                {"role": "user", "content": f"Available input files: {fastq_files}\n\n{prompt}"}
            ]
            
            response = await self._call_openai(messages, response_format={"type": "json_object"})
            workflow_plan = json.loads(response)
            
            self.logger.info(f"Generated workflow plan:\n{json.dumps(workflow_plan, indent=2)}")
            
            # Validate workflow structure
            if not isinstance(workflow_plan, dict) or "steps" not in workflow_plan:
                raise ValueError("Invalid workflow plan: missing 'steps' key")
            
            if not isinstance(workflow_plan["steps"], list):
                raise ValueError("Invalid workflow plan: 'steps' must be a list")
            
            for step in workflow_plan["steps"]:
                required_keys = {"name", "tool", "action", "parameters", "dependencies"}
                if not all(key in step for key in required_keys):
                    raise ValueError(f"Invalid step: missing required keys. Step: {step}")
            
            return workflow_plan
                
        except Exception as e:
            self.logger.error(f"Error generating workflow plan: {str(e)}")
            raise

    async def generate_command(self, step: Dict[str, Any]) -> str:
        """Generate command for workflow step."""
        try:
            # Get available input files for context
            fastq_files = glob.glob("*.fastq.gz")
            
            self.logger.info(f"Generating command for step: {step['name']}")
            self.logger.info(f"Step details:\n{json.dumps(step, indent=2)}")
            
            if step["tool"] == "System" and step["action"] == "mkdir":
                dirs = step["parameters"].get("dirs", [])
                if not dirs:
                    raise ValueError("No directories specified for mkdir")
                return f"mkdir -p {' '.join(dirs)}"
            
            messages = [
                {"role": "system", "content": f"""Generate shell commands for bioinformatics tools.
                Available input files in current directory: {fastq_files}
                
                CRITICAL RULES:
                1. NEVER modify input file paths - use them exactly as provided
                2. Input FASTQ files ({', '.join(fastq_files)}) are in current directory
                3. Only add paths to output files and directories
                4. For single-end RNA-seq:
                   - Process each file separately
                   - Create a separate output directory for each sample
                   - Return one command per line
                
                Command templates (DO NOT CHANGE INPUT FILE NAMES):
                - FastQC:
                  fastqc test1.fastq.gz -o results/fastqc
                  fastqc test2.fastq.gz -o results/fastqc
                
                - Kallisto index:
                  kallisto index -i results/kallisto/index.idx reference.fa
                
                - Kallisto quant (single-end):
                  kallisto quant -i results/kallisto/index.idx --single -l 200 -s 20 -o results/kallisto/test1 test1.fastq.gz
                  kallisto quant -i results/kallisto/index.idx --single -l 200 -s 20 -o results/kallisto/test2 test2.fastq.gz
                
                - MultiQC:
                  multiqc . -o results/rna_seq_analysis/qc
                
                Return ONLY command strings, one per line."""},
                {"role": "user", "content": json.dumps(step)}
            ]
            
            commands = await self._call_openai(messages)
            commands = commands.strip()
            
            # Validate no path prefixes were added to input files
            for fastq in fastq_files:
                if f"results/{fastq}" in commands or f"output/{fastq}" in commands:
                    commands = commands.replace(f"results/{fastq}", fastq)
                    commands = commands.replace(f"output/{fastq}", fastq)
                    self.logger.warning(f"Removed incorrect path prefix from input file {fastq}")
            
            # For single-end RNA-seq, we expect multiple commands
            if step.get("tool") == "Kallisto" and step.get("action") == "quant":
                self.logger.info(f"Generated commands for {step['name']}:")
                for cmd in commands.split("\n"):
                    if cmd.strip():
                        self.logger.info(f"  {cmd.strip()}")
                return commands
            else:
                self.logger.info(f"Generated command for {step['name']}: {commands}")
                return commands
            
        except Exception as e:
            self.logger.error(f"Error generating command: {str(e)}")
            raise

    async def analyze_error(self, error_msg: str, step: Dict[str, Any]) -> str:
        """Analyze error message and provide diagnosis."""
        try:
            self.logger.info(f"Analyzing error for step {step['name']}: {error_msg}")
            
            messages = [
                {"role": "system", "content": """Analyze bioinformatics tool errors and provide diagnosis.
                Common error patterns:
                1. Directory errors:
                   - Could not create directory: Parent directory doesn't exist
                   - Permission denied: Need to create parent directories first
                   - Path doesn't exist: Directory not created before use
                
                2. File errors:
                   - File not found: Input file doesn't exist or wrong path
                   - Permission denied: File access issues
                   - Invalid format: File format not supported
                
                3. Tool-specific errors:
                   - Kallisto index not found: Index not created or wrong path
                   - Kallisto version mismatch: Index created with different version
                   - MultiQC no input files: Source directory empty or wrong path
                
                Return a clear, actionable diagnosis with:
                1. Root cause of the error
                2. Specific steps to fix it
                3. How to prevent it in future"""},
                {"role": "user", "content": f"Tool: {step['tool']}\nAction: {step['action']}\nError:\n{error_msg}\n\nStep details:\n{json.dumps(step, indent=2)}"}
            ]
            
            diagnosis = await self._call_openai(messages)
            diagnosis = diagnosis.strip()
            
            self.logger.info(f"Error diagnosis for {step['name']}: {diagnosis}")
            return diagnosis
            
        except Exception as e:
            self.logger.error(f"Error analyzing error: {str(e)}")
            raise
