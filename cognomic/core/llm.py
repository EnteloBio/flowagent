"""LLM interface for workflow planning and execution."""

from typing import Dict, Any, List, Set
import json
import os
from pathlib import Path
import openai
from openai import AsyncOpenAI

from ..utils.logging import get_logger
from ..config.settings import settings
from .tool_tracker import ToolTracker
from .agent_system import PLAN_agent, TASK_agent, DEBUG_agent

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
        self.tool_tracker = ToolTracker()
        
        self.logger.info(f"Initialized LLM interface with model: {self.model}")

    async def _get_workflow_planning_prompt(self, request: str) -> str:
        """Get the prompt for workflow planning."""
        logger.info("Getting workflow planning prompt")
        return f"""Plan a bioinformatics workflow based on the following request:
{request}

IMPORTANT: You must return a valid JSON object with ALL required fields. The response MUST be parseable JSON.

REQUIRED FIELDS:
1. "workflow_type": Type of workflow (e.g. "rna_seq")
2. "steps": List of steps, where each step MUST have:
   - "name": (REQUIRED) Unique identifier for the step
   - "tool": (REQUIRED) Name of the tool to use
   - "action": (REQUIRED) Specific action to perform
   - "parameters": (REQUIRED) Dictionary of parameters

RULES:
1. Each step MUST have ALL required fields
2. Step names must be unique (e.g. "fastqc_analysis", "kallisto_index")
3. Never use glob patterns (*) in file paths
4. Each file must be processed individually
5. Always include quality control steps
6. Always specify output directories

Here is an example of a VALID response:
{{
    "workflow_type": "rna_seq",
    "steps": [
        {{
            "name": "fastqc_analysis",
            "tool": "FastQC",
            "action": "analyze",
            "parameters": {{
                "input_files": step.parameters["input_files"],  # Ensure this is populated correctly
                "output_dir": "results/rna_seq_analysis/fastqc_reports"
            }}
        }},
        {{
            "name": "kallisto_index",
            "tool": "kallisto",
            "action": "index",
            "parameters": {{
                "reference": "Homo_sapiens.GRCh38.cdna.all.fa",
                "output": "results/rna_seq_analysis/kallisto_index"
            }}
        }},
        {{
            "name": "kallisto_quant",
            "tool": "kallisto",
            "action": "quant",
            "parameters": {{
                "index": "results/rna_seq_analysis/kallisto_index",
                "input_files": step.parameters["input_files"],  # Ensure this is populated correctly
                "output_dir": "results/rna_seq_analysis/kallisto_output",
                "single": true,
                "fragment_length": 200,
                "sd": 20
            }}
        }},
        {{
            "name": "multiqc_report",
            "tool": "MultiQC",
            "action": "aggregate",
            "parameters": {{
                "input_dir": "results/rna_seq_analysis",
                "output_dir": "results/rna_seq_analysis/multiqc_report"
            }}
        }}
    ]
}}"""

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
            self.logger.info(f"LLM Response: {workflow_text}")
            
            try:
                workflow_plan = json.loads(workflow_text)
                
                # Validate workflow structure
                if not isinstance(workflow_plan, dict):
                    raise ValueError("Workflow plan must be a dictionary")
                if 'workflow_type' not in workflow_plan:
                    workflow_plan['workflow_type'] = 'rna_seq'  # Default to RNA-seq
                if 'steps' not in workflow_plan:
                    raise ValueError("Workflow plan must contain steps")
                
                # Post-process steps to ensure all required fields
                processed_steps = []
                for i, step in enumerate(workflow_plan['steps']):
                    if not isinstance(step, dict):
                        raise ValueError(f"Step {i + 1} must be a dictionary")
                        
                    # Ensure all required fields exist with proper defaults
                    processed_step = {
                        'name': step.get('name', f'step_{i + 1}'),
                        'tool': step.get('tool'),
                        'action': step.get('action'),
                        'parameters': step.get('parameters', {}),
                        'dependencies': step.get('dependencies', []),
                        'type': step.get('type', 'command'),
                        'description': step.get('description', '')
                    }
                    
                    # Validate required fields
                    if not processed_step['tool']:
                        raise ValueError(f"Step {i + 1} missing required field: tool")
                    if not processed_step['action']:
                        raise ValueError(f"Step {i + 1} missing required field: action")
                        
                    # Add step-specific defaults
                    if processed_step['tool'] == 'kallisto' and processed_step['action'] == 'quant':
                        if processed_step['parameters'].get('single', False):
                            processed_step['parameters'].setdefault('fragment_length', 200)
                            processed_step['parameters'].setdefault('fragment_sd', 20)
                            
                    processed_steps.append(processed_step)
                
                workflow_plan['steps'] = processed_steps
                return workflow_plan
                
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in workflow plan: {str(e)}")
                
        except Exception as e:
            self.logger.error(f"Error generating workflow plan: {str(e)}")
            raise

    async def generate_workflow_plan_from_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a workflow plan for RNA-seq analysis."""
        self.logger.info("Generating workflow plan")
        
        # Define the workflow steps
        workflow_plan = {
            "workflow_type": "RNA-seq analysis",
            "steps": [
                {
                    "name": "find_fastq",
                    "tool": "find_by_name",
                    "action": "SearchDirectory",
                    "parameters": {
                        "SearchDirectory": ".",
                        "Pattern": "*.fastq.gz",
                        "Type": "file"
                    },
                    "dependencies": []
                },
                {
                    "name": "kallisto_index",
                    "tool": "kallisto",
                    "action": "index",
                    "parameters": {
                        "reference": "Homo_sapiens.GRCh38.cdna.all.fa",
                        "output": "index.idx"
                    },
                    "dependencies": []
                },
                {
                    "name": "kallisto_quant",
                    "tool": "kallisto",
                    "action": "quant",
                    "parameters": {
                        "index": "index.idx",
                        "output_dir": "results/rna_seq_analysis",
                        "single": True,
                        "fragment_length": 200,
                        "fragment_sd": 20
                    },
                    "dependencies": ["find_fastq", "kallisto_index"]
                },
                {
                    "name": "fastqc",
                    "tool": "fastqc",
                    "action": "analyze",
                    "parameters": {
                        "output_dir": "results/rna_seq_analysis/fastqc_reports"
                    },
                    "dependencies": ["find_fastq"]
                },
                {
                    "name": "multiqc",
                    "tool": "multiqc",
                    "action": "aggregate",
                    "parameters": {
                        "input_dir": "results/rna_seq_analysis/fastqc_reports",
                        "output_dir": "results/rna_seq_analysis/multiqc_report"
                    },
                    "dependencies": ["fastqc"]
                }
            ]
        }
        
        self.logger.info("Workflow plan generated successfully")
        return workflow_plan

    async def generate_command(self, step: Dict[str, Any]) -> str:
        """Generate a command for a workflow step."""
        from .agent_system import WorkflowStep  # Local import to avoid circular dependency
        
        # Convert dict to WorkflowStep if needed
        if isinstance(step, dict):
            step = WorkflowStep(
                name=step.get('name', 'unnamed_step'),
                tool=step['tool'],
                action=step['action'],
                parameters=step.get('parameters', {}),
                dependencies=step.get('dependencies', []),
                type=step.get('type', 'command'),
                description=step.get('description', '')
            )
        
        # Log the step being processed
        self.logger.info(f"Generating command for step: {step.name}")
        tool = step.tool
        action = step.action
        parameters = step.parameters

        # Log the parameters being used
        self.logger.info(f"Parameters: {parameters}")

        # Adjust parameters if necessary
        if tool == "kallisto" and action == "quant":
            if parameters.get("single"):
                parameters["single"] = True
                parameters["fragment_length"] = parameters.get("fragment_length", 200)
                parameters["fragment_sd"] = parameters.get("fragment_sd", 20)
        elif tool == "fastqc":
            if "input_files" not in parameters:
                raise ValueError("No input files specified for fastqc")
            if not isinstance(parameters["input_files"], list):
                parameters["input_files"] = [parameters["input_files"]]

        # Generate command prompt
        command_prompt = await self._get_command_generation_prompt()
        command_prompt += f"\n{json.dumps(step.to_dict())}\n"

        # Generate command using LLM
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": command_prompt}
            ]
        )

        command_text = response.choices[0].message.content.strip()
        
        # Post-process command for specific tools
        if tool == "fastqc":
            # Combine multiple input files into a single FastQC command
            output_dir = parameters.get("output_dir", ".")
            input_files = parameters["input_files"]
            command_text = f"fastqc -o {output_dir} {' '.join(input_files)}"
        
        self.logger.info(f"Generated command: {command_text}")
        return command_text

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
                logger.error("Failed to parse error diagnosis as JSON")
                raise ValueError("Invalid diagnosis format")
            
            return diagnosis
            
        except Exception as e:
            logger.error(f"Failed to diagnose error: {str(e)}")
            raise

    async def generate_analysis(self, data: Dict[str, Any], query: str) -> str:
        """Generate analysis of workflow outputs using LLM."""
        # Create a structured prompt for comprehensive analysis
        prompt = f"""Analyze the following workflow output data and provide a detailed report addressing:

1. Workflow Summary & Metadata
- Workflow type and version
- Parameters and settings used
- Date of execution
- Sample information and experimental design

2. Quality Control Analysis
- FastQC metrics and red flags
- Read quality distribution
- Adapter contamination
- Overrepresented sequences
- Duplication rates

3. Alignment Statistics (if applicable)
- Overall alignment rates
- Uniquely mapped reads
- Multi-mapping statistics
- Read distribution (exonic/intronic/intergenic)
- Insert size metrics
- rRNA/mitochondrial content

4. Key Findings and Recommendations
- Major quality concerns
- Potential experimental design issues
- Suggestions for improvement
- Next steps

Data to analyze:
{json.dumps(data, indent=2)}

User query: {query}

Focus on actionable insights and potential issues that could affect interpretation."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a bioinformatics analysis expert. Provide a detailed report based on the given data and query."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1000,
                response_format={"type": "text"}
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating analysis: {e}")
            return f"Error generating analysis: {str(e)}"

    async def execute_tool(self, tool_name: str, **inputs) -> Any:
        """Execute a tool and track its inputs/outputs"""
        # Start tracking this tool call
        call_id = self.tool_tracker.start_tool_call(tool_name, inputs)
        
        try:
            # Get command from LLM
            cmd = await self._get_command_from_llm(tool_name, inputs.get('action', ''), inputs)
            
            # Execute command
            from ..utils.command import run_command
            from pathlib import Path
            
            # Create output directory if specified
            output_dir = inputs.get('output_dir')
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Run command
            result = await run_command(cmd)
            
            # Detect outputs
            outputs = self._detect_outputs(tool_name, inputs, result)
            
            # Register completion
            self.tool_tracker.finish_tool_call(call_id, outputs)
            
            return result
            
        except Exception as e:
            self.tool_tracker.tool_calls[call_id].status = "failed"
            raise

    async def _get_command_from_llm(self, tool: str, action: str, parameters: Dict[str, Any]) -> str:
        """Get command string from LLM based on tool, action and parameters."""
        try:
            # Prepare the prompt
            messages = [
                {"role": "system", "content": """You are a bioinformatics command generator. Generate shell commands for bioinformatics tools.
                
Available tools and their command formats:
- fastqc: Generate quality control reports for FASTQ files
  Format: fastqc [input files] --outdir=[output directory]

- kallisto: RNA-seq quantification
  - index: Build an index
    Format: kallisto index -i [output index] [reference fasta]
  - quant: Quantify RNA-seq data
    Format: kallisto quant -i [index] -o [output dir] [--single] [-l fragment_length] [-s sd] [input files]

- multiqc: Aggregate reports from bioinformatics analyses
  Format: multiqc [input directory] -o [output directory]

- find_by_name: Find files matching patterns
  Format: find [directory] -name "[pattern]" -type f

Return ONLY the exact command to run, nothing else."""},
                {"role": "user", "content": f"""Generate command for:
Tool: {tool}
Action: {action}
Parameters: {json.dumps(parameters, indent=2)}"""}
            ]
            
            # Get LLM response
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1
            )
            
            # Get command from response
            cmd = response.choices[0].message.content.strip()
            logger.info(f"Generated command: {cmd}")
            
            return cmd
            
        except Exception as e:
            logger.error(f"Error generating command: {str(e)}")
            raise

    async def execute_workflow(self, query: str):
        """Execute a workflow based on natural language query."""
        try:
            # Get workflow plan
            workflow_plan = await self._get_workflow_plan(query)
            logger.info(f"Generated workflow plan: {workflow_plan}")
            
            # Track step outputs for dependency resolution
            step_outputs = {}
            
            # Execute each step
            for i, step in enumerate(workflow_plan['steps']):
                step_name = step.get('name', f'step_{i + 1}')
                
                try:
                    # Get step parameters and resolve any dependencies
                    params = step['parameters'].copy()
                    
                    # Resolve file dependencies
                    for param_name, param_value in params.items():
                        if isinstance(param_value, list):
                            resolved_values = []
                            for item in param_value:
                                if isinstance(item, str) and '<' in item and '>' in item:
                                    step_ref = item.strip('<>').split('_')[0]
                                    if step_ref == 'step':
                                        step_num = int(item.strip('<>').split('_')[1].split('_')[0])
                                        if step_num in step_outputs:
                                            resolved_values.extend(step_outputs[step_num])
                                else:
                                    resolved_values.append(item)
                            params[param_name] = resolved_values
                        elif isinstance(param_value, str) and '<' in param_value and '>' in param_value:
                            step_ref = param_value.strip('<>').split('_')[0]
                            if step_ref == 'step':
                                step_num = int(param_value.strip('<>').split('_')[1].split('_')[0])
                                if step_num in step_outputs:
                                    params[param_name] = step_outputs[step_num][0] if step_outputs[step_num] else None
                    
                    # Get command from LLM
                    cmd = await self._get_command_from_llm(step['tool'], step['action'], params)
                    
                    # Execute command
                    from ..utils.command import run_command
                    from pathlib import Path
                    
                    # Create output directory if specified
                    output_dir = params.get('output_dir')
                    if output_dir:
                        Path(output_dir).mkdir(parents=True, exist_ok=True)
                    
                    # Run command
                    result = await run_command(cmd)
                    
                    # Store outputs
                    if output_dir:
                        step_outputs[i] = [str(Path(output_dir))]
                        logger.info(f"Step {step_name} outputs: {output_dir}")
                        
                except Exception as e:
                    logger.error(f"Step {step_name} failed: {str(e)}")
                    raise
                    
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            raise

    async def _get_workflow_plan(self, query: str) -> Dict[str, Any]:
        """Get a workflow plan from the LLM based on the query."""
        logger.info(f"Getting workflow plan for query: {query}")
        
        # Call the generate_workflow_plan_from_context method
        return await self.generate_workflow_plan_from_context({})

    def _predict_outputs(self, tool_name: str, parameters: Dict[str, Any]) -> Set[Path]:
        """Predict output files/directories that will be created by a tool"""
        outputs = set()
        
        if tool_name == "fastqc":
            # FastQC creates HTML and ZIP reports
            for file in parameters.get('input_files', []):
                base = Path(file).stem
                output_dir = Path(parameters.get('output_dir', 'fastqc_output'))
                outputs.add(output_dir / f"{base}_fastqc.html")
                outputs.add(output_dir / f"{base}_fastqc.zip")
                
        elif tool_name == "kallisto":
            action = parameters.get('action', '')
            if action == 'index':
                outputs.add(Path(parameters['output']))
            elif action == 'quant':
                outputs.add(Path(parameters['output_dir']))
                
        elif tool_name == "multiqc":
            outputs.add(Path(parameters['output_dir']))
            
        return outputs

    def _detect_outputs(self, tool_name: str, inputs: Dict[str, Any], result: Any) -> set:
        """Detect files/directories created by the tool"""
        outputs = set()
        base_output_dir = Path(inputs.get('output_dir', 'results'))
        
        if tool_name == "fastqc":
            # FastQC creates HTML and ZIP reports
            for file in inputs.get('input_files', []):
                base = Path(file).stem
                outputs.add(base_output_dir / f"{base}_fastqc.html")
                outputs.add(base_output_dir / f"{base}_fastqc.zip")
                
        elif tool_name == "kallisto":
            action = inputs.get('action', '')
            if action == 'index':
                outputs.add(Path(inputs['output']))
            elif action == 'quant':
                outputs.add(Path(inputs['output_dir']))
                
        elif tool_name == "multiqc":
            outputs.add(Path(inputs['output_dir']))
            
        return outputs
