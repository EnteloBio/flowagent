"""
===========================
Pseudoalignment pipeline for bulkRNAseq data
===========================
This module implements the pseudobulk pipeline using the cgatcore pipeline framework.
"""

from typing import Any, Dict, List
import os
import json
import time
import logging
import asyncio
import subprocess
from pydantic import BaseModel
from ..utils.logging import get_logger
from ..agents.llm_agent import LLMAgent

logger = get_logger(__name__)

class PseudoBulkParams(BaseModel):
    """Parameters for PseudoBulk workflow."""
    
    input_files: List[str]
    reference_transcriptome: str
    output_dir: str
    paired_end: bool = True
    threads: int = 4
    memory: str = "16G"


class ExecutionAgent:
    """Base class for execution agents."""
    
    def __init__(self):
        """Initialize the execution agent."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.command_logger = logging.getLogger("cognomic.commands")
        
        # Initialize with LLM disabled
        self.llm_agent = None
        self.has_llm = False
        
        # Try to initialize LLM agent
        try:
            self.llm_agent = LLMAgent()
            self.has_llm = True
            logger.info("LLM agent initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize LLM agent: {str(e)}")
            logger.warning("Continuing without LLM assistance")
            self.disable_llm()
            
    def disable_llm(self):
        """Disable LLM functionality and cleanup."""
        if self.has_llm:
            logger.info("Disabling LLM functionality")
            self.has_llm = False
            self.llm_agent = None
            
    async def execute_command(self, cmd: str, **kwargs) -> Dict[str, Any]:
        """Execute a shell command with LLM assistance if available.
        
        Args:
            cmd: Command to execute
            **kwargs: Additional arguments for command execution
            
        Returns:
            Dict containing execution results
        """
        self.command_logger.info(f"Preparing to execute command: {cmd}")
        
        execution_plan = []
        
        # Get command analysis from LLM if available
        if self.has_llm and self.llm_agent:
            try:
                analysis = await self.llm_agent.analyze_data({
                    'command': cmd,
                    'parameters': kwargs,
                    'context': 'command_execution'
                })
                execution_plan = await self.llm_agent.plan_execution(analysis)
            except Exception as e:
                logger.warning(f"Error getting LLM execution plan: {str(e)}")
                self.disable_llm()
                execution_plan = [{'command': cmd, 'parameters': kwargs}]
        else:
            execution_plan = [{'command': cmd, 'parameters': kwargs}]
            
        # Execute command
        start_time = time.time()
        try:
            for step in execution_plan:
                cmd_to_run = step.get('command', cmd)
                cmd_params = {**kwargs, **step.get('parameters', {})}
                
                process = await asyncio.create_subprocess_shell(
                    cmd_to_run,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    **cmd_params
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    if self.has_llm and self.llm_agent:
                        try:
                            error_handling = await self.llm_agent.handle_error(
                                Exception(f"Command failed with code {process.returncode}"),
                                {
                                    'command': cmd_to_run,
                                    'stdout': stdout.decode() if stdout else "",
                                    'stderr': stderr.decode() if stderr else ""
                                }
                            )
                            if not error_handling.get('continue', False):
                                raise RuntimeError(f"Command failed: {cmd_to_run}")
                        except Exception as llm_error:
                            logger.warning(f"Error getting LLM error handling: {str(llm_error)}")
                            self.disable_llm()
                            raise RuntimeError(f"Command failed: {cmd_to_run}")
                    else:
                        raise RuntimeError(f"Command failed: {cmd_to_run}")
                        
            duration = time.time() - start_time
            self.command_logger.info(f"Command completed in {duration:.2f}s")
            
            return {
                'stdout': stdout.decode() if stdout else "",
                'stderr': stderr.decode() if stderr else "",
                'returncode': process.returncode,
                'duration': duration
            }
            
        except Exception as e:
            self.command_logger.error(f"Command failed: {str(e)}")
            raise

    async def plan(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Plan is not applicable for ExecutionAgent."""
        raise NotImplementedError("ExecutionAgent does not plan tasks")

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task."""
        self.logger.info(f"Starting execution with params: {json.dumps(params, indent=2)}")
        raise NotImplementedError("ExecutionAgent must implement execute")

    async def validate(self, result: Dict[str, Any]) -> bool:
        """Validate task results."""
        self.logger.info(f"Validating results: {json.dumps(result, indent=2)}")
        raise NotImplementedError("ExecutionAgent must implement validate")


class FastQCAgent(ExecutionAgent):
    """Agent for running FastQC quality control."""

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("Starting FastQC analysis")
        input_files = params["input_files"]
        output_dir = os.path.join(params["output_dir"], "fastqc")
        os.makedirs(output_dir, exist_ok=True)
        threads = params.get("threads", 1)
        
        results = []
        for input_file in input_files:
            self.logger.info(f"Processing file: {input_file}")
            
            cmd = (
                f"fastqc "
                f"--outdir={output_dir} "
                f"--threads={threads} "
                f"{input_file}"
            )
            
            result = await self.execute_command(cmd)
            
            output = {
                "input_file": input_file,
                "output_dir": output_dir,
                "success": result["returncode"] == 0,
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "duration": result["duration"]
            }
            
            results.append(output)
            
            if not output["success"]:
                self.logger.error(f"FastQC failed for {input_file}")
                self.logger.error(f"Error: {output['stderr']}")
                raise RuntimeError(f"FastQC failed for {input_file}")
                
        return {"results": results}

    async def validate(self, result: Dict[str, Any]) -> bool:
        """Validate FastQC results."""
        if not result.get("results"):
            return False
            
        for r in result["results"]:
            if not r.get("success"):
                return False
                
            output_dir = r.get("output_dir")
            if not output_dir or not os.path.exists(output_dir):
                return False
                
        return True


class MultiQCAgent(ExecutionAgent):
    """Agent for running MultiQC."""

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("Starting MultiQC analysis")
        input_dir = os.path.join(params['output_dir'], 'fastqc')
        output_dir = os.path.join(params['output_dir'], 'reports')
        os.makedirs(output_dir, exist_ok=True)

        # Ensure input directory exists
        if not os.path.exists(input_dir):
            raise RuntimeError(f"FastQC output directory not found: {input_dir}")
            
        # Check if there are FastQC reports
        fastqc_files = [f for f in os.listdir(input_dir) if f.endswith('_fastqc.html')]
        if not fastqc_files:
            raise RuntimeError(f"No FastQC reports found in {input_dir}")

        cmd = f"multiqc {input_dir} -f -d -s -o {output_dir}"
        
        result = await self.execute_command(cmd)
        
        output = {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "success": result["returncode"] == 0,
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "duration": result["duration"]
        }
        
        if not output["success"]:
            self.logger.error(f"MultiQC failed for {input_dir}")
            self.logger.error(f"Error: {output['stderr']}")
            raise RuntimeError(f"MultiQC failed for {input_dir}")
            
        return {"result": output}

    async def validate(self, result: Dict[str, Any]) -> bool:
        """Validate MultiQC results."""
        if not result.get("result"):
            return False
            
        r = result["result"]
        if not r.get("success"):
            return False
            
        output_dir = r.get("output_dir")
        if not output_dir or not os.path.exists(output_dir):
            return False
            
        # Check for MultiQC report
        report_path = os.path.join(output_dir, "multiqc_report.html")
        if not os.path.exists(report_path):
            return False
            
        return True


class KallistoIndexAgent(ExecutionAgent):
    """Agent for creating Kallisto index."""

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("Starting Kallisto index creation")
        reference = params['reference_transcriptome']
        output_dir = os.path.join(params['output_dir'], 'kalindex')
        os.makedirs(output_dir, exist_ok=True)

        output_index = os.path.join(output_dir, 'transcripts.idx')
        cmd = f"kallisto index -i {output_index} {reference}"
        
        result = await self.execute_command(cmd)
        
        output = {
            "reference": reference,
            "output_index": output_index,
            "success": result["returncode"] == 0,
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "duration": result["duration"]
        }
        
        if not output["success"]:
            self.logger.error(f"Kallisto index creation failed for {reference}")
            self.logger.error(f"Error: {output['stderr']}")
            raise RuntimeError(f"Kallisto index creation failed for {reference}")
            
        return {"result": output}

    async def validate(self, result: Dict[str, Any]) -> bool:
        """Validate Kallisto index results."""
        if not result.get("result"):
            return False
            
        r = result["result"]
        if not r.get("success"):
            return False
            
        output_index = r.get("output_index")
        if not output_index or not os.path.exists(output_index):
            return False
            
        return True


class KallistoQuantAgent(ExecutionAgent):
    """Agent for running Kallisto quantification."""

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("Starting Kallisto quantification")
        input_files = params['input_files']
        output_dir = os.path.join(params['output_dir'], 'quant')
        os.makedirs(output_dir, exist_ok=True)
        paired_end = params.get('paired_end', True)
        threads = params.get('threads', 4)

        index_file = os.path.join(params['output_dir'], 'kalindex', 'transcripts.idx')
        
        results = []
        # Handle paired-end vs single-end reads
        if paired_end:
            # Ensure even number of files for paired-end
            if len(input_files) % 2 != 0:
                raise ValueError("Odd number of input files provided for paired-end data")
            
            # Process pairs of files
            for i in range(0, len(input_files), 2):
                read1 = input_files[i]
                read2 = input_files[i + 1]
                sample_name = os.path.splitext(os.path.basename(read1))[0].replace('_1', '')
                sample_dir = os.path.join(output_dir, sample_name)
                os.makedirs(sample_dir, exist_ok=True)

                cmd = f"kallisto quant -i {index_file} -o {sample_dir} -t {threads} {read1} {read2}"
                
                result = await self.execute_command(cmd)
                output = {
                    "input_files": [read1, read2],
                    "output_dir": sample_dir,
                    "success": result["returncode"] == 0,
                    "stdout": result["stdout"],
                    "stderr": result["stderr"],
                    "duration": result["duration"]
                }
                results.append(output)
        else:
            # Process single-end reads
            for input_file in input_files:
                sample_name = os.path.splitext(os.path.basename(input_file))[0]
                sample_dir = os.path.join(output_dir, sample_name)
                os.makedirs(sample_dir, exist_ok=True)

                # For single-end reads, we need to specify an estimated fragment length
                cmd = f"kallisto quant -i {index_file} -o {sample_dir} -t {threads} --single -l 200 -s 20 {input_file}"
                
                result = await self.execute_command(cmd)
                output = {
                    "input_file": input_file,
                    "output_dir": sample_dir,
                    "success": result["returncode"] == 0,
                    "stdout": result["stdout"],
                    "stderr": result["stderr"],
                    "duration": result["duration"]
                }
                results.append(output)
            
        # Check for any failures
        failed_samples = [r for r in results if not r["success"]]
        if failed_samples:
            error_msg = "\n".join([f"Failed sample: {r.get('input_file', r.get('input_files', []))} - {r['stderr']}" 
                                 for r in failed_samples])
            raise RuntimeError(f"Kallisto quantification failed for some samples:\n{error_msg}")
                
        return {"results": results}

    async def validate(self, result: Dict[str, Any]) -> bool:
        """Validate Kallisto quant results."""
        if not result.get("results"):
            return False
            
        for r in result["results"]:
            if not r.get("success"):
                return False
                
            output_dir = r.get("output_dir")
            if not output_dir or not os.path.exists(output_dir):
                return False
                
            # Check for abundance files
            abundance_h5 = os.path.join(output_dir, "abundance.h5")
            abundance_tsv = os.path.join(output_dir, "abundance.tsv")
            if not os.path.exists(abundance_h5) or not os.path.exists(abundance_tsv):
                return False
                
        return True


class KallistoMultiQCAgent(ExecutionAgent):
    """Agent for running MultiQC on Kallisto results."""

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("Starting MultiQC for Kallisto")
        output_dir = os.path.join(params["output_dir"], "kallisto_multiqc")
        os.makedirs(output_dir, exist_ok=True)
        
        # Run MultiQC on Kallisto results
        cmd = f"multiqc {params['output_dir']}/kallisto -o {output_dir}"
        result = await self.execute_command(cmd)
        
        if result["returncode"] != 0:
            raise RuntimeError(f"MultiQC failed: {result['stderr']}")
            
        return {
            "output_dir": output_dir,
            "report": os.path.join(output_dir, "multiqc_report.html"),
            "command_result": result
        }
        
    async def validate(self, result: Dict[str, Any]) -> bool:
        """Validate MultiQC on Kallisto results."""
        if not os.path.exists(result["report"]):
            self.logger.error("MultiQC report not found")
            return False
            
        if result["command_result"]["returncode"] != 0:
            self.logger.error("MultiQC command failed")
            return False
            
        return True


class PseudoBulkWorkflow:
    """Workflow for pseudobulk RNA-seq analysis."""
    
    def __init__(self, params: PseudoBulkParams):
        """Initialize workflow with parameters."""
        self.params = params
        self.logger = get_logger(__name__)
        
        # Try to initialize LLM agent, but make it optional
        try:
            self.llm_agent = LLMAgent()
            self.has_llm = True
        except Exception as e:
            self.logger.warning(f"Could not initialize LLM agent: {str(e)}")
            self.logger.warning("Continuing without LLM assistance")
            self.has_llm = False
        
        # Initialize execution agents
        self.agents = {
            'quality_control': FastQCAgent(),
            'multiqc': MultiQCAgent(),
            'kallisto_index': KallistoIndexAgent(),
            'kal_quant': KallistoQuantAgent(),
            'kallisto_multiqc': KallistoMultiQCAgent()
        }
        
    async def execute(self) -> None:
        """Execute the workflow."""
        try:
            # Convert Pydantic model to dict for analysis
            params_dict = self.params.model_dump()
            
            if self.has_llm:
                try:
                    # Get analysis and recommendations from LLM
                    analysis = await self.llm_agent.analyze_data({
                        'input_files': self.params.input_files,
                        'parameters': {
                            'threads': self.params.threads,
                            'memory': self.params.memory
                        }
                    })
                    
                    # Get execution plan from LLM
                    execution_plan = await self.llm_agent.plan_execution(analysis)
                except Exception as e:
                    self.logger.warning(f"Error getting LLM execution plan: {str(e)}")
                    self.has_llm = False  # Disable LLM for the rest of the workflow
                    execution_plan = [
                        {
                            'name': 'quality_control',
                            'parameters': {}
                        },
                        {
                            'name': 'multiqc',
                            'parameters': {}
                        },
                        {
                            'name': 'kallisto_index',
                            'parameters': {}
                        },
                        {
                            'name': 'kal_quant',
                            'parameters': {}
                        },
                        {
                            'name': 'kallisto_multiqc',
                            'parameters': {}
                        }
                    ]
            else:
                execution_plan = [
                    {
                        'name': 'quality_control',
                        'parameters': {}
                    },
                    {
                        'name': 'multiqc',
                        'parameters': {}
                    },
                    {
                        'name': 'kallisto_index',
                        'parameters': {}
                    },
                    {
                        'name': 'kal_quant',
                        'parameters': {}
                    },
                    {
                        'name': 'kallisto_multiqc',
                        'parameters': {}
                    }
                ]
            
            # Execute steps according to plan
            for step in execution_plan:
                self.logger.info(f"Executing step: {step['name']}")
                
                try:
                    # Get agent for this step
                    agent = self.agents[step['name']]
                    
                    # Convert step parameters to dict and merge with workflow params
                    step_params = {
                        'input_files': self.params.input_files,
                        'output_dir': self.params.output_dir,
                        'reference_transcriptome': self.params.reference_transcriptome,
                        'threads': self.params.threads,
                        'memory': self.params.memory,
                        **step.get('parameters', {})
                    }
                    
                    # Execute step with parameters
                    result = await agent.execute(step_params)
                    
                    # Validate results
                    if not await agent.validate(result):
                        raise RuntimeError(f"Validation failed for step {step['name']}.")
                        
                    self.logger.info(f"Step {step['name']} completed successfully.")
                    
                except Exception as e:
                    if self.has_llm:
                        # Get error handling from LLM
                        error_handling = await self.llm_agent.handle_error(e, {
                            'step': step,
                            'params': params_dict,
                            'result': result if 'result' in locals() else None
                        })
                        
                        # If error is recoverable, try recovery steps
                        if error_handling.get('recoverable', False):
                            self.logger.info(f"Attempting recovery for step {step['name']}")
                            for recovery_step in error_handling['recovery_steps']:
                                # Execute recovery step
                                pass  # TODO: Implement recovery execution
                        else:
                            self.has_llm = False  # Disable LLM after error
                            raise RuntimeError(f"Error executing step {step['name']}: {str(e)}")
                    else:
                        # Without LLM, just raise the error
                        raise RuntimeError(f"Error executing step {step['name']}: {str(e)}")
                        
        except Exception as e:
            self.logger.error(f"Workflow failed: {str(e)}")
            raise
