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
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.command_logger = logging.getLogger("cognomic.commands")
        
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

    async def _run_command(self, cmd: str, **kwargs) -> Dict[str, Any]:
        """Run a shell command with logging."""
        self.command_logger.info(f"Executing command: {cmd}")
        self.command_logger.info(f"Command parameters: {json.dumps(kwargs, indent=2)}")
        
        start_time = time.time()
        try:
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **kwargs
            )
            stdout, stderr = await process.communicate()
            
            duration = time.time() - start_time
            self.command_logger.info(f"Command completed in {duration:.2f}s")
            self.command_logger.info(f"Exit code: {process.returncode}")
            
            stdout_str = stdout.decode() if stdout else ""
            stderr_str = stderr.decode() if stderr else ""
            
            if stdout_str:
                self.command_logger.debug(f"STDOUT:\n{stdout_str}")
            if stderr_str:
                self.command_logger.debug(f"STDERR:\n{stderr_str}")
                
            return {
                "returncode": process.returncode,
                "stdout": stdout_str,
                "stderr": stderr_str,
                "duration": duration
            }
            
        except Exception as e:
            self.command_logger.error(f"Command failed: {str(e)}")
            raise


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
            
            result = await self._run_command(cmd)
            
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
        
        result = await self._run_command(cmd)
        
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
        
        result = await self._run_command(cmd)
        
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

        index_file = os.path.join(params['output_dir'], 'kalindex', 'transcripts.idx')
        
        results = []
        for input_file in input_files:
            sample_name = os.path.splitext(os.path.basename(input_file))[0]
            sample_dir = os.path.join(output_dir, sample_name)
            os.makedirs(sample_dir, exist_ok=True)

            cmd = f"kallisto quant -i {index_file} -o {sample_dir} {input_file}"
            
            result = await self._run_command(cmd)
            
            output = {
                "input_file": input_file,
                "output_dir": sample_dir,
                "success": result["returncode"] == 0,
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "duration": result["duration"]
            }
            
            results.append(output)
            
            if not output["success"]:
                self.logger.error(f"Kallisto quantification failed for {input_file}")
                self.logger.error(f"Error: {output['stderr']}")
                raise RuntimeError(f"Kallisto quantification failed for {input_file}")
                
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
        self.logger.info("Starting MultiQC on Kallisto results")
        input_dir = os.path.join(params['output_dir'], 'quant')
        output_file = os.path.join(input_dir, 'kallisto_multiqc.html')

        cmd = f"multiqc -f -n kallisto_multiqc.html -o {input_dir} {input_dir}"
        
        result = await self._run_command(cmd)
        
        output = {
            "input_dir": input_dir,
            "output_file": output_file,
            "success": result["returncode"] == 0,
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "duration": result["duration"]
        }
        
        if not output["success"]:
            self.logger.error(f"MultiQC on Kallisto results failed for {input_dir}")
            self.logger.error(f"Error: {output['stderr']}")
            raise RuntimeError(f"MultiQC on Kallisto results failed for {input_dir}")
            
        return {"result": output}

    async def validate(self, result: Dict[str, Any]) -> bool:
        """Validate MultiQC on Kallisto results."""
        if not result.get("result"):
            return False
            
        r = result["result"]
        if not r.get("success"):
            return False
            
        output_file = r.get("output_file")
        if not output_file or not os.path.exists(output_file):
            return False
            
        return True


class PseudoBulkWorkflow:
    """Implementation of PseudoBulk analysis workflow."""

    def __init__(self, params: PseudoBulkParams) -> None:
        """Initialize PseudoBulk workflow."""
        self.params = params
        self.steps = self._create_workflow_steps()

    def _create_workflow_steps(self) -> List[Dict[str, Any]]:
        """Create workflow steps for PseudoBulk analysis."""
        return [
            {
                "name": "quality_control",
                "agent": FastQCAgent(),
                "params": {
                    "input_files": self.params.input_files,
                    "output_dir": self.params.output_dir,
                }
            },
            {
                "name": "multiqc",
                "agent": MultiQCAgent(),
                "params": {
                    "input_files": self.params.input_files,
                    "output_dir": self.params.output_dir,
                }
            },
            {
                "name": "kallisto_index",
                "agent": KallistoIndexAgent(),
                "params": {
                    "reference_transcriptome": self.params.reference_transcriptome,
                    "output_dir": self.params.output_dir,
                }
            },
            {
                "name": "kal_quant",
                "agent": KallistoQuantAgent(),
                "params": {
                    "input_files": self.params.input_files,
                    "reference_transcriptome": self.params.reference_transcriptome,
                    "output_dir": self.params.output_dir,
                }
            },
            {
                "name": "kallisto_multiqc",
                "agent": KallistoMultiQCAgent(),
                "params": {
                    "input_files": self.params.input_files,
                    "output_dir": self.params.output_dir,
                }
            },
        ]

    async def execute(self) -> None:
        """Execute the PseudoBulk workflow."""
        for step in self.steps:
            logger.info(f"Executing step: {step['name']}")
            agent = step['agent']
            params = step['params']
            try:
                # Execute the step using the agent
                result = await agent.execute(params)
                # Validate the results
                if await agent.validate(result):
                    logger.info(f"Step {step['name']} completed successfully.")
                else:
                    logger.error(f"Validation failed for step {step['name']}.")
            except Exception as e:
                logger.error(f"Error executing step {step['name']}: {e}")
                raise
