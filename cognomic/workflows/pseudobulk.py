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
from pathlib import Path

logger = get_logger(__name__)

class PseudoBulkParams(BaseModel):
    """Parameters for PseudoBulk workflow."""
    
    input_files: List[str]
    reference_transcriptome: str
    output_dir: str
    paired_end: bool = False
    threads: int = 4
    memory: str = "16G"
    fragment_length: int = 200  # For single-end reads
    fragment_sd: int = 20  # For single-end reads

class ExecutionAgent:
    """Base class for execution agents."""
    
    def __init__(self, task_agent: TASK_agent):
        """Initialize the execution agent."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.command_logger = logging.getLogger("cognomic.commands")
        self.task_agent = task_agent
        
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
        
        if not self.has_llm or not self.llm_agent:
            raise RuntimeError("LLM agent required for command generation")
            
        try:
            # Get input files
            input_files = params.get("input_files", [])
            if not input_files:
                # Find all FASTQ files in directory
                try:
                    result = await self.execute_command("find . -maxdepth 1 -type f -name '*.fastq.gz' -o -name '*.fastq' -o -name '*.fq.gz' -o -name '*.fq'")
                    if result["returncode"] == 0:
                        input_files = [f.strip() for f in result["stdout"].split("\n") if f.strip()]
                except Exception as e:
                    self.logger.error(f"Error finding FASTQ files: {str(e)}")
                    raise
            
            if not input_files:
                raise RuntimeError("No FASTQ files found")
                
            # Create output directory
            output_dir = os.path.join(params["output_dir"], "fastqc")
            os.makedirs(output_dir, exist_ok=True)
            
            # Prepare command context for LLM
            command_context = {
                "tool": "fastqc",
                "action": "analyze",
                "parameters": {
                    "input_files": input_files,
                    "output_dir": output_dir,
                    "threads": params.get("threads", 1),
                    "extract": True,  # Extract zip files for easier processing
                    "quiet": True,    # Reduce output noise
                    "nogroup": True   # Disable grouping of similar sequences
                }
            }
            
            # Get command from LLM
            command = await self.llm_agent.generate_command(command_context)
            
            self.logger.info(f"Running FastQC on files: {input_files}")
            self.logger.info(f"Generated command: {command}")
            
            # Execute the command
            result = await self.execute_command(command)
            
            if result["returncode"] != 0:
                error_context = {
                    "command": command,
                    "output": result["stdout"],
                    "error": result["stderr"]
                }
                # Get error analysis from LLM
                error_analysis = await self.llm_agent.handle_error(
                    Exception(f"Command failed with code {result['returncode']}"),
                    error_context
                )
                if not error_analysis.get("continue", False):
                    raise RuntimeError(f"FastQC failed: {result['stderr']}")
            
            return {
                "results": [{
                    "input_files": input_files,
                    "output_dir": output_dir,
                    "command": command,
                    "command_result": result
                }]
            }
            
        except Exception as e:
            self.logger.error(f"Error running FastQC: {str(e)}")
            raise

    async def validate(self, result: Dict[str, Any]) -> bool:
        """Validate FastQC results."""
        if not result.get("results"):
            return False
            
        for r in result["results"]:
            output_dir = r.get("output_dir")
            if not output_dir or not os.path.exists(output_dir):
                self.logger.error(f"Output directory not found: {output_dir}")
                return False
                
            # Check for HTML reports
            input_files = r.get("input_files", [])
            for input_file in input_files:
                base_name = os.path.basename(input_file)
                # Remove all possible extensions
                for ext in ['.gz', '.fastq', '.fq']:
                    base_name = os.path.splitext(base_name)[0]
                    
                html_report = os.path.join(output_dir, f"{base_name}_fastqc.html")
                if not os.path.exists(html_report):
                    self.logger.error(f"FastQC report not found: {html_report}")
                    return False
                    
                # Check file size
                if os.path.getsize(html_report) == 0:
                    self.logger.error(f"Empty FastQC report: {html_report}")
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

        # Get command from LLM
        command = await self.task_agent.get_tool_command(
            tool_name="multiqc",
            action="report",
            parameters={
                "input_dir": input_dir,
                "output_dir": output_dir
            }
        )
        
        result = await self.execute_command(command)
        
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
            # Check for alternative default naming
            alt_report = os.path.join(output_dir, "multiqc_report.html")
            if os.path.exists(alt_report):
                report_path = alt_report
            else:
                raise RuntimeError("MultiQC report not found at either expected path")

        return True


class KallistoIndexAgent(ExecutionAgent):
    """Agent for creating Kallisto index."""

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("Starting Kallisto index creation")
        reference = params['reference_transcriptome']
        output_dir = os.path.join(params['output_dir'], 'kalindex')
        os.makedirs(output_dir, exist_ok=True)

        output_index = os.path.join(output_dir, 'transcripts.idx')
        
        # Check Kallisto version first
        version_cmd = await self.task_agent.get_tool_command(
            tool_name="kallisto",
            action="version",
            parameters={}
        )
        version_result = await self.execute_command(version_cmd)
        if version_result["returncode"] != 0:
            raise RuntimeError("Failed to get Kallisto version")
        
        # Parse version string (expected format: "kallisto, version X.Y.Z")
        version_str = version_result["stdout"].strip()
        try:
            version = version_str.split("version")[1].strip()
            self.logger.info(f"Using Kallisto version {version}")
        except:
            self.logger.warning("Could not parse Kallisto version")
            version = "unknown"
        
        # Get command from LLM
        command = await self.task_agent.get_tool_command(
            tool_name="kallisto",
            action="index",
            parameters={
                "reference": reference,
                "output_index": output_index,
                "version": version  # Pass version to LLM for command generation
            }
        )
        
        result = await self.execute_command(command)
        
        output = {
            "reference": reference,
            "output_index": output_index,
            "kallisto_version": version,
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
            
        # Verify index version compatibility
        try:
            # Run kallisto inspect on the index
            inspect_cmd = await self.task_agent.get_tool_command(
                tool_name="kallisto",
                action="inspect",
                parameters={"index": output_index}
            )
            inspect_result = await self.execute_command(inspect_cmd)
            
            if inspect_result["returncode"] != 0:
                self.logger.error("Failed to inspect Kallisto index")
                return False
                
            # Check for version compatibility in inspect output
            if "version" in inspect_result["stdout"].lower():
                index_version = None
                for line in inspect_result["stdout"].split("\n"):
                    if "version" in line.lower():
                        try:
                            index_version = line.split(":")[1].strip()
                            break
                        except:
                            pass
                
                if index_version and index_version != r.get("kallisto_version"):
                    self.logger.error(f"Index version mismatch. Index: {index_version}, Kallisto: {r.get('kallisto_version')}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error validating index version: {str(e)}")
            return False
            
        return True


class KallistoQuantAgent(ExecutionAgent):
    """Agent for running Kallisto quantification."""
    
    async def _check_kallisto_version(self) -> str:
        """Get Kallisto version and verify installation."""
        try:
            result = await self.execute_command("kallisto version")
            if result["returncode"] != 0:
                raise RuntimeError("Failed to get Kallisto version")
            version = result["stdout"].strip()
            self.logger.info(f"Kallisto version: {version}")
            return version
        except Exception as e:
            self.logger.error(f"Error checking Kallisto version: {str(e)}")
            raise

    async def _check_index_version(self, index_path: str) -> str:
        """Check Kallisto index version."""
        try:
            result = await self.execute_command(f"kallisto inspect {index_path}")
            if result["returncode"] != 0:
                raise RuntimeError(f"Failed to inspect Kallisto index: {result['stderr']}")
            
            # Parse version from inspect output
            version_line = [line for line in result["stdout"].split("\n") 
                          if "version" in line.lower()]
            if not version_line:
                raise RuntimeError("Could not determine index version")
                
            index_version = version_line[0].split(":")[1].strip()
            self.logger.info(f"Index version: {index_version}")
            return index_version
            
        except Exception as e:
            self.logger.error(f"Error checking index version: {str(e)}")
            raise

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Kallisto quantification."""
        try:
            # Check Kallisto version
            kallisto_version = await self._check_kallisto_version()
            
            # Get input files
            input_files = params.get("input_files", [])
            if not input_files:
                # Find all FASTQ files in directory
                try:
                    result = await self.execute_command("find . -maxdepth 1 -type f -name '*.fastq.gz' -o -name '*.fastq' -o -name '*.fq.gz' -o -name '*.fq'")
                    if result["returncode"] == 0:
                        input_files = [f.strip() for f in result["stdout"].split("\n") if f.strip()]
                except Exception as e:
                    self.logger.error(f"Error finding FASTQ files: {str(e)}")
                    raise
            
            if not input_files:
                raise RuntimeError("No FASTQ files found")
                
            # Check if index exists
            index_path = params.get("index")
            if not os.path.exists(index_path):
                # Build index
                reference = params.get("reference")
                if not reference or not os.path.exists(reference):
                    raise RuntimeError(f"Reference file not found: {reference}")
                    
                index_dir = os.path.dirname(index_path)
                os.makedirs(index_dir, exist_ok=True)
                
                # Build index command
                index_cmd = f"kallisto index -i {index_path} {reference}"
                result = await self.execute_command(index_cmd)
                if result["returncode"] != 0:
                    raise RuntimeError(f"Failed to build Kallisto index: {result['stderr']}")
            
            # Check index version compatibility
            index_version = await self._check_index_version(index_path)
            if index_version != kallisto_version:
                self.logger.warning(f"Index version ({index_version}) does not match Kallisto version ({kallisto_version})")
                # Rebuild index
                self.logger.info("Rebuilding index with current Kallisto version")
                reference = params.get("reference")
                index_cmd = f"kallisto index -i {index_path} {reference}"
                result = await self.execute_command(index_cmd)
                if result["returncode"] != 0:
                    raise RuntimeError(f"Failed to rebuild Kallisto index: {result['stderr']}")
            
            # Create output directory
            output_dir = params.get("output_dir")
            os.makedirs(output_dir, exist_ok=True)
            
            # Process each input file
            results = []
            for input_file in input_files:
                # Create file-specific output directory
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                file_output_dir = os.path.join(output_dir, base_name)
                os.makedirs(file_output_dir, exist_ok=True)
                
                # Build quantification command
                quant_cmd = [
                    "kallisto quant",
                    f"-i {index_path}",
                    f"-o {file_output_dir}",
                    "--single" if params.get("single", True) else "",
                    f"-l {params.get('fragment_length', 200)}",
                    f"-s {params.get('sd', 20)}",
                    f"-t {params.get('threads', 1)}",
                    input_file
                ]
                
                command = " ".join(filter(None, quant_cmd))
                self.logger.info(f"Running Kallisto quantification for {input_file}")
                self.logger.info(f"Command: {command}")
                
                result = await self.execute_command(command)
                if result["returncode"] != 0:
                    raise RuntimeError(f"Kallisto quantification failed: {result['stderr']}")
                    
                results.append({
                    "input_file": input_file,
                    "output_dir": file_output_dir,
                    "command": command,
                    "result": result
                })
            
            return {
                "results": results,
                "metadata": {
                    "kallisto_version": kallisto_version,
                    "index_version": index_version,
                    "index_path": index_path,
                    "output_dir": output_dir
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error running Kallisto quantification: {str(e)}")
            raise

    async def validate(self, result: Dict[str, Any]) -> bool:
        """Validate Kallisto quantification results."""
        if not result.get("results"):
            return False
            
        for r in result["results"]:
            output_dir = r.get("output_dir")
            if not output_dir or not os.path.exists(output_dir):
                self.logger.error(f"Output directory not found: {output_dir}")
                return False
                
            # Check for required output files
            required_files = ["abundance.h5", "abundance.tsv", "run_info.json"]
            for file in required_files:
                file_path = os.path.join(output_dir, file)
                if not os.path.exists(file_path):
                    self.logger.error(f"Required output file not found: {file_path}")
                    return False
                    
                # Check file size
                if os.path.getsize(file_path) == 0:
                    self.logger.error(f"Empty output file: {file_path}")
                    return False
                    
        return True


class KallistoMultiQCAgent(ExecutionAgent):
    """Agent for running MultiQC on Kallisto results."""

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("Starting MultiQC for Kallisto")
        
        # Validate input directory
        input_dir = os.path.abspath(os.path.join(params["output_dir"], "quant"))
        if not os.path.exists(input_dir):
            raise RuntimeError(f"Kallisto output directory not found: {input_dir}")

        # Create output directory with proper permissions
        output_dir = os.path.abspath(os.path.join(params["output_dir"], "kallisto_multiqc"))
        os.makedirs(output_dir, exist_ok=True, mode=0o755)

        # Verify Kallisto outputs
        required_files = [
            'abundance.h5',
            'abundance.tsv',
            'run_info.json'
        ]

        sample_dirs = []
        for d in os.listdir(input_dir):
            dir_path = os.path.join(input_dir, d)
            if os.path.isdir(dir_path):
                missing = [f for f in required_files if not os.path.exists(os.path.join(dir_path, f))]
                if not missing:
                    sample_dirs.append(d)
                else:
                    self.logger.warning(f"Skipping incomplete Kallisto directory {d}: Missing {', '.join(missing)}")

        if not sample_dirs:
            raise RuntimeError(f"No valid Kallisto directories found in {input_dir}")

        self.logger.info(f"Processing {len(sample_dirs)} Kallisto samples")

        # Get command from LLM
        command = await self.task_agent.get_tool_command(
            tool_name="multiqc",
            action="report",
            parameters={
                "input_dir": input_dir,
                "output_dir": output_dir
            }
        )
        
        result = await self.execute_command(command)
        
        # Debug output
        self.logger.debug(f"MultiQC stdout:\n{result['stdout']}")
        self.logger.debug(f"MultiQC stderr:\n{result['stderr']}")

        # Handle MultiQC's output paths
        report_path = os.path.join(output_dir, "kallisto_multiqc_report.html")
        if not os.path.exists(report_path):
            # Check for alternative default naming
            alt_report = os.path.join(output_dir, "multiqc_report.html")
            if os.path.exists(alt_report):
                report_path = alt_report
            else:
                raise RuntimeError("MultiQC report not found at either expected path")

        return {
            "output_dir": output_dir,
            "report": report_path,
            "command_result": result
        }

    async def validate(self, result: Dict[str, Any]) -> bool:
        """Validate MultiQC on Kallisto results."""
        # Check if the report exists
        if not os.path.exists(result["report"]):
            self.logger.error(f"MultiQC report not found at: {result['report']}")
            return False
            
        # Check if the report has content
        if os.path.getsize(result["report"]) == 0:
            self.logger.error("MultiQC report is empty")
            return False
            
        # Check command result
        if result["command_result"]["returncode"] != 0:
            self.logger.error("MultiQC command failed")
            return False
            
        return True


class PseudoBulkWorkflow:
    """Workflow for pseudobulk RNA-seq analysis."""
    
    name = "pseudobulk"
    description = "Pseudobulk RNA-seq analysis using Kallisto"
    required_tools = ["kallisto", "fastqc", "multiqc"]
    
    def __init__(self, params: Dict[str, Any], task_agent: Any):
        """Initialize workflow with parameters."""
        self.params = PseudoBulkParams(**params)
        
        # Initialize execution agents
        self.agents = {
            'quality_control': FastQCAgent(task_agent),
            'multiqc': MultiQCAgent(task_agent),
            'kallisto_index': KallistoIndexAgent(task_agent),
            'kal_quant': KallistoQuantAgent(task_agent),
            'kallisto_multiqc': KallistoMultiQCAgent(task_agent)
        }
    
    async def validate_params(self) -> bool:
        """Validate workflow parameters."""
        # Check input files exist
        for file in self.params.input_files:
            if not os.path.exists(file):
                self.logger.error(f"Input file not found: {file}")
                return False
        
        # Check reference exists
        if not os.path.exists(self.params.reference_transcriptome):
            self.logger.error(f"Reference not found: {self.params.reference_transcriptome}")
            return False
            
        # Create output directory
        os.makedirs(self.params.output_dir, exist_ok=True)
        
        return True
    
    async def execute(self) -> Dict[str, Any]:
        """Execute the workflow."""
        results = {}
        
        # 1. Create Kallisto index
        index_results = await self.agents['kallisto_index'].execute({
            "reference": self.params.reference_transcriptome,
            "output_dir": self.params.output_dir
        })
        results['index'] = index_results
        
        # 2. Run Kallisto quantification
        quant_results = await self.agents['kal_quant'].execute({
            "index": index_results['result']['output_index'],
            "input_files": self.params.input_files,
            "output_dir": os.path.join(self.params.output_dir, "kallisto"),
            "paired_end": self.params.paired_end,
            "fragment_length": self.params.fragment_length,
            "fragment_sd": self.params.fragment_sd,
            "threads": self.params.threads
        })
        results['quantification'] = quant_results
        
        # 3. Run FastQC
        fastqc_results = await self.agents['quality_control'].execute({
            "input_files": self.params.input_files,
            "output_dir": os.path.join(self.params.output_dir, "fastqc")
        })
        results['fastqc'] = fastqc_results
        
        # 4. Run MultiQC
        multiqc_results = await self.agents['multiqc'].execute({
            "input_dirs": [
                os.path.join(self.params.output_dir, "fastqc"),
                os.path.join(self.params.output_dir, "kallisto")
            ],
            "output_dir": os.path.join(self.params.output_dir, "multiqc")
        })
        results['multiqc'] = multiqc_results
        
        return results
    
    async def validate_results(self, results: Dict[str, Any]) -> bool:
        """Validate workflow results."""
        # Check index results
        if not await self.agents['kallisto_index'].validate(results.get('index', {})):
            return False
            
        # Check quantification results
        if not await self.agents['kal_quant'].validate(results.get('quantification', {})):
            return False
            
        # Check FastQC results
        if not await self.agents['quality_control'].validate(results.get('fastqc', {})):
            return False
            
        # Check MultiQC results
        if not await self.agents['multiqc'].validate(results.get('multiqc', {})):
            return False
            
        return True

# Register the workflow
# WorkflowRegistry.register(PseudoBulkWorkflow)
