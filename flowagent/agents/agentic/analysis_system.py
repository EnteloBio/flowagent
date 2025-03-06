"""Agent-based analysis system for workflow outputs."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np
import os

from .analysis_agents import QualityAnalysisAgent, QuantificationAnalysisAgent, TechnicalQCAgent
from ...core.api_usage import APIUsageTracker

logger = logging.getLogger(__name__)

class AgenticAnalysisSystem:
    """System for analyzing workflow outputs using specialized agents."""
    
    def __init__(self):
        """Initialize analysis agents."""
        self.api_usage_tracker = APIUsageTracker()
        self.quality_agent = QualityAnalysisAgent(self.api_usage_tracker)
        self.quantification_agent = QuantificationAnalysisAgent(self.api_usage_tracker)
        self.technical_agent = TechnicalQCAgent(self.api_usage_tracker)
    
    async def _prepare_analysis_data(self, results_dir: Path) -> Dict[str, Any]:
        """Prepare data for analysis by organizing files and metrics."""
        data = {
            "workflow_type": "rna_seq",
            "tools_used": [],
            "qc_data": {},
            "quantification_data": {},
            "technical_data": {
                "resource_usage": {},
                "tool_versions": {},
                "logs": []
            }
        }
        
        try:
            # Define potential results directories to search
            search_dirs = [results_dir]
            
            # Check for results in parent directory (common case)
            parent_results = Path(os.path.dirname(os.path.dirname(results_dir))) / "results"
            if parent_results.exists():
                search_dirs.append(parent_results)
                logger.info(f"Found results directory at {parent_results}")
            
            # Check for results in the workflow directory itself
            workflow_results = results_dir / "results"
            if workflow_results.exists():
                search_dirs.append(workflow_results)
                logger.info(f"Found results directory at {workflow_results}")
            
            # Check if we have a workflow.json that might point to a results directory
            workflow_json = results_dir / "workflow.json"
            if workflow_json.exists():
                try:
                    with open(workflow_json) as f:
                        workflow_data = json.load(f)
                        # Extract results directory from commands if present
                        for step in workflow_data.get("steps", []):
                            if step.get("command", "").startswith("mkdir -p "):
                                dirs = step.get("command", "").replace("mkdir -p ", "").split()
                                for dir_path in dirs:
                                    if "results" in dir_path:
                                        potential_path = Path(os.path.join(os.path.dirname(results_dir), dir_path))
                                        if potential_path.exists():
                                            search_dirs.append(potential_path)
                                            logger.info(f"Found potential results directory from workflow.json: {potential_path}")
                except Exception as e:
                    logger.warning(f"Error reading workflow.json: {str(e)}")
            
            # Log the directories we're searching
            logger.info(f"Searching for output files in: {[str(d) for d in search_dirs]}")
            
            # Recursively find all relevant files from all search directories
            kallisto_data = {}
            for search_dir in search_dirs:
                dir_kallisto_data = self._find_kallisto_outputs(search_dir)
                if dir_kallisto_data:
                    # Merge with existing data
                    if "abundance_files" not in kallisto_data:
                        kallisto_data["abundance_files"] = []
                    kallisto_data["abundance_files"].extend(dir_kallisto_data.get("abundance_files", []))
                    
                    if "run_info" not in kallisto_data:
                        kallisto_data["run_info"] = {}
                    kallisto_data["run_info"].update(dir_kallisto_data.get("run_info", {}))
                    
                    if "samples" not in kallisto_data:
                        kallisto_data["samples"] = {}
                    kallisto_data["samples"].update(dir_kallisto_data.get("samples", {}))
            
            if kallisto_data and kallisto_data.get("abundance_files"):
                data["tools_used"].append("kallisto")
                data["quantification_data"]["kallisto"] = kallisto_data
            
            # Find QC outputs recursively from all search directories
            qc_data = {}
            for search_dir in search_dirs:
                dir_qc_data = self._find_qc_outputs(search_dir)
                # Merge with existing data
                for key, files in dir_qc_data.items():
                    if key not in qc_data:
                        qc_data[key] = []
                    qc_data[key].extend(files)
            data["qc_data"].update(qc_data)
            
            # Find technical data recursively from all search directories
            tech_data = {
                "resource_usage": {},
                "tool_versions": {},
                "logs": []
            }
            for search_dir in search_dirs:
                dir_tech_data = self._find_technical_data(search_dir)
                # Merge with existing data
                tech_data["resource_usage"].update(dir_tech_data.get("resource_usage", {}))
                tech_data["tool_versions"].update(dir_tech_data.get("tool_versions", {}))
                tech_data["logs"].extend(dir_tech_data.get("logs", []))
            data["technical_data"].update(tech_data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error preparing analysis data: {str(e)}")
            return data
    
    def _normalize_sample_name(self, name: str) -> str:
        """Normalize sample name by removing common suffixes and standardizing separators."""
        # Remove common suffixes
        name = name.replace('.fastq', '')
        name = name.replace('.fq', '')
        name = name.replace('.gz', '')
        name = name.replace('.1', '')
        name = name.replace('.2', '')
        # Standardize separators
        name = name.replace('-', '_')
        return name

    def _find_kallisto_outputs(self, directory: Path) -> Dict[str, Any]:
        """Recursively find Kallisto output files in any directory structure."""
        kallisto_data = {
            "abundance_files": [],
            "run_info": {},
            "samples": {}
        }
        
        logger.info(f"Searching for Kallisto outputs in {directory}")
        
        # Find all abundance.h5 files recursively
        for abundance_file in directory.rglob("abundance.h5"):
            # Get sample name from parent directory
            sample_name = None
            parent_dir = abundance_file.parent
            
            # Try to find a valid sample name by walking up the directory tree
            while parent_dir != directory:
                if parent_dir.name not in ["kallisto", "output", "kallisto_quant"]:
                    sample_name = parent_dir.name
                    break
                parent_dir = parent_dir.parent
            
            # Skip if we couldn't determine a valid sample name
            if not sample_name:
                logger.warning(f"Could not determine sample name for {abundance_file}")
                continue
                
            # Normalize sample name
            sample_name = self._normalize_sample_name(sample_name)
            
            kallisto_data["abundance_files"].append(str(abundance_file))
            
            # Look for run_info.json in the same directory
            run_info_file = abundance_file.parent / "run_info.json"
            if run_info_file.exists():
                try:
                    with open(run_info_file) as f:
                        run_info = json.load(f)
                        kallisto_data["run_info"][sample_name] = run_info
                except Exception as e:
                    logger.error(f"Error reading run info for {sample_name}: {str(e)}")
                    continue
            
            # Look for abundance.tsv for easier parsing
            abundance_tsv = abundance_file.parent / "abundance.tsv"
            if abundance_tsv.exists():
                try:
                    metrics = self._parse_abundance_file(abundance_tsv)
                    kallisto_data["samples"][sample_name] = {
                        "abundance_file": str(abundance_tsv),
                        "metrics": metrics
                    }
                except Exception as e:
                    logger.error(f"Error parsing abundance file for {sample_name}: {str(e)}")
                    continue
        
        # Also look for abundance.tsv files directly since our test creates only these
        logger.info(f"Looking for abundance.tsv files in {directory}")
        for abundance_tsv in directory.rglob("abundance.tsv"):
            logger.info(f"Found abundance.tsv file at {abundance_tsv}")
            # Get sample name from parent directory
            sample_name = None
            parent_dir = abundance_tsv.parent
            
            # Try to find a valid sample name by walking up the directory tree
            while parent_dir != directory:
                if parent_dir.name not in ["kallisto", "output", "kallisto_quant"]:
                    sample_name = parent_dir.name
                    break
                parent_dir = parent_dir.parent
            
            # Skip if we couldn't determine a valid sample name
            if not sample_name:
                logger.warning(f"Could not determine sample name for {abundance_tsv}")
                continue
                
            # Normalize sample name
            sample_name = self._normalize_sample_name(sample_name)
            logger.info(f"Using sample name: {sample_name}")
            
            # Add the abundance file to the list if not already present
            if str(abundance_tsv) not in kallisto_data["abundance_files"]:
                kallisto_data["abundance_files"].append(str(abundance_tsv))
                
                # Look for run_info.json in the same directory
                run_info_file = abundance_tsv.parent / "run_info.json"
                if run_info_file.exists():
                    try:
                        logger.info(f"Found run_info.json file at {run_info_file}")
                        with open(run_info_file) as f:
                            run_info = json.load(f)
                            kallisto_data["run_info"][sample_name] = run_info
                    except Exception as e:
                        logger.error(f"Error reading run info for {sample_name}: {str(e)}")
                        continue
                
                # Parse abundance.tsv file
                try:
                    metrics = self._parse_abundance_file(abundance_tsv)
                    kallisto_data["samples"][sample_name] = {
                        "abundance_file": str(abundance_tsv),
                        "metrics": metrics
                    }
                    logger.info(f"Successfully parsed abundance.tsv for {sample_name}")
                except Exception as e:
                    logger.error(f"Error parsing abundance file for {sample_name}: {str(e)}")
                    continue
        
        # Log what we found
        if kallisto_data["abundance_files"]:
            logger.info(f"Found {len(kallisto_data['abundance_files'])} Kallisto abundance files")
            logger.info(f"Found {len(kallisto_data['run_info'])} Kallisto run info files")
            logger.info(f"Found {len(kallisto_data['samples'])} Kallisto samples")
        else:
            logger.warning(f"No Kallisto outputs found in {directory}")
            
        return kallisto_data if kallisto_data["abundance_files"] else None
    
    def _find_qc_outputs(self, directory: Path) -> Dict[str, Any]:
        """Recursively find QC output files in any directory structure."""
        qc_data = {}
        
        # Find FastQC reports recursively
        fastqc_files = list(directory.rglob("*_fastqc.html"))
        if fastqc_files:
            qc_data["fastqc"] = [str(f) for f in fastqc_files]
        
        # Find MultiQC reports recursively
        multiqc_files = list(directory.rglob("multiqc_report.html"))
        if multiqc_files:
            qc_data["multiqc"] = [str(f) for f in multiqc_files]
        
        return qc_data
    
    def _find_technical_data(self, directory: Path) -> Dict[str, Any]:
        """Recursively find technical data files in any directory structure."""
        tech_data = {
            "resource_usage": {},
            "tool_versions": {},
            "logs": []
        }
        
        # Find all log files recursively
        log_files = list(directory.rglob("*.log"))
        if log_files:
            tech_data["logs"] = [str(f) for f in log_files]
        
        # Look for workflow metadata files recursively
        for metadata_file in directory.rglob("*metadata*.json"):
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    if "tool_versions" in metadata:
                        tech_data["tool_versions"].update(metadata["tool_versions"])
                    if "resource_usage" in metadata:
                        tech_data["resource_usage"].update(metadata["resource_usage"])
            except Exception as e:
                logger.error(f"Error reading metadata file {metadata_file}: {str(e)}")
        
        # Try to get versions from run info files
        for run_info in directory.rglob("run_info.json"):
            try:
                with open(run_info) as f:
                    info = json.load(f)
                    if "kallisto_version" in info:
                        tech_data["tool_versions"]["kallisto"] = info["kallisto_version"]
            except Exception as e:
                logger.error(f"Error reading run info file {run_info}: {str(e)}")
        
        return tech_data
    
    def _parse_abundance_file(self, abundance_file: Path) -> Dict[str, Any]:
        """Parse Kallisto abundance.tsv file to extract key metrics."""
        metrics = {
            "total_transcripts": 0,
            "expressed_transcripts": 0,
            "median_tpm": 0.0,
            "mean_tpm": 0.0
        }
        
        try:
            tpm_values = []
            with open(abundance_file) as f:
                # Skip header
                next(f)
                metrics["total_transcripts"] = sum(1 for _ in f)  # Count total lines first
                
            # Now read the actual values
            with open(abundance_file) as f:
                next(f)  # Skip header again
                for line in f:
                    fields = line.strip().split("\t")
                    if len(fields) >= 5:  # Expect target_id, length, eff_length, est_counts, tpm
                        tpm = float(fields[4])
                        tpm_values.append(tpm)
                        if tpm > 0:
                            metrics["expressed_transcripts"] += 1
            
            if tpm_values:
                metrics["median_tpm"] = float(np.median(tpm_values))
                metrics["mean_tpm"] = float(np.mean(tpm_values))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error parsing abundance file {abundance_file}: {str(e)}")
            return metrics
    
    async def analyze_results(self, results_dir: Path) -> Dict[str, Any]:
        """Analyze workflow results using specialized agents."""
        # Start tracking API usage for analysis
        analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.api_usage_tracker.start_workflow(analysis_id, f"Analysis of {results_dir}")
        
        try:
            # Prepare data for analysis
            data = await self._prepare_analysis_data(results_dir)
            
            # Run specialized agent analyses
            quality_results = await self.quality_agent.analyze(data)
            quant_results = await self.quantification_agent.analyze(data)
            tech_results = await self.technical_agent.analyze(data)
            
            # Combine results
            analysis = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "directory": str(results_dir),
                "workflow_info": {
                    "type": data["workflow_type"],
                    "tools_used": data["tools_used"]
                },
                "quality_analysis": {
                    "fastqc_available": bool(data["qc_data"].get("fastqc")),
                    "multiqc_available": bool(data["qc_data"].get("multiqc")),
                    **quality_results
                },
                "quantification_analysis": {
                    "tools": {
                        "kallisto": {
                            "sample_count": len(data["quantification_data"].get("kallisto", {}).get("samples", {})),
                            "samples": data["quantification_data"].get("kallisto", {}).get("samples", {})
                        }
                    },
                    **quant_results
                },
                "technical_analysis": {
                    "tool_versions": data["technical_data"]["tool_versions"],
                    "log_files": len(data["technical_data"]["logs"]),
                    **tech_results
                },
                "recommendations": []
            }
            
            # Add recommendations based on findings
            if not data["qc_data"].get("fastqc"):
                analysis["recommendations"].append("Run FastQC on all input files to ensure data quality")
            if not data["qc_data"].get("multiqc"):
                analysis["recommendations"].append("Generate MultiQC report to compare samples")
            if not data["technical_data"]["tool_versions"]:
                analysis["recommendations"].append("Record tool versions for reproducibility")
            
            # Add quantification-specific recommendations
            kallisto_data = data["quantification_data"].get("kallisto", {})
            if kallisto_data.get("samples"):
                low_expressed = []
                for sample, metrics in kallisto_data["samples"].items():
                    if metrics["metrics"]["expressed_transcripts"] < 1000:
                        low_expressed.append(sample)
                if low_expressed:
                    analysis["recommendations"].append(
                        f"Review samples with low expressed transcript counts: {', '.join(low_expressed)}"
                    )
            
            # End API usage tracking and display stats
            self.api_usage_tracker.end_workflow(analysis_id)
            self.api_usage_tracker.display_usage(analysis_id)
            
            return analysis
            
        except Exception as e:
            # End API usage tracking even on error
            if self.api_usage_tracker.current_workflow_id == analysis_id:
                self.api_usage_tracker.end_workflow(analysis_id)
                
            logger.error(f"Error in agentic analysis: {str(e)}")
            return {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "directory": str(results_dir),
                "error": str(e)
            }
