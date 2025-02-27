"""Agent-based analysis system for workflow outputs."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import numpy as np

from .analysis_agents import QualityAnalysisAgent, QuantificationAnalysisAgent, TechnicalQCAgent

logger = logging.getLogger(__name__)

class AgenticAnalysisSystem:
    """System for analyzing workflow outputs using specialized agents."""
    
    def __init__(self):
        """Initialize analysis agents."""
        self.quality_agent = QualityAnalysisAgent()
        self.quantification_agent = QuantificationAnalysisAgent()
        self.technical_agent = TechnicalQCAgent()
    
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
            # Recursively find all relevant files
            kallisto_data = self._find_kallisto_outputs(results_dir)
            if kallisto_data:
                data["tools_used"].append("kallisto")
                data["quantification_data"]["kallisto"] = kallisto_data
            
            # Find QC outputs recursively
            qc_data = self._find_qc_outputs(results_dir)
            data["qc_data"].update(qc_data)
            
            # Find technical data recursively
            tech_data = self._find_technical_data(results_dir)
            data["technical_data"].update(tech_data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error preparing analysis data: {str(e)}")
            return data
    
    def _find_kallisto_outputs(self, directory: Path) -> Dict[str, Any]:
        """Recursively find Kallisto output files in any directory structure."""
        kallisto_data = {
            "abundance_files": [],
            "run_info": {},
            "samples": {}
        }
        
        # Find all abundance.h5 files recursively
        for abundance_file in directory.rglob("abundance.h5"):
            # Get sample name from parent directory
            sample_name = abundance_file.parent.name
            if sample_name == "kallisto" or sample_name == "output":  # Skip generic directory names
                sample_name = abundance_file.parent.parent.name
            
            kallisto_data["abundance_files"].append(str(abundance_file))
            
            # Look for run_info.json in the same directory
            run_info_file = abundance_file.parent / "run_info.json"
            if run_info_file.exists():
                with open(run_info_file) as f:
                    run_info = json.load(f)
                    kallisto_data["run_info"][sample_name] = run_info
            
            # Look for abundance.tsv for easier parsing
            abundance_tsv = abundance_file.parent / "abundance.tsv"
            if abundance_tsv.exists():
                kallisto_data["samples"][sample_name] = {
                    "abundance_file": str(abundance_tsv),
                    "metrics": self._parse_abundance_file(abundance_tsv)
                }
        
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
                for line in f:
                    fields = line.strip().split("\t")
                    if len(fields) >= 4:  # Expect target_id, length, eff_length, est_counts, tpm
                        tpm = float(fields[4])
                        tpm_values.append(tpm)
                        if tpm > 0:
                            metrics["expressed_transcripts"] += 1
                            
            if tpm_values:
                metrics["total_transcripts"] = len(tpm_values)
                metrics["median_tpm"] = float(np.median(tpm_values))
                metrics["mean_tpm"] = float(np.mean(tpm_values))
                
        except Exception as e:
            logger.error(f"Error parsing abundance file {abundance_file}: {str(e)}")
            
        return metrics
    
    async def analyze_results(self, results_dir: Path) -> Dict[str, Any]:
        """Analyze workflow results using specialized agents."""
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
                            "samples": len(data["quantification_data"].get("kallisto", {}).get("samples", {})),
                            "metrics": data["quantification_data"].get("kallisto", {}).get("samples", {})
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
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in agentic analysis: {str(e)}")
            return {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "directory": str(results_dir),
                "error": str(e)
            }
