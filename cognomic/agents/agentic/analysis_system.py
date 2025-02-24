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
            "workflow_type": "rna_seq",  # Default to RNA-seq for now
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
            # Find RNA-seq Kallisto directory
            rna_seq_dir = results_dir / "rna_seq_kallisto"
            if rna_seq_dir.exists():
                data["tools_used"].append("kallisto")
                kallisto_data = {
                    "abundance_files": [],
                    "run_info": {},
                    "tool_versions": {},
                    "index_version": None,
                    "index_dir": None
                }
                
                # Get Kallisto index info
                index_dir = rna_seq_dir / "kallisto_index"
                if index_dir.exists():
                    kallisto_data["index_dir"] = str(index_dir)
                    # Try to read index version from metadata
                    metadata_file = index_dir / "index_metadata.json"
                    if metadata_file.exists():
                        try:
                            with open(metadata_file) as f:
                                metadata = json.load(f)
                                kallisto_data["index_version"] = metadata.get("kallisto_version")
                        except Exception as e:
                            logger.error(f"Error reading Kallisto index metadata: {str(e)}")
                
                # Collect abundance files from kallisto_quant directory
                quant_dir = rna_seq_dir / "kallisto_quant"
                if quant_dir.exists():
                    for abundance_file in quant_dir.glob("*/abundance.h5"):
                        kallisto_data["abundance_files"].append(str(abundance_file))
                        sample_name = abundance_file.parent.name
                        
                        # Read run info from h5 file
                        try:
                            import h5py
                            with h5py.File(abundance_file, 'r') as f:
                                if 'aux' in f and 'n_processed' in f['aux']:
                                    n_processed = f['aux']['n_processed'][()]
                                    p_pseudoaligned = f['aux']['p_pseudoaligned'][()]
                                    n_unique = f['aux'].get('n_unique', 0)[()]
                                    n_expressed = len([x for x in f['est_counts'][()] if x > 0])
                                    tpm_values = f['abundance'][()] if 'abundance' in f else []
                                    median_tpm = float(np.median(tpm_values)) if len(tpm_values) > 0 else 0.0
                                    
                                    kallisto_data["run_info"][sample_name] = {
                                        "n_processed": int(n_processed),
                                        "p_pseudoaligned": float(p_pseudoaligned) * 100,
                                        "n_unique": int(n_unique),
                                        "n_expressed": n_expressed,
                                        "median_tpm": median_tpm
                                    }
                        except Exception as e:
                            logger.error(f"Error reading Kallisto h5 file: {str(e)}")
                
                # Get Kallisto version from workflow results
                workflow_results = results_dir / "workflow_results.json"
                if workflow_results.exists():
                    try:
                        with open(workflow_results) as f:
                            results_data = json.load(f)
                            if "tool_versions" in results_data:
                                kallisto_data["tool_versions"] = results_data["tool_versions"]
                    except Exception as e:
                        logger.error(f"Error reading workflow results: {str(e)}")
                
                data["quantification_data"]["kallisto"] = kallisto_data
            
            # Find FastQC outputs in rna_seq_kallisto/fastqc
            fastqc_dir = rna_seq_dir / "fastqc" if rna_seq_dir.exists() else None
            if fastqc_dir and fastqc_dir.exists():
                fastqc_files = list(fastqc_dir.glob("*_fastqc.html"))
                data["qc_data"]["fastqc"] = [str(f) for f in fastqc_files]
            
            # Find MultiQC output in rna_seq_kallisto/qc
            qc_dir = rna_seq_dir / "qc" if rna_seq_dir.exists() else None
            if qc_dir and qc_dir.exists():
                multiqc_file = qc_dir / "multiqc_report.html"
                if multiqc_file.exists():
                    data["qc_data"]["multiqc"] = str(multiqc_file)
            
            # Read workflow results for resource usage and tool versions
            workflow_results = results_dir / "workflow_results.json"
            if workflow_results.exists():
                try:
                    with open(workflow_results) as f:
                        results_data = json.load(f)
                        if "resource_usage" in results_data:
                            data["technical_data"]["resource_usage"] = results_data["resource_usage"]
                        if "tool_versions" in results_data:
                            data["technical_data"]["tool_versions"] = results_data["tool_versions"]
                except Exception as e:
                    logger.error(f"Error reading workflow results: {str(e)}")
            
            # Collect logs from workflow directory
            workflow_dir = results_dir / "workflow"
            if workflow_dir.exists():
                for log_file in workflow_dir.glob("*.log"):
                    try:
                        with open(log_file) as f:
                            data["technical_data"]["logs"].extend(f.readlines())
                    except Exception as e:
                        logger.error(f"Error reading log file {log_file}: {str(e)}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error preparing analysis data: {str(e)}")
            return data
    
    async def analyze_results(self, results_dir: Path) -> Dict[str, Any]:
        """Analyze workflow results using specialized agents."""
        try:
            # Prepare data for analysis
            data = await self._prepare_analysis_data(results_dir)
            
            # Run analysis agents
            quality_results = await self.quality_agent.analyze(data)
            quantification_results = await self.quantification_agent.analyze(data)
            technical_results = await self.technical_agent.analyze(data)
            
            # Combine results
            all_issues = []
            all_recommendations = []
            
            if quality_results.get("issues"):
                all_issues.extend(quality_results["issues"])
            if quality_results.get("recommendations"):
                all_recommendations.extend(quality_results["recommendations"])
                
            if quantification_results.get("issues"):
                all_issues.extend(quantification_results["issues"])
            if quantification_results.get("recommendations"):
                all_recommendations.extend(quantification_results["recommendations"])
                
            if technical_results.get("issues"):
                all_issues.extend(technical_results["issues"])
            if technical_results.get("recommendations"):
                all_recommendations.extend(technical_results["recommendations"])
            
            # Generate report
            report = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "directory": str(results_dir),
                "overall_assessment": {
                    "quality_summary": quality_results.get("summary", "No quality analysis available"),
                    "quantification_summary": quantification_results.get("summary", "No quantification analysis available"),
                    "technical_summary": technical_results.get("summary", "No technical analysis available")
                },
                "issues": all_issues,
                "recommendations": all_recommendations
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return {
                "error": str(e),
                "issues": [{
                    "severity": "high",
                    "description": f"Analysis failed: {str(e)}"
                }]
            }
