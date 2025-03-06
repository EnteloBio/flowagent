"""Analysis agents for different aspects of workflow results."""

import json
import logging
from typing import Dict, Any, List
from pathlib import Path
import h5py

logger = logging.getLogger(__name__)

class BaseAnalysisAgent:
    """Base class for analysis agents."""
    
    def __init__(self, api_usage_tracker):
        """Initialize base analysis agent."""
        self.api_usage_tracker = api_usage_tracker
    
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Base analyze method to be implemented by subclasses."""
        raise NotImplementedError

class QualityAnalysisAgent(BaseAnalysisAgent):
    """Analyzes quality control metrics and reports."""
    
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze quality control reports."""
        try:
            # Get QC data
            qc_data = data.get("qc_data", {})
            workflow_type = data.get("workflow_type", "unknown")
            
            issues = []
            recommendations = []
            
            # Analyze FastQC reports
            fastqc_reports = qc_data.get("fastqc", [])
            if not fastqc_reports:
                issues.append({
                    "severity": "high",
                    "description": "No FastQC reports found. Quality control may not have run successfully."
                })
                recommendations.append("Run FastQC on all input files to ensure data quality")
            
            # Analyze MultiQC report
            multiqc_report = qc_data.get("multiqc")
            if not multiqc_report:
                issues.append({
                    "severity": "medium",
                    "description": "No MultiQC report found. This makes it harder to compare samples."
                })
                recommendations.append("Generate MultiQC report to compare samples")
            
            # Analyze workflow-specific QC reports
            other_qc = qc_data.get("other_qc", [])
            if workflow_type == "rna_seq":
                # Check for RNA-seq specific QC
                rseqc_reports = [qc for qc in other_qc if qc["tool"] == "rseqc"]
                if not rseqc_reports:
                    recommendations.append("Consider running RSeQC for RNA-seq specific quality metrics")
            elif workflow_type == "chip_seq":
                # Check for ChIP-seq specific QC
                if not any(qc["tool"] == "phantompeakqualtools" for qc in other_qc):
                    recommendations.append("Run phantompeakqualtools for ChIP-seq quality assessment")
            
            # Generate summary
            summary = "Quality control analysis completed."
            if issues:
                summary = f"Found {len(issues)} quality-related issues."
            
            self.api_usage_tracker.track_api_usage("quality_analysis")
            
            return {
                "summary": summary,
                "issues": issues,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Quality analysis failed: {str(e)}")
            return {
                "error": str(e),
                "issues": [{
                    "severity": "high",
                    "description": f"Quality analysis failed: {str(e)}"
                }]
            }

class QuantificationAnalysisAgent(BaseAnalysisAgent):
    """Analyzes quantification results from various tools."""
    
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze quantification results."""
        try:
            quant_data = data.get("quantification_data", {})
            workflow_type = data.get("workflow_type", "unknown")
            tools_used = data.get("tools_used", [])
            
            issues = []
            recommendations = []
            
            # Analyze based on tools used
            if "kallisto" in tools_used:
                self._analyze_kallisto(quant_data.get("kallisto", {}), issues, recommendations)
            
            if "star" in tools_used:
                self._analyze_star(quant_data.get("star", {}), issues, recommendations)
            
            if "salmon" in tools_used:
                self._analyze_salmon(quant_data.get("salmon", {}), issues, recommendations)
            
            if "hisat2" in tools_used:
                self._analyze_hisat2(quant_data.get("hisat2", {}), issues, recommendations)
            
            # Check if any quantification was performed
            if not any(tool in tools_used for tool in ["kallisto", "star", "salmon", "hisat2"]):
                issues.append({
                    "severity": "high",
                    "description": "No quantification results found from any supported tool."
                })
            
            # Generate summary
            summary = "Quantification analysis completed."
            if issues:
                summary = f"Found {len(issues)} quantification-related issues."
            
            self.api_usage_tracker.track_api_usage("quantification_analysis")
            
            return {
                "summary": summary,
                "issues": issues,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Quantification analysis failed: {str(e)}")
            return {
                "error": str(e),
                "issues": [{
                    "severity": "high",
                    "description": f"Quantification analysis failed: {str(e)}"
                }]
            }
    
    def _analyze_kallisto(self, kallisto_data: Dict[str, Any], issues: List[Dict[str, Any]], recommendations: List[str]):
        """Analyze Kallisto output."""
        if not kallisto_data.get("abundance_files"):
            issues.append({
                "severity": "high",
                "description": "No Kallisto abundance files found. Quantification may have failed."
            })
            recommendations.append("Check Kallisto logs for quantification errors")
            return
        
        # Check Kallisto version compatibility
        tool_versions = kallisto_data.get("tool_versions", {})
        if "kallisto" in tool_versions:
            kallisto_version = tool_versions["kallisto"]
            index_version = kallisto_data.get("index_version")
            if index_version and kallisto_version != index_version:
                issues.append({
                    "severity": "high",
                    "description": f"Kallisto version mismatch: tool version {kallisto_version} != index version {index_version}"
                })
                recommendations.append("Rebuild Kallisto index with the current version of Kallisto")
        else:
            issues.append({
                "severity": "medium",
                "description": "Kallisto version information not found. Cannot verify index compatibility."
            })
            recommendations.append("Record Kallisto version information for reproducibility")
        
        # Analyze run info for each sample
        run_info = kallisto_data.get("run_info", {})
        if run_info:
            for sample_name, info in run_info.items():
                n_processed = info.get("n_processed", 0)
                if n_processed < 1000000:
                    issues.append({
                        "severity": "medium",
                        "description": f"Low number of processed reads ({n_processed}) for sample {sample_name}. This may affect quantification accuracy."
                    })
                    recommendations.append(f"Consider increasing sequencing depth for sample {sample_name}")
                
                p_pseudoaligned = info.get("p_pseudoaligned", 0)
                if p_pseudoaligned < 70:
                    issues.append({
                        "severity": "medium",
                        "description": f"Low pseudoalignment rate ({p_pseudoaligned}%) for sample {sample_name}. Check reference transcriptome."
                    })
                    recommendations.append("Verify reference transcriptome is appropriate for your samples")
                
                # Check for expressed genes
                expressed_genes = info.get("n_expressed", 0)
                if expressed_genes < 10000:
                    issues.append({
                        "severity": "medium",
                        "description": f"Low number of expressed genes ({expressed_genes}) for sample {sample_name}"
                    })
                    recommendations.append("Check RNA quality and library complexity")
                
                # Check TPM distribution
                median_tpm = info.get("median_tpm", 0)
                if median_tpm == 0:
                    issues.append({
                        "severity": "high",
                        "description": f"Median TPM is 0 for sample {sample_name}, indicating potential quantification issues"
                    })
                    recommendations.append("Review expression quantification parameters and RNA quality")
        
        # Check index
        index_dir = kallisto_data.get("index_dir")
        if not index_dir:
            issues.append({
                "severity": "medium",
                "description": "Kallisto index directory not found"
            })
            recommendations.append("Verify Kallisto index was properly created")
        elif not any(Path(index_dir).glob("*.idx")):
            issues.append({
                "severity": "high",
                "description": "No Kallisto index file found"
            })
            recommendations.append("Create Kallisto index before running quantification")
    
    def _analyze_star(self, star_data: Dict[str, Any], issues: List[Dict[str, Any]], recommendations: List[str]):
        """Analyze STAR output."""
        if not star_data.get("bam_files"):
            issues.append({
                "severity": "high",
                "description": "No STAR BAM files found. Alignment may have failed."
            })
            recommendations.append("Check STAR logs for alignment errors")
            return
        
        # Analyze log files for mapping rates
        log_files = star_data.get("logs", [])
        if not log_files:
            issues.append({
                "severity": "medium",
                "description": "No STAR log files found. Cannot assess alignment quality."
            })
    
    def _analyze_salmon(self, salmon_data: Dict[str, Any], issues: List[Dict[str, Any]], recommendations: List[str]):
        """Analyze Salmon output."""
        if not salmon_data.get("quant_files"):
            issues.append({
                "severity": "high",
                "description": "No Salmon quantification files found. Quantification may have failed."
            })
            recommendations.append("Check Salmon logs for quantification errors")
            return
        
        meta_info = salmon_data.get("meta_info", {})
        if meta_info:
            num_processed = meta_info.get("num_processed", 0)
            if num_processed < 1000000:
                issues.append({
                    "severity": "medium",
                    "description": f"Low number of processed reads ({num_processed}). This may affect quantification accuracy."
                })
                recommendations.append("Consider increasing sequencing depth for better quantification")
            
            mapping_rate = meta_info.get("mapping_rate", 0)
            if mapping_rate < 70:
                issues.append({
                    "severity": "medium",
                    "description": f"Low mapping rate ({mapping_rate}%). Check reference transcriptome."
                })
                recommendations.append("Verify reference transcriptome is appropriate for your samples")
    
    def _analyze_hisat2(self, hisat2_data: Dict[str, Any], issues: List[Dict[str, Any]], recommendations: List[str]):
        """Analyze HISAT2 output."""
        if not hisat2_data.get("bam_files"):
            issues.append({
                "severity": "high",
                "description": "No HISAT2 BAM files found. Alignment may have failed."
            })
            recommendations.append("Check HISAT2 logs for alignment errors")
            return
        
        # Analyze summary files for mapping rates
        summary_files = hisat2_data.get("summary", [])
        if not summary_files:
            issues.append({
                "severity": "medium",
                "description": "No HISAT2 summary files found. Cannot assess alignment quality."
            })

class TechnicalQCAgent(BaseAnalysisAgent):
    """Analyzes technical aspects of workflow execution."""
    
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze resource usage and technical metrics."""
        try:
            tech_data = data.get("technical_data", {})
            workflow_type = data.get("workflow_type", "unknown")
            tools_used = data.get("tools_used", [])
            
            resource_usage = tech_data.get("resource_usage", {})
            tool_versions = tech_data.get("tool_versions", {})
            logs = tech_data.get("logs", [])
            
            issues = []
            recommendations = []
            
            # Check resource usage
            if resource_usage:
                max_memory = resource_usage.get("max_memory_mb", 0)
                if max_memory > 32000:
                    issues.append({
                        "severity": "medium",
                        "description": f"High memory usage ({max_memory} MB). Consider optimizing memory usage."
                    })
                    recommendations.append("Consider using memory-efficient parameters")
                
                cpu_usage = resource_usage.get("cpu_percent", 0)
                if cpu_usage < 50:
                    recommendations.append("Consider increasing parallel processing to improve CPU utilization")
            
            # Check tool versions
            if not tool_versions:
                issues.append({
                    "severity": "low",
                    "description": "No tool version information available."
                })
                recommendations.append("Record tool versions for reproducibility")
            else:
                # Check for version compatibility
                for tool in tools_used:
                    if tool not in tool_versions:
                        issues.append({
                            "severity": "medium",
                            "description": f"Version information missing for {tool}"
                        })
            
            # Check logs
            error_logs = [log for log in logs if "error" in log.lower()]
            if error_logs:
                issues.append({
                    "severity": "high",
                    "description": f"Found {len(error_logs)} error messages in logs."
                })
                recommendations.append("Review error messages in workflow logs")
            
            # Check workflow-specific technical requirements
            if workflow_type == "single_cell":
                if resource_usage.get("max_memory_mb", 0) < 64000:
                    recommendations.append("Consider increasing memory allocation for single-cell analysis")
            
            # Generate summary
            summary = {
                "resource_efficiency": "Good" if not resource_usage else "Needs optimization",
                "error_count": len(error_logs),
                "tool_versions_available": bool(tool_versions)
            }
            
            self.api_usage_tracker.track_api_usage("technical_analysis")
            
            return {
                "summary": summary,
                "issues": issues,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Technical analysis failed: {str(e)}")
            return {
                "error": str(e),
                "issues": [{
                    "severity": "high",
                    "description": f"Technical analysis failed: {str(e)}"
                }]
            }
