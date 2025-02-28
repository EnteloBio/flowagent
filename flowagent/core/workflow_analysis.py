"""Multi-agent workflow analysis system."""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from .llm import LLMInterface
from ..utils.logging import get_logger
from datetime import datetime

class WorkflowAnalyzer:
    """Multi-agent workflow analysis system."""
    
    def __init__(self, llm_interface: LLMInterface):
        """Initialize workflow analyzer.
        
        Args:
            llm_interface: LLM interface for analysis
        """
        self.llm = llm_interface
        self.logger = get_logger(__name__)
        
    async def analyze_workflow(self, results_dir: str, workflow_type: str) -> Dict[str, Any]:
        """Analyze workflow results using multi-agent approach.
        
        Args:
            results_dir: Directory containing workflow results
            workflow_type: Type of workflow (e.g., "rna_seq", "chip_seq")
            
        Returns:
            Dict containing analysis results
        """
        try:
            self.logger.info(f"Starting multi-agent analysis of {workflow_type} workflow")
            results_path = Path(results_dir)
            
            # Search in base directory and all subdirectories
            search_paths = [results_path]
            for subdir in results_path.rglob("*"):
                if subdir.is_dir():
                    search_paths.append(subdir)
            
            self.logger.info(f"Searching in directories: {[str(p) for p in search_paths]}")
            
            # 1. Data Quality & Preprocessing
            quality_results = await self._analyze_data_quality(search_paths)
            
            # 2. Alignment & Quantification
            alignment_results = await self._analyze_alignment(search_paths)
            
            # 3. Analysis-Specific Metrics
            analysis_results = await self._analyze_workflow_specific(search_paths, workflow_type)
            
            # 4. Resource Usage & Performance
            resource_results = await self._analyze_resource_usage(search_paths)
            
            # 5. Visualization & Results
            viz_results = await self._analyze_visualizations(search_paths)
            
            # Combine all results
            return {
                "data_quality": quality_results,
                "alignment": alignment_results,
                "analysis": analysis_results,
                "resources": resource_results,
                "visualization": viz_results,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze workflow: {e}")
            raise
            
    def _find_files_in_paths(self, search_paths: List[Path], patterns: List[str]) -> List[Path]:
        """Find files matching any of the patterns in any of the search paths."""
        found_files = []
        for path in search_paths:
            for pattern in patterns:
                found_files.extend(path.rglob(pattern))
        return found_files
            
    async def _analyze_data_quality(self, search_paths: List[Path]) -> Dict[str, Any]:
        """Analyze data quality metrics."""
        try:
            # Look for FastQC and MultiQC reports with flexible naming
            fastqc_patterns = ["*_fastqc.html", "*fastqc_report.html", "**/fastqc/*", "*.zip"]
            multiqc_patterns = ["multiqc_report.html", "**/multiqc/*report.html", "**/multiqc_data/*"]
            
            fastqc_reports = self._find_files_in_paths(search_paths, fastqc_patterns)
            multiqc_reports = self._find_files_in_paths(search_paths, multiqc_patterns)
            
            # Parse FastQC results if available
            fastqc_metrics = {}
            for report in fastqc_reports:
                if report.suffix == '.zip':
                    # Extract sample name from FastQC zip
                    sample_name = report.stem.replace('_fastqc', '')
                    fastqc_metrics[sample_name] = {
                        'report': str(report),
                        'status': 'completed'
                    }
            
            return {
                "fastqc_reports": [str(p) for p in fastqc_reports],
                "multiqc_reports": [str(p) for p in multiqc_reports],
                "num_samples": len(fastqc_metrics),
                "sample_metrics": fastqc_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing data quality: {e}")
            return {"error": str(e)}
            
    async def _analyze_alignment(self, search_paths: List[Path]) -> Dict[str, Any]:
        """Analyze alignment and quantification metrics."""
        try:
            # Look for Kallisto output with flexible naming
            abundance_patterns = ["abundance.h5", "abundance.tsv", "**/kallisto/**/*abundance*"]
            run_info_patterns = ["run_info.json", "**/kallisto/**/*info*.json"]
            
            abundance_files = self._find_files_in_paths(search_paths, abundance_patterns)
            run_info_files = self._find_files_in_paths(search_paths, run_info_patterns)
            
            results = {
                "tool": "kallisto",
                "samples": [],
                "num_samples": len(abundance_files)
            }
            
            # Parse run info for each sample
            for info_file in run_info_files:
                try:
                    with open(info_file) as f:
                        run_info = json.load(f)
                    
                    # Try to find corresponding abundance file
                    sample_dir = info_file.parent
                    abundance_file = next((f for f in abundance_files if f.parent == sample_dir), None)
                    
                    sample_info = {
                        "sample": info_file.parent.name,
                        "n_processed": run_info.get("n_processed", 0),
                        "n_pseudoaligned": run_info.get("n_pseudoaligned", 0),
                        "n_unique": run_info.get("n_unique", 0),
                        "abundance_file": str(abundance_file) if abundance_file else None
                    }
                    
                    results["samples"].append(sample_info)
                    
                except Exception as e:
                    self.logger.warning(f"Error parsing run info {info_file}: {e}")
                    
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing alignment: {e}")
            return {"error": str(e)}
            
    async def _analyze_workflow_specific(self, search_paths: List[Path], workflow_type: str) -> Dict[str, Any]:
        """Analyze workflow-specific metrics."""
        try:
            if workflow_type == "rna_seq":
                # RNA-seq specific outputs
                # 1. Expression Quantification
                abundance_files = self._find_files_in_paths(search_paths, ["*abundance*", "*.h5", "*.tsv"])
                kallisto_dir = self._find_files_in_paths(search_paths, ["**/kallisto/*", "**/kallisto_quant/*"])
                
                # Parse abundance files for expression metrics
                expression_metrics = {}
                for abundance_file in abundance_files:
                    try:
                        sample_name = abundance_file.parent.name
                        with open(abundance_file) as f:
                            header = f.readline()  # Skip header
                            tpm_values = []
                            count_values = []
                            for line in f:
                                parts = line.strip().split("\t")
                                if len(parts) >= 4:  # Typical Kallisto abundance.tsv format
                                    try:
                                        tpm = float(parts[3])
                                        count = float(parts[2])
                                        if tpm > 0:
                                            tpm_values.append(tpm)
                                        if count > 0:
                                            count_values.append(count)
                                    except (ValueError, IndexError):
                                        continue
                            
                            if tpm_values:
                                expression_metrics[sample_name] = {
                                    "total_transcripts": len(tpm_values),
                                    "expressed_transcripts": len([t for t in tpm_values if t > 1.0]),
                                    "median_tpm": sorted(tpm_values)[len(tpm_values)//2],
                                    "mean_tpm": sum(tpm_values) / len(tpm_values),
                                    "total_counts": sum(count_values) if count_values else 0
                                }
                    except Exception as e:
                        self.logger.warning(f"Error parsing abundance file {abundance_file}: {e}")

                # 2. Differential Expression
                deseq_files = self._find_files_in_paths(search_paths, ["*deseq*", "*differential*", "*DE_results*"])
                diff_expr_metrics = {}
                for deseq_file in deseq_files:
                    try:
                        with open(deseq_file) as f:
                            header = f.readline().lower()
                            sig_genes = 0
                            total_genes = 0
                            for line in f:
                                total_genes += 1
                                parts = line.strip().split("\t")
                                try:
                                    if "padj" in header:
                                        padj_idx = header.split("\t").index("padj")
                                        if float(parts[padj_idx]) < 0.05:
                                            sig_genes += 1
                                except (ValueError, IndexError):
                                    continue
                            
                            diff_expr_metrics[deseq_file.stem] = {
                                "total_genes": total_genes,
                                "significant_genes": sig_genes,
                                "significance_threshold": 0.05
                            }
                    except Exception as e:
                        self.logger.warning(f"Error parsing DE file {deseq_file}: {e}")

                # 3. Clustering & QC
                clustering_files = self._find_files_in_paths(search_paths, ["*pca*", "*clustering*", "*heatmap*"])
                qc_files = self._find_files_in_paths(search_paths, ["*qc*", "*quality*", "*metrics*"])
                
                # 4. Resource Usage
                log_files = self._find_files_in_paths(search_paths, ["*.log", "**/logs/*"])
                resource_metrics = {}
                for log_file in log_files:
                    try:
                        with open(log_file) as f:
                            content = f.read()
                            if "kallisto" in content.lower():
                                resource_metrics["kallisto"] = {
                                    "found_logs": True,
                                    "log_file": str(log_file)
                                }
                    except Exception as e:
                        self.logger.warning(f"Error parsing log file {log_file}: {e}")
                
                return {
                    "expression_quantification": {
                        "kallisto_output": bool(abundance_files),
                        "num_abundance_files": len(abundance_files),
                        "abundance_files": [str(p) for p in abundance_files],
                        "sample_metrics": expression_metrics
                    },
                    "differential_expression": {
                        "deseq_output": bool(deseq_files),
                        "num_deseq_files": len(deseq_files),
                        "deseq_files": [str(p) for p in deseq_files],
                        "analysis_metrics": diff_expr_metrics
                    },
                    "clustering_analysis": {
                        "clustering_output": bool(clustering_files),
                        "num_clustering_files": len(clustering_files),
                        "clustering_files": [str(p) for p in clustering_files],
                        "qc_files": [str(p) for p in qc_files],
                        "has_qc_metrics": bool(qc_files)
                    },
                    "resource_usage": resource_metrics
                }
            
            elif workflow_type == "chip_seq":
                # ChIP-seq specific outputs
                peak_files = self._find_files_in_paths(search_paths, ["*peaks.bed", "*peaks.narrowPeak", "*peaks.broadPeak"])
                signal_files = self._find_files_in_paths(search_paths, ["*signal*", "*.bw", "*.bigwig", "*.bedgraph"])
                qc_files = self._find_files_in_paths(search_paths, ["*phantompeakqualtools*", "*cross_correlation*"])
                
                return {
                    "peak_calling": {
                        "peak_output": bool(peak_files),
                        "num_peak_files": len(peak_files),
                        "peak_files": [str(p) for p in peak_files]
                    },
                    "signal_tracks": {
                        "signal_output": bool(signal_files),
                        "num_signal_files": len(signal_files),
                        "signal_files": [str(p) for p in signal_files]
                    },
                    "quality_metrics": {
                        "qc_output": bool(qc_files),
                        "num_qc_files": len(qc_files),
                        "qc_files": [str(p) for p in qc_files]
                    }
                }
                
            elif workflow_type == "hic":
                # Hi-C specific outputs
                contact_files = self._find_files_in_paths(search_paths, ["*.cool", "*.mcool", "*.hic"])
                tad_files = self._find_files_in_paths(search_paths, ["*tads*", "*domains*", "*boundaries*"])
                interaction_files = self._find_files_in_paths(search_paths, ["*loops*", "*interactions*", "*anchors*"])
                
                return {
                    "contact_matrices": {
                        "matrix_output": bool(contact_files),
                        "num_matrix_files": len(contact_files),
                        "matrix_files": [str(p) for p in contact_files]
                    },
                    "domain_analysis": {
                        "tad_output": bool(tad_files),
                        "num_tad_files": len(tad_files),
                        "tad_files": [str(p) for p in tad_files]
                    },
                    "interaction_analysis": {
                        "interaction_output": bool(interaction_files),
                        "num_interaction_files": len(interaction_files),
                        "interaction_files": [str(p) for p in interaction_files]
                    }
                }
            
            return {}
        except Exception as e:
            self.logger.error(f"Error analyzing workflow metrics: {e}")
            return {"error": str(e)}
            
    async def _analyze_resource_usage(self, search_paths: List[Path]) -> Dict[str, Any]:
        """Analyze resource usage and performance metrics."""
        try:
            # Look for log files with flexible naming
            log_patterns = ["*.log", "**/*log*", "**/logs/*"]
            log_files = self._find_files_in_paths(search_paths, log_patterns)
            
            return {
                "num_log_files": len(log_files),
                "log_files": [str(p) for p in log_files]
            }
        except Exception as e:
            self.logger.error(f"Error analyzing resource usage: {e}")
            return {"error": str(e)}
            
    async def _analyze_visualizations(self, search_paths: List[Path]) -> Dict[str, Any]:
        """Analyze visualization outputs."""
        try:
            # Look for common plot formats with flexible naming
            plot_patterns = ["*.png", "*.pdf", "*.svg", "**/plots/*", "**/figures/*"]
            plots = self._find_files_in_paths(search_paths, plot_patterns)
            
            return {
                "num_plots": len(plots),
                "plot_files": [str(p) for p in plots]
            }
        except Exception as e:
            self.logger.error(f"Error analyzing visualizations: {e}")
            return {"error": str(e)}
            
    def format_analysis_report(self, analysis: Dict[str, Any]) -> str:
        """Format analysis results into a readable report."""
        try:
            sections = []
            
            # Add report header with current time
            sections.append("# Workflow Analysis Report")
            sections.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # 1. Data Quality Section
            quality = analysis.get("data_quality", {})
            sections.append("## 1. Data Quality & Preprocessing")
            sections.append("### Raw Data Quality Metrics")
            if "error" in quality:
                sections.append(f"Error: {quality['error']}")
            else:
                sections.append(f"- FastQC Reports: {len(quality.get('fastqc_reports', []))}")
                sections.append(f"- MultiQC Reports: {len(quality.get('multiqc_reports', []))}")
                sections.append(f"- Number of Samples: {quality.get('num_samples', 0)}\n")
            
            # 2. Alignment Section
            alignment = analysis.get("alignment", {})
            sections.append("## 2. Alignment & Quantification")
            if "error" in alignment:
                sections.append(f"Error: {alignment['error']}")
            else:
                sections.append(f"- Tool: {alignment.get('tool', 'Unknown')}")
                sections.append(f"- Number of Samples: {alignment.get('num_samples', 0)}")
                if alignment.get("samples"):
                    sections.append("\nSample Statistics:")
                    for sample in alignment["samples"]:
                        sections.append(f"- {sample['sample']}:")
                        sections.append(f"  - Processed Reads: {sample.get('n_processed', 0):,}")
                        sections.append(f"  - Pseudoaligned Reads: {sample.get('n_pseudoaligned', 0):,}")
                        sections.append(f"  - Unique Alignments: {sample.get('n_unique', 0):,}\n")
            
            # 3. Workflow-Specific Section
            workflow = analysis.get("analysis", {})
            sections.append("## 3. Analysis-Specific Metrics")
            if "error" in workflow:
                sections.append(f"Error: {workflow['error']}")
            else:
                # RNA-seq specific metrics
                if "expression_quantification" in workflow:
                    sections.append("### RNA-seq Analysis")
                    expr = workflow["expression_quantification"]
                    sections.append("#### Expression Quantification")
                    sections.append(f"- Kallisto Output Present: {expr['kallisto_output']}")
                    sections.append(f"- Number of Abundance Files: {expr['num_abundance_files']}")
                    sections.append(f"- TPM Metrics: {expr['sample_metrics']}")
                    
                    diff = workflow["differential_expression"]
                    sections.append("\n#### Differential Expression")
                    sections.append(f"- DESeq Output Present: {diff['deseq_output']}")
                    sections.append(f"- Number of DE Results: {diff['num_deseq_files']}")
                    sections.append(f"- Analysis Metrics: {diff['analysis_metrics']}")
                    
                    clust = workflow["clustering_analysis"]
                    sections.append("\n#### Clustering Analysis")
                    sections.append(f"- Clustering Output Present: {clust['clustering_output']}")
                    sections.append(f"- Number of Clustering Files: {clust['num_clustering_files']}")
                    sections.append(f"- QC Metrics Present: {clust['has_qc_metrics']}")
                    
                    batch = workflow.get("batch_correction", {})
                    sections.append("\n#### Batch Effect Analysis")
                    sections.append(f"- Batch Output Present: {batch.get('batch_output', False)}")
                    sections.append(f"- Number of Batch Files: {batch.get('num_batch_files', 0)}")
                    
                    pathway = workflow.get("pathway_analysis", {})
                    sections.append("\n#### Pathway Analysis")
                    sections.append(f"- Pathway Output Present: {pathway.get('pathway_output', False)}")
                    sections.append(f"- Number of Pathway Files: {pathway.get('num_pathway_files', 0)}")
                
                # ChIP-seq specific metrics
                elif "peak_calling" in workflow:
                    sections.append("### ChIP-seq Analysis")
                    peaks = workflow["peak_calling"]
                    sections.append("#### Peak Calling")
                    sections.append(f"- Peak Files Present: {peaks['peak_output']}")
                    sections.append(f"- Number of Peak Files: {peaks['num_peak_files']}")
                    
                    signal = workflow["signal_tracks"]
                    sections.append("\n#### Signal Tracks")
                    sections.append(f"- Signal Files Present: {signal['signal_output']}")
                    sections.append(f"- Number of Signal Files: {signal['num_signal_files']}")
                    
                    qc = workflow["quality_metrics"]
                    sections.append("\n#### Quality Metrics")
                    sections.append(f"- QC Files Present: {qc['qc_output']}")
                    sections.append(f"- Number of QC Files: {qc['num_qc_files']}\n")
                
                # Hi-C specific metrics
                elif "contact_matrices" in workflow:
                    sections.append("### Hi-C Analysis")
                    matrices = workflow["contact_matrices"]
                    sections.append("#### Contact Matrices")
                    sections.append(f"- Matrix Files Present: {matrices['matrix_output']}")
                    sections.append(f"- Number of Matrix Files: {matrices['num_matrix_files']}")
                    
                    domains = workflow["domain_analysis"]
                    sections.append("\n#### Domain Analysis")
                    sections.append(f"- TAD Files Present: {domains['tad_output']}")
                    sections.append(f"- Number of TAD Files: {domains['num_tad_files']}")
                    
                    interactions = workflow["interaction_analysis"]
                    sections.append("\n#### Interaction Analysis")
                    sections.append(f"- Interaction Files Present: {interactions['interaction_output']}")
                    sections.append(f"- Number of Interaction Files: {interactions['num_interaction_files']}\n")
            
            # 4. Resource Usage Section
            resources = analysis.get("resources", {})
            sections.append("## 4. Resource Usage & Performance")
            if "error" in resources:
                sections.append(f"Error: {resources['error']}")
            else:
                sections.append(f"- Number of Log Files: {resources.get('num_log_files', 0)}\n")
            
            # 5. Visualization Section
            viz = analysis.get("visualization", {})
            sections.append("## 5. Visualization & Results")
            if "error" in viz:
                sections.append(f"Error: {viz['error']}")
            else:
                sections.append(f"- Number of Plot Files: {viz.get('num_plots', 0)}\n")
            
            return "\n".join(sections)
            
        except Exception as e:
            self.logger.error(f"Error formatting analysis report: {e}")
            return f"Error generating report: {e}"
