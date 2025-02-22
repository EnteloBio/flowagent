"""Module for generating automated analysis reports for any workflow tool."""

import json
import os
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional, Type
from abc import ABC, abstractmethod
import importlib
import pkg_resources
from ..utils.logging import get_logger
from ..core.llm import LLMInterface

logger = get_logger(__name__)

class ToolAnalyzer(ABC):
    """Base class for tool-specific output analyzers."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    def analyze(self) -> Dict[str, Any]:
        """Analyze tool outputs and return results."""
        pass
    
    @abstractmethod
    def get_tool_name(self) -> str:
        """Return the name of the tool this analyzer handles."""
        pass
    
    def find_output_files(self, pattern: str) -> List[Path]:
        """Find tool output files matching pattern."""
        return list(self.output_dir.rglob(pattern))

class MultiQCAnalyzer(ToolAnalyzer):
    """Analyzer for MultiQC outputs."""
    
    def get_tool_name(self) -> str:
        return "multiqc"
    
    def analyze(self) -> Dict[str, Any]:
        """Analyze MultiQC output."""
        results = {
            'status': 'success',
            'issues': [],
            'metrics': {}
        }
        
        # Find MultiQC data file
        data_files = self.find_output_files("multiqc_data.json")
        if not data_files:
            return {'status': 'error', 'message': 'No MultiQC data file found'}
            
        try:
            with open(data_files[0]) as f:
                data = json.load(f)
                
            # Extract general statistics
            if 'general_stats_data' in data:
                results['metrics']['general_stats'] = data['general_stats_data']
                
            # Extract tool-specific metrics
            for key, value in data.items():
                if isinstance(value, dict) and key != 'general_stats_data':
                    results['metrics'][key] = value
            
            return results
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

class LogAnalyzer(ToolAnalyzer):
    """Analyzer for workflow log files."""
    
    def get_tool_name(self) -> str:
        return "logs"
    
    def analyze(self) -> Dict[str, Any]:
        """Analyze workflow logs."""
        results = {
            'status': 'success',
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        log_files = self.find_output_files("*.log")
        for log_file in log_files:
            try:
                with open(log_file) as f:
                    for line in f:
                        if 'ERROR' in line:
                            results['errors'].append(line.strip())
                        elif 'WARNING' in line:
                            results['warnings'].append(line.strip())
                        elif 'INFO' in line:
                            results['info'].append(line.strip())
            except Exception as e:
                results['errors'].append(f"Error reading log file {log_file}: {str(e)}")
        
        return results

class OutputAnalyzer(ToolAnalyzer):
    """Generic analyzer for tool outputs."""
    
    def __init__(self, output_dir: Path, tool_name: str):
        super().__init__(output_dir)
        self.tool_name = tool_name
    
    def get_tool_name(self) -> str:
        return self.tool_name
    
    def analyze(self) -> Dict[str, Any]:
        """Analyze tool outputs."""
        results = {
            'status': 'success',
            'files': [],
            'metrics': {}
        }
        
        # Find all output files for this tool
        tool_dir = self.output_dir / self.tool_name
        if tool_dir.exists():
            results['files'] = [str(p.relative_to(self.output_dir)) 
                              for p in tool_dir.rglob('*') if p.is_file()]
            
            # Try to parse common output formats
            for file in tool_dir.rglob('*'):
                if file.suffix == '.json':
                    try:
                        with open(file) as f:
                            results['metrics'][file.stem] = json.load(f)
                    except:
                        pass
                elif file.suffix in ['.txt', '.log', '.out']:
                    try:
                        with open(file) as f:
                            results['metrics'][file.stem] = f.read()
                    except:
                        pass
        
        return results

class FastQCAnalyzer(ToolAnalyzer):
    """Analyzer for FastQC outputs."""
    
    def get_tool_name(self) -> str:
        return "fastqc"
    
    def analyze(self) -> Dict[str, Any]:
        """Analyze FastQC output."""
        results = {
            'status': 'success',
            'issues': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Look in fastqc_reports directory
        fastqc_dir = self.output_dir / "fastqc_reports"
        if not fastqc_dir.exists():
            return {'status': 'error', 'message': 'FastQC output directory not found'}
            
        # Find all FastQC data files
        data_files = list(fastqc_dir.glob("*_fastqc/fastqc_data.txt"))
        if not data_files:
            data_files = list(fastqc_dir.glob("*.zip"))  # Try zipped files
            
        if not data_files:
            return {'status': 'error', 'message': 'No FastQC data files found'}
            
        for data_file in data_files:
            sample_name = data_file.parent.name.replace('_fastqc', '')
            try:
                metrics = self._parse_fastqc_data(data_file)
                results['metrics'][sample_name] = metrics
                
                # Check for warnings and failures
                for module, status in metrics.get('module_status', {}).items():
                    if status == 'FAIL':
                        results['issues'].append(f"{sample_name}: {module} failed QC")
                    elif status == 'WARN':
                        results['warnings'].append(f"{sample_name}: {module} has warnings")
                        
            except Exception as e:
                results['issues'].append(f"Error parsing {sample_name}: {str(e)}")
                
        return results
        
    def _parse_fastqc_data(self, data_file: Path) -> Dict[str, Any]:
        """Parse FastQC data file."""
        metrics = {'module_status': {}}
        current_module = None
        
        try:
            with open(data_file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('>>'):
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            module_name = parts[0][2:]
                            status = parts[1]
                            metrics['module_status'][module_name] = status
                            current_module = module_name
                            metrics[current_module] = []
                    elif line and not line.startswith('##') and current_module:
                        metrics[current_module].append(line.split('\t'))
        except:
            # If can't read directly (e.g. zipped), try unzipping first
            import zipfile
            import io
            if data_file.suffix == '.zip':
                with zipfile.ZipFile(data_file) as zf:
                    data_name = data_file.stem + '/fastqc_data.txt'
                    with zf.open(data_name) as f:
                        content = io.TextIOWrapper(f)
                        for line in content:
                            line = line.strip()
                            if line.startswith('>>'):
                                parts = line.split('\t')
                                if len(parts) >= 2:
                                    module_name = parts[0][2:]
                                    status = parts[1]
                                    metrics['module_status'][module_name] = status
                                    current_module = module_name
                                    metrics[current_module] = []
                            elif line and not line.startswith('##') and current_module:
                                metrics[current_module].append(line.split('\t'))
                                
        return metrics

class KallistoAnalyzer(ToolAnalyzer):
    """Analyzer for Kallisto outputs."""
    
    def get_tool_name(self) -> str:
        return "kallisto"
    
    def analyze(self) -> Dict[str, Any]:
        """Analyze Kallisto output."""
        results = {
            'status': 'success',
            'issues': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Look in kallisto_output directory
        kallisto_dir = self.output_dir / "kallisto_output"
        if not kallisto_dir.exists():
            return {'status': 'error', 'message': 'Kallisto output directory not found'}
            
        # Find run_info.json files
        info_files = list(kallisto_dir.glob("*/run_info.json"))
        if not info_files:
            return {'status': 'error', 'message': 'No Kallisto run info files found'}
            
        for info_file in info_files:
            sample_name = info_file.parent.name
            try:
                # Parse run info
                with open(info_file) as f:
                    run_info = json.load(f)
                results['metrics'][sample_name] = run_info
                
                # Check for potential issues
                if run_info.get('n_processed', 0) == 0:
                    results['issues'].append(f"{sample_name}: No reads processed")
                elif run_info.get('n_pseudoaligned', 0) / run_info.get('n_processed', 1) < 0.5:
                    results['warnings'].append(
                        f"{sample_name}: Low alignment rate "
                        f"({run_info.get('n_pseudoaligned', 0) / run_info.get('n_processed', 1) * 100:.1f}%)"
                    )
                    
            except Exception as e:
                results['issues'].append(f"Error parsing {sample_name}: {str(e)}")
                
        return results

class ReportGenerator:
    """Generates comprehensive analysis reports for any workflow."""
    
    def __init__(self, output_dir: str, workflow_type: str = None):
        """Initialize report generator.
        
        Args:
            output_dir: Directory containing workflow outputs
            workflow_type: Optional type of workflow (e.g. 'rna_seq', 'chip_seq', etc.)
                         If not provided, will be inferred from outputs
        """
        self.output_dir = Path(output_dir)
        self.workflow_type = workflow_type
        self.logger = get_logger(__name__)
        self.llm = LLMInterface()
        
    def _infer_workflow_type(self, tool_outputs: Dict[str, List]) -> str:
        """Infer workflow type from available tool outputs."""
        # Look for characteristic tool outputs
        tools = set()
        for dir_name in tool_outputs.keys():
            tools.add(dir_name.lower())
            
        # Common workflow signatures
        signatures = {
            'rna_seq': {'kallisto', 'star', 'fastqc', 'multiqc', 'deseq2', 'salmon'},
            'chip_seq': {'bowtie2', 'macs2', 'fastqc', 'multiqc', 'homer'},
            'atac_seq': {'bowtie2', 'macs2', 'fastqc', 'multiqc', 'homer'},
            'wgs': {'bwa', 'samtools', 'gatk', 'fastqc', 'multiqc'},
            'metagenomics': {'kraken2', 'metaphlan', 'fastqc', 'multiqc'},
            'variant_calling': {'gatk', 'samtools', 'bcftools', 'fastqc', 'multiqc'}
        }
        
        # Find best matching workflow type
        best_match = None
        best_score = 0
        for wf_type, signature in signatures.items():
            score = len(tools & signature)
            if score > best_score:
                best_score = score
                best_match = wf_type
                
        return best_match or 'unknown'
    
    async def analyze_tool_outputs(self) -> Dict[str, Any]:
        """Use LLM to analyze tool outputs."""
        tool_outputs = {}
        
        # Collect all output files from subdirectories
        for dir_path in self.output_dir.iterdir():
            if dir_path.is_dir() and not dir_path.name.startswith('.'):
                tool_outputs[dir_path.name] = []
                # Collect all file contents
                for file in dir_path.rglob('*'):
                    if file.is_file():
                        try:
                            if file.suffix in ['.json', '.txt', '.log', '.out', '.tsv', '.csv']:
                                with open(file) as f:
                                    content = f.read()
                                    # Truncate very large files
                                    if len(content) > 10000:
                                        content = content[:10000] + "\n... (truncated)"
                                tool_outputs[dir_path.name].append({
                                    'file': str(file.relative_to(dir_path)),
                                    'content': content
                                })
                            elif file.suffix in ['.html', '.pdf', '.png', '.jpg']:
                                tool_outputs[dir_path.name].append({
                                    'file': str(file.relative_to(dir_path)),
                                    'type': file.suffix[1:]  # Remove leading dot
                                })
                        except Exception as e:
                            self.logger.warning(f"Error reading {file}: {str(e)}")
        
        # Infer workflow type if not provided
        if not self.workflow_type:
            self.workflow_type = self._infer_workflow_type(tool_outputs)
        
        # Ask LLM to analyze outputs
        prompt = f"""
        Analyze the following {self.workflow_type} workflow outputs and provide a comprehensive summary:

        Workflow type: {self.workflow_type}
        Tool outputs available:
        {json.dumps(tool_outputs, indent=2)}

        Please provide:
        1. Overall quality assessment
        2. Key metrics and findings specific to this type of analysis
        3. Any potential issues or warnings
        4. Recommendations for the user

        Focus on the most important information that would be relevant to a bioinformatician.
        If you see any concerning patterns or unusual results, highlight them.
        
        For any metrics, provide context about what values are considered good/bad.
        
        If you see output from tools you don't recognize, focus on general quality metrics
        and any clear error messages or warnings.
        """
        
        try:
            analysis = await self.llm.generate_analysis(prompt)
            return analysis
        except Exception as e:
            self.logger.error(f"Error generating analysis: {str(e)}")
            return {
                'status': 'error',
                'message': f'Failed to analyze outputs: {str(e)}'
            }
    
    async def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        try:
            # Analyze outputs using LLM
            analysis = await self.analyze_tool_outputs()
            
            report = {
                'summary': {
                    'status': 'success',
                    'workflow_type': self.workflow_type,
                    'analysis': analysis
                }
            }
            
            # Save report
            report_file = self.output_dir / "analysis_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return {
                'error': str(e),
                'status': 'error',
                'message': 'Failed to generate analysis report'
            }
