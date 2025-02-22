"""Module for generating automated analysis reports for any workflow tool."""

import json
import os
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional, Type, Union
from abc import ABC, abstractmethod
import importlib
import pkg_resources
from ..utils.logging import get_logger
from ..core.llm import LLMInterface
import fnmatch

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
            'metrics': {
                'summary': {},
                'samples': {}
            }
        }
        
        # Check kallisto index
        index_dir = self.output_dir / "kallisto_index"
        if not index_dir.exists():
            results['warnings'].append("Kallisto index directory not found")
        else:
            index_files = list(index_dir.glob("*.idx"))
            if not index_files:
                results['warnings'].append("No Kallisto index file found")
            else:
                results['metrics']['summary']['index'] = str(index_files[0].name)
        
        # Look in kallisto_output directory
        kallisto_dir = self.output_dir / "kallisto_output"
        if not kallisto_dir.exists():
            return {'status': 'error', 'message': 'Kallisto output directory not found'}
            
        # Find abundance.h5 and run_info.json files
        sample_dirs = [d for d in kallisto_dir.iterdir() if d.is_dir()]
        if not sample_dirs:
            return {'status': 'error', 'message': 'No Kallisto output directories found'}
            
        total_reads = 0
        total_pseudoaligned = 0
        
        for sample_dir in sample_dirs:
            sample_name = sample_dir.name
            results['metrics']['samples'][sample_name] = {}
            
            # Check run info
            run_info_file = sample_dir / "run_info.json"
            if run_info_file.exists():
                try:
                    with open(run_info_file) as f:
                        run_info = json.load(f)
                    results['metrics']['samples'][sample_name]['run_info'] = run_info
                    
                    # Update totals
                    n_processed = run_info.get('n_processed', 0)
                    n_pseudoaligned = run_info.get('n_pseudoaligned', 0)
                    total_reads += n_processed
                    total_pseudoaligned += n_pseudoaligned
                    
                    # Check alignment rate
                    if n_processed == 0:
                        results['issues'].append(f"{sample_name}: No reads processed")
                    else:
                        alignment_rate = n_pseudoaligned / n_processed
                        results['metrics']['samples'][sample_name]['alignment_rate'] = alignment_rate
                        
                        if alignment_rate < 0.5:
                            results['warnings'].append(
                                f"{sample_name}: Low alignment rate ({alignment_rate * 100:.1f}%)"
                            )
                        elif alignment_rate < 0.3:
                            results['issues'].append(
                                f"{sample_name}: Very low alignment rate ({alignment_rate * 100:.1f}%)"
                            )
                            
                except Exception as e:
                    results['issues'].append(f"Error parsing run info for {sample_name}: {str(e)}")
            else:
                results['issues'].append(f"{sample_name}: No run_info.json found")
            
            # Check abundance files
            abundance_file = sample_dir / "abundance.h5"
            if not abundance_file.exists():
                results['issues'].append(f"{sample_name}: No abundance.h5 file found")
                
        # Add summary metrics
        if total_reads > 0:
            overall_alignment_rate = total_pseudoaligned / total_reads
            results['metrics']['summary'].update({
                'total_reads': total_reads,
                'total_pseudoaligned': total_pseudoaligned,
                'overall_alignment_rate': overall_alignment_rate
            })
            
            if overall_alignment_rate < 0.5:
                results['warnings'].append(
                    f"Overall low alignment rate ({overall_alignment_rate * 100:.1f}%)"
                )
                
        return results

class ReportGenerator:
    """Generates comprehensive analysis reports for any workflow."""
    
    def __init__(self):
        """Initialize the report generator."""
        self.logger = get_logger(__name__)
        self.llm = LLMInterface()
    
    async def _infer_workflow_type(self, outputs: Dict[str, Any]) -> str:
        """Infer workflow type from available outputs."""
        # Extract directory names and file patterns
        directories = set()
        file_patterns = set()
        
        def collect_patterns(data: Dict[str, Any]):
            # Add directory names
            for dirname in data.get('subdirs', {}).keys():
                directories.add(dirname.lower())
                
            # Add file patterns
            for file in data.get('files', []):
                name = Path(file['path']).name.lower()
                file_patterns.add(name)
                
            # Recurse into subdirectories
            for subdir in data.get('subdirs', {}).values():
                collect_patterns(subdir)
        
        collect_patterns(outputs.get('raw_outputs', {}))
        
        # Define workflow signatures
        signatures = {
            'rna_seq': {
                'dirs': {'fastqc', 'kallisto', 'star', 'salmon', 'multiqc'},
                'files': {'abundance.h5', 'run_info.json', '*_fastqc.html', 'multiqc_report.html'}
            },
            'chip_seq': {
                'dirs': {'bowtie2', 'macs2', 'fastqc', 'multiqc', 'homer'},
                'files': {'*.bam', '*.bed', '*_peaks.narrowPeak', '*_fastqc.html'}
            },
            'atac_seq': {
                'dirs': {'bowtie2', 'macs2', 'fastqc', 'multiqc'},
                'files': {'*.bam', '*.bed', '*_peaks.narrowPeak', '*_fastqc.html'}
            },
            'single_cell': {
                'dirs': {'cellranger', '10x', 'seurat', 'scanpy'},
                'files': {'matrix.mtx', 'barcodes.tsv', 'features.tsv', 'web_summary.html'}
            },
            'variant_calling': {
                'dirs': {'gatk', 'samtools', 'bcftools'},
                'files': {'*.vcf', '*.bam', '*.bai', '*.g.vcf'}
            },
            'metagenomics': {
                'dirs': {'kraken2', 'metaphlan', 'humann'},
                'files': {'*.kreport', '*.biom', 'metaphlan_bugs_list.tsv'}
            }
        }
        
        # Score each workflow type
        scores = {}
        for wf_type, signature in signatures.items():
            dir_score = len(directories & signature['dirs'])
            file_score = 0
            for pattern in signature['files']:
                if any(fnmatch.fnmatch(f, pattern) for f in file_patterns):
                    file_score += 1
            scores[wf_type] = dir_score + file_score
        
        # Find best match
        if scores:
            best_match = max(scores.items(), key=lambda x: x[1])
            if best_match[1] > 0:  # Only return if we have some match
                return best_match[0]
        
        return 'unknown'
    
    async def _collect_file_content(self, file_path: Path, max_size: int = 50000) -> Dict[str, Any]:
        """Safely collect file content with size limits and format detection."""
        result = {
            'path': str(file_path),
            'size': file_path.stat().st_size if file_path.exists() else 0,
            'is_json': False,
            'content': None,
            'error': None
        }
        
        try:
            # Skip if file is too large
            if result['size'] > max_size:
                result['content'] = f"File too large ({result['size']} bytes)"
                return result
                
            # Read file content
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Try to parse as JSON
            try:
                if file_path.suffix == '.json':
                    result['content'] = json.loads(content)
                    result['is_json'] = True
                else:
                    result['content'] = content
            except json.JSONDecodeError:
                result['content'] = content
                
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error reading {file_path}: {e}")
            
        return result

    async def analyze_tool_outputs(self, output_dir: Union[str, Path]) -> Dict[str, Any]:
        """Analyze tool outputs using LLM."""
        output_dir = Path(output_dir)
        if not output_dir.exists():
            return {
                'status': 'error',
                'message': f'Output directory {output_dir} does not exist'
            }
        
        try:
            self.logger.debug(f"Starting analysis of directory: {output_dir}")
            tool_outputs = await self._collect_tool_outputs(output_dir)
            
            if not isinstance(tool_outputs, dict):
                self.logger.error(f"Expected dict for tool_outputs, got {type(tool_outputs)}")
                return {
                    'status': 'error',
                    'message': f'Invalid tool outputs format: expected dict, got {type(tool_outputs)}'
                }
            
            raw_outputs = tool_outputs.get('raw_outputs')
            metrics = tool_outputs.get('metrics')
            
            if not raw_outputs or not metrics:
                self.logger.debug(f"No outputs found. Raw outputs: {raw_outputs}, Metrics: {metrics}")
                return {
                    'status': 'error',
                    'message': 'No tool outputs found to analyze'
                }
            
            # Prepare analysis prompt
            self.logger.debug("Preparing analysis prompt...")
            prompt = f"""
            Analyze the following workflow outputs and provide a comprehensive summary:
            
            Output Structure:
            {json.dumps(metrics, indent=2)}
            
            Please provide:
            1. Overall workflow success assessment
            2. Quality metrics analysis
            3. Tool-specific findings
            4. Any warnings or potential issues
            5. Recommendations for improvement
            """
            
            # Generate analysis
            self.logger.debug("Generating analysis with LLM...")
            analysis = await self.llm.generate_analysis(prompt)
            
            return {
                'status': 'success',
                'analysis': analysis,
                'metrics': metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing outputs: {str(e)}")
            return {
                'status': 'error',
                'message': f'Failed to analyze outputs: {str(e)}'
            }
    
    async def _extract_metrics_from_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from collected outputs."""
        self.logger.debug(f"Starting metrics extraction from outputs: {outputs}")
        
        metrics = {
            'file_counts': {},
            'file_sizes': {},
            'error_counts': 0,
            'warning_counts': 0,
            'tool_metrics': {}
        }
        
        async def process_directory(data: Dict[str, Any], prefix: str):
            self.logger.debug(f"Processing directory with prefix '{prefix}': {data}")
            
            # Process files
            files = data.get('files', [])
            if not isinstance(files, list):
                self.logger.error(f"Expected list for files, got {type(files)}")
                return
                
            for file in files:
                if not isinstance(file, dict):
                    self.logger.error(f"Expected dict for file, got {type(file)}")
                    continue
                    
                # Get file extension
                ext = Path(file.get('path', '')).suffix
                metrics['file_counts'][ext] = metrics['file_counts'].get(ext, 0) + 1
                metrics['file_sizes'][ext] = metrics['file_sizes'].get(ext, 0) + file.get('size', 0)
                
                # Check for errors and warnings in content
                content = file.get('content', '')
                if isinstance(content, str):
                    metrics['error_counts'] += content.lower().count('error')
                    metrics['warning_counts'] += content.lower().count('warning')
                
                # Extract JSON metrics if present
                if file.get('is_json') and isinstance(file.get('content'), dict):
                    tool_name = prefix.strip('/') or 'unknown'
                    if tool_name not in metrics['tool_metrics']:
                        metrics['tool_metrics'][tool_name] = []
                    metrics['tool_metrics'][tool_name].append(file['content'])
            
            # Process subdirectories
            subdirs = data.get('subdirs', {})
            if not isinstance(subdirs, dict):
                self.logger.error(f"Expected dict for subdirs, got {type(subdirs)}")
                return
                
            for name, subdir in subdirs.items():
                new_prefix = f"{prefix}/{name}" if prefix else name
                await process_directory(subdir, new_prefix)
        
        # Start processing from root
        if isinstance(outputs, dict):
            await process_directory(outputs, '')
        else:
            self.logger.error(f"Expected dict for outputs, got {type(outputs)}")
        
        self.logger.debug(f"Extracted metrics: {metrics}")
        return metrics

    async def _collect_tool_outputs(self, output_dir: Path) -> Dict[str, Any]:
        """Collect all tool outputs from the workflow directory."""
        try:
            # Collect all outputs
            self.logger.debug("Starting to collect directory outputs...")
            outputs = await self._collect_directory_outputs(output_dir)
            self.logger.debug(f"Raw directory outputs: {outputs}")
            
            # Extract metrics and metadata
            self.logger.debug("Extracting metrics from outputs...")
            metrics = await self._extract_metrics_from_outputs(outputs)
            self.logger.debug(f"Extracted metrics: {metrics}")
            
            return {
                'raw_outputs': outputs,
                'metrics': metrics
            }
        except Exception as e:
            self.logger.error(f"Error in _collect_tool_outputs: {str(e)}")
            raise

    async def _collect_directory_outputs(self, directory: Path) -> Dict[str, Any]:
        """Recursively collect outputs from a directory."""
        if not directory.exists():
            return {'files': [], 'subdirs': {}}

        result = {'files': [], 'subdirs': {}}
        
        try:
            for item in directory.iterdir():
                if item.name.startswith('.'):
                    continue
                    
                if item.is_file():
                    result['files'].append(await self._collect_file_content(item))
                elif item.is_dir():
                    result['subdirs'][item.name] = await self._collect_directory_outputs(item)
        except Exception as e:
            self.logger.error(f"Error collecting outputs from {directory}: {str(e)}")
            
        return result

    async def generate_report(self, output_dir: Union[str, Path]) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        output_dir = Path(output_dir)
        if not output_dir.exists():
            return {
                'success': False,
                'message': f'Output directory {output_dir} does not exist'
            }
        
        try:
            # Analyze outputs
            analysis_results = await self.analyze_tool_outputs(output_dir)
            
            # Save report
            report_file = output_dir / "analysis_report.json"
            with open(report_file, 'w') as f:
                json.dump(analysis_results, f, indent=2)
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return {
                'status': 'error',
                'message': 'Failed to generate analysis report',
                'error': str(e)
            }
