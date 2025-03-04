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
from datetime import datetime

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
    
    async def _collect_file_content(self, file_path: Path, max_size: int = 500000) -> Dict[str, Any]:
        """Safely collect file content with focus on log and data files."""
        result = {
            'path': str(file_path),
            'size': file_path.stat().st_size if file_path.exists() else 0,
            'is_json': False,
            'is_log': False,
            'is_data': False,
            'content': None,
            'parsed_content': None,
            'error': None
        }
        
        try:
            # Skip if file is too large
            if result['size'] > max_size:
                result['content'] = f"File too large ({result['size']} bytes)"
                return result
            
            # Determine file type
            suffix = file_path.suffix.lower()
            name = file_path.name.lower()
            
            # Identify log files
            result['is_log'] = any([
                suffix in ['.log', '.out', '.err'],
                'log' in name,
                'output' in name,
                'error' in name
            ])
            
            # Identify data files
            result['is_data'] = any([
                suffix in ['.tsv', '.csv', '.txt', '.json', '.stats', '.metrics'],
                'metrics' in name,
                'stats' in name,
                'report' in name
            ])
            
            # Read and parse content based on type
            with open(file_path, 'r') as f:
                content = f.read()
                
                if suffix == '.json' or name.endswith('.json'):
                    try:
                        result['content'] = json.loads(content)
                        result['is_json'] = True
                        result['parsed_content'] = {
                            'type': 'json',
                            'metrics': result['content']
                        }
                    except json.JSONDecodeError:
                        result['content'] = content
                
                elif result['is_log']:
                    result['content'] = content
                    # Parse log content for key information
                    result['parsed_content'] = {
                        'type': 'log',
                        'summary': self._parse_log_content(content)
                    }
                
                elif result['is_data']:
                    result['content'] = content
                    # Parse data file content
                    result['parsed_content'] = {
                        'type': 'data',
                        'summary': self._parse_data_content(content, suffix)
                    }
                
                else:
                    result['content'] = content
                    
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error reading {file_path}: {e}")
            
        return result
        
    def _parse_log_content(self, content: str) -> Dict[str, Any]:
        """Parse log content for key information."""
        lines = content.split('\n')
        return {
            'total_lines': len(lines),
            'errors': [l for l in lines if 'error' in l.lower()],
            'warnings': [l for l in lines if 'warning' in l.lower()],
            'stats': [l for l in lines if any(x in l.lower() for x in ['processed', 'aligned', 'total', 'rate', 'percentage'])],
            'parameters': [l for l in lines if any(x in l.lower() for x in ['parameter', 'option', 'setting', 'config', '--', '-p'])]
        }
        
    def _parse_data_content(self, content: str, suffix: str) -> Dict[str, Any]:
        """Parse data file content based on type."""
        lines = content.split('\n')
        
        # Handle TSV/CSV files
        if suffix in ['.tsv', '.csv']:
            delimiter = '\t' if suffix == '.tsv' else ','
            try:
                # Get header and first few data rows
                rows = [line.split(delimiter) for line in lines if line.strip()]
                if len(rows) > 0:
                    return {
                        'headers': rows[0],
                        'num_columns': len(rows[0]),
                        'num_rows': len(rows) - 1,
                        'sample_rows': rows[1:min(6, len(rows))]
                    }
            except Exception as e:
                self.logger.error(f"Error parsing {suffix} content: {e}")
                
        # Handle metrics/stats files
        elif suffix in ['.stats', '.metrics']:
            return {
                'metrics': [l for l in lines if ':' in l],
                'total_metrics': len([l for l in lines if ':' in l])
            }
            
        return {
            'total_lines': len(lines),
            'content_preview': '\n'.join(lines[:5])
        }

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
            
            # Helper function to recursively find files
            def find_files(directory: Path) -> List[Path]:
                files = []
                try:
                    for item in directory.iterdir():
                        if item.is_file():
                            files.append(item)
                        elif item.is_dir():
                            files.extend(find_files(item))
                except (PermissionError, OSError) as e:
                    self.logger.warning(f"Error accessing directory {directory}: {e}")
                return files
            
            # Find all files
            files = find_files(output_dir)
            if not files:
                return {
                    'status': 'error',
                    'message': f'No files found in directory {output_dir} or its subdirectories'
                }
                
            # Count files by type
            file_types = {}
            for f in files:
                ext = f.suffix.lower()
                file_types[ext] = file_types.get(ext, 0) + 1
                    
            self.logger.debug(f"Found files by type: {file_types}")
            
            # Look for specific file types we care about
            log_files = [f for f in files if f.suffix.lower() == '.log']
            data_files = [f for f in files if f.suffix.lower() in ['.tsv', '.csv']]
            qc_files = [f for f in files if 'multiqc_data' in str(f)]
            
            if not any([log_files, data_files, qc_files]):
                return {
                    'status': 'error',
                    'message': 'No analyzable files found. Looking for: .log, .tsv, .csv, or multiqc data files'
                }
            
            # Extract content from files
            log_content = self._extract_log_content(log_files)
            data_content = self._extract_data_content(data_files)
            qc_content = self._extract_qc_content(qc_files)
            
            # Build analysis prompt with actual content
            prompt = f"""
            Please analyze these workflow outputs:
            
            Files Found:
            {json.dumps(file_types, indent=2)}
            
            Log Content:
            {log_content}
            
            Data Content:
            {data_content}
            
            QC Content:
            {qc_content}
            
            Please provide:
            1. Summary of Files:
               - Types and counts of files found
               - Any missing expected files
            
            2. Content Analysis:
               - Key information from logs
               - Important metrics from data files
               - Quality control findings
            
            3. Issues and Recommendations:
               - Any problems identified
               - Suggested next steps
            """
            
            # Generate analysis
            self.logger.debug("Generating analysis with LLM...")
            analysis = await self.llm.generate_analysis(prompt)
            
            return {
                'status': 'success',
                'analysis': analysis,
                'file_summary': {
                    'total_files': len(files),
                    'by_type': file_types,
                    'log_files': len(log_files),
                    'data_files': len(data_files),
                    'qc_files': len(qc_files)
                }
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
        outputs = {
            'metadata': {
                'workflow_date': datetime.now().isoformat(),
                'workflow_dir': str(output_dir),
                'tool_versions': {},
                'parameters': {},
            },
            'quality_control': {
                'fastqc': {},
                'adapter_content': {},
                'sequence_quality': {},
                'duplication_rates': {},
                'overrepresented_sequences': {},
            },
            'alignment': {
                'overall_rate': None,
                'unique_rate': None,
                'multi_mapped': None,
                'unmapped': None,
                'read_distribution': {},
                'insert_size': {},
                'contamination': {},
            },
            'expression': {
                'total_reads': None,
                'mapped_reads': None,
                'quantified_targets': None,
                'samples': {}
            },
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        try:
            # Define potential results directories to search
            search_dirs = [output_dir]
            
            # Check for results in parent directory (common case)
            parent_results = Path(os.path.dirname(os.path.dirname(output_dir))) / "results"
            if parent_results.exists():
                search_dirs.append(parent_results)
                self.logger.info(f"Found results directory at {parent_results}")
            
            # Check for results in the workflow directory itself
            workflow_results = output_dir / "results"
            if workflow_results.exists():
                search_dirs.append(workflow_results)
                self.logger.info(f"Found results directory at {workflow_results}")
            
            # Check if we have a workflow.json that might point to a results directory
            workflow_json = output_dir / "workflow.json"
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
                                        potential_path = Path(os.path.join(os.path.dirname(output_dir), dir_path))
                                        if potential_path.exists():
                                            search_dirs.append(potential_path)
                                            self.logger.info(f"Found potential results directory from workflow.json: {potential_path}")
                except Exception as e:
                    self.logger.warning(f"Error reading workflow.json: {str(e)}")
            
            # Log the directories we're searching
            self.logger.info(f"Searching for output files in: {[str(d) for d in search_dirs]}")
            
            # Find all relevant files recursively across all search directories
            fastqc_files = []
            multiqc_files = []
            run_info_files = []
            
            for search_dir in search_dirs:
                fastqc_files.extend(list(search_dir.rglob('*_fastqc.zip')))
                fastqc_files.extend(list(search_dir.rglob('*_fastqc.html')))
                multiqc_files.extend(list(search_dir.rglob('multiqc_data.json')))
                run_info_files.extend(list(search_dir.rglob('run_info.json')))
            
            self.logger.info(f"Found {len(fastqc_files)} FastQC files, {len(multiqc_files)} MultiQC files, and {len(run_info_files)} run info files")
            
            # Process FastQC outputs
            for file in fastqc_files:
                try:
                    if file.suffix == '.zip':
                        # Handle zip files
                        import zipfile
                        with zipfile.ZipFile(file, 'r') as zip_ref:
                            data_file = next((f for f in zip_ref.namelist() if f.endswith('fastqc_data.txt')), None)
                            if data_file:
                                with zip_ref.open(data_file) as f:
                                    content = f.read().decode('utf-8')
                            else:
                                continue
                    else:
                        # Handle HTML files
                        with open(file) as f:
                            content = f.read()
                    
                    sample_name = file.stem.replace('_fastqc', '')
                    outputs['quality_control']['fastqc'][sample_name] = {}
                    
                    if 'Per base sequence quality' in content:
                        quality_data = self._parse_fastqc_section(content, 'Per base sequence quality')
                        outputs['quality_control']['sequence_quality'][sample_name] = quality_data
                    if 'Adapter Content' in content:
                        adapter_data = self._parse_fastqc_section(content, 'Adapter Content')
                        outputs['quality_control']['adapter_content'][sample_name] = adapter_data
                    if 'Sequence Duplication Levels' in content:
                        dup_data = self._parse_fastqc_section(content, 'Sequence Duplication Levels')
                        outputs['quality_control']['duplication_rates'][sample_name] = dup_data
                except Exception as e:
                    outputs['issues'].append(f"Error parsing FastQC file {file.name}: {str(e)}")
            
            # Process MultiQC data
            for file in multiqc_files:
                try:
                    with open(file) as f:
                        mqc_data = json.load(f)
                        if 'report_general_stats_data' in mqc_data:
                            for stats in mqc_data['report_general_stats_data']:
                                for sample, metrics in stats.items():
                                    if 'FastQC' in metrics:
                                        outputs['quality_control']['fastqc'][sample] = metrics['FastQC']
                except Exception as e:
                    outputs['issues'].append(f"Error parsing MultiQC data from {file.name}: {str(e)}")
            
            # Process expression data (Kallisto outputs)
            for info_file in run_info_files:
                try:
                    # Find corresponding abundance file in the same directory
                    abundance_file = info_file.parent / 'abundance.tsv'
                    if not abundance_file.exists():
                        continue
                        
                    # Get run info
                    with open(info_file) as f:
                        run_info = json.load(f)
                        # Use parent directory name as sample name, removing any .fastq extension
                        sample_name = info_file.parent.name
                        if sample_name.endswith('.fastq'):
                            sample_name = sample_name[:-6]
                        outputs['expression']['samples'][sample_name] = {
                            'n_processed': run_info.get('n_processed', 0),
                            'n_pseudoaligned': run_info.get('n_pseudoaligned', 0),
                            'n_unique': run_info.get('n_unique', 0),
                            'p_pseudoaligned': run_info.get('p_pseudoaligned', 0)
                        }
                    
                    # Get abundance data
                    with open(abundance_file) as f:
                        header = f.readline().strip().split('\t')
                        if 'tpm' in [h.lower() for h in header]:
                            tpm_idx = [h.lower() for h in header].index('tpm')
                            tpm_values = []
                            for line in f:
                                fields = line.strip().split('\t')
                                if len(fields) > tpm_idx:
                                    try:
                                        tpm = float(fields[tpm_idx])
                                        tpm_values.append(tpm)
                                    except ValueError:
                                        continue
                            
                            if tpm_values:
                                outputs['expression']['samples'][sample_name]['expressed_genes'] = len([t for t in tpm_values if t > 1])
                                outputs['expression']['samples'][sample_name]['median_tpm'] = sorted(tpm_values)[len(tpm_values)//2]
                except Exception as e:
                    outputs['issues'].append(f"Error processing expression data for {info_file.parent.name}: {str(e)}")
            
            # Add recommendations based on metrics
            for sample, metrics in outputs['expression']['samples'].items():
                if metrics.get('p_pseudoaligned', 0) < 70:
                    outputs['recommendations'].append(
                        f"Low alignment rate ({metrics['p_pseudoaligned']:.1f}%) for sample {sample}. "
                        "Consider checking for contamination or updating reference transcriptome."
                    )
            
            if outputs['quality_control']['fastqc']:
                for sample, metrics in outputs['quality_control']['fastqc'].items():
                    if metrics.get('per_base_quality', {}).get('mean', 0) < 30:
                        outputs['recommendations'].append(
                            f"Low base quality scores detected in {sample}. Consider more stringent quality filtering."
                        )
            
            return outputs
            
        except Exception as e:
            self.logger.error(f"Error collecting tool outputs: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }    

    def _parse_fastqc_section(self, content: str, section_name: str) -> Dict[str, Any]:
        """Parse a specific section from FastQC output."""
        try:
            # Split content into lines and find section
            lines = content.split('\n')
            section_start = -1
            section_end = -1
            
            # Find section boundaries
            for i, line in enumerate(lines):
                if line.startswith(f'>>{section_name}'):
                    section_start = i
                elif line.startswith('>>END_MODULE') and section_start != -1:
                    section_end = i
                    break
            
            if section_start == -1:
                self.logger.debug(f"Section {section_name} not found")
                return {}
                
            if section_end == -1:
                section_end = len(lines)
            
            # Extract section lines
            section_lines = lines[section_start:section_end]
            self.logger.debug(f"Found section {section_name} with {len(section_lines)} lines")
            self.logger.debug(f"First few lines:\n" + '\n'.join(section_lines[:5]))
            
            # Remove module header and column header lines
            data_lines = []
            for line in section_lines:
                if not line.startswith('>>') and not line.startswith('#'):
                    data_lines.append(line)
            
            if not data_lines:
                self.logger.debug(f"No data lines found in section {section_name}")
                return {}
            
            # Parse based on section type
            data = {}
            
            if section_name == "Per base sequence quality":
                # First line is column headers, skip it
                data['base_positions'] = []
                data['mean_scores'] = []
                data['median_scores'] = []
                data['lower_quartile'] = []
                data['upper_quartile'] = []
                
                for line in data_lines[1:]:  # Skip header row
                    if not line.strip():
                        continue
                    try:
                        parts = line.strip().split('\t')
                        self.logger.debug(f"Processing line: {parts}")
                        if len(parts) >= 6:
                            base = parts[0]
                            mean = float(parts[1])
                            median = float(parts[2])
                            lower = float(parts[3])
                            upper = float(parts[4])
                            
                            data['base_positions'].append(base)
                            data['mean_scores'].append(mean)
                            data['median_scores'].append(median)
                            data['lower_quartile'].append(lower)
                            data['upper_quartile'].append(upper)
                    except (ValueError, IndexError) as e:
                        self.logger.debug(f"Error parsing line {line}: {str(e)}")
                        continue
                
                if data['mean_scores']:
                    data['overall_mean'] = sum(data['mean_scores']) / len(data['mean_scores'])
                    data['overall_median'] = sum(data['median_scores']) / len(data['median_scores'])
                
            elif section_name == "Adapter Content":
                # Get adapter names from first data line
                header = data_lines[0].strip().split('\t')
                if len(header) > 1:
                    adapter_names = header[1:]  # Skip 'Position' column
                    data['positions'] = []
                    data['adapters'] = {name: [] for name in adapter_names}
                    
                    for line in data_lines[1:]:
                        if not line.strip():
                            continue
                        try:
                            parts = line.strip().split('\t')
                            self.logger.debug(f"Processing adapter line: {parts}")
                            if len(parts) >= len(adapter_names) + 1:
                                position = parts[0]
                                data['positions'].append(position)
                                for i, adapter in enumerate(adapter_names):
                                    value = parts[i + 1]
                                    if value.endswith('%'):
                                        value = value[:-1]
                                    data['adapters'][adapter].append(float(value))
                        except (ValueError, IndexError) as e:
                            self.logger.debug(f"Error parsing adapter line {line}: {str(e)}")
                            continue
                    
                    # Calculate max adapter content
                    max_content = 0
                    for values in data['adapters'].values():
                        if values:
                            max_content = max(max_content, max(values))
                    data['max_adapter_content'] = max_content
                
            elif section_name == "Sequence Duplication Levels":
                data['duplication_levels'] = []
                data['percentages'] = []
                
                for line in data_lines[1:]:  # Skip header
                    if not line.strip() or line.startswith('Total'):  # Skip empty lines and summary
                        continue
                    try:
                        parts = line.strip().split('\t')
                        self.logger.debug(f"Processing duplication line: {parts}")
                        if len(parts) >= 2:
                            level = parts[0]
                            percentage = parts[1]
                            if percentage.endswith('%'):
                                percentage = percentage[:-1]
                            data['duplication_levels'].append(level)
                            data['percentages'].append(float(percentage))
                    except (ValueError, IndexError) as e:
                        self.logger.debug(f"Error parsing duplication line {line}: {str(e)}")
                        continue
                
                if data['percentages']:
                    data['total_duplication'] = sum(data['percentages'])
                    data['max_duplication'] = max(data['percentages'])
            
            return data
            
        except Exception as e:
            self.logger.warning(f"Error parsing FastQC section {section_name}: {str(e)}")
            return {}

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

    async def generate_analysis_report(self, workflow_dir: Path, query: str = None) -> str:
        """Generate analysis report for workflow outputs."""
        try:
            # Collect all outputs
            outputs = await self._collect_tool_outputs(workflow_dir)
            
            if 'status' in outputs and outputs['status'] == 'error':
                return f"Error analyzing outputs: {outputs.get('message', 'Unknown error')}"
                
            # Generate analysis using LLM
            analysis = await self.llm.generate_analysis(outputs, query or "Analyze the workflow outputs and provide key findings")
            
            # Extract sections from analysis
            sections = {}
            current_section = None
            current_content = []
            
            for line in analysis.split('\n'):
                line = line.strip()
                if line.startswith('###'):
                    if current_section:
                        sections[current_section] = '\n'.join(current_content).strip()
                    current_section = line.lstrip('#').strip()
                    current_content = []
                else:
                    current_content.append(line)
            
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()
            
            # Format report with clean sections
            report = f"""# Workflow Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Directory: {workflow_dir}

## Summary
- Log files analyzed: {len(outputs.get('metadata', {}).get('tool_versions', {}))}
- QC metrics analyzed: {len(outputs.get('quality_control', {}).get('fastqc', {}))} samples
- Issues found: {len(outputs.get('issues', []))}
- Recommendations: {len(outputs.get('recommendations', []))}

"""
            # Add each section
            for section, content in sections.items():
                report += f"## {section}\n{content}\n\n"
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error analyzing outputs: {e}")
            return f"Error analyzing outputs: {str(e)}"

    def _extract_log_content(self, log_files: List[Path]) -> str:
        """Extract relevant content from log files."""
        content = []
        for file in log_files:
            try:
                content.append(f"\n# Log File: {file.name}")
                with open(file, 'r') as f:
                    # Read file in chunks to handle large files
                    chunk_size = 8192  # 8KB chunks
                    buffer = []
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        buffer.append(chunk)
                        
                    text = ''.join(buffer)
                    lines = text.split('\n')
                    
                    # Extract key information
                    version_lines = [l.strip() for l in lines if 'version' in l.lower()]
                    error_lines = [l.strip() for l in lines if 'error' in l.lower()]
                    warning_lines = [l.strip() for l in lines if 'warning' in l.lower()]
                    stat_lines = [l.strip() for l in lines if any(x in l.lower() for x in ['processed', 'aligned', 'total', 'rate', 'percentage'])]
                    param_lines = [l.strip() for l in lines if any(x in l.lower() for x in ['parameter', 'option', 'setting', 'config', '--', '-p'])]
                    
                    # Add organized sections
                    if version_lines:
                        content.append("Versions:")
                        content.extend(version_lines)
                    if error_lines:
                        content.append("\nErrors:")
                        content.extend(error_lines)
                    if warning_lines:
                        content.append("\nWarnings:")
                        content.extend(warning_lines)
                    if stat_lines:
                        content.append("\nStatistics:")
                        content.extend(stat_lines)
                    if param_lines:
                        content.append("\nParameters:")
                        content.extend(param_lines)
                        
            except Exception as e:
                self.logger.warning(f"Could not read {file}: {str(e)}")
                content.append(f"Error reading file: {str(e)}")
                
        return '\n'.join(content) if content else "No log content found"
        
    def _extract_data_content(self, data_files: List[Path]) -> str:
        """Extract relevant content from data files."""
        content = []
        for file in data_files:
            try:
                content.append(f"\n# Data File: {file.name}")
                with open(file, 'r') as f:
                    # Read header and sample data
                    header = f.readline().strip()
                    content.append(f"Header: {header}")
                    
                    # Get column names
                    columns = header.split('\t')
                    content.append(f"Number of columns: {len(columns)}")
                    content.append("Column names:")
                    content.extend([f"  {i+1}. {col}" for i, col in enumerate(columns)])
                    
                    # Read sample data
                    data = []
                    for _ in range(100):  # Read up to 100 lines
                        line = f.readline()
                        if not line:
                            break
                        data.append(line.strip().split('\t'))
                        
                    if data:
                        content.append(f"\nRows analyzed: {len(data)}")
                        
                        # Try to get stats for numeric columns
                        for col_idx, col_name in enumerate(columns):
                            try:
                                col_data = []
                                for row in data:
                                    if col_idx < len(row):  # Ensure column exists
                                        try:
                                            val = float(row[col_idx])
                                            col_data.append(val)
                                        except ValueError:
                                            continue
                                
                                if col_data:  # Only show stats if we found numeric values
                                    content.append(f"\nStats for column '{col_name}':")
                                    content.append(f"  Min: {min(col_data):.2f}")
                                    content.append(f"  Max: {max(col_data):.2f}")
                                    content.append(f"  Mean: {sum(col_data)/len(col_data):.2f}")
                                    content.append(f"  Values found: {len(col_data)}")
                            except Exception as e:
                                self.logger.debug(f"Could not analyze column {col_idx}: {e}")
                                
            except Exception as e:
                self.logger.warning(f"Could not read {file}: {str(e)}")
                content.append(f"Error reading file: {str(e)}")
                
        return '\n'.join(content) if content else "No data content found"

    def _extract_qc_content(self, qc_files: List[Path]) -> str:
        """Extract relevant content from QC files."""
        content = []
        for file in qc_files:
            try:
                if file.suffix in ['.txt', '.tsv', '.csv', '.json']:
                    with open(file, 'r') as f:
                        content.append(f"\n# QC File: {file.name}")
                        for line in f:
                            line = line.strip()
                            # Only include informative QC metrics
                            if any(key in line.lower() for key in [
                                'quality', 'score', 'metric', 'stat',
                                'pass', 'fail', 'warning', 'error',
                                'total', 'mean', 'median', 'std'
                            ]):
                                content.append(line)
            except Exception as e:
                self.logger.warning(f"Could not read {file}: {str(e)}")
        return "\n".join(content)
