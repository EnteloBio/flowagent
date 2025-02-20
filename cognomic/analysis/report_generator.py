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

class ReportGenerator:
    """Generates comprehensive analysis reports for any workflow."""
    
    def __init__(self, output_dir: str):
        """Initialize report generator."""
        self.output_dir = Path(output_dir)
        self.logger = get_logger(__name__)
        self.analyzers: Dict[str, ToolAnalyzer] = {}
        
        # Register built-in analyzers
        self.register_analyzer(MultiQCAnalyzer(self.output_dir))
        self.register_analyzer(LogAnalyzer(self.output_dir))
        
        # Load plugin analyzers
        self._load_plugins()
    
    def register_analyzer(self, analyzer: ToolAnalyzer):
        """Register a tool analyzer."""
        self.analyzers[analyzer.get_tool_name()] = analyzer
    
    def _load_plugins(self):
        """Load analyzer plugins."""
        try:
            for entry_point in pkg_resources.iter_entry_points('cognomic.analyzers'):
                analyzer_class = entry_point.load()
                if issubclass(analyzer_class, ToolAnalyzer):
                    analyzer = analyzer_class(self.output_dir)
                    self.register_analyzer(analyzer)
        except Exception as e:
            self.logger.warning(f"Error loading analyzer plugins: {str(e)}")
    
    def analyze_tool_output(self, tool_name: str) -> Dict[str, Any]:
        """Analyze output from a specific tool."""
        if tool_name in self.analyzers:
            return self.analyzers[tool_name].analyze()
        else:
            # Use generic analyzer for unknown tools
            analyzer = OutputAnalyzer(self.output_dir, tool_name)
            return analyzer.analyze()
    
    def generate_report(self, workflow_tools: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        report = {
            'summary': {
                'status': 'success',
                'issues': [],
                'warnings': [],
                'recommendations': []
            },
            'tools': {}
        }
        
        try:
            # If no specific tools provided, analyze all tool outputs in directory
            if not workflow_tools:
                workflow_tools = [d.name for d in self.output_dir.iterdir() 
                                if d.is_dir() and not d.name.startswith('.')]
            
            # Analyze each tool's output
            for tool in workflow_tools:
                tool_results = self.analyze_tool_output(tool)
                report['tools'][tool] = tool_results
                
                # Check for issues
                if tool_results.get('status') == 'error':
                    report['summary']['issues'].append(
                        f"Error analyzing {tool}: {tool_results.get('message')}"
                    )
                if tool_results.get('issues'):
                    report['summary']['issues'].extend(
                        f"{tool}: {issue}" for issue in tool_results['issues']
                    )
                if tool_results.get('warnings'):
                    report['summary']['warnings'].extend(
                        f"{tool}: {warning}" for warning in tool_results['warnings']
                    )
            
            # Analyze workflow logs
            log_results = self.analyze_tool_output('logs')
            if log_results['errors']:
                report['summary']['issues'].extend(log_results['errors'])
            if log_results['warnings']:
                report['summary']['warnings'].extend(log_results['warnings'])
            
            # Set overall status
            if report['summary']['issues']:
                report['summary']['status'] = 'failed'
                report['summary']['recommendations'].append(
                    "Review and address major issues before proceeding"
                )
            elif report['summary']['warnings']:
                report['summary']['status'] = 'warning'
                report['summary']['recommendations'].append(
                    "Review warnings and consider their impact"
                )
            
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
