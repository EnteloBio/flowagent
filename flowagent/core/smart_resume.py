"""
Smart resume functionality for FlowAgent workflows.

This module adds the ability to detect which steps of a workflow have been completed
and which need to be rerun based on the presence of output files.
"""

import os
import logging
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Any, Callable, Pattern, Tuple, Optional, Union

logger = logging.getLogger(__name__)

# Type for tool validator functions
ToolValidatorType = Callable[[str, Dict[str, Any]], bool]

def check_directory_exists(dir_path: str) -> bool:
    """Check if directory exists."""
    return os.path.isdir(dir_path)

def check_file_exists(file_path: str) -> bool:
    """Check if file exists."""
    return os.path.isfile(file_path)

def check_files_match_pattern(directory: str, pattern: str) -> bool:
    """Check if any files in the directory match the given pattern."""
    if not check_directory_exists(directory):
        return False
    
    regex = re.compile(pattern)
    return any(regex.search(f) for f in os.listdir(directory))

def check_file_not_empty(file_path: str) -> bool:
    """Check if file exists and is not empty."""
    return check_file_exists(file_path) and os.path.getsize(file_path) > 0

def extract_output_paths_from_command(command: str) -> List[str]:
    """
    Extract output paths from a command string.
    Handles common patterns like -o, --output, etc.
    
    Returns:
        List of output paths extracted from the command
    """
    output_paths = []
    
    # Common output flags used by bioinformatics and data analysis tools
    output_flags = [
        # General outputs
        (r'-o\s+([^\s]+)', 1),
        (r'--output\s+([^\s]+)', 1),
        (r'--outdir\s+([^\s]+)', 1),
        (r'-outdir\s+([^\s]+)', 1),
        (r'--out\s+([^\s]+)', 1),
        (r'-out\s+([^\s]+)', 1),
        (r'>\s+([^\s]+)', 1),
        (r'--results\s+([^\s]+)', 1),
        (r'-results\s+([^\s]+)', 1),
        
        # Tool-specific output patterns
        (r'-i\s+([^\s]+)', 1),  # Index outputs
        (r'--index\s+([^\s]+)', 1),
        (r'--db\s+([^\s]+)', 1),  # Database outputs
        (r'-db\s+([^\s]+)', 1),
        (r'--prefix\s+([^\s]+)', 1),  # Prefix-based outputs
        (r'-prefix\s+([^\s]+)', 1),
        (r'--report\s+([^\s]+)', 1),  # Report outputs
        (r'-report\s+([^\s]+)', 1),
        
        # Directory creation
        (r'mkdir\s+-p\s+(.+)', 1),
    ]
    
    for pattern, group in output_flags:
        matches = re.finditer(pattern, command)
        for match in matches:
            output_path = match.group(group).strip()
            
            # Remove any surrounding quotes
            if output_path.startswith('"') and output_path.endswith('"'):
                output_path = output_path[1:-1]
            elif output_path.startswith("'") and output_path.endswith("'"):
                output_path = output_path[1:-1]
                
            # Add to output paths
            output_paths.append(output_path)
    
    # For mkdir commands with multiple directories, split them
    if 'mkdir -p ' in command:
        # Extract directories after mkdir -p
        mkdir_match = re.search(r'mkdir\s+-p\s+(.+)', command)
        if mkdir_match:
            dir_part = mkdir_match.group(1)
            # Split by spaces, but not spaces within quoted strings
            if '"' in dir_part or "'" in dir_part:
                # Handle quoted paths
                import shlex
                dirs = shlex.split(dir_part)
            else:
                # Simple space split for unquoted paths
                dirs = dir_part.split()
            output_paths.extend(dirs)
    
    # Look for output patterns in command
    # Common suffixes for output files
    output_suffixes = ['.txt', '.csv', '.tsv', '.json', '.xml', '.html', 
                      '.pdf', '.png', '.jpg', '.h5', '.hdf5', '.bam', 
                      '.sam', '.vcf', '.gff', '.gtf', '.bed', '.idx']
    
    # Check for words that look like file paths with output suffixes
    words = re.findall(r'\S+', command)
    for word in words:
        # Skip if it looks like a flag
        if word.startswith('-'):
            continue
        
        # Check if it has an output suffix
        if any(word.endswith(suffix) for suffix in output_suffixes):
            # Ensure it's not a quoted string already captured
            if not (word.startswith('"') or word.startswith("'")):
                output_paths.append(word)
    
    return output_paths

# Dictionary mapping tool names (from command) to their specific validators
_TOOL_VALIDATORS: Dict[str, ToolValidatorType] = {}

def register_tool_validator(tool_pattern: str, validator_func: ToolValidatorType) -> None:
    """
    Register a validator function for a specific tool pattern.
    
    Args:
        tool_pattern: Regex pattern to match the tool in the command
        validator_func: Function that validates if the tool's output exists
    """
    _TOOL_VALIDATORS[tool_pattern] = validator_func

# Define tool-specific validators

def kallisto_index_validator(command: str, step: Dict[str, Any]) -> bool:
    """Validate kallisto index outputs."""
    # Extract the index path from the command
    index_path_match = re.search(r'-i\s+([^\s]+)', command)
    if index_path_match:
        index_path = index_path_match.group(1)
        
        # Check that the index file exists and has a minimum size
        # Kallisto index files are typically several MB in size
        MIN_INDEX_SIZE = 1000000  # 1MB minimum size for index file
        
        if check_file_exists(index_path) and os.path.getsize(index_path) > MIN_INDEX_SIZE:
            return True
    
    return False

def kallisto_quant_validator(command: str, step: Dict[str, Any]) -> bool:
    """Validate kallisto quantification outputs."""
    output_dir_match = re.search(r'-o\s+([^\s]+)', command)
    if output_dir_match:
        output_dir = output_dir_match.group(1)
        
        # Check output directory exists
        if not check_directory_exists(output_dir):
            return False
            
        # Check for required output files in kallisto quant output directory
        required_files = ['abundance.h5', 'abundance.tsv', 'run_info.json']
        for file in required_files:
            if not check_file_exists(os.path.join(output_dir, file)):
                return False
                
        # Verify run_info.json contains completion data
        try:
            with open(os.path.join(output_dir, 'run_info.json'), 'r') as f:
                run_info = json.load(f)
                # Check for n_processed and n_pseudoaligned fields
                if 'n_processed' not in run_info or 'n_pseudoaligned' not in run_info:
                    return False
                # Ensure some reads were processed
                if run_info['n_processed'] <= 0:
                    return False
        except (json.JSONDecodeError, FileNotFoundError, IOError):
            return False
            
        return True
    
    return False

def fastqc_validator(command: str, step: Dict[str, Any]) -> bool:
    """Validate FastQC outputs."""
    output_dir_match = re.search(r'-o\s+([^\s]+)', command)
    if output_dir_match:
        output_dir = output_dir_match.group(1)
        
        # Check for directory existence
        if not check_directory_exists(output_dir):
            return False
        
        # Look for HTML files with _fastqc.html suffix
        html_files = [f for f in os.listdir(output_dir) if f.endswith('_fastqc.html')]
        if not html_files:
            return False
            
        # Look for zip files with _fastqc.zip suffix
        zip_files = [f for f in os.listdir(output_dir) if f.endswith('_fastqc.zip')]
        if not zip_files:
            return False
            
        # Ensure we have at least one HTML and one ZIP file
        return len(html_files) > 0 and len(zip_files) > 0
    
    return False

def multiqc_validator(command: str, step: Dict[str, Any]) -> bool:
    """Validate MultiQC outputs."""
    # Extract output directory path
    output_dir = None
    outdir_match = re.search(r'-o\s+([^\s]+)', command)
    if outdir_match:
        output_dir = outdir_match.group(1)
    else:
        # If no output directory is specified, MultiQC creates a 'multiqc_data' directory
        output_dir = 'multiqc_data'
    
    # Check for directory existence
    if not check_directory_exists(output_dir):
        return False
    
    # Check for multiqc_report.html
    html_report = os.path.join(output_dir, "..", "multiqc_report.html")
    if not check_file_exists(html_report):
        # Try alternative location
        html_report = "multiqc_report.html"
        if not check_file_exists(html_report):
            return False
    
    # Check for required MultiQC data files
    required_files = ['multiqc_general_stats.txt', 'multiqc_sources.txt']
    for file in required_files:
        if not check_file_exists(os.path.join(output_dir, file)):
            return False
    
    return True

def bwa_index_validator(command: str, step: Dict[str, Any]) -> bool:
    """Validate BWA index outputs."""
    ref_path = command.split()[-1]
    index_files = [f"{ref_path}.{ext}" for ext in ['amb', 'ann', 'bwt', 'pac', 'sa']]
    return all(check_file_exists(f) for f in index_files)

def samtools_sort_validator(command: str, step: Dict[str, Any]) -> bool:
    """Validate samtools sort outputs."""
    output_match = re.search(r'-o\s+([^\s]+)', command)
    if output_match:
        output_file = output_match.group(1)
        return check_file_not_empty(output_file)
    return False

def bcftools_call_validator(command: str, step: Dict[str, Any]) -> bool:
    """Validate bcftools call outputs."""
    output_match = re.search(r'-o\s+([^\s]+)', command)
    if output_match:
        output_file = output_match.group(1)
        return check_file_not_empty(output_file)
    return False

def bowtie2_build_validator(command: str, step: Dict[str, Any]) -> bool:
    """Validate bowtie2-build outputs."""
    parts = command.split()
    if len(parts) >= 3:
        ref_idx = parts[-1]
        for ext in ['1.bt2', '2.bt2', '3.bt2', '4.bt2', 'rev.1.bt2', 'rev.2.bt2']:
            if not check_file_exists(f"{ref_idx}.{ext}"):
                return False
        return True
    return False

def generic_validator(command: str, step: Dict[str, Any]) -> bool:
    """Generic validator that checks for output files or directories."""
    # Don't consider incomplete command (like commented ones) as completed
    if command.strip().startswith("#") or len(command.strip()) < 3:
        return False
        
    # Extract output files or directories from the command
    # Look for common output redirection patterns
    output_files = []
    
    # Check for output redirection to files
    output_redirects = re.findall(r'>\s*(\S+)', command)
    output_files.extend(output_redirects)
    
    # Check for -o/--output arguments (common in bioinformatics tools)
    output_args = re.findall(r'(?:-o|--output)\s+(\S+)', command)
    output_files.extend(output_args)
    
    # Check for -outdir/--outdir arguments
    outdir_args = re.findall(r'(?:--outdir|\-outdir|\-output_dir|--output_dir)\s+(\S+)', command)
    for outdir in outdir_args:
        if check_directory_exists(outdir):
            # Make sure it's not just an empty directory
            if os.path.isdir(outdir) and len(os.listdir(outdir)) > 0:
                return True
    
    # Check if all identified output files exist
    for output_file in output_files:
        if not check_file_exists(output_file):
            return False
            
    # If we identified output files and they all exist, consider the step completed
    return len(output_files) > 0

# Register tool validators
register_tool_validator(r'kallisto\s+index', kallisto_index_validator)
register_tool_validator(r'kallisto\s+quant', kallisto_quant_validator)
register_tool_validator(r'fastqc\b', fastqc_validator)
register_tool_validator(r'multiqc\b', multiqc_validator)
register_tool_validator(r'bwa\s+index', bwa_index_validator)
register_tool_validator(r'samtools\s+sort', samtools_sort_validator)
register_tool_validator(r'bcftools\s+call', bcftools_call_validator)
register_tool_validator(r'bowtie2-build', bowtie2_build_validator)

def detect_completed_steps(workflow_steps: List[Dict[str, Any]]) -> Set[str]:
    """
    Detects which steps have already been completed based on the existence of their outputs.
    
    Args:
        workflow_steps: List of workflow steps.
        
    Returns:
        Set of step names that have been completed.
    """
    logger.info("Detecting completed steps...")
    
    completed_steps = set()
    
    for step in workflow_steps:
        step_name = step.get("name", "")
        logger.info(f"Checking step: {step_name}")
        command = step.get("command", "")
        
        # Skip steps with no command
        if not command:
            continue
            
        # Check if this is literally just a directory creation step and nothing else
        if command.strip().startswith("mkdir") and "&&" not in command and ";" not in command and "|" not in command:
            logger.info(f"Step {step_name} is only a directory creation step, marking as completed")
            completed_steps.add(step_name)
            continue
        
        # Check for tool-specific validators
        for pattern, validator_func in _TOOL_VALIDATORS.items():
            if re.search(pattern, command):
                logger.info(f"Using {validator_func.__name__} for step {step_name}")
                if validator_func(command, step):
                    logger.info(f"Tool-specific validator confirmed step {step_name} is completed")
                    completed_steps.add(step_name)
                    break
                else:
                    logger.info(f"Tool-specific validator determined step {step_name} is NOT completed")
                    break
        else:
            # Use generic validator if no tool-specific validator matched
            logger.info(f"Using generic validator for step {step_name}")
            if generic_validator(command, step):
                logger.info(f"Generic validator confirmed step {step_name} is completed")
                completed_steps.add(step_name)
            else:
                logger.info(f"Generic validator determined step {step_name} is NOT completed")
    
    logger.info(f"Detected {len(completed_steps)} completed steps: {', '.join(completed_steps)}")
    
    return completed_steps

def filter_workflow_steps(workflow_steps: List[Dict[str, Any]], 
                         completed_steps: Set[str]) -> List[Dict[str, Any]]:
    """
    Filter workflow steps to only include those that need to be run.
    
    Args:
        workflow_steps: List of workflow steps
        completed_steps: Set of names of completed steps
        
    Returns:
        Filtered list of workflow steps
    """
    # Steps we always run regardless of completion status
    always_run_steps = {"create_directories"}
    
    # Filter steps
    filtered_steps = []
    for step in workflow_steps:
        step_name = step.get("name", "")
        
        # Always include steps that should always run
        if step_name in always_run_steps:
            filtered_steps.append(step)
            continue
            
        # Skip completed steps
        if step_name in completed_steps:
            logger.info(f"Skipping completed step: {step_name}")
            continue
            
        # Include steps that need to be run
        filtered_steps.append(step)
    
    return filtered_steps

def register_custom_validator(tool_name: str, validator_func: ToolValidatorType) -> None:
    """
    Public API to register a custom validator for a specific tool.
    This allows users to extend the smart resume functionality with their own validators.
    
    Args:
        tool_name: Name of the tool (regex pattern)
        validator_func: Function that validates if tool outputs exist
    """
    register_tool_validator(tool_name, validator_func)
