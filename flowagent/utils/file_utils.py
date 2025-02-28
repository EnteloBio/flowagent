"""Utility functions for file operations."""

import os
from pathlib import Path
from typing import List, Dict, Any, Set
import logging

logger = logging.getLogger(__name__)

def find_files(directory: str, pattern: str, file_type: str = 'file', max_depth: int = None, 
               extensions: List[str] = None, excludes: List[str] = None, full_path: bool = False) -> List[str]:
    """Find files matching pattern in directory.
    
    Args:
        directory: Directory to search in
        pattern: Pattern to match (glob pattern)
        file_type: Type of files to find ('file' or 'directory')
        max_depth: Maximum depth to search
        extensions: List of file extensions to include
        excludes: List of patterns to exclude
        full_path: Whether to return full paths
        
    Returns:
        List of matching file paths
    """
    results = []
    directory = Path(directory)
    
    def should_include(path: Path) -> bool:
        """Check if path should be included in results."""
        # Check if path matches pattern
        if not path.match(pattern):
            return False
            
        # Check file type
        if file_type == 'file' and not path.is_file():
            return False
        if file_type == 'directory' and not path.is_dir():
            return False
            
        # Check extensions
        if extensions and path.suffix[1:] not in extensions:
            return False
            
        # Check excludes
        if excludes:
            for exclude in excludes:
                if path.match(exclude):
                    return False
                    
        return True
    
    def walk(current_dir: Path, depth: int = 0):
        """Recursively walk directory."""
        if max_depth is not None and depth > max_depth:
            return
            
        try:
            for item in current_dir.iterdir():
                if should_include(item):
                    results.append(str(item) if full_path else item.name)
                if item.is_dir():
                    walk(item, depth + 1)
        except PermissionError:
            pass  # Skip directories we can't access
    
    walk(directory)
    return sorted(results)  # Sort for consistent ordering

def to_relative_path(abs_path: str, base_dir: str) -> str:
    """Convert absolute path to relative path based on base directory."""
    try:
        return os.path.relpath(abs_path, base_dir)
    except ValueError:
        # If paths are on different drives, return original path
        return abs_path

def find_fastq_files(directory: str) -> List[str]:
    """Find all FASTQ files in a directory.
    
    Supports all common FASTQ naming conventions:
    - Standard: .fastq, .fastq.gz, .fq, .fq.gz
    - Paired-end suffixes:
        - .fastq.1.gz, .fastq.2.gz
        - .fq.1.gz, .fq.2.gz
        - _1.fastq.gz, _2.fastq.gz
        - _R1.fastq.gz, _R2.fastq.gz
        - .R1.fastq.gz, .R2.fastq.gz
        - Same patterns without .gz compression
    
    Args:
        directory: Directory to search for FASTQ files
        
    Returns:
        List of relative paths to FASTQ files found
    """
    fastq_files = []
    # Define base extensions
    base_exts = ('.fastq', '.fq')
    # Define paired-end patterns
    pair_patterns = (
        '.1', '.2',  # .fastq.1.gz
        '_1', '_2',  # _1.fastq.gz
        '_R1', '_R2',  # _R1.fastq.gz
        '.R1', '.R2'  # .R1.fastq.gz
    )
    
    # Build full list of extensions with all combinations
    fastq_extensions = []
    for base in base_exts:
        # Add basic extensions
        fastq_extensions.append(base)
        fastq_extensions.append(f"{base}.gz")
        # Add paired-end patterns
        for pattern in pair_patterns:
            fastq_extensions.append(f"{pattern}{base}")
            fastq_extensions.append(f"{pattern}{base}.gz")
            fastq_extensions.append(f"{base}{pattern}")
            fastq_extensions.append(f"{base}{pattern}.gz")
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in fastq_extensions):
                abs_path = os.path.join(root, file)
                rel_path = to_relative_path(abs_path, directory)
                fastq_files.append(rel_path)
    return sorted(fastq_files)  # Sort for consistent ordering

def ensure_directory(path: str) -> None:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure exists
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    
def get_file_size(path: str) -> int:
    """Get size of a file in bytes.
    
    Args:
        path: Path to file
        
    Returns:
        File size in bytes
    """
    try:
        return os.path.getsize(path)
    except (OSError, FileNotFoundError):
        logger.error(f"File {path} does not exist")
        return None
