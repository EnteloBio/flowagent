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
    """Find all FASTQ files in a directory."""
    fastq_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.fastq', '.fastq.gz', '.fq', '.fq.gz')):
                abs_path = os.path.join(root, file)
                rel_path = to_relative_path(abs_path, directory)
                fastq_files.append(rel_path)
    return fastq_files

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
