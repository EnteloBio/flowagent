"""Utility functions for file operations."""

import os
from pathlib import Path
from typing import List, Dict, Any, Set

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
