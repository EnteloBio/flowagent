"""Utility functions for running shell commands."""

import asyncio
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional, Union, List

logger = logging.getLogger(__name__)

async def run_command(cmd: str, cwd: Optional[Union[str, Path]] = None, 
                     env: Optional[dict] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command asynchronously.
    
    Args:
        cmd: Command to run
        cwd: Working directory for command
        env: Environment variables
        check: Whether to raise exception on non-zero return code
        
    Returns:
        CompletedProcess instance
    """
    # Ensure we have a clean environment
    if env is None:
        env = os.environ.copy()
    
    # Convert working directory to string if it's a Path
    if isinstance(cwd, Path):
        cwd = str(cwd)
    
    logger.info(f"Running command: {cmd}")
    if cwd:
        logger.info(f"Working directory: {cwd}")
    
    try:
        # Run command and capture output
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env
        )
        
        # Wait for command to complete and get output
        stdout, stderr = await process.communicate()
        
        # Convert bytes to string
        stdout = stdout.decode() if stdout else ""
        stderr = stderr.decode() if stderr else ""
        
        # Log output
        if stdout:
            logger.debug(f"Command stdout:\n{stdout}")
        if stderr:
            logger.warning(f"Command stderr:\n{stderr}")
            
        # Check return code
        if check and process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, cmd, stdout, stderr
            )
            
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr
        )
        
    except Exception as e:
        logger.error(f"Command failed: {str(e)}")
        raise
