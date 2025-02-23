"""Tool tracking and execution for workflow management."""

from typing import Dict, List, Set, Any
from dataclasses import dataclass
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import json
from datetime import datetime
import subprocess
import os
import logging

from ..utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class ToolCall:
    tool_name: str
    inputs: Dict[str, Any]
    outputs: Set[Path]  # Files/directories generated
    dependencies: Set[str]  # Tool call IDs this depends on
    call_id: str
    status: str = "pending"
    start_time: datetime = None
    end_time: datetime = None

    def to_dict(self):
        return {
            'tool_name': self.tool_name,
            'inputs': self.inputs,
            'outputs': [str(p) for p in self.outputs],
            'dependencies': list(self.dependencies),
            'call_id': self.call_id,
            'status': self.status,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None
        }

class ToolTracker:
    def __init__(self):
        self.tool_calls: Dict[str, ToolCall] = {}
        self.current_context: List[str] = []  # Stack of tool call IDs
        self.logger = get_logger(__name__)
    
    async def execute_tool(self, step: Dict[str, Any], llm) -> Dict[str, Any]:
        """Execute a tool based on the step configuration."""
        try:
            # Get or generate command
            command = step.get("command")
            if not command:
                command = await llm.generate_command(step)
            
            # Create tool call record
            tool_call = ToolCall(
                tool_name=step["name"],
                inputs=step.get("parameters", {}),
                outputs=set(step.get("outputs", [])),
                dependencies=set(step.get("dependencies", [])),
                call_id=step["name"]
            )
            
            # Record start time and log
            tool_call.status = "running"
            tool_call.start_time = datetime.now()
            self.tool_calls[tool_call.call_id] = tool_call
            
            # Log command execution
            self.logger.info(f"Executing command for step {step['name']}:")
            self.logger.info(f"  Command: {command}")
            self.logger.info(f"  Working directory: {os.getcwd()}")
            
            # Execute command
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            # Real-time logging of stdout and stderr
            stdout_lines = []
            stderr_lines = []
            
            while True:
                stdout_line = process.stdout.readline()
                stderr_line = process.stderr.readline()
                
                if stdout_line:
                    self.logger.info(f"  [stdout] {stdout_line.strip()}")
                    stdout_lines.append(stdout_line)
                
                if stderr_line:
                    self.logger.warning(f"  [stderr] {stderr_line.strip()}")
                    stderr_lines.append(stderr_line)
                
                if not stdout_line and not stderr_line and process.poll() is not None:
                    break
            
            stdout = "".join(stdout_lines)
            stderr = "".join(stderr_lines)
            returncode = process.wait()
            
            # Record completion and log
            tool_call.end_time = datetime.now()
            duration = (tool_call.end_time - tool_call.start_time).total_seconds()
            
            if returncode == 0:
                tool_call.status = "completed"
                self.logger.info(f"Step {step['name']} completed successfully in {duration:.1f}s")
                result = {
                    "status": "success",
                    "stdout": stdout,
                    "stderr": stderr,
                    "returncode": returncode,
                    "duration": duration
                }
            else:
                tool_call.status = "failed"
                self.logger.error(f"Step {step['name']} failed after {duration:.1f}s")
                self.logger.error(f"Exit code: {returncode}")
                if stderr:
                    self.logger.error(f"Error output:\n{stderr}")
                result = {
                    "status": "failed",
                    "stdout": stdout,
                    "stderr": stderr,
                    "returncode": returncode,
                    "duration": duration
                }
                raise Exception(f"Command failed with exit code {returncode}")
            
            # Update step with results
            step.update({
                "status": tool_call.status,
                "result": result,
                "start_time": tool_call.start_time.isoformat(),
                "end_time": tool_call.end_time.isoformat(),
                "duration": duration
            })
            
            return step
            
        except Exception as e:
            self.logger.error(f"Error executing tool: {str(e)}")
            raise
