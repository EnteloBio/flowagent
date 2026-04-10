"""Tool definitions for LLM function-calling.

Each tool is described in the OpenAI-compatible tool schema format.
Anthropic and Google providers normalise this automatically.
"""

from typing import Any, Dict, List

AGENT_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List files in a directory, optionally filtered by a glob pattern. "
                "Use this to discover input data files such as FASTQ, BAM, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory path to list",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern e.g. '*.fastq.gz'",
                    },
                },
                "required": ["directory"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_tool",
            "description": "Check whether a bioinformatics tool is installed and return its version.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Tool name, e.g. 'kallisto', 'fastqc', 'samtools'",
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "install_dependency",
            "description": "Install a bioinformatics tool via conda or pip.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Package name to install",
                    },
                    "manager": {
                        "type": "string",
                        "enum": ["conda", "mamba", "pip"],
                        "description": "Package manager to use",
                    },
                    "channel": {
                        "type": "string",
                        "description": "Conda channel (e.g. 'bioconda')",
                    },
                },
                "required": ["name", "manager"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": (
                "Execute a shell command and return stdout, stderr, and exit code. "
                "Use for running bioinformatics tools."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory",
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file (or first N lines of a large file).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to file",
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Maximum number of lines to read (default: 200)",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file (creates parent directories as needed).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Output file path",
                    },
                    "content": {
                        "type": "string",
                        "description": "File content to write",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
]
