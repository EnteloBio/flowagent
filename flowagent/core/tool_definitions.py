"""Tool definitions for LLM function-calling.

Each tool is described in the OpenAI-compatible tool schema format.
Anthropic and Google providers normalise this automatically.
"""

from typing import Any, Dict, List


# ── Workflow-aware tools (plan / execute / export / analyze) ──

WORKFLOW_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "plan_workflow",
            "description": (
                "Generate a bioinformatics workflow plan from a natural-language "
                "description. Returns the plan JSON with steps, commands, and "
                "dependencies. Does NOT execute anything."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Natural-language description of the desired analysis",
                    },
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_workflow",
            "description": (
                "Execute a previously generated workflow plan. Runs each step "
                "sequentially, manages dependencies, and streams logs. "
                "Returns execution results with status per step."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The original prompt or plan description to execute",
                    },
                    "checkpoint_dir": {
                        "type": "string",
                        "description": "Directory for checkpoints (enables resume)",
                    },
                    "resume": {
                        "type": "boolean",
                        "description": "Resume from last checkpoint if True",
                    },
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "export_pipeline",
            "description": (
                "Export a workflow plan as a Nextflow or Snakemake pipeline file. "
                "Returns the path to the generated file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Natural-language description of the pipeline",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["nextflow", "snakemake"],
                        "description": "Pipeline format to generate",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory to write the pipeline file",
                    },
                },
                "required": ["prompt", "format"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_results",
            "description": (
                "Analyze workflow output in a given directory. Uses both "
                "rule-based and LLM analysis to produce a report."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Path to the workflow results directory",
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
            "name": "load_preset",
            "description": (
                "Load a pre-built workflow preset by ID. Available presets: "
                "rnaseq-kallisto, rnaseq-star, chipseq, atacseq. "
                "Returns the full plan JSON."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "preset_id": {
                        "type": "string",
                        "description": "Preset identifier e.g. 'rnaseq-kallisto'",
                    },
                },
                "required": ["preset_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_workflow_status",
            "description": (
                "Read a workflow checkpoint to see which steps have completed. "
                "Returns the checkpoint data including completed steps list."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "checkpoint_dir": {
                        "type": "string",
                        "description": "Directory containing checkpoint.json",
                    },
                },
                "required": ["checkpoint_dir"],
                "additionalProperties": False,
            },
        },
    },
]

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

# Combine: agent gets both low-level and workflow-aware tools
AGENT_TOOLS.extend(WORKFLOW_TOOLS)
