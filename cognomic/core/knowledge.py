import chromadb
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

def initialize_knowledge_base(persist_dir: str = "knowledge_base") -> chromadb.Client:
    """Initialize and populate the knowledge base"""
    # Create ChromaDB client
    client = chromadb.Client(chromadb.config.Settings(
        persist_directory=persist_dir,
        is_persistent=True
    ))
    
    # Create collections if they don't exist
    collections = {
        "workflow_patterns": client.get_or_create_collection("workflow_patterns"),
        "tool_documentation": client.get_or_create_collection("tool_documentation"),
        "error_solutions": client.get_or_create_collection("error_solutions")
    }
    
    # Load default knowledge if collections are empty
    if collections["workflow_patterns"].count() == 0:
        _load_default_patterns(collections["workflow_patterns"])
    
    if collections["tool_documentation"].count() == 0:
        _load_default_tool_docs(collections["tool_documentation"])
        
    if collections["error_solutions"].count() == 0:
        _load_default_error_solutions(collections["error_solutions"])
    
    return client

def _load_default_patterns(collection: chromadb.Collection):
    """Load default workflow patterns"""
    patterns = [
        {
            "name": "RNA-seq Basic",
            "description": "Basic RNA-seq analysis workflow",
            "steps": [
                {
                    "tool": "fastqc",
                    "action": "analyze",
                    "description": "Quality control of raw reads"
                },
                {
                    "tool": "kallisto",
                    "action": "index",
                    "description": "Create Kallisto index from reference"
                },
                {
                    "tool": "kallisto",
                    "action": "quant",
                    "description": "Quantify transcript abundance"
                },
                {
                    "tool": "multiqc",
                    "action": "report",
                    "description": "Generate QC report"
                }
            ]
        }
    ]
    
    collection.add(
        documents=[json.dumps(p) for p in patterns],
        metadatas=[{"type": "workflow_pattern"} for _ in patterns],
        ids=[f"pattern_{i}" for i in range(len(patterns))]
    )

def _load_default_tool_docs(collection: chromadb.Collection):
    """Load default tool documentation"""
    tools = [
        {
            "name": "fastqc",
            "description": "Quality control tool for high throughput sequence data",
            "actions": {
                "analyze": {
                    "description": "Analyze FASTQ files for quality metrics",
                    "parameters": {
                        "input_files": "List of FASTQ files to analyze",
                        "output_dir": "Directory for output reports",
                        "threads": "Number of analysis threads (default: 4)"
                    }
                }
            }
        },
        {
            "name": "kallisto",
            "description": "Fast and accurate RNA-seq quantification",
            "actions": {
                "index": {
                    "description": "Create Kallisto index from reference transcriptome",
                    "parameters": {
                        "reference": "Path to reference transcriptome FASTA",
                        "output_index": "Path for output index file"
                    }
                },
                "quant": {
                    "description": "Quantify transcript abundance",
                    "parameters": {
                        "index": "Path to Kallisto index",
                        "input_files": "FASTQ files to quantify",
                        "output_dir": "Directory for output files",
                        "threads": "Number of threads (default: 4)"
                    }
                }
            }
        },
        {
            "name": "multiqc",
            "description": "Aggregate results from bioinformatics analyses",
            "actions": {
                "report": {
                    "description": "Create MultiQC report",
                    "parameters": {
                        "input_dir": "Directory containing analysis results",
                        "output_dir": "Directory for MultiQC report"
                    }
                }
            }
        }
    ]
    
    collection.add(
        documents=[json.dumps(t) for t in tools],
        metadatas=[{"type": "tool_documentation"} for _ in tools],
        ids=[f"tool_{t['name']}" for t in tools]
    )

def _load_default_error_solutions(collection: chromadb.Collection):
    """Load default error solutions"""
    solutions = [
        {
            "error_pattern": "index.*not found",
            "diagnosis": "Kallisto index file not found or not created",
            "action": "fix",
            "solution": "Ensure reference transcriptome exists and create index"
        },
        {
            "error_pattern": "permission denied",
            "diagnosis": "Insufficient permissions to access files",
            "action": "fix",
            "solution": "Check file permissions and ownership"
        },
        {
            "error_pattern": "not enough memory",
            "diagnosis": "System memory exhausted",
            "action": "retry",
            "solution": "Reduce number of threads or increase memory limit"
        }
    ]
    
    collection.add(
        documents=[json.dumps(s) for s in solutions],
        metadatas=[{"type": "error_solution"} for _ in solutions],
        ids=[f"error_{i}" for i in range(len(solutions))]
    )
