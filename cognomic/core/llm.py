"""LLM interface for workflow generation and command creation."""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import openai
from openai import AsyncOpenAI
import glob
import networkx as nx
import os
from pathlib import Path

from ..utils import file_utils
from ..utils.logging import get_logger

logger = get_logger(__name__)

class LLMInterface:
    """Interface for LLM-based workflow generation."""
    
    WORKFLOW_TYPES = {
        "rna_seq_kallisto": {
            "keywords": ["rna-seq", "rnaseq", "kallisto", "pseudoalignment", "transcript", "expression"],
            "tools": ["fastqc", "kallisto", "multiqc"],
            "dir_structure": [
                "results/rna_seq_kallisto/fastqc",
                "results/rna_seq_kallisto/kallisto_index",
                "results/rna_seq_kallisto/kallisto_quant",
                "results/rna_seq_kallisto/qc"
            ],
            "rules": [
                "FastQC: fastqc file.fastq.gz -o results/rna_seq_kallisto/fastqc",
                "Kallisto index: kallisto index -i results/rna_seq_kallisto/kallisto_index/transcripts.idx reference.fa",
                "Kallisto quant: kallisto quant -o results/rna_seq_kallisto/kallisto_quant/sample_name --single -l 200 -s 20 -i results/rna_seq_kallisto/kallisto_index/transcripts.idx file.fastq.gz",
                "MultiQC: multiqc results/rna_seq_kallisto/fastqc results/rna_seq_kallisto/kallisto_quant -o results/rna_seq_kallisto/qc"
            ]
        },
        "rna_seq_hisat": {
            "keywords": ["rna-seq", "rnaseq", "hisat", "hisat2", "mapping", "spliced alignment"],
            "tools": ["fastqc", "hisat2", "samtools", "featureCounts", "multiqc"],
            "dir_structure": [
                "results/rna_seq_hisat/fastqc",
                "results/rna_seq_hisat/hisat2_index",
                "results/rna_seq_hisat/hisat2_align",
                "results/rna_seq_hisat/counts",
                "results/rna_seq_hisat/qc"
            ],
            "rules": [
                "FastQC: fastqc file.fastq.gz -o results/rna_seq_hisat/fastqc",
                "HISAT2 index: hisat2-build reference.fa results/rna_seq_hisat/hisat2_index/genome",
                "HISAT2 align: hisat2 -x results/rna_seq_hisat/hisat2_index/genome -U file.fastq.gz -S results/rna_seq_hisat/hisat2_align/sample.sam",
                "SAM to BAM: samtools view -bS results/rna_seq_hisat/hisat2_align/sample.sam > results/rna_seq_hisat/hisat2_align/sample.bam",
                "Sort BAM: samtools sort results/rna_seq_hisat/hisat2_align/sample.bam -o results/rna_seq_hisat/hisat2_align/sample.sorted.bam",
                "Index BAM: samtools index results/rna_seq_hisat/hisat2_align/sample.sorted.bam",
                "FeatureCounts: featureCounts -a annotation.gtf -o results/rna_seq_hisat/counts/counts.txt results/rna_seq_hisat/hisat2_align/sample.sorted.bam",
                "MultiQC: multiqc results/rna_seq_hisat/fastqc results/rna_seq_hisat/hisat2_align results/rna_seq_hisat/counts -o results/rna_seq_hisat/qc"
            ]
        },
        "chip_seq": {
            "keywords": ["chip-seq", "chipseq", "chip", "peaks", "binding sites"],
            "tools": ["fastqc", "bowtie2", "samtools", "macs2", "multiqc"],
            "dir_structure": [
                "results/chip_seq/fastqc",
                "results/chip_seq/bowtie2_index",
                "results/chip_seq/bowtie2_align",
                "results/chip_seq/peaks",
                "results/chip_seq/qc"
            ],
            "rules": [
                "FastQC: fastqc file.fastq.gz -o results/chip_seq/fastqc",
                "Bowtie2 index: bowtie2-build reference.fa results/chip_seq/bowtie2_index/genome",
                "Bowtie2 align: bowtie2 -x results/chip_seq/bowtie2_index/genome -U file.fastq.gz -S results/chip_seq/bowtie2_align/sample.sam",
                "SAM to BAM: samtools view -bS results/chip_seq/bowtie2_align/sample.sam > results/chip_seq/bowtie2_align/sample.bam",
                "Sort BAM: samtools sort results/chip_seq/bowtie2_align/sample.bam -o results/chip_seq/bowtie2_align/sample.sorted.bam",
                "Index BAM: samtools index results/chip_seq/bowtie2_align/sample.sorted.bam",
                "MACS2 peaks: macs2 callpeak -t results/chip_seq/bowtie2_align/sample.sorted.bam -c results/chip_seq/bowtie2_align/control.sorted.bam -f BAM -g hs -n sample -o results/chip_seq/peaks",
                "MultiQC: multiqc results/chip_seq/fastqc results/chip_seq/bowtie2_align -o results/chip_seq/qc"
            ]
        },
        "single_cell_10x": {
            "keywords": ["single-cell", "single cell", "scRNA-seq", "10x", "cellranger", "10x genomics"],
            "tools": ["cellranger", "fastqc", "multiqc"],
            "dir_structure": [
                "results/single_cell_10x/fastqc",
                "results/single_cell_10x/cellranger",
                "results/single_cell_10x/qc"
            ],
            "rules": [
                "FastQC: fastqc file.fastq.gz -o results/single_cell_10x/fastqc",
                "CellRanger count: cellranger count --id=sample_name --fastqs=fastq_path --transcriptome=transcriptome_path --localcores=8 --localmem=64",
                "MultiQC: multiqc results/single_cell_10x/fastqc results/single_cell_10x/cellranger -o results/single_cell_10x/qc"
            ]
        },
        "single_cell_kb": {
            "keywords": ["single-cell", "single cell", "scRNA-seq", "kallisto", "bustools", "kb", "kb-python"],
            "tools": ["kb-python", "fastqc", "multiqc"],
            "dir_structure": [
                "results/single_cell_kb/fastqc",
                "results/single_cell_kb/kb_count",
                "results/single_cell_kb/qc"
            ],
            "rules": [
                "FastQC: fastqc file.fastq.gz -o results/single_cell_kb/fastqc",
                "KB ref: kb ref -i index.idx -g t2g.txt -f1 cdna.fa --workflow standard",
                "KB count: kb count -i index.idx -g t2g.txt -x 10xv3 -o results/single_cell_kb/kb_count/sample_name --workflow standard -t 8 read_1.fastq.gz read_2.fastq.gz",
                "MultiQC: multiqc results/single_cell_kb/fastqc results/single_cell_kb/kb_count -o results/single_cell_kb/qc"
            ]
        },
        "single_cell_salmon": {
            "keywords": ["single-cell", "single cell", "scRNA-seq", "salmon", "alevin", "alevin-fry"],
            "tools": ["salmon", "alevin-fry", "fastqc", "multiqc"],
            "dir_structure": [
                "results/single_cell_salmon/fastqc",
                "results/single_cell_salmon/salmon_index",
                "results/single_cell_salmon/alevin_map",
                "results/single_cell_salmon/alevin_quant",
                "results/single_cell_salmon/qc"
            ],
            "rules": [
                "FastQC: fastqc file.fastq.gz -o results/single_cell_salmon/fastqc",
                "Salmon index: salmon index -t transcripts.fa -i results/single_cell_salmon/salmon_index/transcriptome -p 8",
                "Salmon alevin: salmon alevin -l ISR -i results/single_cell_salmon/salmon_index/transcriptome --chromium -1 read_1.fastq.gz -2 read_2.fastq.gz -p 8 -o results/single_cell_salmon/alevin_map/sample_name --tgMap transcript_to_gene.txt",
                "Alevin-fry generate permit: alevin-fry generate-permit-list -d fw -k -i results/single_cell_salmon/alevin_map/sample_name",
                "Alevin-fry collate: alevin-fry collate -t 8 -i results/single_cell_salmon/alevin_map/sample_name",
                "Alevin-fry quant: alevin-fry quant -t 8 -i results/single_cell_salmon/alevin_map/sample_name -o results/single_cell_salmon/alevin_quant/sample_name --resolution cr-like --use-mtx",
                "MultiQC: multiqc results/single_cell_salmon/fastqc results/single_cell_salmon/alevin_map results/single_cell_salmon/alevin_quant -o results/single_cell_salmon/qc"
            ]
        }
    }
    
    WORKFLOW_CONFIG = {
        "execution": {
            "environments": {
                "local": {
                    "executor": "LocalExecutor",
                    "description": "Run tasks on local machine"
                },
                "slurm": {
                    "executor": "CGATExecutor",
                    "description": "Run tasks on SLURM HPC cluster"
                },
                "kubernetes": {
                    "executor": "KubernetesExecutor",
                    "description": "Run tasks on Kubernetes cluster",
                    "default_namespace": "cognomic",
                    "service_account": "workflow-runner"
                }
            }
        },
        "containers": {
            # Quality Control
            "fastqc": {
                "image": "quay.io/biocontainers/fastqc:0.12.1--hdfd78af_0",
                "pull_policy": "IfNotPresent"
            },
            "multiqc": {
                "image": "quay.io/biocontainers/multiqc:1.19--pyhdfd78af_0",
                "pull_policy": "IfNotPresent"
            },
            
            # RNA-seq Tools
            "kallisto": {
                "image": "quay.io/biocontainers/kallisto:0.50.1--h05f6578_0",
                "pull_policy": "IfNotPresent"
            },
            "hisat2": {
                "image": "quay.io/biocontainers/hisat2:2.2.1--h87f3376_4",
                "pull_policy": "IfNotPresent"
            },
            "featureCounts": {
                "image": "quay.io/biocontainers/subread:2.0.6--he4a0461_0",
                "pull_policy": "IfNotPresent"
            },
            
            # ChIP-seq Tools
            "bowtie2": {
                "image": "quay.io/biocontainers/bowtie2:2.5.3--py310h8d7afc0_0",
                "pull_policy": "IfNotPresent"
            },
            "macs2": {
                "image": "quay.io/biocontainers/macs2:2.2.9.1--py39hf95cd2a_0",
                "pull_policy": "IfNotPresent"
            },
            
            # BAM Processing
            "samtools": {
                "image": "quay.io/biocontainers/samtools:1.19.2--h50ea8bc_0",
                "pull_policy": "IfNotPresent"
            },
            
            # Single-cell Tools
            "cellranger": {
                "image": "nfcore/cellranger:7.1.0",
                "pull_policy": "IfNotPresent"
            },
            "alevin-fry": {
                "image": "quay.io/biocontainers/alevin-fry:0.8.2--h4ac6f70_0",
                "pull_policy": "IfNotPresent"
            },
            "kb-python": {
                "image": "quay.io/biocontainers/kb-python:0.27.3--pyhdfd78af_0",
                "pull_policy": "IfNotPresent"
            }
        },
        "resources": {
            "minimal": {
                "cpus": 1,
                "memory_mb": 2000,
                "time_min": 30,
                "description": "For lightweight tasks like FastQC, MultiQC, and basic file operations"
            },
            "default": {
                "cpus": 1,
                "memory_mb": 4000,
                "time_min": 60,
                "description": "Standard profile for most preprocessing tasks"
            },
            "high_memory": {
                "cpus": 1,
                "memory_mb": 32000,
                "time_min": 120,
                "description": "For memory-intensive tasks like genome indexing"
            },
            "multi_thread": {
                "cpus": 8,
                "memory_mb": 16000,
                "time_min": 120,
                "description": "For CPU-intensive tasks that can be parallelized"
            },
            "high_memory_parallel": {
                "cpus": 8,
                "memory_mb": 64000,
                "time_min": 240,
                "description": "For memory and CPU-intensive tasks like large BAM processing"
            }
        },
        "task_resources": {
            # Quality Control
            "fastqc": "minimal",
            "multiqc": "minimal",
            
            # RNA-seq Tasks
            "kallisto_index": "high_memory",  # Genome indexing needs more memory
            "kallisto_quant": "multi_thread", # Quantification can be parallelized
            "hisat2_index": "high_memory",    # Genome indexing
            "hisat2_align": "multi_thread",   # Alignment benefits from parallelization
            "featureCounts": "high_memory",   # Counting needs good memory
            
            # ChIP-seq Tasks
            "bowtie2_index": "high_memory",
            "bowtie2_align": "multi_thread",
            "macs2_callpeak": "high_memory",  # Peak calling can be memory intensive
            
            # BAM Processing
            "samtools_sort": "high_memory_parallel",  # Sorting large BAMs needs memory and CPU
            "samtools_index": "default",
            "samtools_view": "multi_thread",
            
            # Single-cell Tasks
            "cellranger_count": "high_memory_parallel",  # Cell Ranger needs lots of resources
            "alevin_fry": "high_memory_parallel",
            "kb_count": "multi_thread"
        },
        "scaling_factors": {
            # File size based scaling
            "size_tiers": [
                {"threshold_gb": 5, "memory_multiplier": 1.0, "time_multiplier": 1.0},
                {"threshold_gb": 10, "memory_multiplier": 1.2, "time_multiplier": 1.3},
                {"threshold_gb": 20, "memory_multiplier": 1.5, "time_multiplier": 1.5},
                {"threshold_gb": 50, "memory_multiplier": 2.0, "time_multiplier": 2.0},
                {"threshold_gb": 100, "memory_multiplier": 3.0, "time_multiplier": 2.5}
            ],
            # Tool type scaling characteristics
            "tool_characteristics": {
                "memory_intensive": {
                    "patterns": ["index", "build", "merge", "sort", "assembly", "peak"],
                    "memory_weight": 1.5,
                    "time_weight": 1.2
                },
                "cpu_intensive": {
                    "patterns": ["align", "map", "quant", "call", "analyze"],
                    "memory_weight": 1.2,
                    "time_weight": 1.5
                },
                "io_intensive": {
                    "patterns": ["compress", "decompress", "convert", "dump"],
                    "memory_weight": 1.0,
                    "time_weight": 1.3
                }
            }
        }
    }
    
    def __init__(self):
        """Initialize LLM interface."""
        self.logger = get_logger(__name__)
        self.client = AsyncOpenAI()
    
    def _detect_workflow_type(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Detect workflow type from prompt.
        
        Returns:
            Tuple containing:
            - workflow_type: String identifier for the workflow
            - config: Dictionary containing workflow configuration
              For custom workflows, this includes base templates that can be modified
        """
        prompt_lower = prompt.lower()
        
        # Score each workflow type based on keyword matches
        scores = {}
        for wf_type, config in self.WORKFLOW_TYPES.items():
            score = sum(1 for kw in config["keywords"] if kw in prompt_lower)
            scores[wf_type] = score
        
        # Get workflow type with highest score
        best_match = max(scores.items(), key=lambda x: x[1])
        
        # If no strong match (score of 0 or 1), return custom workflow template
        if best_match[1] <= 1:
            return "custom", {
                "keywords": [],
                "tools": [],
                "dir_structure": [
                    "results/workflow/raw_data",
                    "results/workflow/processed_data",
                    "results/workflow/qc",
                    "results/workflow/analysis",
                    "results/workflow/output"
                ],
                "rules": [
                    "# This is a custom workflow. Consider the following guidelines:",
                    "1. Choose appropriate tools based on the specific analysis requirements",
                    "2. Include quality control steps suitable for the data type",
                    "3. Process data in a logical order with clear dependencies",
                    "4. Store intermediate files in organized directories",
                    "5. Generate comprehensive QC reports",
                    "6. Document all parameters and decisions"
                ]
            }
            
        return best_match[0], self.WORKFLOW_TYPES[best_match[0]]

    async def _call_openai(self, messages: List[Dict[str, Any]], response_format: Optional[Dict[str, str]] = None, **kwargs) -> str:
        """Call OpenAI API with retry logic."""
        try:
            params = {
                "model": "gpt-4-turbo-preview",
                "messages": messages,
                "temperature": 0,
                **kwargs
            }
            if response_format:
                params["response_format"] = response_format
                
            response = await self.client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {str(e)}")
            raise

    def _clean_llm_response(self, response: str) -> str:
        """Clean LLM response by removing markdown formatting."""
        # Remove markdown code block if present
        if response.startswith('```'):
            # Find the first and last ``` and extract content
            start = response.find('\n', response.find('```')) + 1
            end = response.rfind('```')
            if end > start:
                response = response[start:end].strip()
            
        # Remove any "json" language identifier
        if response.lower().startswith('json'):
            response = response[4:].strip()
            
        return response.strip()

    async def _suggest_tool_resources(self, tool_name: str, tool_description: str = "") -> Dict[str, str]:
        """Use LLM to suggest appropriate resource profile for unknown tools."""
        # First check if we have a predefined mapping for common tools
        common_tools = {
            "fastqc": {"profile": "minimal", "reason": "FastQC is a lightweight QC tool"},
            "multiqc": {"profile": "minimal", "reason": "MultiQC aggregates reports with minimal resources"},
            "kallisto_index": {"profile": "high_memory", "reason": "Indexing requires significant memory"},
            "kallisto_quant": {"profile": "multi_thread", "reason": "Quantification benefits from parallelization"},
            "create_directories": {"profile": "minimal", "reason": "Basic file system operations"}
        }
        
        # Check if it's a common tool
        tool_base = tool_name.split('_')[0] if '_' in tool_name else tool_name
        if tool_base in common_tools:
            return {
                "profile_name": common_tools[tool_base]["profile"],
                "reasoning": common_tools[tool_base]["reason"],
                "suggested_time_min": 60
            }

        # For unknown tools, ask LLM
        prompt = f"""
Analyze this bioinformatics tool and suggest computational resources.
Tool: {tool_name}
Description: {tool_description}

Choose ONE resource profile from:
- minimal (1 CPU, 2GB RAM): For lightweight tasks
- default (1 CPU, 4GB RAM): For standard preprocessing
- high_memory (1 CPU, 32GB RAM): For memory-intensive tasks
- multi_thread (8 CPUs, 16GB RAM): For parallel tasks
- high_memory_parallel (8 CPUs, 64GB RAM): For heavy processing

Return a JSON object in this EXACT format:
{{
    "profile_name": "chosen_profile",
    "reasoning": "brief explanation",
    "suggested_time_min": estimated_minutes
}}
"""
        try:
            messages = [
                {"role": "system", "content": "You are a bioinformatics resource allocation expert. Return ONLY the JSON object, nothing else."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self._call_openai(messages)
            
            # Clean the response
            cleaned_response = self._clean_llm_response(response)
            
            # Ensure we get valid JSON
            try:
                suggestion = json.loads(cleaned_response)
                # Validate the response
                required_fields = ["profile_name", "reasoning", "suggested_time_min"]
                valid_profiles = ["minimal", "default", "high_memory", "multi_thread", "high_memory_parallel"]
                
                if not all(field in suggestion for field in required_fields):
                    raise ValueError("Missing required fields in LLM response")
                
                if suggestion["profile_name"] not in valid_profiles:
                    raise ValueError(f"Invalid profile name: {suggestion['profile_name']}")
                
                return suggestion
                
            except json.JSONDecodeError:
                self.logger.warning(f"Invalid JSON response from LLM for {tool_name}. Response: {response}")
                raise
                
        except Exception as e:
            # Fallback based on tool name patterns
            if any(x in tool_name.lower() for x in ["index", "build"]):
                return {"profile_name": "high_memory", "reasoning": "Indexing typically needs more memory", "suggested_time_min": 120}
            elif any(x in tool_name.lower() for x in ["align", "map", "quant"]):
                return {"profile_name": "multi_thread", "reasoning": "Alignment/quantification benefits from multiple cores", "suggested_time_min": 90}
            elif any(x in tool_name.lower() for x in ["qc", "check", "stat"]):
                return {"profile_name": "minimal", "reasoning": "QC tasks usually need minimal resources", "suggested_time_min": 30}
            else:
                return {"profile_name": "default", "reasoning": "Using safe default profile", "suggested_time_min": 60}

    async def _get_execution_environment(self, prompt: str, default_env: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Determine execution environment from prompt and configuration.
        
        Args:
            prompt: User prompt or workflow description
            default_env: Optional default environment to use
            
        Returns:
            Dictionary with execution environment configuration
        """
        environments = self.WORKFLOW_CONFIG["execution"]["environments"]
        
        # If default environment specified, use it
        if default_env and default_env.get("type") in environments:
            return {
                "type": default_env["type"],
                **environments[default_env["type"]],
                **default_env  # Allow overriding specific settings
            }
            
        # Use LLM to analyze execution requirements
        messages = [
            {"role": "system", "content": """You are an expert at determining computational requirements for bioinformatics workflows.
            Analyze the prompt and determine the most appropriate execution environment:
            - local: For small datasets and testing
            - slurm: For production HPC workloads
            - kubernetes: For containerized, cloud-native workloads
            
            Consider:
            1. Data size and computational requirements
            2. Reproducibility needs
            3. Infrastructure requirements
            4. Scaling requirements
            """},
            {"role": "user", "content": f"""Determine the best execution environment for this workflow:
            {prompt}
            
            Respond with a JSON object containing:
            - type: The environment type (local, slurm, or kubernetes)
            - reason: Brief explanation of the choice
            - requirements: Any specific requirements (e.g. memory, storage)"""}
        ]
        
        try:
            response = await self._call_openai(
                messages,
                response_format={"type": "json_object"}
            )
            
            recommendation = json.loads(self._clean_llm_response(response))
            env_type = recommendation["type"]
            
            # Log the reasoning
            self.logger.info(
                f"Selected {env_type} execution environment:\n"
                f"Reason: {recommendation['reason']}\n"
                f"Requirements: {recommendation.get('requirements', {})}"
            )
            
            # Get base environment config
            env_config = environments[env_type].copy()
            
            # Add any specific requirements
            if "requirements" in recommendation:
                env_config.update(recommendation["requirements"])
            
            return {
                "type": env_type,
                **env_config,
                "recommendation_reason": recommendation["reason"]
            }
            
        except Exception as e:
            self.logger.warning(
                f"Failed to determine execution environment from prompt: {str(e)}\n"
                "Defaulting to local execution"
            )
            return {
                "type": "local",
                **environments["local"]
            }
    
    async def generate_workflow_plan(self, prompt: str, execution_env: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a workflow plan from a prompt.
        
        Args:
            prompt: User prompt describing the workflow
            execution_env: Optional execution environment to use, overrides automatic detection
        """
        # Detect workflow type
        workflow_type, config = self._detect_workflow_type(prompt)
        
        # Get execution environment
        execution = await self._get_execution_environment(prompt, execution_env)
        
        # Get available input files
        fastq_files = glob.glob("*.fastq.gz")
        if not fastq_files:
            raise ValueError("No .fastq.gz files found in current directory")
        
        # Extract output directory from prompt
        output_dir = "results"  # Default
        if "save" in prompt.lower() and "in" in prompt.lower():
            # Try to extract output directory from prompt
            parts = prompt.lower().split("save")
            if len(parts) > 1:
                after_save = parts[1]
                if "in" in after_save:
                    dir_part = after_save.split("in")[1].strip().strip("'\"./")
                    if dir_part:
                        output_dir = dir_part
        
        # Generate workflow plan using OpenAI
        messages = [
            {"role": "system", "content": """You are a bioinformatics workflow expert. Generate workflow plans as JSON objects with this structure:
{
    "workflow_type": "fastqc",
    "steps": [
        {
            "name": "step_name",
            "command": "command_to_execute",
            "parameters": {
                "input_file": "input.fastq.gz",
                "output_dir": "results/",
                "threads": 1
            },
            "dependencies": [],
            "outputs": ["expected_output1"]
        }
    ]
}

For FastQC analysis:
1. Include --outdir parameter to specify output directory
2. Include --threads parameter for parallel processing
3. Process each FASTQ file in a separate step
4. Include --noextract to keep files zipped (optional)
5. Include --quiet for less verbose output (optional)"""},
            {"role": "user", "content": f"""Generate a workflow plan for: {prompt}
            
Available input files: {fastq_files}
Output directory: {output_dir}
Execution environment: {execution['type']}
Workflow type: {workflow_type}

Rules:
1. Each step needs a unique name
2. Process each file individually, no wildcards
3. Dependencies must form a valid DAG (no cycles)
4. Return ONLY the JSON object"""}
        ]
        
        response = await self._call_openai(
            messages,
            response_format={"type": "json_object"}
        )
        
        try:
            workflow_plan = json.loads(self._clean_llm_response(response))
            
            # Log workflow plan
            self.logger.info("Generated workflow plan:")
            self.logger.info(f"Workflow type: {workflow_plan['workflow_type']}")
            
            # Convert rules to steps if needed
            if "rules" in workflow_plan and "steps" not in workflow_plan:
                workflow_plan["steps"] = workflow_plan.pop("rules")
            
            # Update each step with resources and container info
            updated_steps = []
            for step in workflow_plan.get("steps", []):
                # Ensure proper command formatting for FastQC
                if step["command"].startswith("fastqc"):
                    params = step.get("parameters", {})
                    input_file = params.get("input_file", "")
                    out_dir = params.get("output_dir", output_dir)
                    threads = params.get("threads", 1)
                    
                    # Build complete FastQC command
                    step["command"] = (
                        f"fastqc --outdir={out_dir} --threads={threads} "
                        f"--quiet --noextract {input_file}"
                    )
                
                # Get task resources
                try:
                    resources = await self._get_task_resources(step["command"], step.get("parameters", {}))
                    step["resources"] = resources
                except Exception as e:
                    self.logger.warning(f"Failed to get resources for step {step['name']}: {str(e)}")
                    step["resources"] = {
                        "memory_mb": 4000,
                        "cpus": 1,
                        "time_min": 60,
                        "profile": "default"
                    }
                
                # Update container config if needed
                if execution["type"] == "kubernetes":
                    try:
                        container = self._get_container_config(step["command"].split()[0])
                        step["container"] = container
                    except Exception as e:
                        self.logger.warning(f"Failed to get container config for step {step['name']}: {str(e)}")
                
                # Ensure resources are properly formatted
                resources = step.get("resources", {})
                if not isinstance(resources, dict):
                    resources = {
                        "memory_mb": 4000,
                        "cpus": 1,
                        "time_min": 60,
                        "profile": "default"
                    }
                
                # Convert numeric values to integers
                for key in ["memory_mb", "cpus", "time_min"]:
                    if key in resources:
                        try:
                            resources[key] = int(float(resources[key]))
                        except (ValueError, TypeError):
                            resources[key] = 4000 if key == "memory_mb" else 1 if key == "cpus" else 60
                
                step["resources"] = resources
                updated_steps.append(step)
                
                # Log step details
                self.logger.info(f"  Step: {step['name']}")
                self.logger.info(f"    Command: {step['command']}")
                self.logger.info(f"    Dependencies: {step.get('dependencies', [])}")
                self.logger.info(f"    Resources: {step['resources']}")
            
            workflow_plan["steps"] = updated_steps
            workflow_plan["execution"] = execution
            
            return workflow_plan
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse workflow plan: {str(e)}")
            self.logger.error(f"Raw response: {response}")
            raise ValueError("Failed to generate valid workflow plan")

    async def generate_command(self, step: Dict[str, Any]) -> str:
        """Generate a command for a workflow step."""
        try:
            # If command is already provided, use it
            if "command" in step and step["command"]:
                return step["command"]
            
            messages = [
                {"role": "system", "content": "You are a bioinformatics workflow expert. Generate precise shell commands."},
                {"role": "user", "content": f"Generate a shell command for the following workflow step:\n{json.dumps(step, indent=2)}"}
            ]
            
            response = await self._call_openai(messages)
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Failed to generate command: {str(e)}")
            raise

    async def generate_analysis(self, outputs: Dict[str, Any], query: str) -> str:
        """Generate analysis of workflow outputs."""
        try:
            # Prepare the prompt for analysis
            analysis_prompt = f"""
Analyze the following bioinformatics workflow outputs and provide a detailed report.
Focus on: {query}

Output Data:
{json.dumps(outputs, indent=2)}

Provide analysis in this format:
1. Overall Quality Assessment
2. Key Metrics and Statistics
3. Issues Found (if any)
4. Recommendations
"""
            messages = [
                {"role": "system", "content": "You are a bioinformatics analysis expert. Provide detailed analysis of workflow outputs."},
                {"role": "user", "content": analysis_prompt}
            ]
            
            # Don't specify response_format for text output
            response = await self._call_openai(messages)
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to generate analysis: {str(e)}")
            raise

    def _get_tool_characteristics(self, task_name: str) -> Dict[str, float]:
        """Determine tool characteristics based on task name patterns."""
        weights = {"memory_weight": 1.0, "time_weight": 1.0}
        
        task_lower = task_name.lower()
        for char_type, info in self.WORKFLOW_CONFIG["scaling_factors"]["tool_characteristics"].items():
            if any(pattern in task_lower for pattern in info["patterns"]):
                weights["memory_weight"] *= info["memory_weight"]
                weights["time_weight"] *= info["time_weight"]
        
        return weights

    def _get_size_multipliers(self, input_size_gb: float) -> Dict[str, float]:
        """Get scaling multipliers based on input size."""
        tiers = self.WORKFLOW_CONFIG["scaling_factors"]["size_tiers"]
        
        # Find appropriate tier
        for tier in reversed(tiers):  # Start from largest tier
            if input_size_gb >= tier["threshold_gb"]:
                return {
                    "memory_multiplier": tier["memory_multiplier"],
                    "time_multiplier": tier["time_multiplier"]
                }
        
        # If smaller than smallest tier
        return {"memory_multiplier": 1.0, "time_multiplier": 1.0}

    async def _get_task_resources(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get resource requirements for a task.
        
        Args:
            command: Command to execute
            parameters: Command parameters
            
        Returns:
            Dictionary with resource requirements
        """
        # Get base tool name
        tool = command.split()[0]
        
        # Default resources
        resources = {
            "memory_mb": 4096,  # 4GB
            "cpus": 1,
            "time_min": 60,
            "profile": "default"  # Add profile for CGAT compatibility
        }
        
        # Tool-specific resource adjustments
        if tool == "kallisto":
            if "index" in command:
                resources.update({
                    "memory_mb": 16384,  # 16GB
                    "cpus": 4,
                    "time_min": 120,
                    "profile": "high_memory"
                })
            elif "quant" in command:
                resources.update({
                    "memory_mb": 8192,  # 8GB
                    "cpus": 4,
                    "time_min": 90,
                    "profile": "multi_thread"
                })
        elif tool == "fastqc":
            # Get threads from command if specified
            threads = 1
            if "--threads=" in command:
                try:
                    threads_part = [p for p in command.split() if p.startswith("--threads=")][0]
                    threads = int(threads_part.split("=")[1])
                except (IndexError, ValueError):
                    threads = 1
            
            resources.update({
                "memory_mb": max(2048 * threads, 2048),  # 2GB per thread, minimum 2GB
                "cpus": threads,
                "time_min": 30,
                "profile": "minimal" if threads == 1 else "multi_thread"
            })
        
        # Convert all numeric values to integers
        for key in ["memory_mb", "cpus", "time_min"]:
            if key in resources:
                resources[key] = int(resources[key])
        
        return resources

    def _estimate_input_size(self, input_pattern: str) -> float:
        """Estimate input size in GB for resource scaling."""
        total_size = 0
        for file in glob.glob(input_pattern):
            if os.path.exists(file):
                total_size += os.path.getsize(file)
        return total_size / (1024 * 1024 * 1024)  # Convert to GB

    def _get_container_config(self, tool_name: str) -> Dict[str, Any]:
        """Get container configuration for a tool.
        
        Args:
            tool_name: Name of the tool (e.g., 'kallisto', 'fastqc')
            
        Returns:
            Dictionary with container configuration
        """
        # Extract base tool name (e.g., 'kallisto_quant' -> 'kallisto')
        base_tool = tool_name.split('_')[0]
        
        # Get container config
        container_config = self.WORKFLOW_CONFIG["containers"].get(base_tool, {})
        if not container_config:
            self.logger.warning(f"No container configuration found for tool: {tool_name}")
            container_config = {
                "image": "ubuntu:latest",
                "pull_policy": "IfNotPresent"
            }
        
        return container_config
