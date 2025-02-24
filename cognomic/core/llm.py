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

from ..config.settings import settings
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
                "model": settings.OPENAI_MODEL,  # Use model from settings
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

    async def generate_workflow_plan(self, prompt: str) -> Dict[str, Any]:
        """Generate a workflow plan from a prompt."""
        try:
            # Get available input files
            fastq_files = glob.glob("*.fastq.gz")
            if not fastq_files:
                raise ValueError("No .fastq.gz files found in current directory")
            
            self.logger.info(f"Found input files: {fastq_files}")
            
            # Detect workflow type
            workflow_type, workflow_config = self._detect_workflow_type(prompt)
            self.logger.info(f"Detected workflow type: {workflow_type}")
            
            # Create directory structure command
            dir_structure = " ".join(workflow_config["dir_structure"])
            mkdir_command = f"mkdir -p {dir_structure}"
            
            # Prepare workflow-specific instructions
            if workflow_type == "custom":
                tool_instructions = """
You are designing a custom bioinformatics workflow. Please:
1. Analyze the input files and requirements carefully
2. Suggest appropriate tools and methods based on the specific needs
3. Create a logical directory structure that matches the analysis flow
4. Include necessary quality control and validation steps
5. Follow best practices for the given analysis type
6. Document any assumptions or requirements

The base directory structure can be modified to better suit the analysis:
- raw_data: Store input files
- processed_data: Store intermediate processing results
- qc: Quality control reports and metrics
- analysis: Main analysis outputs
- output: Final results and reports
"""
            else:
                tool_instructions = f"Use these specific tool commands:\n{json.dumps(workflow_config['rules'], indent=4)}"
            
            # Add specific instructions for dependency specification
            enhanced_prompt = f"""
You are a bioinformatics workflow expert. Generate a workflow plan as a JSON object with the following structure:
{{
    "workflow_type": "{workflow_type}",
    "steps": [
        {{
            "name": "step_name",
            "command": "command_to_execute",
            "parameters": {{"param1": "value1"}},
            "dependencies": ["dependent_step_name1"],
            "outputs": ["expected_output1"]
        }}
    ]
}}

Available input files: {fastq_files}
Task: {prompt}

Rules:
1. First step MUST be directory creation with this EXACT command:
   "{mkdir_command}"

2. {tool_instructions}

3. Dependencies must form a valid DAG (no cycles)
4. Each step needs a unique name
5. Process each file individually, no wildcards
6. Return ONLY the JSON object, no markdown formatting or other text
"""

            # Add resource management instructions to the prompt
            resource_instructions = """
Resource Management Rules:
1. Each task must specify resource requirements and include a brief description
2. Consider input data size for resource scaling
3. Available resource profiles:
   - minimal: 1 CPU, 2GB RAM (lightweight tasks)
   - default: 1 CPU, 4GB RAM (standard preprocessing)
   - high_memory: 1 CPU, 32GB RAM (genome indexing)
   - multi_thread: 8 CPUs, 16GB RAM (parallel tasks)
   - high_memory_parallel: 8 CPUs, 64GB RAM (heavy processing)

4. For unknown tools, provide:
   - Brief description of the tool's purpose
   - Expected computational characteristics (CPU, memory, I/O intensive)
   - Any parallelization capabilities
   - Typical input data sizes and types

5. Resource scaling:
   - Large BAM files (>10GB): 1.5x memory, 2x time
   - Large FASTQ files (>20GB): 1.2x memory, 1.5x time
"""
            
            # Update the enhanced prompt with resource instructions
            enhanced_prompt = f"""
{enhanced_prompt}

{resource_instructions}

"""
            messages = [
                {"role": "system", "content": "You are a bioinformatics workflow expert. Return only valid JSON."},
                {"role": "user", "content": enhanced_prompt}
            ]
            
            response = await self._call_openai(messages, response_format={"type": "json_object"})
            
            try:
                workflow_plan = json.loads(self._clean_llm_response(response))
                # Log workflow plan
                self.logger.info("Generated workflow plan:")
                self.logger.info(f"Workflow type: {workflow_plan['workflow_type']}")
                for step in workflow_plan["steps"]:
                    self.logger.info(f"  Step: {step['name']}")
                    self.logger.info(f"    Command: {step['command']}")
                    self.logger.info(f"    Dependencies: {step['dependencies']}")
                
                # Update resources for each rule
                for i, step in enumerate(workflow_plan["steps"]):
                    workflow_plan["steps"][i] = await self._update_rule_resources(step)
                
                return workflow_plan
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse workflow plan: {str(e)}")
                self.logger.error(f"Raw response: {response}")
                raise
                
        except Exception as e:
            self.logger.error(f"Failed to generate workflow plan: {str(e)}")
            raise

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

    async def _update_rule_resources(self, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Update rule with appropriate resource requirements."""
        task_name = rule["name"].lower().replace("-", "_")
        
        # Get task description from rule if available
        task_description = rule.get("description", "")
        
        # Estimate input size if there are input files
        input_size = 0
        if "inputs" in rule:
            for input_pattern in rule["inputs"]:
                input_size += self._estimate_input_size(input_pattern)
        
        # Get resource requirements with task description
        resources = await self._get_task_resources(task_name, input_size, task_description)
        rule["resources"] = resources
        
        return rule

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

    async def _get_task_resources(self, task_name: str, input_size_gb: float = 0, task_description: str = "") -> Dict[str, Any]:
        """Get resource requirements for a task, scaling based on input size and characteristics."""
        # Get base resource profile
        profile_name = self.WORKFLOW_CONFIG["task_resources"].get(task_name)
        
        # If task not in predefined config, get suggestion from LLM
        if profile_name is None:
            suggestion = await self._suggest_tool_resources(task_name, task_description)
            profile_name = suggestion["profile_name"]
            
            # Log the suggestion for future reference
            self.logger.info(f"Resource suggestion for {task_name}: {suggestion['profile_name']} - {suggestion['reasoning']}")
            
            # Cache the suggestion for future use
            self.WORKFLOW_CONFIG["task_resources"][task_name] = profile_name
        
        base_resources = self.WORKFLOW_CONFIG["resources"][profile_name].copy()
        
        # Get tool-specific characteristics
        tool_weights = self._get_tool_characteristics(task_name)
        
        # Get size-based scaling
        size_multipliers = self._get_size_multipliers(input_size_gb)
        
        # Apply both tool-specific and size-based scaling
        final_memory_multiplier = tool_weights["memory_weight"] * size_multipliers["memory_multiplier"]
        final_time_multiplier = tool_weights["time_weight"] * size_multipliers["time_multiplier"]
        
        # Apply scaling to resources
        base_resources["memory_mb"] = int(base_resources["memory_mb"] * final_memory_multiplier)
        base_resources["time_min"] = int(base_resources["time_min"] * final_time_multiplier)
        
        # Log scaling decisions
        self.logger.debug(f"Resource scaling for {task_name}: memory_multiplier={final_memory_multiplier:.2f}, time_multiplier={final_time_multiplier:.2f}")
        
        return base_resources

    def _estimate_input_size(self, input_pattern: str) -> float:
        """Estimate input size in GB for resource scaling."""
        total_size = 0
        for file in glob.glob(input_pattern):
            if os.path.exists(file):
                total_size += os.path.getsize(file)
        return total_size / (1024 * 1024 * 1024)  # Convert to GB
