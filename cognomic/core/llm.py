"""LLM interface for workflow generation and command creation."""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import openai
from openai import AsyncOpenAI
import glob
import networkx as nx

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
            messages = [
                {"role": "system", "content": "You are a bioinformatics workflow expert. Return only valid JSON."},
                {"role": "user", "content": enhanced_prompt}
            ]
            
            response = await self._call_openai(messages, response_format={"type": "json_object"})
            
            try:
                workflow_plan = json.loads(response)
                # Log workflow plan
                self.logger.info("Generated workflow plan:")
                self.logger.info(f"Workflow type: {workflow_plan['workflow_type']}")
                for step in workflow_plan["steps"]:
                    self.logger.info(f"  Step: {step['name']}")
                    self.logger.info(f"    Command: {step['command']}")
                    self.logger.info(f"    Dependencies: {step['dependencies']}")
                
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
