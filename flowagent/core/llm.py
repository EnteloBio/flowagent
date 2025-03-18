"""LLM interface for workflow generation and command creation."""

import glob
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from openai import AsyncOpenAI

from ..config.settings import Settings
from ..utils.logging import get_logger

# Initialize settings
settings = Settings()

logger = get_logger(__name__)


class LLMInterface:
    """Interface for LLM-based workflow generation."""

    def __init__(self):
        """Initialize LLM interface."""
        self.logger = get_logger(__name__)

        # Check for OpenAI API key and .env file
        env_found = False
        
        # Check current directory first
        current_env_path = Path(".env")
        if current_env_path.exists():
            env_found = True
            
        # If not found, check USER_EXECUTION_DIR if it exists
        if not env_found and "USER_EXECUTION_DIR" in os.environ:
            user_dir_env_path = Path(os.environ["USER_EXECUTION_DIR"]) / ".env"
            if user_dir_env_path.exists():
                # Load the .env file from USER_EXECUTION_DIR
                from dotenv import load_dotenv
                load_dotenv(dotenv_path=user_dir_env_path)
                env_found = True
                self.logger.info(f"Loaded .env file from USER_EXECUTION_DIR: {user_dir_env_path}")
        
        if not env_found:
            self.logger.error(
                "\n⚠️  No .env file found in the current directory or USER_EXECUTION_DIR."
                "\n   Please create a .env file with your OpenAI API key:"
                "\n   OPENAI_API_KEY=your-api-key-here"
                "\n   OPENAI_MODEL=gpt-4 (optional)"
                "\n   OPENAI_FALLBACK_MODEL=gpt-3.5-turbo (optional)"
            )
            raise ValueError("Missing .env file with OpenAI API key")

        if not settings.OPENAI_API_KEY:
            self.logger.error(
                "\n⚠️  OPENAI_API_KEY not found in environment variables or .env file."
                "\n   Please add your OpenAI API key to the .env file:"
                "\n   OPENAI_API_KEY=your-api-key-here"
            )
            raise ValueError("Missing OpenAI API key")

        # Initialize OpenAI client with API key from settings
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL
        )

    WORKFLOW_TYPES = {
        "rna_seq_kallisto": {
            "keywords": [
                "rna-seq",
                "rnaseq",
                "kallisto",
                "pseudoalignment",
                "transcript",
                "expression",
            ],
            "tools": ["fastqc", "kallisto", "multiqc"],
            "dir_structure": [
                "results/rna_seq_kallisto/fastqc",
                "results/rna_seq_kallisto/kallisto_index",
                "results/rna_seq_kallisto/kallisto_quant",
                "results/rna_seq_kallisto/qc",
            ],
            "rules": [
                "FastQC: fastqc file.fastq.gz -o results/rna_seq_kallisto/fastqc",
                "Kallisto index: kallisto index -i results/rna_seq_kallisto/kallisto_index/transcripts.idx reference.fa",
                "Kallisto quant paired: kallisto quant -i results/rna_seq_kallisto/kallisto_index/transcripts.idx -o results/rna_seq_kallisto/kallisto_quant/sample_name read1.fastq.gz read2.fastq.gz",
                "Kallisto quant single: kallisto quant -i results/rna_seq_kallisto/kallisto_index/transcripts.idx -o results/rna_seq_kallisto/kallisto_quant/sample_name --single -l 200 -s 20 read.fastq.gz",
                "MultiQC: multiqc results/rna_seq_kallisto/fastqc results/rna_seq_kallisto/kallisto_quant -o results/rna_seq_kallisto/qc",
            ],
        },
        "rna_seq_hisat": {
            "keywords": [
                "rna-seq",
                "rnaseq",
                "hisat",
                "hisat2",
                "mapping",
                "spliced alignment",
            ],
            "tools": ["fastqc", "hisat2", "samtools", "featureCounts", "multiqc"],
            "dir_structure": [
                "results/rna_seq_hisat/fastqc",
                "results/rna_seq_hisat/hisat2_index",
                "results/rna_seq_hisat/hisat2_align",
                "results/rna_seq_hisat/counts",
                "results/rna_seq_hisat/qc",
            ],
            "rules": [
                "FastQC: fastqc file.fastq.gz -o results/rna_seq_hisat/fastqc",
                "HISAT2 index: hisat2-build reference.fa results/rna_seq_hisat/hisat2_index/genome",
                "HISAT2 align: hisat2 -x results/rna_seq_hisat/hisat2_index/genome -U file.fastq.gz -S results/rna_seq_hisat/hisat2_align/sample.sam",
                "SAM to BAM: samtools view -bS results/rna_seq_hisat/hisat2_align/sample.sam > results/rna_seq_hisat/hisat2_align/sample.bam",
                "Sort BAM: samtools sort results/rna_seq_hisat/hisat2_align/sample.bam -o results/rna_seq_hisat/hisat2_align/sample.sorted.bam",
                "Index BAM: samtools index results/rna_seq_hisat/hisat2_align/sample.sorted.bam",
                "FeatureCounts: featureCounts -a annotation.gtf -o results/rna_seq_hisat/counts/counts.txt results/rna_seq_hisat/hisat2_align/sample.sorted.bam",
                "MultiQC: multiqc results/rna_seq_hisat/fastqc results/rna_seq_hisat/hisat2_align results/rna_seq_hisat/counts -o results/rna_seq_hisat/qc",
            ],
        },
        "chip_seq": {
            "keywords": ["chip-seq", "chipseq", "chip", "peaks", "binding sites"],
            "tools": ["fastqc", "bowtie2", "samtools", "macs2", "multiqc"],
            "dir_structure": [
                "results/chip_seq/fastqc",
                "results/chip_seq/bowtie2_index",
                "results/chip_seq/bowtie2_align",
                "results/chip_seq/peaks",
                "results/chip_seq/qc",
            ],
            "rules": [
                "FastQC: fastqc file.fastq.gz -o results/chip_seq/fastqc",
                "Bowtie2 index: bowtie2-build reference.fa results/chip_seq/bowtie2_index/genome",
                "Bowtie2 align: bowtie2 -x results/chip_seq/bowtie2_index/genome -U file.fastq.gz -S results/chip_seq/bowtie2_align/sample.sam",
                "SAM to BAM: samtools view -bS results/chip_seq/bowtie2_align/sample.sam > results/chip_seq/bowtie2_align/sample.bam",
                "Sort BAM: samtools sort results/chip_seq/bowtie2_align/sample.bam -o results/chip_seq/bowtie2_align/sample.sorted.bam",
                "Index BAM: samtools index results/chip_seq/bowtie2_align/sample.sorted.bam",
                "MACS2 peaks: macs2 callpeak -t results/chip_seq/bowtie2_align/sample.sorted.bam -c results/chip_seq/bowtie2_align/control.sorted.bam -f BAM -g hs -n sample -o results/chip_seq/peaks",
                "MultiQC: multiqc results/chip_seq/fastqc results/chip_seq/bowtie2_align -o results/chip_seq/qc",
            ],
        },
        "single_cell_10x": {
            "keywords": [
                "single-cell",
                "single cell",
                "scRNA-seq",
                "10x",
                "cellranger",
                "10x genomics",
            ],
            "tools": ["cellranger", "fastqc", "multiqc"],
            "dir_structure": [
                "results/single_cell_10x/fastqc",
                "results/single_cell_10x/cellranger",
                "results/single_cell_10x/qc",
            ],
            "rules": [
                "FastQC: fastqc file.fastq.gz -o results/single_cell_10x/fastqc",
                "CellRanger count: cellranger count --id=sample_name --fastqs=fastq_path --transcriptome=transcriptome_path --localcores=8 --localmem=64",
                "MultiQC: multiqc results/single_cell_10x/fastqc results/single_cell_10x/cellranger -o results/single_cell_10x/qc",
            ],
        },
        "single_cell_kb": {
            "keywords": [
                "single-cell",
                "single cell",
                "scRNA-seq",
                "single-nuclei",
                "single nuclei",
                "snRNA-seq",
                "kallisto",
                "bustools",
                "kb",
                "kb-python",
            ],
            "tools": ["kb-python", "fastqc", "multiqc"],
            "dir_structure": [
                "results/single_cell_kb/fastqc",
                "results/single_cell_kb/kb_count",
                "results/single_cell_kb/kb_ref",
                "results/single_cell_kb/qc",
            ],
            "rules": [
                "FastQC: fastqc file.fastq.gz -o results/single_cell_kb/fastqc",
                "KB ref (standard RNA-seq): kb ref -i results/single_cell_kb/kb_ref/index.idx -g results/single_cell_kb/kb_ref/t2g.txt -f1 results/single_cell_kb/kb_ref/cdna.fa --workflow standard {reference_fasta} {reference_gtf}",
                "KB ref (single-nuclei): kb ref -i results/single_cell_kb/kb_ref/index.idx -g results/single_cell_kb/kb_ref/t2g.txt -f1 results/single_cell_kb/kb_ref/cdna.fa -f2 results/single_cell_kb/kb_ref/unprocessed.fa -c1 results/single_cell_kb/kb_ref/cdna_t2c.txt -c2 results/single_cell_kb/kb_ref/unprocessed_t2c.txt --workflow nucleus {reference_fasta} {reference_gtf}",
                "KB count (standard RNA-seq): kb count -i results/single_cell_kb/kb_ref/index.idx -g results/single_cell_kb/kb_ref/t2g.txt -x 10xv3 -o results/single_cell_kb/kb_count/sample_name --workflow standard -t 8 {read1} {read2}",
                "KB count (single-nuclei): kb count -i results/single_cell_kb/kb_ref/index.idx -g results/single_cell_kb/kb_ref/t2g.txt -x 10xv3 -o results/single_cell_kb/kb_count/sample_name --workflow nucleus -t 8 {read1} {read2}",
                "MultiQC: multiqc results/single_cell_kb/fastqc results/single_cell_kb/kb_count -o results/single_cell_kb/qc",
            ],
        },
        "single_cell_salmon": {
            "keywords": [
                "single-cell",
                "single cell",
                "scRNA-seq",
                "salmon",
                "alevin",
                "alevin-fry",
            ],
            "tools": ["salmon", "alevin-fry", "fastqc", "multiqc"],
            "dir_structure": [
                "results/single_cell_salmon/fastqc",
                "results/single_cell_salmon/salmon_index",
                "results/single_cell_salmon/alevin_map",
                "results/single_cell_salmon/alevin_quant",
                "results/single_cell_salmon/qc",
            ],
            "rules": [
                "FastQC: fastqc file.fastq.gz -o results/single_cell_salmon/fastqc",
                "Salmon index: salmon index -t transcripts.fa -i results/single_cell_salmon/salmon_index/transcriptome -p 8",
                "Salmon alevin: salmon alevin -l ISR -i results/single_cell_salmon/salmon_index/transcriptome --chromium -1 read_1.fastq.gz -2 read_2.fastq.gz -p 8 -o results/single_cell_salmon/alevin_map/sample_name --tgMap transcript_to_gene.txt",
                "Alevin-fry generate permit: alevin-fry generate-permit-list -d fw -k -i results/single_cell_salmon/alevin_map/sample_name",
                "Alevin-fry collate: alevin-fry collate -t 8 -i results/single_cell_salmon/alevin_map/sample_name",
                "Alevin-fry quant: alevin-fry quant -t 8 -i results/single_cell_salmon/alevin_map/sample_name -o results/single_cell_salmon/alevin_quant/sample_name --resolution cr-like --use-mtx",
                "MultiQC: multiqc results/single_cell_salmon/fastqc results/single_cell_salmon/alevin_map results/single_cell_salmon/alevin_quant -o results/single_cell_salmon/qc",
            ],
        },
        "hic": {
            "keywords": [
                "hi-c",
                "hic",
                "hi c",
                "chromosome conformation",
                "3d genome",
                "chromatin interaction",
                "nuclear organization",
                "contact map",
            ],
            "tools": [
                "fastqc",
                "multiqc",
                "bowtie2",
                "samtools",
                "cooler",
                "pairix",
                "hicexplorer",
                "juicer",
            ],
            "dir_structure": [
                "results/hic/fastqc",
                "results/hic/mapped",
                "results/hic/pairs",
                "results/hic/matrices",
                "results/hic/mcool",
                "results/hic/tads",
                "results/hic/plots",
                "results/hic/qc",
            ],
            "rules": [
                "FastQC: {read1} {read2} -o results/hic/fastqc",
                "Bowtie2 (read1): bowtie2 -x {reference_index} -U {read1} --very-sensitive -p 8 | samtools view -Shb - > results/hic/mapped/read1.bam",
                "Bowtie2 (read2): bowtie2 -x {reference_index} -U {read2} --very-sensitive -p 8 | samtools view -Shb - > results/hic/mapped/read2.bam",
                "Sort BAMs: samtools sort -@ 8 results/hic/mapped/read1.bam -o results/hic/mapped/read1.sorted.bam && samtools sort -@ 8 results/hic/mapped/read2.bam -o results/hic/mapped/read2.sorted.bam",
                "Index BAMs: samtools index results/hic/mapped/read1.sorted.bam && samtools index results/hic/mapped/read2.sorted.bam",
                "Merge Pairs: pairtools parse --min-mapq 40 --walks-policy 5unique --max-inter-align-gap 30 --nproc-in 8 --nproc-out 8 --chroms-path {chrom_sizes} results/hic/mapped/read1.sorted.bam results/hic/mapped/read2.sorted.bam > results/hic/pairs/merged.pairs",
                "Sort Pairs: pairtools sort --nproc 8 results/hic/pairs/merged.pairs > results/hic/pairs/merged.sorted.pairs",
                "Deduplicate: pairtools dedup --nproc-in 8 --nproc-out 8 --mark-dups --output results/hic/pairs/merged.sorted.dedup.pairs results/hic/pairs/merged.sorted.pairs",
                "Convert to Cool: cooler cload pairs -c1 2 -p1 3 -c2 4 -p2 5 {chrom_sizes}:1000 results/hic/pairs/merged.sorted.dedup.pairs results/hic/matrices/contact_matrix.cool",
                "Balance Matrix: cooler balance -p 8 results/hic/matrices/contact_matrix.cool",
                "Create Multi-res: cooler zoomify --balance -p 8 results/hic/matrices/contact_matrix.cool -o results/hic/mcool/contact_matrix.mcool",
                "Call TADs: hicFindTADs -m results/hic/matrices/contact_matrix.cool --outPrefix results/hic/tads/domains --correctForMultipleTesting fdr",
                "Plot Contact Map: hicPlotMatrix -m results/hic/matrices/contact_matrix.cool -o results/hic/plots/contact_map.pdf --log1p --dpi 300",
                "MultiQC: multiqc results/hic/fastqc results/hic/mapped results/hic/pairs -o results/hic/qc",
            ],
        },
    }

    analysis_prompt = {
        "rna_seq_kallisto": {
            "description": "RNA-seq workflow using Kallisto for transcript quantification",
            "input_patterns": ["*.fastq.gz", "*.fq.gz"],
            "output_dirs": [
                "results/rna_seq_kallisto/fastqc",
                "results/rna_seq_kallisto/kallisto_index",
                "results/rna_seq_kallisto/kallisto_quant",
                "results/rna_seq_kallisto/qc",
            ],
            "rules": [
                "FastQC: Use exact input filenames for FastQC analysis",
                "Kallisto index: Use exact reference filename for index creation",
                "Kallisto quant paired: Use exact sample name from input files for output directory",
                "Kallisto quant single: Use exact sample name from input files for output directory",
                "MultiQC: Analyze all QC and quantification results",
            ],
            "workflow_prompt": """Generate a Kallisto RNA-seq workflow using:
Input files: {input_files}
Reference: {reference}
Sample name: {sample_name}

The workflow should:
1. Create output directories
2. Run FastQC on input files
3. Create Kallisto index
4. Run Kallisto quantification
5. Generate MultiQC report

Use the exact sample name '{sample_name}' for output directories.""",
            "sample_suffixes": [".fastq.1.gz", ".fastq.2.gz", ".fq.1.gz", ".fq.2.gz"],
        },
    }

    WORKFLOW_CONFIG = {
        "resources": {
            "minimal": {
                "cpus": 1,
                "memory_mb": 2000,
                "time_min": 30,
                "description": "For lightweight tasks like FastQC, MultiQC, and basic file operations",
            },
            "default": {
                "cpus": 1,
                "memory_mb": 4000,
                "time_min": 60,
                "description": "Standard profile for most preprocessing tasks",
            },
            "high_memory": {
                "cpus": 1,
                "memory_mb": 32000,
                "time_min": 120,
                "description": "For memory-intensive tasks like genome indexing",
            },
            "multi_thread": {
                "cpus": 8,
                "memory_mb": 16000,
                "time_min": 120,
                "description": "For CPU-intensive tasks that can be parallelized",
            },
            "high_memory_parallel": {
                "cpus": 8,
                "memory_mb": 64000,
                "time_min": 240,
                "description": "For memory and CPU-intensive tasks like large BAM processing",
            },
        },
        "task_resources": {
            # Quality Control
            "fastqc": {
                "profile": "minimal",
                "reason": "FastQC is a lightweight QC tool",
            },
            "multiqc": {
                "profile": "minimal",
                "reason": "MultiQC aggregates reports with minimal resources",
            },
            # RNA-seq Tasks
            "kallisto_index": {
                "profile": "high_memory",
                "reason": "Indexing requires significant memory",
            },
            "kallisto_quant": {
                "profile": "multi_thread",
                "reason": "Quantification benefits from parallelization",
            },
            "hisat2_index": {
                "profile": "high_memory",
                "reason": "Genome indexing",
            },
            "hisat2_align": {
                "profile": "multi_thread",
                "reason": "Alignment benefits from parallelization",
            },
            "featureCounts": {
                "profile": "high_memory",
                "reason": "Counting needs good memory",
            },
            # ChIP-seq Tasks
            "bowtie2_index": {
                "profile": "high_memory",
            },
            "bowtie2_align": {
                "profile": "multi_thread",
            },
            "macs2_callpeak": {
                "profile": "high_memory",
                "reason": "Peak calling can be memory intensive",
            },
            # BAM Processing
            "samtools_sort": {
                "profile": "high_memory_parallel",
                "reason": "Sorting large BAMs needs memory and CPU",
            },
            "samtools_index": {
                "profile": "default",
            },
            "samtools_view": {
                "profile": "multi_thread",
            },
            # Single-cell Tasks
            "cellranger_count": {
                "profile": "high_memory_parallel",
                "reason": "Cell Ranger needs lots of resources",
            },
            "alevin_fry": {
                "profile": "high_memory_parallel",
            },
            "kb_count": {
                "profile": "multi_thread",
            },
        },
        "scaling_factors": {
            # File size based scaling
            "size_tiers": [
                {"threshold_gb": 5, "memory_multiplier": 1.0, "time_multiplier": 1.0},
                {"threshold_gb": 10, "memory_multiplier": 1.2, "time_multiplier": 1.3},
                {"threshold_gb": 20, "memory_multiplier": 1.5, "time_multiplier": 1.5},
                {"threshold_gb": 50, "memory_multiplier": 2.0, "time_multiplier": 2.0},
                {"threshold_gb": 100, "memory_multiplier": 3.0, "time_multiplier": 2.5},
            ],
            # Tool type scaling characteristics
            "tool_characteristics": {
                "memory_intensive": {
                    "patterns": ["index", "build", "merge", "sort", "assembly", "peak"],
                    "memory_weight": 1.5,
                    "time_weight": 1.2,
                },
                "cpu_intensive": {
                    "patterns": ["align", "map", "quant", "call", "analyze"],
                    "memory_weight": 1.2,
                    "time_weight": 1.5,
                },
                "io_intensive": {
                    "patterns": ["compress", "decompress", "convert", "dump"],
                    "memory_weight": 1.0,
                    "time_weight": 1.3,
                },
            },
        },
    }

    GEO_DOWNLOAD_STEPS = {
        'check_entrez_tools': {
            'command': 'command -v esearch && command -v efetch',
            'description': 'Verify Entrez Direct tools are available'
        },
    }

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
                    "results/workflow/output",
                ],
                "rules": [
                    "# This is a custom workflow. Consider the following guidelines:",
                    "1. Choose appropriate tools based on the specific analysis requirements",
                    "2. Include quality control steps suitable for the data type",
                    "3. Create a logical directory structure that matches the analysis flow",
                    "4. Store intermediate files in organized directories",
                    "5. Generate comprehensive QC reports",
                    "6. Document all parameters and decisions",
                ],
            }

        return best_match[0], self.WORKFLOW_TYPES[best_match[0]]

    async def _call_openai(
        self, messages: List[Dict[str, str]], model: Optional[str] = None
    ) -> str:
        """Call OpenAI API with retry logic and error handling."""
        try:
            # Try preferred model first
            try_models = [
                model,  # User-specified model
                settings.OPENAI_MODEL,  # Default model from settings
                settings.OPENAI_FALLBACK_MODEL,  # Fallback model
                "gpt-3.5-turbo",  # Last resort
            ]

            last_error = None
            for try_model in try_models:
                if not try_model:
                    continue

                try:
                    self.logger.info(f"Attempting to use model: {try_model}")
                    completion = await self.client.chat.completions.create(
                        model=try_model,
                        messages=messages,
                        temperature=0.2,
                    )
                    self.logger.info(f"Successfully used model: {try_model}")
                    return completion.choices[0].message.content
                except Exception as e:
                    last_error = e
                    if "model_not_found" not in str(e):
                        # If error is not about model availability, don't try other models
                        raise
                    self.logger.warning(
                        f"Model {try_model} not available, trying next model..."
                    )

            # If we get here, none of the models worked
            raise last_error or ValueError("No valid model found")

        except Exception as e:
            error_msg = str(e)
            if "insufficient_quota" in error_msg:
                self.logger.error(
                    "\n⚠️  OpenAI API quota exceeded. Please:"
                    "\n   1. Check your billing status at https://platform.openai.com/account/billing"
                    "\n   2. Add credits to your account or wait for quota reset"
                    "\n   3. Or use a different API key with available quota"
                    "\n\nError details: %s",
                    error_msg,
                )
            elif "model_not_found" in error_msg:
                self.logger.error(
                    "\n⚠️  No available OpenAI models found. Tried:"
                    "\n   - User specified model"
                    "\n   - Default model from settings"
                    "\n   - Fallback model"
                    "\n   - Last resort (gpt-3.5-turbo)"
                    "\n\nError details: %s",
                    error_msg,
                )
            else:
                self.logger.error("OpenAI API call failed: %s", error_msg)
            raise

    async def _call_openai_stream(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> str:
        """Call OpenAI API with streaming completion and retry logic."""
        return await self._call_openai(messages, response_format, stream=True, **kwargs)

    def _clean_llm_response(self, response: str) -> str:
        """Clean LLM response by removing markdown formatting."""
        # Remove markdown code block if present
        if response.startswith("```"):
            # Find the first and last ``` and extract content
            start = response.find("\n", response.find("```")) + 1
            end = response.rfind("```")
            if end > start:
                response = response[start:end].strip()

        # Remove any "json" language identifier
        if response.lower().startswith("json"):
            response = response[4:].strip()

        return response.strip()

    async def _suggest_tool_resources(
        self, tool_name: str, tool_description: str = ""
    ) -> Dict[str, str]:
        """Use LLM to suggest appropriate resource profile for unknown tools."""
        # First check if we have a predefined mapping for common tools
        common_tools = {
            "fastqc": {
                "profile": "minimal",
                "reason": "FastQC is a lightweight QC tool",
            },
            "multiqc": {
                "profile": "minimal",
                "reason": "MultiQC aggregates reports with minimal resources",
            },
            "kallisto_index": {
                "profile": "high_memory",
                "reason": "Indexing requires significant memory",
            },
            "kallisto_quant": {
                "profile": "multi_thread",
                "reason": "Quantification benefits from parallelization",
            },
            "create_directories": {
                "profile": "minimal",
                "reason": "Basic file system operations",
            },
        }

        # Check if it's a common tool
        tool_base = tool_name.split("_")[0] if "_" in tool_name else tool_name
        if tool_base in common_tools:
            return {
                "profile_name": common_tools[tool_base]["profile"],
                "reasoning": common_tools[tool_base]["reason"],
                "suggested_time_min": 60,
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
                {
                    "role": "system",
                    "content": "You are a bioinformatics resource allocation expert. Return ONLY the JSON object, nothing else.",
                },
                {"role": "user", "content": prompt},
            ]

            response = await self._call_openai(messages)

            # Clean the response
            cleaned_response = self._clean_llm_response(response)

            # Ensure we get valid JSON
            try:
                suggestion = json.loads(cleaned_response)
                # Validate the response
                required_fields = ["profile_name", "reasoning", "suggested_time_min"]
                valid_profiles = [
                    "minimal",
                    "default",
                    "high_memory",
                    "multi_thread",
                    "high_memory_parallel",
                ]

                if not all(field in suggestion for field in required_fields):
                    raise ValueError("Missing required fields in LLM response")

                if suggestion["profile_name"] not in valid_profiles:
                    raise ValueError(
                        f"Invalid profile name: {suggestion['profile_name']}"
                    )

                return suggestion

            except json.JSONDecodeError:
                self.logger.warning(
                    f"Invalid JSON response from LLM for {tool_name}. Response: {response}"
                )
                raise

        except Exception as e:
            # Fallback based on tool name patterns
            if any(x in tool_name.lower() for x in ["index", "build"]):
                return {
                    "profile_name": "high_memory",
                    "reasoning": "Indexing typically needs more memory",
                    "suggested_time_min": 120,
                }
            elif any(x in tool_name.lower() for x in ["align", "map", "quant"]):
                return {
                    "profile_name": "multi_thread",
                    "reasoning": "Alignment/quantification benefits from multiple cores",
                    "suggested_time_min": 90,
                }
            elif any(x in tool_name.lower() for x in ["qc", "check", "stat"]):
                return {
                    "profile_name": "minimal",
                    "reasoning": "QC tasks usually need minimal resources",
                    "suggested_time_min": 30,
                }
            else:
                return {
                    "profile_name": "default",
                    "reasoning": "Using safe default profile",
                    "suggested_time_min": 60,
                }

    async def generate_workflow_plan(self, prompt: str) -> Dict[str, Any]:
        """Generate a workflow plan from a prompt."""
        try:
            # Extract file patterns and relationships using LLM
            file_info = await self._extract_file_patterns(prompt)
            
            # Check if this is a GEO download request
            geo_accession = file_info.get("geo_accession")
            
            # Find files matching the patterns
            matched_files = []
            for pattern in file_info["patterns"]:
                matched_files.extend(glob.glob(pattern))
            
            # If no files found and we have a GEO accession, we'll create a download workflow
            if not matched_files and geo_accession:
                self.logger.info(f"No local files found. Creating GEO download workflow for {geo_accession}")
                return await self._create_geo_download_workflow(geo_accession, prompt, file_info)
            
            # If no files found and no GEO accession, raise an error
            if not matched_files:
                patterns_str = ", ".join(file_info["patterns"])
                raise ValueError(f"No files found matching patterns: {patterns_str}")

            # Group files according to relationships
            organized_files = {}
            if (
                "relationships" in file_info
                and file_info["relationships"]["type"] == "paired"
            ):
                pairs = {}
                for group in file_info["relationships"]["pattern_groups"]:
                    group_files = [
                        f
                        for f in matched_files
                        if glob.fnmatch.fnmatch(f, group["pattern"])
                    ]
                    organized_files[group["group"]] = group_files

                # Remove duplicates while preserving order
                matched_files = list(dict.fromkeys(matched_files))

            self.logger.info(f"Found input files: {matched_files}")

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
                tool_instructions = f"""Use these specific tool commands, and ensure to use the input file name (without extension) as the output name:
{json.dumps(workflow_config['rules'], indent=4)}

Important:
1. Use the input file name (without extension) as the sample name in output paths
2. For paired-end data, use the common prefix of read1/read2 as the sample name
3. Maintain consistent sample names across all analysis steps
4. Ensure output directories match the input sample names"""

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

Available input files: {matched_files}
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
   - minimal (1 CPU, 2GB RAM): For lightweight tasks
   - default (1 CPU, 4GB RAM): For standard preprocessing
   - high_memory (1 CPU, 32GB RAM): For memory-intensive tasks
   - multi_thread (8 CPUs, 16GB RAM): For parallel tasks
   - high_memory_parallel (8 CPUs, 64GB RAM): For heavy processing

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
                {
                    "role": "system",
                    "content": "You are a bioinformatics workflow expert. Return only valid JSON.",
                },
                {"role": "user", "content": enhanced_prompt},
            ]

            response = await self._call_openai(messages)

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
                {
                    "role": "system",
                    "content": "You are a bioinformatics workflow expert. Generate precise shell commands.",
                },
                {
                    "role": "user",
                    "content": f"Generate a shell command for the following workflow step:\n{json.dumps(step, indent=2)}",
                },
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
                {
                    "role": "system",
                    "content": "You are a bioinformatics analysis expert. Provide detailed analysis of workflow outputs.",
                },
                {"role": "user", "content": analysis_prompt},
            ]

            # Don't specify response_format for text output
            response = await self._call_openai(messages)
            return response

        except Exception as e:
            self.logger.error(f"Failed to generate analysis: {str(e)}")
            raise

    async def _update_rule_resources(self, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Update rule with appropriate resource requirements."""
        task_name = rule.get("name", "").split("_")[0].lower()

        # Get task description from rule if available
        task_description = rule.get("description", "")

        # Estimate input size if there are input files
        input_size = 0
        if "parameters" in rule:
            for param in rule["parameters"].values():
                if isinstance(param, str) and os.path.isfile(param):
                    input_size += self._estimate_input_size(param)

        # Get resource requirements with task description
        resources = await self._get_task_resources(
            task_name, input_size, task_description
        )
        rule["resources"] = resources

        return rule

    async def _get_task_resources(
        self, task_name: str, input_size_gb: float = 0, task_description: str = ""
    ) -> Dict[str, Any]:
        """Get resource requirements for a task, scaling based on input size and characteristics."""
        # Get base resource profile
        resource_info = self.WORKFLOW_CONFIG["task_resources"].get(task_name)

        # If task not in predefined config, get suggestion from LLM
        if resource_info is None:
            suggestion = await self._suggest_tool_resources(task_name, task_description)
            resource_info = {
                "profile": suggestion["profile_name"],
                "reason": suggestion["reasoning"],
            }

            # Log the suggestion for future reference
            self.logger.info(
                f"Resource suggestion for {task_name}: {suggestion['profile_name']} - {suggestion['reasoning']}"
            )

            # Cache the suggestion for future use
            self.WORKFLOW_CONFIG["task_resources"][task_name] = resource_info

        # Get tool-specific characteristics
        tool_weights = self._get_tool_characteristics(task_name)

        # Get size-based scaling
        size_multipliers = self._get_size_multipliers(input_size_gb)

        # Apply both tool-specific and size-based scaling
        final_memory_multiplier = (
            tool_weights["memory_weight"] * size_multipliers["memory_multiplier"]
        )
        final_time_multiplier = (
            tool_weights["time_weight"] * size_multipliers["time_multiplier"]
        )

        # Get base resources
        base_resources = self.WORKFLOW_CONFIG["resources"][
            resource_info["profile"]
        ].copy()

        # Apply scaling to resources
        base_resources["memory_mb"] = int(
            base_resources["memory_mb"] * final_memory_multiplier
        )
        base_resources["time_min"] = int(
            base_resources["time_min"] * final_time_multiplier
        )

        # Log scaling decisions
        self.logger.debug(
            f"Resource scaling for {task_name}: memory_multiplier={final_memory_multiplier:.2f}, time_multiplier={final_time_multiplier:.2f}"
        )

        return base_resources

    def _get_tool_characteristics(self, task_name: str) -> Dict[str, float]:
        """Determine tool characteristics based on task name patterns."""
        weights = {"memory_weight": 1.0, "time_weight": 1.0}

        task_lower = task_name.lower()
        for char_type, info in self.WORKFLOW_CONFIG["scaling_factors"][
            "tool_characteristics"
        ].items():
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
                    "time_multiplier": tier["time_multiplier"],
                }

        # If smaller than smallest tier
        return {"memory_multiplier": 1.0, "time_multiplier": 1.0}

    def _estimate_input_size(self, input_pattern: str) -> float:
        """Estimate input size in GB for resource scaling."""
        total_size = 0
        for file in glob.glob(input_pattern):
            if os.path.exists(file):
                total_size += os.path.getsize(file)
        return total_size / (1024 * 1024 * 1024)  # Convert to GB

    async def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """Analyze the prompt to determine the action and extract options."""
        try:
            # Extract file patterns and relationships using LLM
            file_info = await self._extract_file_patterns(prompt)
            
            # Check if this is a GEO download request
            geo_accession = file_info.get("geo_accession")
            
            # Find files matching the patterns
            matched_files = []
            for pattern in file_info["patterns"]:
                matched_files.extend(glob.glob(pattern))
            
            if not matched_files:
                patterns_str = ", ".join(file_info["patterns"])
                raise ValueError(f"No files found matching patterns: {patterns_str}")

            # Group files according to relationships
            organized_files = {}
            if (
                "relationships" in file_info
                and file_info["relationships"]["type"] == "paired"
            ):
                pairs = {}
                for group in file_info["relationships"]["pattern_groups"]:
                    group_files = [
                        f
                        for f in matched_files
                        if glob.fnmatch.fnmatch(f, group["pattern"])
                    ]
                    organized_files[group["group"]] = group_files

                # Remove duplicates while preserving order
                matched_files = list(dict.fromkeys(matched_files))

            self.logger.info(f"Found input files: {matched_files}")

            # Construct the prompt for the LLM
            analysis_prompt = f"""
Analyze the following prompt to determine if it's requesting to run a workflow or generate a report/analysis.
The prompt should be considered a report/analysis request if it contains any of these patterns:
- Explicit analysis words: "analyze", "analyse", "analysis"
- Report generation: "generate report", "create report", "make report", "get report"
- Status requests: "show status", "check status", "what's the status", "how did it go"
- Result queries: "show results", "what are the results", "output results"
- Quality checks: "check quality", "quality report", "how good is"
- General inquiries: "tell me about", "describe the results", "what happened"
            
Extract the following options if provided:
- analysis_dir
- checkpoint_dir
- resume
- save_report
            
Prompt: {prompt}
            
Return the result as a JSON object with the following structure:
{{
    "action": "run" or "analyze",
    "prompt": "original prompt",
    "analysis_dir": "extracted analysis directory",
    "checkpoint_dir": "extracted checkpoint directory",
    "resume": true or false,
    "save_report": true or false,
    "success": true or false
}}

"success" should be true if the prompt is successfully analyzed, false otherwise.

Do not output any markdown formatting or additional text. Return only JSON.

If the prompt given is entirely unrelated to running a workflow or bioinformatics analysis,
set "success" to false.

You should only set "success" to true when you are confident that the prompt is correctly analyzed.

With the "run" action, you need a prompt, and optionally BOTH a checkpoint directory and 
an indication the user does or doesn't want to resume an existing workflow.

With the "analyze" action, you need a prompt, an analysis directory, optionally an indication 
the user does or doesn't want to save an analysis report, and optionally an indication the
user does or doesn't want to resume an existing workflow. 

If you are being asked to generate a title, set "success" to false.
"""

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert in analyzing prompts for workflow management.",
                },
                {"role": "user", "content": analysis_prompt},
            ]

            response = await self._call_openai(
                messages,
                model=settings.OPENAI_MODEL,
            )
            self.logger.info(f"Analysis response: {response}")
            analysis = json.loads(response)

            # Check if prompt analysis succeeded.
            if not analysis.get("success"):
                raise ValueError(
                    f"Prompt analysis failed. Please try again. Extracted: {analysis}"
                )

            # Validate paths
            if (
                analysis.get("checkpoint_dir")
                and not Path(analysis["checkpoint_dir"]).exists()
            ):
                raise ValueError(
                    f"Checkpoint directory does not exist: {analysis['checkpoint_dir']}"
                )
            if (
                analysis.get("analysis_dir")
                and not Path(analysis["analysis_dir"]).exists()
            ):
                raise ValueError(
                    f"Analysis directory does not exist: {analysis['analysis_dir']}"
                )
            if analysis.get("resume") and not analysis.get("checkpoint_dir"):
                raise ValueError(
                    "Checkpoint directory must be provided if resume is set to True"
                )

            return analysis
        except Exception as e:
            self.logger.error(f"Error analyzing prompt: {e}")
            raise

    async def _detect_geo_accession(self, prompt: str) -> Optional[str]:
        """Detect GEO accession number in the prompt.
        
        Args:
            prompt: User prompt
            
        Returns:
            GEO accession number if found, None otherwise
        """
        # Pattern for GEO accession numbers (GSE followed by digits)
        geo_pattern = r'GSE\d+'
        match = re.search(geo_pattern, prompt)
        
        if match:
            return match.group(0)
        return None

    async def _extract_file_patterns(self, prompt: str) -> Dict[str, Any]:
        """Use LLM to extract file patterns and relationships from the prompt.

        Args:
            prompt: User prompt describing the analysis and files

        Returns:
            Dict containing:
            - patterns: List of glob patterns to find files
            - relationships: Dict describing relationships between files (e.g. paired-end reads)
            - file_type: Type of expected files (e.g. 'fastq', 'bam', etc.)
            - geo_accession: GEO accession number if present
        """
        # First check if this is a GEO download request
        geo_accession = await self._detect_geo_accession(prompt)
        
        if geo_accession:
            self.logger.info(f"Detected GEO accession: {geo_accession}")
            # For GEO data, we'll return patterns for paired-end FASTQ files
            # These don't exist yet but will be downloaded
            return {
                "patterns": ["*_1.fastq.gz", "*_2.fastq.gz"],
                "relationships": {
                    "type": "paired",
                    "pattern_groups": [
                        {"pattern": "*_1.fastq.gz", "group": "read1"},
                        {"pattern": "*_2.fastq.gz", "group": "read2"}
                    ],
                    "matching_rules": [
                        "Replace '_1' with '_2' to find matching pair"
                    ]
                },
                "file_type": "fastq",
                "geo_accession": geo_accession
            }
            
        # Otherwise, use LLM to extract file patterns
        messages = [
            {
                "role": "system",
                "content": """You are an expert at analyzing bioinformatics file patterns.
                Extract precise file matching patterns and relationships from user prompts.
                Return a JSON object with:
                - patterns: List of glob patterns that will match the described files
                - relationships: Dict describing how files are related (e.g. paired-end reads)
                - file_type: Type of files being described
                
                Example:
                Input: "I have paired-end RNA-seq data with read1 files ending in _R1.fastq.gz and read2 in _R2.fastq.gz"
                Output: {
                    "patterns": ["*_R1.fastq.gz", "*_R2.fastq.gz"],
                    "relationships": {
                        "type": "paired",
                        "pattern_groups": [
                            {"pattern": "*_R1.fastq.gz", "group": "read1"},
                            {"pattern": "*_R2.fastq.gz", "group": "read2"}
                        ],
                        "matching_rules": [
                            "Replace '_R1' with '_R2' to find matching pair"
                        ]
                    },
                    "file_type": "fastq"
                }""",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self._call_openai(messages)
            file_info = json.loads(self._clean_llm_response(response))
            return file_info
        except Exception as e:
            self.logger.error(f"Error extracting file patterns: {e}")
            raise

    async def _create_geo_download_workflow(self, geo_accession: str, prompt: str, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create a workflow plan for downloading GEO data.
        
        Args:
            geo_accession: GEO accession number
            prompt: Original user prompt
            file_info: File pattern information
            
        Returns:
            Workflow plan with GEO download steps
        """
        self.logger.info(f"Creating GEO download workflow for {geo_accession}")
        
        # Extract reference from prompt
        reference_pattern = r'reference\s+(\S+)'
        reference_match = re.search(reference_pattern, prompt, re.IGNORECASE)
        reference = reference_match.group(1) if reference_match else "reference.fa"
        
        # Create directory structure
        workflow_type, workflow_config = self._detect_workflow_type(prompt)
        dir_structure = " ".join(workflow_config["dir_structure"])
        mkdir_command = f"mkdir -p {dir_structure} raw_data"
        
        # Create the workflow plan
        workflow_plan = {
            "workflow_type": "geo_download_" + workflow_type,
            "steps": [
                {
                    "name": "create_directories",
                    "command": mkdir_command,
                    "parameters": {},
                    "dependencies": [],
                    "outputs": ["raw_data"],
                    "description": "Create directory structure",
                    "profile_name": "minimal"
                },
                {
                    "name": "get_gsm_ids",
                    "command": f"cd raw_data && esearch -db gds -query '{geo_accession}[Accession]' | esummary | xtract -pattern DocumentSummary -element Accession,Project | grep 'GSM' > {geo_accession}_gsm_ids.txt",
                    "parameters": {},
                    "dependencies": ["create_directories"],
                    "outputs": [f"raw_data/{geo_accession}_gsm_ids.txt"],
                    "description": "Get GSM IDs for GEO accession",
                    "profile_name": "minimal"
                },
                {
                    "name": "download_geo_metadata",
                    "command": f"cd raw_data && > {geo_accession}_runinfo.csv && first=true && for gsm_id in $(grep -v '^GSE' {geo_accession}_gsm_ids.txt); do echo \"Processing $gsm_id...\" && if [ \"$first\" = true ]; then esearch -db sra -query \"$gsm_id\" | efetch -format runinfo >> {geo_accession}_runinfo.csv && first=false; else esearch -db sra -query \"$gsm_id\" | efetch -format runinfo | tail -n +2 >> {geo_accession}_runinfo.csv; fi; done",
                    "parameters": {},
                    "dependencies": ["get_gsm_ids"],
                    "outputs": [f"raw_data/{geo_accession}_runinfo.csv"],
                    "description": "Download metadata for all GSM IDs",
                    "profile_name": "minimal"
                },
                {
                    "name": "extract_srr_ids",
                    "command": f"cd raw_data && tail -n +2 {geo_accession}_runinfo.csv | cut -d',' -f1 > {geo_accession}_srr_ids.txt",
                    "parameters": {},
                    "dependencies": ["download_geo_metadata"],
                    "outputs": [f"raw_data/{geo_accession}_srr_ids.txt"],
                    "description": "Extract SRR IDs from runinfo",
                    "profile_name": "minimal"
                },
                {
                    "name": "download_fastq_files",
                    "command": f"cd raw_data && cat {geo_accession}_srr_ids.txt | xargs -I{{}} sh -c 'prefetch {{}} && fasterq-dump {{}} --split-files && gzip {{}}*.fastq'",
                    "parameters": {},
                    "dependencies": ["extract_srr_ids"],
                    "outputs": ["raw_data/*.fastq.gz"],
                    "description": "Download and convert SRA files to FASTQ",
                    "profile_name": "minimal"
                },
                {
                    "name": "fastqc",
                    "command": f"cd raw_data && mkdir -p ../results/rna_seq_kallisto/fastqc && fastqc -o ../results/rna_seq_kallisto/fastqc *.fastq.gz",
                    "parameters": {},
                    "dependencies": ["download_fastq_files"],
                    "outputs": ["results/rna_seq_kallisto/fastqc"],
                    "description": "Run FastQC on FASTQ files",
                    "profile_name": "minimal"
                },
                {
                    "name": "kallisto_index",
                    "command": f"kallisto index -i {reference.rstrip('.')}.idx {reference.rstrip('.')}",
                    "parameters": {},
                    "dependencies": ["fastqc"],
                    "outputs": [f"{reference.rstrip('.')}.idx"],
                    "description": "Create kallisto index",
                    "profile_name": "high_memory"
                },
                {
                    "name": "kallisto_quant",
                    "command": f"for srr in $(cat raw_data/{geo_accession}_srr_ids.txt); do mkdir -p results/rna_seq_kallisto/kallisto_quant/$srr && kallisto quant -i {reference.rstrip('.')}.idx -o results/rna_seq_kallisto/kallisto_quant/$srr raw_data/${{srr}}_1.fastq.gz raw_data/${{srr}}_2.fastq.gz; done",
                    "parameters": {},
                    "dependencies": ["kallisto_index"],
                    "outputs": ["results/rna_seq_kallisto/kallisto_quant"],
                    "description": "Quantify transcripts using kallisto",
                    "profile_name": "multi_thread"
                },
                {
                    "name": "multiqc",
                    "command": "multiqc -o results/rna_seq_kallisto/qc results/rna_seq_kallisto/fastqc results/rna_seq_kallisto/kallisto_quant",
                    "parameters": {},
                    "dependencies": ["kallisto_quant"],
                    "outputs": ["results/rna_seq_kallisto/qc/multiqc_report.html"],
                    "description": "Generate MultiQC report",
                    "profile_name": "minimal"
                },
            ]
        }
        
        return workflow_plan

    async def analyze_run_prompt(self, prompt: str) -> Dict[str, Any]:
        """Analyze the prompt to extract options for running a workflow."""
        analysis_prompt = f"""
        Analyze the following prompt to extract options for running a workflow.
        Extract the following options if provided:
        - checkpoint_dir
        - resume

        Prompt: {prompt}

        Return the result as a JSON object with the following structure:
        {{
            "prompt": "original prompt",
            "checkpoint_dir": "extracted checkpoint directory" or null,
            "resume": true or false,
            "success": true or false
        }}

        "success" should be true if the prompt is successfully analyzed, false otherwise.

        Do not output any markdown formatting or additional text. Return only JSON.

        If the prompt given is entirely unrelated to running a workflow, set "success" to false.

        You should only set "success" to true when you are confident that the prompt is correctly analyzed.

        With the "run" action, you need a prompt, and optionally BOTH a checkpoint directory and 
        an indication the user does or doesn't want to resume an existing workflow.

        With the "analyze" action, you need a prompt, an analysis directory, optionally an indication 
        the user does or doesn't want to save an analysis report, and optionally an indication the
        user does or doesn't want to resume an existing workflow. 

        If you are being asked to generate a title, set "success" to false.
"""

        messages = [
            {
                "role": "system",
                "content": "You are an expert in analyzing prompts for workflow management.",
            },
            {"role": "user", "content": analysis_prompt},
        ]

        response = await self._call_openai(
            messages,
            model=settings.OPENAI_MODEL,
        )
        self.logger.info(f"Analysis response: {response}")
        analysis = json.loads(response)

        return analysis

    async def analyze_analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """Analyze the prompt to extract options for analyzing a workflow."""
        analysis_prompt = f"""
        Analyze the following prompt to extract options for analyzing a workflow.
        Extract the following options if provided:
        - analysis_dir
        - save_report

        Prompt: {prompt}

        Return the result as a JSON object with the following structure:
        {{
            "prompt": "original prompt",
            "analysis_dir": "extracted analysis directory" or null,
            "save_report": true or false,
            "success": true or false
        }}

        "success" should be true if the prompt is successfully analyzed, false otherwise.

        Do not output any markdown formatting or additional text. Return only JSON.

        If the prompt given is entirely unrelated to analyzing a workflow, set "success" to false.

        You should only set "success" to true when you are confident that the prompt is correctly analyzed.
        """

        messages = [
            {
                "role": "system",
                "content": "You are an expert in analyzing prompts for workflow management.",
            },
            {"role": "user", "content": analysis_prompt},
        ]

        response = await self._call_openai(
            messages,
            model=settings.OPENAI_MODEL,
        )
        self.logger.info(f"Analysis response: {response}")
        analysis = json.loads(response)

        return analysis

    def _get_workflow_prompt(
        self, workflow_type: str, input_files: List[str], reference: str
    ) -> str:
        """Get workflow-specific prompt."""
        if not input_files:
            return ""

        # Extract base sample name from first file
        sample_name = input_files[0]
        for suffix in [".fastq.1.gz", ".fastq.2.gz", ".fq.1.gz", ".fq.2.gz"]:
            if sample_name.endswith(suffix):
                sample_name = sample_name[: -len(suffix)]
                break

        if workflow_type == "rna_seq_kallisto":
            return f"""Generate a Kallisto RNA-seq workflow using:
Input files: {' '.join(input_files)}
Reference: {reference}
Sample name: {sample_name}

The workflow should:
1. Create output directories
2. Run FastQC on input files
3. Create Kallisto index
4. Run Kallisto quantification
5. Generate MultiQC report

Use the exact sample name '{sample_name}' for output directories."""
        else:
            return ""
