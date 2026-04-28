"""LLM interface for workflow generation and command creation."""

import asyncio
import glob
import json
import logging
import os
import platform
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from openai import AsyncOpenAI

from ..config.settings import Settings
from ..utils.logging import get_logger
from .providers import create_provider, LLMProvider
from .schemas import PipelineContext, WorkflowPlanSchema, to_json_schema

# Initialize settings
settings = Settings()

logger = get_logger(__name__)


class LLMInterface:
    """Interface for LLM-based workflow generation.

    Now delegates API calls to the provider abstraction layer so that
    OpenAI, Anthropic, Google Gemini, and Ollama all work identically.
    """

    def __init__(self):
        """Initialize LLM interface."""
        self.logger = get_logger(__name__)

        # Check for .env file (warn, don't hard-fail -- keys may come from env)
        env_found = False
        current_env_path = Path(".env")
        if current_env_path.exists():
            env_found = True
        if not env_found and "USER_EXECUTION_DIR" in os.environ:
            user_dir_env_path = Path(os.environ["USER_EXECUTION_DIR"]) / ".env"
            if user_dir_env_path.exists():
                from dotenv import load_dotenv
                load_dotenv(dotenv_path=user_dir_env_path)
                env_found = True
                self.logger.info(f"Loaded .env file from USER_EXECUTION_DIR: {user_dir_env_path}")

        if not env_found:
            self.logger.warning(
                "No .env file found. Ensure LLM API keys are set via environment variables."
            )

        # Build the provider using new multi-provider settings
        api_key = settings.active_api_key or settings.OPENAI_API_KEY
        if not api_key:
            self.logger.error(
                "\n⚠️  No API key found for provider '%s'."
                "\n   Set the appropriate key in .env or environment variables."
                "\n   OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY",
                settings.LLM_PROVIDER,
            )
            raise ValueError(f"Missing API key for LLM provider '{settings.LLM_PROVIDER}'")

        # Create multi-provider LLM backend
        self.provider: LLMProvider = create_provider(
            settings.LLM_PROVIDER,
            model=settings.LLM_MODEL,
            api_key=api_key,
            base_url=settings.LLM_BASE_URL or (
                settings.OPENAI_BASE_URL if settings.LLM_PROVIDER == "openai" else None
            ),
        )

        # Keep a raw OpenAI client for backwards-compat paths that use it directly
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY or "unused",
            base_url=settings.OPENAI_BASE_URL,
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
            "tools": ["fastqc", "kallisto", "tximport", "deseq2", "multiqc"],
            "dir_structure": [
                "results/rna_seq_kallisto/fastqc",
                "results/rna_seq_kallisto/kallisto_index",
                "results/rna_seq_kallisto/kallisto_quant",
                "results/rna_seq_kallisto/deseq2",
                "results/rna_seq_kallisto/qc",
            ],
            "rules": [
                "FastQC: fastqc file.fastq.gz -o results/rna_seq_kallisto/fastqc",
                "Kallisto index: kallisto index -i results/rna_seq_kallisto/kallisto_index/transcripts.idx reference.fa",
                "Kallisto quant paired: kallisto quant -i results/rna_seq_kallisto/kallisto_index/transcripts.idx -o results/rna_seq_kallisto/kallisto_quant/sample_name read1.fastq.gz read2.fastq.gz",
                "Kallisto quant single: kallisto quant -i results/rna_seq_kallisto/kallisto_index/transcripts.idx -o results/rna_seq_kallisto/kallisto_quant/sample_name --single -l 200 -s 20 read.fastq.gz",
                "tximport: Rscript -e 'library(tximport); library(rtracklayer); gtf <- rtracklayer::import(\"raw_data/reference/annotation.gtf\"); tx <- gtf[gtf$type == \"transcript\"]; tx2gene <- unique(data.frame(TXNAME=tx$transcript_id, GENEID=tx$gene_id)); tx2gene$TXNAME <- sub(\"[.].*$\", \"\", tx2gene$TXNAME); quant_dir <- \"results/rna_seq_kallisto/kallisto_quant\"; samples <- list.dirs(quant_dir, recursive=FALSE, full.names=FALSE); files <- setNames(file.path(quant_dir, samples, \"abundance.h5\"), samples); txi <- tximport(files, type=\"kallisto\", tx2gene=tx2gene, ignoreTxVersion=TRUE, ignoreAfterBar=TRUE); saveRDS(txi, \"results/rna_seq_kallisto/deseq2/txi.rds\")'",
                "DESeq2: Rscript -e 'library(DESeq2); txi <- readRDS(\"results/rna_seq_kallisto/deseq2/txi.rds\"); coldata <- read.table(\"sample_conditions.tsv\", header=TRUE, row.names=1, sep=\"\\t\"); coldata$condition <- factor(coldata$condition); dds <- DESeqDataSetFromTximport(txi, colData=coldata[colnames(txi$counts),,drop=FALSE], design=~condition); dds <- DESeq(dds); write.csv(as.data.frame(results(dds)), \"results/rna_seq_kallisto/deseq2/deseq2_results.csv\")'",
                "MultiQC: multiqc results/rna_seq_kallisto/fastqc results/rna_seq_kallisto/kallisto_quant -o results/rna_seq_kallisto/qc",
            ],
        },
        "rna_seq_star": {
            "keywords": [
                "rna-seq",
                "rnaseq",
                "star",
                "featurecounts",
                "deseq2",
                "differential expression",
            ],
            "tools": ["fastqc", "star", "samtools", "featureCounts", "deseq2", "multiqc"],
            "dir_structure": [
                "results/rna_seq_star/fastqc",
                "results/rna_seq_star/star_index",
                "results/rna_seq_star/star_align",
                "results/rna_seq_star/counts",
                "results/rna_seq_star/deseq2",
                "results/rna_seq_star/qc",
            ],
            "rules": [
                "FastQC: fastqc file.fastq.gz -o results/rna_seq_star/fastqc",
                "STAR index: STAR --runMode genomeGenerate --genomeDir results/rna_seq_star/star_index --genomeFastaFiles reference.fa --sjdbGTFfile annotation.gtf --runThreadN 8",
                "STAR align paired: STAR --runMode alignReads --genomeDir results/rna_seq_star/star_index --readFilesIn read1.fastq.gz read2.fastq.gz --readFilesCommand zcat --outSAMtype BAM SortedByCoordinate --outFileNamePrefix results/rna_seq_star/star_align/sample_ --runThreadN 8",
                "featureCounts: featureCounts -a annotation.gtf -o results/rna_seq_star/counts/counts.txt -p --countReadPairs -T 8 results/rna_seq_star/star_align/*Aligned.sortedByCoord.out.bam",
                "DESeq2: Rscript -e 'library(DESeq2); cts <- read.table(\"results/rna_seq_star/counts/counts.txt\", header=TRUE, row.names=1, sep=\"\\t\", comment.char=\"#\"); coldata <- read.table(\"sample_conditions.tsv\", header=TRUE, row.names=1); dds <- DESeqDataSetFromMatrix(countData=cts[,rownames(coldata)], colData=coldata, design=~condition); dds <- DESeq(dds); write.csv(as.data.frame(results(dds)), \"results/rna_seq_star/deseq2/deseq2_results.csv\")'",
                "MultiQC: multiqc results/rna_seq_star/fastqc results/rna_seq_star/star_align results/rna_seq_star/counts -o results/rna_seq_star/qc",
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

    # Primary analysis tools whose explicit mention in a prompt signals that
    # the user wants that specific tool, not a generic keyword-matched workflow.
    # Generic/supporting tools (fastqc, multiqc, samtools, picard, etc.) are
    # excluded because they appear across many workflow types.
    # Genomes we know how to fetch from Ensembl. When a prompt names one of
    # these keys, the STAR workflow inserts a ``download_reference`` step
    # that pulls the matching primary-assembly FASTA and GTF from Ensembl
    # so the user does not have to stage reference files manually.
    _GENOME_REFERENCES = {
        # prompt keyword (lowercase): (species, assembly, ensembl_release)
        # Mouse: GRCm39 is current; GRCm38 (mm10) was last in Ensembl
        # release-102 (Nov 2020) and is still widely cited.
        "grcm39": ("mus_musculus", "GRCm39", 112),
        "mm39":   ("mus_musculus", "GRCm39", 112),
        "grcm38": ("mus_musculus", "GRCm38", 102),
        "mm10":   ("mus_musculus", "GRCm38", 102),
        # Human: GRCh38 is current; GRCh37 (hg19) is still common in
        # clinical pipelines and was last in Ensembl release-75.
        "grch38": ("homo_sapiens", "GRCh38", 112),
        "hg38":   ("homo_sapiens", "GRCh38", 112),
        "grch37": ("homo_sapiens", "GRCh37", 75),
        "hg19":   ("homo_sapiens", "GRCh37", 75),
        "grcz11": ("danio_rerio", "GRCz11", 112),
        "wbcel235": ("caenorhabditis_elegans", "WBcel235", 112),
        "bdgp6":  ("drosophila_melanogaster", "BDGP6.46", 112),
        # Rat — GSE-style RNA-seq case studies sometimes use these
        "mratbn7":   ("rattus_norvegicus", "mRatBN7.2", 112),
        "rnor6":     ("rattus_norvegicus", "Rnor_6.0", 104),
    }

    _PRIMARY_TOOLS = {
        # Aligners / quantifiers / callers — the tools that define a workflow
        "kallisto", "salmon", "star", "starsolo", "hisat2", "bowtie2",
        "bwa", "minimap2", "bismark", "cellranger",
        "macs2", "gatk", "haplotypecaller", "bcftools",
        "featurecounts", "htseq-count", "htseq", "stringtie", "cufflinks",
        "kraken2", "bracken", "metaphlan",
        "manta", "delly", "medaka", "clair3",
        "kb-python", "fastp",
    }

    # Mapping of search patterns to canonical names (handles "trim galore" vs
    # "trim_galore", "bwa-mem" vs "bwa", etc.)
    _TOOL_ALIASES = {
        "trim galore": "trim_galore",
        "trim_galore": "trim_galore",
        "bwa-mem": "bwa",
        "haplotypecaller": "gatk",
        "kb-python": "kb",
        "htseq": "htseq-count",
    }

    def _detect_mentioned_tools(self, prompt: str) -> Set[str]:
        """Extract primary bioinformatics tools explicitly named in *prompt*."""
        prompt_lower = prompt.lower()
        found: Set[str] = set()
        # Check aliases first (multi-word patterns)
        for pattern, canonical in self._TOOL_ALIASES.items():
            if pattern in prompt_lower:
                found.add(canonical)
        # Check single-word primary tools
        for tool in self._PRIMARY_TOOLS:
            if tool in prompt_lower:
                found.add(tool)
        return found

    def _detect_reference_genome(self, prompt: str) -> Optional[Dict[str, str]]:
        """Return Ensembl URLs for a reference genome named in *prompt*.

        Looks for a case-insensitive whole-word match against
        ``_GENOME_REFERENCES``. Returns ``None`` if no known genome is
        mentioned — callers should then fall back to placeholder paths so
        the user can supply their own reference.
        """
        p = prompt.lower()
        for key, (species, assembly, release) in self._GENOME_REFERENCES.items():
            if re.search(rf"\b{re.escape(key)}\b", p):
                species_filename = species[0].upper() + species[1:]
                base = f"https://ftp.ensembl.org/pub/release-{release}"
                return {
                    "assembly": assembly,
                    "fasta_url": (
                        f"{base}/fasta/{species}/dna/"
                        f"{species_filename}.{assembly}.dna.primary_assembly.fa.gz"
                    ),
                    "cdna_url": (
                        f"{base}/fasta/{species}/cdna/"
                        f"{species_filename}.{assembly}.cdna.all.fa.gz"
                    ),
                    "gtf_url": (
                        f"{base}/gtf/{species}/"
                        f"{species_filename}.{assembly}.{release}.gtf.gz"
                    ),
                }
        return None

    def _custom_workflow_config(self) -> Dict[str, Any]:
        """Return the template config used for custom (free-form) workflows."""
        return {
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

    def _detect_workflow_type(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Detect workflow type from prompt.

        Uses keyword scoring to pick the best predefined workflow, then
        verifies that any primary tools explicitly named in the prompt are
        compatible with that workflow's tool list.  If the user asked for
        tools the matched workflow doesn't include, we fall back to a
        "custom" workflow so the LLM has full freedom.

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
            return "custom", self._custom_workflow_config()

        # ── Tool-compatibility guard ────────────────────────────────
        # If the prompt explicitly names primary analysis tools that are NOT
        # in the matched workflow's tool list, the keyword match is
        # misleading (e.g. prompt says "align with STAR" but keyword scoring
        # picked rna_seq_kallisto because of "RNA-seq").  Fall back to
        # "custom" so the LLM can use the tools the user actually asked for.
        mentioned = self._detect_mentioned_tools(prompt)
        if mentioned:
            wf_tools = {t.lower() for t in
                        self.WORKFLOW_TYPES[best_match[0]].get("tools", [])}
            # Only check primary/specific tools — ignore generic ones that
            # appear in many workflows (fastqc, multiqc, samtools …)
            _GENERIC = {"fastqc", "multiqc", "samtools", "picard", "bedtools",
                        "deeptools", "trim_galore"}
            specific_mentioned = mentioned - _GENERIC
            specific_workflow = wf_tools - _GENERIC
            if specific_mentioned and not specific_mentioned.issubset(specific_workflow):
                self.logger.info(
                    "Tool-compatibility guard: prompt mentions %s but %s "
                    "only provides %s — falling back to custom",
                    specific_mentioned, best_match[0], specific_workflow,
                )
                return "custom", self._custom_workflow_config()

        return best_match[0], self.WORKFLOW_TYPES[best_match[0]]

    async def _call_openai(
        self, messages: List[Dict[str, str]], model: Optional[str] = None,
        timeout: float = 120,
    ) -> str:
        """Call the configured LLM provider with retry / fallback logic.

        Despite the legacy name, this now routes through whichever provider
        is configured (OpenAI, Anthropic, Google, Ollama).

        Parameters
        ----------
        timeout : float
            Maximum seconds to wait for the API response (default 120).
        """
        try:
            resp = await asyncio.wait_for(
                self.provider.chat(messages, model=model),
                timeout=timeout,
            )
            self.logger.info("LLM call succeeded (provider=%s)", settings.LLM_PROVIDER)
            return resp.content
        except asyncio.TimeoutError:
            self.logger.error(
                "LLM API call timed out after %.0fs (provider=%s)",
                timeout, settings.LLM_PROVIDER,
            )
            raise
        except Exception as e:
            error_msg = str(e)
            if "insufficient_quota" in error_msg:
                self.logger.error("API quota exceeded – check billing. %s", error_msg)
            elif "model_not_found" in error_msg:
                self.logger.error("Model not found – check LLM_MODEL setting. %s", error_msg)
            else:
                self.logger.error("LLM API call failed: %s", error_msg)
            raise

    async def _call_openai_stream(
        self,
        messages: List[Dict[str, Any]],
        **kwargs,
    ):
        """Stream tokens from the configured LLM provider.

        Yields text chunks as they arrive.
        """
        async for token in self.provider.stream(messages):
            yield token

    def _clean_llm_response(self, response: str) -> str:
        """Clean LLM response and attempt to repair truncated JSON.

        Handles markdown fences, language identifiers, trailing garbage,
        and truncated JSON (missing closing brackets/braces).
        """
        text = response.strip()

        # Strip markdown code fences
        if "```" in text:
            # Extract content between first and last ```
            first = text.find("```")
            after_first_nl = text.find("\n", first)
            last = text.rfind("```")
            if after_first_nl != -1 and last > after_first_nl:
                text = text[after_first_nl + 1:last].strip()
            elif after_first_nl != -1:
                text = text[after_first_nl + 1:].strip()

        # Strip leading language tag (json, JSON, etc.)
        if text.lower().startswith("json"):
            text = text[4:].strip()

        # Find the outermost JSON object
        brace_start = text.find("{")
        if brace_start == -1:
            return text
        text = text[brace_start:]

        # Try parsing as-is first
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        # Trim trailing garbage after the last } or ]
        for i in range(len(text) - 1, -1, -1):
            if text[i] in ('}', ']'):
                candidate = text[:i + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    break

        # Repair truncated JSON by closing open brackets/braces
        repaired = self._repair_truncated_json(text)
        try:
            json.loads(repaired)
            self.logger.warning("Repaired truncated JSON from LLM response")
            return repaired
        except json.JSONDecodeError:
            pass

        return text

    @staticmethod
    def _repair_truncated_json(text: str) -> str:
        """Best-effort repair of JSON truncated mid-stream.

        Walks the string tracking open braces/brackets and string state,
        then appends the necessary closing tokens.
        """
        stack: list[str] = []
        in_string = False
        escape = False

        for ch in text:
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in ('{', '['):
                stack.append('}' if ch == '{' else ']')
            elif ch in ('}', ']'):
                if stack and stack[-1] == ch:
                    stack.pop()

        # If we're inside a string, close it
        if in_string:
            text += '"'

        # Strip trailing comma (invalid before a closing bracket)
        stripped = text.rstrip()
        if stripped.endswith(','):
            text = stripped[:-1]

        # Append missing closers in reverse order
        text += ''.join(reversed(stack))
        return text

    async def _suggest_tool_resources(
        self, tool_name: str, tool_description: str = ""
    ) -> Dict[str, str]:
        """Use LLM to suggest appropriate resource profile for unknown tools."""
        common_tools = {
            # Bioinformatics tools
            "fastqc": {"profile": "minimal", "reason": "Lightweight QC tool"},
            "multiqc": {"profile": "minimal", "reason": "Report aggregation"},
            "kallisto_index": {"profile": "high_memory", "reason": "Indexing requires memory"},
            "kallisto_quant": {"profile": "multi_thread", "reason": "Quantification benefits from parallelization"},
            "kallisto": {"profile": "multi_thread", "reason": "Quantification benefits from parallelization"},
            "salmon": {"profile": "multi_thread", "reason": "Quasi-mapping is CPU-intensive"},
            "hisat2": {"profile": "multi_thread", "reason": "Read alignment is CPU-intensive"},
            "star": {"profile": "high_memory_parallel", "reason": "STAR requires large memory for genome index"},
            "bwa": {"profile": "multi_thread", "reason": "Alignment is CPU-intensive"},
            "bowtie2": {"profile": "multi_thread", "reason": "Alignment is CPU-intensive"},
            "samtools": {"profile": "default", "reason": "BAM processing"},
            "bedtools": {"profile": "default", "reason": "Interval operations"},
            "picard": {"profile": "high_memory", "reason": "Java-based, needs heap"},
            "gatk": {"profile": "high_memory_parallel", "reason": "Variant calling is resource-intensive"},
            "bcftools": {"profile": "default", "reason": "VCF processing"},
            "macs2": {"profile": "high_memory", "reason": "Peak calling"},
            "trimmomatic": {"profile": "multi_thread", "reason": "Read trimming"},
            "trim_galore": {"profile": "default", "reason": "Lightweight trimming wrapper"},
            "cutadapt": {"profile": "default", "reason": "Adapter trimming"},
            "featurecounts": {"profile": "multi_thread", "reason": "Read counting"},
            "htseq": {"profile": "default", "reason": "Read counting"},
            "deseq2": {"profile": "high_memory", "reason": "Statistical testing in R"},
            "cellranger": {"profile": "high_memory_parallel", "reason": "Single-cell pipeline"},
            "prefetch": {"profile": "minimal", "reason": "SRA download utility"},
            "fasterq-dump": {"profile": "multi_thread", "reason": "Parallel FASTQ conversion"},
            # Shell / filesystem primitives -- never need an LLM call
            "mkdir": {"profile": "minimal", "reason": "Filesystem operation"},
            "ls": {"profile": "minimal", "reason": "Filesystem operation"},
            "find": {"profile": "minimal", "reason": "Filesystem operation"},
            "cat": {"profile": "minimal", "reason": "Filesystem operation"},
            "cp": {"profile": "minimal", "reason": "Filesystem operation"},
            "mv": {"profile": "minimal", "reason": "Filesystem operation"},
            "rm": {"profile": "minimal", "reason": "Filesystem operation"},
            "ln": {"profile": "minimal", "reason": "Filesystem operation"},
            "chmod": {"profile": "minimal", "reason": "Filesystem operation"},
            "head": {"profile": "minimal", "reason": "Filesystem operation"},
            "tail": {"profile": "minimal", "reason": "Filesystem operation"},
            "wc": {"profile": "minimal", "reason": "Filesystem operation"},
            "grep": {"profile": "minimal", "reason": "Text search"},
            "awk": {"profile": "minimal", "reason": "Text processing"},
            "sed": {"profile": "minimal", "reason": "Text processing"},
            "sort": {"profile": "minimal", "reason": "Text processing"},
            "cut": {"profile": "minimal", "reason": "Text processing"},
            "echo": {"profile": "minimal", "reason": "Shell built-in"},
            "touch": {"profile": "minimal", "reason": "Filesystem operation"},
            "tar": {"profile": "minimal", "reason": "Archive operation"},
            "gzip": {"profile": "minimal", "reason": "Compression"},
            "gunzip": {"profile": "minimal", "reason": "Decompression"},
            "wget": {"profile": "minimal", "reason": "File download"},
            "curl": {"profile": "minimal", "reason": "File download"},
            "create_directories": {"profile": "minimal", "reason": "Filesystem operation"},
            "create": {"profile": "minimal", "reason": "Filesystem operation"},
            "download": {"profile": "minimal", "reason": "File download"},
            "install": {"profile": "minimal", "reason": "Package installation"},
        }

        # Match by exact name first, then by the base (before first underscore)
        tool_lower = tool_name.lower()
        match = common_tools.get(tool_lower)
        if not match:
            tool_base = tool_lower.split("_")[0] if "_" in tool_lower else tool_lower
            match = common_tools.get(tool_base)
        if match:
            return {
                "profile_name": match["profile"],
                "reasoning": match["reason"],
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

    async def generate_workflow_plan(
        self, prompt: str, *, context: Optional[PipelineContext] = None,
    ) -> Dict[str, Any]:
        """Generate a workflow plan from a prompt.

        Parameters
        ----------
        context : PipelineContext, optional
            Pre-gathered planning context (organism, references, input files).
            When supplied the method skips its own file-scanning and injects
            reference information (including download steps) into the plan.
        """
        try:
            # Check for invalid prompt type
            if not isinstance(prompt, str):
                raise TypeError(f"Prompt must be a string, got {type(prompt).__name__}")
                
            # Extract file patterns and relationships using LLM
            file_info = await self._extract_file_patterns(prompt)
            
            # Check if this is a GEO download request
            geo_accession = file_info.get("geo_accession")

            # If a PipelineContext was provided, prefer its input_files
            if context and context.input_files:
                matched_files = list(context.input_files)
            else:
                # Find files matching the patterns
                matched_files = []
                for pattern in file_info["patterns"]:
                    matched_files.extend(glob.glob(pattern))

                # Fallback: if LLM patterns found nothing, scan cwd for
                # bioinformatics files with common (and uncommon) extensions.
                if not matched_files and not geo_accession:
                    _fallback_globs = [
                        "*.fastq.gz", "*.fq.gz", "*.fastq.bz2", "*.fq.bz2",
                        "*.fastq.*.gz",   # e.g. sample.fastq.1.gz
                        "*.fq.*.gz",
                        "*.fastq", "*.fq",
                        "*.bam", "*.sam", "*.cram",
                        "*.sra",
                    ]
                    for pat in _fallback_globs:
                        matched_files.extend(glob.glob(pat))
                    if matched_files:
                        matched_files = sorted(set(matched_files))
                        self.logger.info(
                            "LLM patterns missed files; fallback scan found %d file(s): %s",
                            len(matched_files), matched_files[:6],
                        )

            # If no files found and we have a GEO accession, we'll create a download workflow
            if not matched_files and geo_accession:
                self.logger.info(f"No local files found. Creating GEO download workflow for {geo_accession}")
                workflow_plan = await self._create_geo_download_workflow(geo_accession, prompt, file_info)
                
                # Check if the user wants to analyze the data after downloading
                prompt_lower = prompt.lower()
                analysis_keywords = ["analyze", "analysis", "run", "kallisto", "hisat", "star", "salmon", "quantify"]
                
                if any(keyword in prompt_lower for keyword in analysis_keywords) and "only download" not in prompt_lower and "just download" not in prompt_lower:
                    workflow_plan = await self._add_analysis_steps_to_geo_workflow(workflow_plan, prompt, file_info)
                
                return workflow_plan
            
            # If no files found and no GEO accession, raise an error
            # Special case for empty prompt in test environment
            if not matched_files:
                if not prompt or prompt.strip() == "":
                    enhanced_prompt = ""
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a bioinformatics workflow expert. Return only valid JSON.",
                        },
                        {"role": "user", "content": enhanced_prompt},
                    ]
                    
                    response = await self._call_openai(messages)
                    workflow_plan = json.loads(self._clean_llm_response(response))
                    return workflow_plan

                # Last resort: no GEO accession in the prompt and no
                # local files match. Ask the LLM to infer where the
                # data could be downloaded from based on the dataset
                # name — covers 10X Genomics public datasets, ENCODE,
                # GIAB / NIST, 1000 Genomes, TCGA/GDC, ArrayExpress,
                # and other public repos the LLM knows from training
                # data. If the LLM can't identify a source, raise a
                # clean error so the user knows to stage manually.
                self.logger.info(
                    "No local files + no GEO accession — asking LLM "
                    "to infer the data source from the prompt."
                )
                workflow_plan = await self._create_inferred_download_workflow(
                    prompt, file_info,
                )
                if workflow_plan is not None:
                    return workflow_plan
                patterns_str = ", ".join(file_info["patterns"])
                raise ValueError(
                    f"Could not infer a download source from the prompt "
                    f"and no local files match patterns: {patterns_str}. "
                    f"Either stage the data manually under raw_data/ or "
                    f"include a recognised accession (GSE…, ENCSR…) in "
                    f"the prompt."
                )

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
6. The bioinformatics tool (fastqc, kallisto, multiqc, etc.) MUST be the first token of the command. Do not prepend 'mkdir -p' or other shell prefixes; rely on the directory-creation step instead.
7. For multiqc, always pass '-f' (force overwrite) and '-n multiqc_report' (fixed filename) so re-runs produce the same output path.
8. Return ONLY the JSON object, no markdown formatting or other text
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

            # Inject reference context so the LLM uses correct paths
            _ref_supplement = ""
            if context:
                from .pipeline_planner import context_to_prompt_supplement
                _ref_supplement = "\nReference files:\n" + context_to_prompt_supplement(context)

            # Update the enhanced prompt with resource instructions
            enhanced_prompt = f"""
{enhanced_prompt}

{resource_instructions}
{_ref_supplement}
"""
            _os_hint = ""
            if platform.system() == "Darwin":
                _os_hint = (
                    " The host is macOS; use BSD-compatible shell commands"
                    " (no GNU extensions like find -printf)."
                    " IMPORTANT: wget is NOT available on macOS by default."
                    " Always use 'curl -fSL -o <file> <url>' instead of wget for downloads."
                )
            elif platform.system() == "Linux":
                _os_hint = (
                    " The host is Linux."
                    " Prefer curl over wget for downloads to maximise portability"
                    " (use 'curl -fSL -o <file> <url>')."
                )

            messages = [
                {
                    "role": "system",
                    "content": f"You are a bioinformatics workflow expert. Return only valid JSON.{_os_hint}",
                },
                {"role": "user", "content": enhanced_prompt},
            ]

            # Try structured output first, then plain chat, repairing JSON if needed
            workflow_plan = None
            last_err = None

            # Attempt 1: structured output (guaranteed JSON schema)
            try:
                schema = to_json_schema(WorkflowPlanSchema)
                resp = await self.provider.chat_structured(messages, schema)
                workflow_plan = json.loads(resp.content) if isinstance(resp.content, str) else resp.content
            except Exception as structured_err:
                self.logger.debug("Structured output failed (%s), trying plain chat", structured_err)
                last_err = structured_err

            # Attempt 2: plain chat + clean/repair
            if workflow_plan is None:
                try:
                    response = await self._call_openai(messages)
                    cleaned = self._clean_llm_response(response)
                    workflow_plan = json.loads(cleaned)
                except (json.JSONDecodeError, Exception) as parse_err:
                    self.logger.debug("First plain-chat parse failed (%s), retrying", parse_err)
                    last_err = parse_err

            # Attempt 3: retry with a shorter prompt asking for fewer details
            if workflow_plan is None:
                try:
                    retry_messages = [
                        messages[0],
                        {"role": "user", "content": (
                            f"Generate a workflow plan as a JSON object for: {prompt}\n"
                            f"Input files: {matched_files}\n"
                            "Return JSON with keys: workflow_type (string), "
                            "steps (array of objects with name, command, dependencies). "
                            "Return ONLY valid JSON, no markdown."
                        )},
                    ]
                    response = await self._call_openai(retry_messages)
                    cleaned = self._clean_llm_response(response)
                    workflow_plan = json.loads(cleaned)
                except Exception as retry_err:
                    self.logger.error("All JSON parse attempts failed")
                    raise last_err or retry_err

            # Prepend reference download steps if the context indicates
            # that references need to be fetched.
            if context:
                from .pipeline_planner import build_reference_download_steps
                dl_steps = build_reference_download_steps(context)
                if dl_steps:
                    dl_names = {s["name"] for s in dl_steps}
                    existing = workflow_plan.get("steps", [])
                    # Remove any LLM-generated steps whose name collides
                    # with the download steps we are about to prepend.
                    existing = [s for s in existing if s.get("name") not in dl_names]
                    # Wire index/align steps to depend on downloads
                    for step in existing:
                        cmd_lower = (step.get("command") or "").lower()
                        name_lower = (step.get("name") or "").lower()
                        needs_ref = any(kw in cmd_lower for kw in [
                            "index", "genome", "reference/", "transcriptome", "genes.gtf",
                        ]) or any(kw in name_lower for kw in [
                            "index", "genome",
                        ])
                        if needs_ref:
                            deps = step.get("dependencies", [])
                            for dl_name in dl_names:
                                if dl_name not in deps:
                                    deps.append(dl_name)
                            step["dependencies"] = deps
                    workflow_plan["steps"] = dl_steps + existing

            # Log workflow plan
            self.logger.info("Generated workflow plan:")
            self.logger.info(f"Workflow type: {workflow_plan.get('workflow_type', 'unknown')}")
            for step in workflow_plan.get("steps", []):
                self.logger.info(f"  Step: {step.get('name', '?')}")
                self.logger.info(f"    Command: {step.get('command', '?')}")
                self.logger.info(f"    Dependencies: {step.get('dependencies', [])}")

            # Update resources for each rule
            for i, step in enumerate(workflow_plan.get("steps", [])):
                workflow_plan["steps"][i] = await self._update_rule_resources(step)

            return workflow_plan

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
                    import fnmatch
                    group_files = [
                        f
                        for f in matched_files
                        if fnmatch.fnmatch(f, group["pattern"])
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
            analysis = json.loads(self._clean_llm_response(response))

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
        """Extract file patterns and relationships from a prompt.
        
        This method uses the LLM to extract file patterns, relationships, and other parameters
        from the user prompt. It also detects GEO accession numbers and references.
        
        Args:
            prompt: User prompt
            
        Returns:
            Dictionary with file patterns, relationships, and other parameters
        """
        # Check for invalid prompt type
        if not isinstance(prompt, str):
            self.logger.warning(f"Invalid prompt type: {type(prompt).__name__}, expected str")
            return {
                "patterns": ["*.fastq.gz", "*.fq.gz", "*.bam", "*.sam"],
                "reference": "",
                "paired_end": False
            }
            
        # Handle empty prompt
        if not prompt or prompt.strip() == "":
            return {
                "patterns": ["*.fastq.gz", "*.fq.gz", "*.bam", "*.sam"],
                "reference": "",
                "paired_end": False
            }
            
        # Check for GEO accession first
        geo_accession = await self._detect_geo_accession(prompt)
        
        # Extract reference file if mentioned
        reference_pattern = r'reference\s+(\S+)'
        reference_match = re.search(reference_pattern, prompt, re.IGNORECASE)
        reference = reference_match.group(1) if reference_match else ""
        
        # Detect if data is paired-end
        paired_end = False
        paired_patterns = ["paired[- ]end", "pair[- ]end", "paired", "PE"]
        if any(re.search(pattern, prompt, re.IGNORECASE) for pattern in paired_patterns):
            paired_end = True
        
        # If we found a GEO accession, we can return early with minimal patterns
        if geo_accession:
            result = {
                "patterns": ["*.fastq.gz", "*.fq.gz"],
                "geo_accession": geo_accession,
                "reference": reference,
                "paired_end": paired_end
            }
            
            # If paired-end, add relationship information
            if paired_end:
                result["relationships"] = {
                    "type": "paired",
                    "pattern_groups": [
                        {"group": "read1", "pattern": "*_1.fastq.gz"},
                        {"group": "read2", "pattern": "*_2.fastq.gz"}
                    ]
                }
            
            return result
        
        # For non-GEO requests, use the LLM to extract patterns
        messages = [
            {
                "role": "system",
                "content": """You are an expert in bioinformatics file pattern recognition.
                Extract file patterns and relationships from user prompts.
                Return only valid JSON.""",
            },
            {
                "role": "user",
                "content": f"""Extract file patterns and relationships from this prompt:
                {prompt}
                
                Return a JSON object with:
                1. "patterns": List of file patterns (e.g., "*.fastq.gz", "sample_*.bam")
                2. "relationships": Object describing relationships between patterns
                   - "type": "paired" for paired-end data
                   - "pattern_groups": List of objects with "group" and "pattern"
                
                Example for paired-end RNA-seq:
                {{
                  "patterns": ["*.fastq.gz", "*.fq.gz"],
                  "relationships": {{
                    "type": "paired",
                    "pattern_groups": [
                      {{"group": "read1", "pattern": "*_1.fastq.gz"}},
                      {{"group": "read2", "pattern": "*_2.fastq.gz"}}
                    ]
                  }}
                }}
                
                Return only the JSON object, no markdown or other text.""",
            },
        ]

        response = await self._call_openai(messages)
        
        try:
            # Clean and parse the response
            cleaned_response = self._clean_llm_response(response)
            result = json.loads(cleaned_response)
            
            # Add reference if found
            if reference:
                result["reference"] = reference
                
            # Add paired_end flag based on relationships
            if "relationships" in result and isinstance(result["relationships"], dict) and result["relationships"].get("type") == "paired":
                result["paired_end"] = True
            else:
                result["paired_end"] = paired_end
                
            return result
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse file patterns: {str(e)}")
            # Return default patterns if parsing fails
            return {
                "patterns": ["*.fastq.gz", "*.fq.gz", "*.bam", "*.sam"],
                "reference": reference,
                "paired_end": paired_end
            }

    async def _create_inferred_download_workflow(
        self, prompt: str, file_info: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Ask the LLM to infer a data source when there's no GEO accession.

        Used by ``generate_workflow_plan`` as the fallback when (a) no
        local files match the inferred patterns and (b) no recognised
        accession (GSE…) is present in the prompt. The LLM is asked to
        identify the dataset, find a download source from its training-
        data knowledge of public bioinformatics repositories
        (10X Genomics CDN, ENCODE, GIAB/NIST, 1000 Genomes, TCGA/GDC,
        ArrayExpress, EGA, …), and emit a complete workflow plan that
        starts with concrete download steps.

        Returns ``None`` if the LLM can't identify a clear source —
        the caller then raises a clean error rather than running a
        plan with hallucinated paths. Hallucinated placeholder paths
        like ``/path/to/your/...`` are filtered out before returning.
        """
        system = (
            "You are a senior bioinformatics workflow engineer. The "
            "user has requested a pipeline but the data is not present "
            "locally and no GEO accession was detected in the prompt. "
            "Your job is:\n"
            "1. Identify what dataset the user is asking about by name.\n"
            "2. Recall (from training-data knowledge) where it can be "
            "downloaded from a public repository.\n"
            "3. Emit a complete workflow plan that begins with concrete "
            "download steps (curl / wget) for that data source, then "
            "proceeds with the requested analysis.\n\n"
            "Common public bioinformatics data sources you should "
            "recognise:\n"
            "- 10X Genomics public datasets — https://cf.10xgenomics.com/"
            "samples/cell-exp/<version>/<dataset>/<dataset>_fastqs.tar\n"
            "- ENCODE — https://www.encodeproject.org/files/<ENCFF…>/"
            "@@download/<ENCFF…>.<ext>.gz\n"
            "- GIAB / NIST — https://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/\n"
            "- 1000 Genomes — https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/\n"
            "- TCGA / GDC — https://api.gdc.cancer.gov/data/<file_uuid>\n"
            "- ArrayExpress / Biostudies — https://www.ebi.ac.uk/"
            "biostudies/files/<E-…>/<file>\n"
            "- Sequence Read Archive (SRA) — via prefetch + "
            "fasterq-dump on an SRR accession\n"
            "- ENA — https://ftp.sra.ebi.ac.uk/vol1/fastq/<SRR-prefix>/<SRR>\n"
            "- HMP DACC — https://hmpdacc.org/hmp/\n"
            "- 4DN portal — https://data.4dnucleome.org/files-fastq/<accession>\n\n"
            "Return EXACTLY a JSON object with these keys:\n"
            "  workflow_type: a short string label (e.g. "
            "  'inferred_10x_pbmc', 'inferred_giab_hg001', 'inferred_encode')\n"
            "  data_source:  one-line description of where the data is from\n"
            "  steps:        list of step dicts with keys "
            "  (name, command, parameters, dependencies, outputs, "
            "  description, profile_name)\n\n"
            "If you CANNOT confidently identify a download source for "
            "the dataset named in the prompt, return JSON: "
            '{\"error\": \"<concise reason>\"} and the planner will '
            "fall back to manual data-staging.\n\n"
            "Hard rules:\n"
            "- NEVER emit placeholder paths like /path/to/your/... or "
            "<insert path here>. If you don't know the URL, return the "
            "error JSON instead.\n"
            "- The first step must always be ``create_directories`` "
            "with mkdir -p creating raw_data and results subdirs.\n"
            "- Download steps must use real, well-formed HTTPS URLs "
            "from your training data, not invented paths.\n"
            "- Subsequent analysis steps follow the same conventions as "
            "the GEO-download workflows: paths under raw_data/ for "
            "inputs, results/<workflow>/ for outputs.\n"
            "- Append a final multiqc step that aggregates per-step "
            "QC reports, when applicable."
        )
        user = (
            f"User request:\n{prompt}\n\n"
            f"Return only the JSON object — no markdown fences, no "
            f"preamble, no commentary."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        try:
            reply = await self._call_openai(messages)
        except Exception as exc:
            self.logger.warning(
                "LLM call for inferred-download workflow failed: %s", exc,
            )
            return None
        cleaned = self._clean_llm_response(reply or "")
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not m:
            self.logger.warning("Inferred-download reply has no JSON object")
            return None
        try:
            obj = json.loads(self._repair_truncated_json(m.group(0)))
        except json.JSONDecodeError as exc:
            self.logger.warning("Inferred-download JSON parse failed: %s", exc)
            return None
        if "error" in obj:
            self.logger.warning(
                "LLM declined to infer download source: %s",
                obj.get("error", "(no reason given)"),
            )
            return None
        steps = obj.get("steps") or []
        if not isinstance(steps, list) or not steps:
            return None

        # Defensive: reject plans that contain placeholder paths
        # (we hit this exact failure mode earlier — the LLM emitting
        # /path/to/your/reference.fa as if it were a real path).
        _PLACEHOLDER_RE = re.compile(
            r"/path/to/(your|the)/|<(insert|your|path|placeholder)|"
            r"<\.\.\.>|REPLACE_ME|TODO_PATH",
            flags=re.IGNORECASE,
        )
        for s in steps:
            cmd = str(s.get("command", ""))
            if _PLACEHOLDER_RE.search(cmd):
                self.logger.warning(
                    "Inferred-download plan rejected — placeholder path "
                    "detected in step %r: %s", s.get("name"),
                    _PLACEHOLDER_RE.search(cmd).group(0),
                )
                return None

        # Light validation — ensure required keys
        valid_steps: List[Dict[str, Any]] = []
        for s in steps:
            if not isinstance(s, dict):
                continue
            if not s.get("name") or not s.get("command"):
                continue
            s.setdefault("parameters", {})
            s.setdefault("dependencies", [])
            s.setdefault("outputs", [])
            s.setdefault("description", "")
            s.setdefault("profile_name", "default")
            valid_steps.append(s)
        if not valid_steps:
            return None
        self.logger.info(
            "Inferred-download workflow: %d steps from source '%s'",
            len(valid_steps), obj.get("data_source", "?"),
        )
        return {
            "workflow_type": obj.get("workflow_type", "inferred_download"),
            "steps": valid_steps,
        }

    async def _create_geo_download_workflow(self, geo_accession: str, prompt: str, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create a workflow plan for downloading data from GEO.

        Routes to supplementary-file download or raw FASTQ download depending
        on what the user asked for.
        """
        self.logger.info(f"Creating GEO download workflow for {geo_accession}")

        prompt_lower = prompt.lower()

        # Matched as whole words / phrases. Bare "counts" and "matrix" are
        # intentionally excluded because they show up inside tool names
        # (e.g. "featureCounts") and genome identifiers — use the explicit
        # phrases below instead.
        supplementary_indicators = [
            "no fastq", "no raw fastq", "no raw", "not raw",
            "supplementary", "suppl", "processed",
            "count matrix", "counts matrix", "count_matrix", "counts_matrix",
            "counts.tsv", "matrix.mtx",
            "barcodes.tsv", "features.tsv",
            "cell_metadata", "metadata",
        ]
        wants_supplementary = any(
            re.search(rf"\b{re.escape(ind)}\b", prompt_lower)
            for ind in supplementary_indicators
        )

        if wants_supplementary:
            self.logger.info("Detected request for supplementary/processed files (not raw FASTQ)")
            return self._create_geo_supplementary_download_workflow(geo_accession, prompt)

        return self._create_geo_fastq_download_workflow(geo_accession, prompt, file_info)

    def _geo_ftp_suppl_url(self, geo_accession: str) -> str:
        """Construct the NCBI FTP URL for a GEO series supplementary directory."""
        numeric = geo_accession.upper().replace("GSE", "")
        prefix = numeric[:-3] if len(numeric) > 3 else numeric
        return f"ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE{prefix}nnn/{geo_accession}/suppl/"

    def _geo_soft_url(self, geo_accession: str) -> str:
        """Construct the NCBI URL for a GEO series SOFT family file."""
        numeric = geo_accession.upper().replace("GSE", "")
        prefix = numeric[:-3] if len(numeric) > 3 else numeric
        return (
            f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE{prefix}nnn/"
            f"{geo_accession}/soft/{geo_accession}_family.soft.gz"
        )

    def _build_sample_sheet_step(self, geo_accession: str, prompt: str) -> Dict[str, Any]:
        """Build the ``build_sample_sheet`` step shared by STAR and Kallisto.

        Downloads the GEO SOFT family file and runs
        ``python -m flowagent.utils.build_sample_sheet`` to produce
        ``sample_conditions.tsv``. When condition labels can be extracted
        from the prompt (``between A and B``), the output is a two-column
        DESeq2-ready TSV; otherwise a template is written for manual fill-in.
        """
        labels = self._extract_condition_labels(prompt)
        labels_arg = f"--labels '{','.join(labels)}' " if labels else ""
        soft_url = self._geo_soft_url(geo_accession)
        return {
            "name": "build_sample_sheet",
            "command": (
                f"mkdir -p raw_data/metadata && "
                f"curl -fSL -o raw_data/metadata/family.soft.gz '{soft_url}' && "
                f"gunzip -f raw_data/metadata/family.soft.gz && "
                f"python3 -m flowagent.utils.build_sample_sheet "
                f"--soft raw_data/metadata/family.soft "
                f"--runinfo raw_data/{geo_accession}_runinfo.csv "
                f"{labels_arg}--out sample_conditions.tsv"
            ),
            "parameters": {},
            "dependencies": ["download_geo_metadata"],
            "outputs": ["sample_conditions.tsv"],
            "description": (
                f"Build sample_conditions.tsv from {geo_accession} SOFT metadata"
                + (f" using labels {labels}" if labels else " (template — fill in condition column)")
            ),
            "profile_name": "minimal",
        }

    def _extract_condition_labels(self, prompt: str) -> List[str]:
        """Pull a pair of condition labels out of a prompt.

        Recognises ``between X and Y``, ``X vs Y`` (and ``X vs. Y``), and
        ``compar(e|ing) X {and|to|with} Y``. Returns ``[]`` when only
        bracketed placeholders (``[condition A]``) are present or when the
        two labels are identical (meaning we probably matched a generic
        phrase like "compare condition and condition").
        """
        patterns = (
            r"between\s+([A-Za-z0-9_\-]+)\s+and\s+([A-Za-z0-9_\-]+)",
            r"\b([A-Za-z0-9_\-]+)\s+vs\.?\s+([A-Za-z0-9_\-]+)",
            r"compar(?:e|ing)\s+([A-Za-z0-9_\-]+)\s+(?:and|to|with)\s+([A-Za-z0-9_\-]+)",
        )
        for pat in patterns:
            m = re.search(pat, prompt, re.IGNORECASE)
            if m:
                a, b = m.group(1), m.group(2)
                if a.lower() == b.lower():
                    continue
                return [a, b]
        return []

    def _geo_download_page_url(self, geo_accession: str) -> str:
        """Construct the GEO bulk-download URL that returns a tar of all supplementary files."""
        return f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={geo_accession}&format=file"

    def _create_geo_supplementary_download_workflow(self, geo_accession: str, prompt: str) -> Dict[str, Any]:
        """Create a workflow to download supplementary (processed) files from GEO."""
        self.logger.info(f"Creating GEO supplementary file download workflow for {geo_accession}")

        ftp_url = self._geo_ftp_suppl_url(geo_accession)

        workflow_plan = {
            "workflow_type": "geo_download",
            "steps": [
                {
                    "name": "create_directories",
                    "command": "mkdir -p raw_data",
                    "parameters": {},
                    "dependencies": [],
                    "outputs": ["raw_data"],
                    "description": "Create directory structure",
                    "profile_name": "minimal"
                },
                {
                    "name": "download_supplementary_files",
                    "command": (
                        f"cd raw_data && "
                        f"curl -fSL -O '{ftp_url}'"
                    ),
                    "parameters": {},
                    "dependencies": ["create_directories"],
                    "outputs": ["raw_data/"],
                    "description": f"Download supplementary files from {geo_accession} via FTP",
                    "profile_name": "minimal"
                },
                {
                    "name": "list_downloaded_files",
                    "command": "ls -lh raw_data/",
                    "parameters": {},
                    "dependencies": ["download_supplementary_files"],
                    "outputs": [],
                    "description": "List downloaded files and verify",
                    "profile_name": "minimal"
                }
            ]
        }
        return workflow_plan

    def _create_geo_fastq_download_workflow(self, geo_accession: str, prompt: str, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create a workflow to download raw FASTQ files from GEO/SRA."""
        workflow_type, _ = self._detect_workflow_type(prompt)
        reference = file_info.get("reference", "")

        workflow_plan = {
            "workflow_type": "geo_download",
            "steps": [
                {
                    "name": "create_directories",
                    "command": "mkdir -p raw_data results",
                    "parameters": {},
                    "dependencies": [],
                    "outputs": ["raw_data"],
                    "description": "Create directory structure",
                    "profile_name": "minimal"
                },
                {
                    "name": "download_geo_metadata",
                    # ``elink`` is the canonical NCBI way to navigate from a
                    # GEO Series to the associated SRA runs. SRA's own
                    # ``[Accession]`` field only matches SRA-prefixed IDs
                    # (SRR/SRP/SRS/SRX) — it will NOT resolve a ``GSE…``
                    # accession, so a naive direct SRA query returns zero
                    # rows and every downstream step silently cascades to
                    # empty files. The elink hop via ``-db gds -> sra``
                    # resolves the cross-reference server-side and returns
                    # exactly the runs belonging to this series.
                    #
                    # ``set -e -o pipefail 2>/dev/null || set -e`` + a row-count guard surface silent
                    # NCBI failures instead of letting empty files cascade.
                    "command": (
                        f"cd raw_data && set -e -o pipefail 2>/dev/null || set -e && "
                        f"esearch -db gds -query '{geo_accession}[Accession]' | "
                        f"elink -target sra | "
                        f"efetch -format runinfo > {geo_accession}_runinfo.csv && "
                        f"[ $(wc -l < {geo_accession}_runinfo.csv) -gt 1 ] || "
                        f"{{ echo 'FAIL: runinfo has no data rows for {geo_accession}'; "
                        f"exit 1; }}"
                    ),
                    "parameters": {},
                    "dependencies": ["create_directories"],
                    "outputs": [f"raw_data/{geo_accession}_runinfo.csv"],
                    "description": "Fetch SRA runinfo via GDS→SRA elink (one query, no fan-out)",
                    "profile_name": "minimal"
                },
                {
                    "name": "extract_srr_ids",
                    # Header-aware, accession-filtered, deduped. The
                    # ``[ -s ... ]`` guard fails loudly if the resulting
                    # file is empty rather than cascading to silent
                    # zero-work downstream steps.
                    "command": (
                        f"cd raw_data && set -e -o pipefail 2>/dev/null || set -e && "
                        f"( if head -1 {geo_accession}_runinfo.csv | grep -q '^Run,'; then "
                        f"tail -n +2 {geo_accession}_runinfo.csv; "
                        f"else cat {geo_accession}_runinfo.csv; fi ) | "
                        f"cut -d',' -f1 | "
                        f"grep -E '^(SRR|ERR|DRR)[0-9]+$' | sort -u "
                        f"> {geo_accession}_srr_ids.txt && "
                        f"[ -s {geo_accession}_srr_ids.txt ] || "
                        f"{{ echo 'FAIL: no SRR/ERR/DRR ids extracted from runinfo'; "
                        f"exit 1; }}"
                    ),
                    "parameters": {},
                    "dependencies": ["download_geo_metadata"],
                    "outputs": [f"raw_data/{geo_accession}_srr_ids.txt"],
                    "description": "Extract SRR IDs from runinfo (deduped; fails loudly if empty)",
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
                }
            ]
        }
        return workflow_plan

    async def _add_analysis_steps_to_geo_workflow(self, workflow_plan: Dict[str, Any], prompt: str, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Add analysis steps to a GEO download workflow if requested.
        
        Args:
            workflow_plan: Existing workflow plan with GEO download steps
            prompt: Original user prompt
            file_info: File pattern information
            
        Returns:
            Updated workflow plan with analysis steps added
        """
        self.logger.info("Adding analysis steps to GEO download workflow")
        
        # Detect workflow type based on the prompt
        workflow_type, workflow_config = self._detect_workflow_type(prompt)
        
        # Extract reference from file_info or prompt
        reference = file_info.get("reference", "")
        if not reference:
            reference_pattern = r'reference\s+(\S+)'
            reference_match = re.search(reference_pattern, prompt, re.IGNORECASE)
            reference = reference_match.group(1) if reference_match else "reference.fa"
        
        # Get GEO accession from the workflow plan. Scan every step's command
        # and description for a GSE identifier — the old code only looked at
        # the "extract_srr_ids" step, which the supplementary-download path
        # never creates, so analysis steps were silently skipped.
        geo_accession = None
        for step in workflow_plan["steps"]:
            haystack = f"{step.get('command', '')} {step.get('description', '')}"
            match = re.search(r'\b(GSE\d+)\b', haystack)
            if match:
                geo_accession = match.group(1)
                break

        if not geo_accession:
            self.logger.warning("Could not extract GEO accession from workflow plan")
            return workflow_plan
            
        # Update workflow type to include analysis
        workflow_plan["workflow_type"] = f"geo_download_and_{workflow_type}"
        
        # Create directory structure for analysis
        dir_structure = " ".join(workflow_config["dir_structure"])
        for step in workflow_plan["steps"]:
            if step["name"] == "create_directories":
                step["command"] = f"{step['command']} {dir_structure}"
                break
        
        # Add analysis steps based on workflow type
        if workflow_type == "rna_seq_kallisto":
            genome = self._detect_reference_genome(prompt)
            if genome:
                transcriptome = "raw_data/reference/transcriptome.fa"
                self.logger.info(
                    "Detected reference genome %s — adding Ensembl download step",
                    genome["assembly"],
                )
                workflow_plan["steps"].append({
                    "name": "download_reference",
                    "command": (
                        f"mkdir -p raw_data/reference && cd raw_data/reference && "
                        f"curl -fSL -o transcriptome.fa.gz '{genome['cdna_url']}' && "
                        f"curl -fSL -o annotation.gtf.gz '{genome['gtf_url']}' && "
                        f"gunzip -f transcriptome.fa.gz annotation.gtf.gz"
                    ),
                    "parameters": {},
                    "dependencies": ["create_directories"],
                    "outputs": [transcriptome, "raw_data/reference/annotation.gtf"],
                    "description": (
                        f"Download {genome['assembly']} transcriptome cDNA and GTF from Ensembl"
                    ),
                    "profile_name": "minimal",
                })
                index_path = "raw_data/reference/transcriptome.idx"
                index_source = transcriptome
                index_deps = ["fastqc", "download_reference"]
            else:
                index_source = reference.rstrip('.')
                index_path = f"{index_source}.idx"
                index_deps = ["fastqc"]

            workflow_plan["steps"].append(
                self._build_sample_sheet_step(geo_accession, prompt)
            )

            workflow_plan["steps"].extend([
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
                    "command": f"kallisto index -i {index_path} {index_source}",
                    "parameters": {},
                    "dependencies": index_deps,
                    "outputs": [index_path],
                    "description": "Create kallisto index",
                    "profile_name": "high_memory"
                },
                {
                    "name": "kallisto_quant",
                    # Self-detecting paired vs single end. ``fasterq-dump
                    # --split-files`` writes ``<srr>_1.fastq.gz`` +
                    # ``<srr>_2.fastq.gz`` for paired libraries, but a
                    # single ``<srr>.fastq.gz`` for single-end ones (e.g.
                    # GSE60450 — Fu et al. 2015 used single-end Illumina).
                    # Without this branch the kallisto_quant step
                    # mis-fires on single-end studies with
                    # "file not found raw_data/<srr>_1.fastq.gz".
                    # The fragment-length defaults (-l 200 -s 20) are
                    # the kallisto-recommended fall-backs when the
                    # library spec isn't known.
                    "command": (
                        f"for srr in $(cat raw_data/{geo_accession}_srr_ids.txt); do "
                        f"mkdir -p results/rna_seq_kallisto/kallisto_quant/$srr && "
                        f"if [ -f raw_data/${{srr}}_1.fastq.gz ] && [ -f raw_data/${{srr}}_2.fastq.gz ]; then "
                        f"kallisto quant -i {index_path} "
                        f"-o results/rna_seq_kallisto/kallisto_quant/$srr "
                        f"raw_data/${{srr}}_1.fastq.gz raw_data/${{srr}}_2.fastq.gz; "
                        f"elif [ -f raw_data/${{srr}}.fastq.gz ]; then "
                        f"kallisto quant -i {index_path} "
                        f"-o results/rna_seq_kallisto/kallisto_quant/$srr "
                        f"--single -l 200 -s 20 raw_data/${{srr}}.fastq.gz; "
                        f"else echo \"FAIL: no FASTQ found for $srr\"; exit 1; "
                        f"fi; done"
                    ),
                    "parameters": {},
                    "dependencies": ["kallisto_index"],
                    "outputs": ["results/rna_seq_kallisto/kallisto_quant"],
                    "description": "Quantify transcripts using kallisto (auto-detects paired vs single end)",
                    "profile_name": "multi_thread"
                },
                {
                    "name": "tximport",
                    # Self-detecting HDF5 vs plaintext output. Conda
                    # kallisto packages on some platforms (notably the
                    # nebius docker image we benchmarked on) are built
                    # without HDF5 support — they silently emit only
                    # ``abundance.tsv`` + ``run_info.json``, no
                    # ``abundance.h5``. tximport defaults to the .h5
                    # path; if it's missing the call fails with no
                    # quantifications. The R block below picks the
                    # available format per-sample, so the same step
                    # works on both kallisto builds.
                    "command": (
                        "mkdir -p results/rna_seq_kallisto/deseq2 && "
                        "Rscript -e 'library(tximport); library(rtracklayer); "
                        "gtf <- rtracklayer::import(\"raw_data/reference/annotation.gtf\"); "
                        "tx <- gtf[gtf$type == \"transcript\"]; "
                        "tx2gene <- unique(data.frame(TXNAME=tx$transcript_id, GENEID=tx$gene_id)); "
                        "tx2gene$TXNAME <- sub(\"[.].*$\", \"\", tx2gene$TXNAME); "
                        "quant_dir <- \"results/rna_seq_kallisto/kallisto_quant\"; "
                        "samples <- list.dirs(quant_dir, recursive=FALSE, full.names=FALSE); "
                        "use_h5 <- all(file.exists(file.path(quant_dir, samples, \"abundance.h5\"))); "
                        "fname <- if (use_h5) \"abundance.h5\" else \"abundance.tsv\"; "
                        "files <- setNames(file.path(quant_dir, samples, fname), samples); "
                        "cat(sprintf(\"tximport: using %s format (%d samples)\\n\", fname, length(samples))); "
                        # tximport's ``type=\"kallisto\"`` auto-detects from the
                        # file extension — .h5 and .tsv both work.
                        "txi <- tximport(files, type=\"kallisto\", tx2gene=tx2gene, ignoreTxVersion=TRUE, ignoreAfterBar=TRUE); "
                        "saveRDS(txi, \"results/rna_seq_kallisto/deseq2/txi.rds\"); "
                        "write.csv(txi$counts, \"results/rna_seq_kallisto/deseq2/gene_counts.csv\")'"
                    ),
                    "parameters": {},
                    "dependencies": ["kallisto_quant"] + (["download_reference"] if genome else []),
                    "outputs": [
                        "results/rna_seq_kallisto/deseq2/txi.rds",
                        "results/rna_seq_kallisto/deseq2/gene_counts.csv",
                    ],
                    "description": "Aggregate kallisto transcript abundance to gene level with tximport",
                    "profile_name": "default"
                },
                {
                    "name": "generate_deseq2_script",
                    "command": (
                        "python -m flowagent.utils.generate_deseq2_script "
                        "--quant-dir results/rna_seq_kallisto/kallisto_quant "
                        "--sample-sheet sample_conditions.tsv "
                        "--gtf raw_data/reference/annotation.gtf "
                        "--txi results/rna_seq_kallisto/deseq2/txi.rds "
                        "--out-csv results/rna_seq_kallisto/deseq2/deseq2_results.csv "
                        f"--prompt {shlex.quote(prompt)} "
                        "--out scripts/run_deseq2.R"
                    ),
                    "parameters": {},
                    "dependencies": ["tximport", "build_sample_sheet"],
                    "outputs": ["scripts/run_deseq2.R"],
                    "description": "Generate a bespoke DESeq2 R script tailored to this dataset via the LLM (falls back to deterministic template if generation fails)",
                    "profile_name": "default"
                },
                {
                    "name": "deseq2",
                    "command": "Rscript scripts/run_deseq2.R",
                    "parameters": {},
                    "dependencies": ["generate_deseq2_script"],
                    "outputs": ["results/rna_seq_kallisto/deseq2/deseq2_results.csv"],
                    "description": "Differential expression with DESeq2 (LLM-generated script at scripts/run_deseq2.R)",
                    "profile_name": "default"
                },
                {
                    "name": "multiqc",
                    "command": "multiqc -o results/rna_seq_kallisto/qc results/rna_seq_kallisto/fastqc results/rna_seq_kallisto/kallisto_quant",
                    "parameters": {},
                    "dependencies": ["deseq2"],
                    "outputs": ["results/rna_seq_kallisto/qc/multiqc_report.html"],
                    "description": "Generate MultiQC report",
                    "profile_name": "minimal"
                }
            ])
        elif workflow_type == "rna_seq_star":
            genome = self._detect_reference_genome(prompt)
            if genome:
                ref = "raw_data/reference/genome.fa"
                gtf = "raw_data/reference/annotation.gtf"
                self.logger.info(
                    "Detected reference genome %s — adding Ensembl download step",
                    genome["assembly"],
                )
                workflow_plan["steps"].append({
                    "name": "download_reference",
                    "command": (
                        f"mkdir -p raw_data/reference && cd raw_data/reference && "
                        f"curl -fSL -o genome.fa.gz '{genome['fasta_url']}' && "
                        f"curl -fSL -o transcriptome.fa.gz '{genome['cdna_url']}' && "
                        f"curl -fSL -o annotation.gtf.gz '{genome['gtf_url']}' && "
                        f"gunzip -f genome.fa.gz transcriptome.fa.gz annotation.gtf.gz"
                    ),
                    "parameters": {},
                    "dependencies": ["create_directories"],
                    "outputs": [
                        ref,
                        "raw_data/reference/transcriptome.fa",
                        gtf,
                    ],
                    "description": (
                        f"Download {genome['assembly']} genome FASTA, transcriptome cDNA and GTF from Ensembl"
                    ),
                    "profile_name": "minimal",
                })
            else:
                ref = reference.rstrip('.') if reference else "reference.fa"
                gtf = file_info.get("annotation", "") or "annotation.gtf"

            workflow_plan["steps"].append(
                self._build_sample_sheet_step(geo_accession, prompt)
            )

            star_index_deps = ["fastqc"]
            if genome:
                star_index_deps.append("download_reference")

            workflow_plan["steps"].extend([
                {
                    "name": "fastqc",
                    "command": "cd raw_data && mkdir -p ../results/rna_seq_star/fastqc && fastqc -o ../results/rna_seq_star/fastqc *.fastq.gz",
                    "parameters": {},
                    "dependencies": ["download_fastq_files"],
                    "outputs": ["results/rna_seq_star/fastqc"],
                    "description": "Run FastQC on FASTQ files",
                    "profile_name": "minimal",
                },
                {
                    "name": "star_index",
                    "command": f"STAR --runMode genomeGenerate --genomeDir results/rna_seq_star/star_index --genomeFastaFiles {ref} --sjdbGTFfile {gtf} --runThreadN 8",
                    "parameters": {},
                    "dependencies": star_index_deps,
                    "outputs": ["results/rna_seq_star/star_index"],
                    "description": "Build STAR genome index",
                    "profile_name": "high_memory",
                },
                {
                    "name": "star_align",
                    "command": (
                        f"for srr in $(cat raw_data/{geo_accession}_srr_ids.txt); do "
                        f"STAR --runMode alignReads --genomeDir results/rna_seq_star/star_index "
                        f"--readFilesIn raw_data/${{srr}}_1.fastq.gz raw_data/${{srr}}_2.fastq.gz "
                        f"--readFilesCommand zcat --outSAMtype BAM SortedByCoordinate "
                        f"--outFileNamePrefix results/rna_seq_star/star_align/${{srr}}_ "
                        f"--runThreadN 8; done"
                    ),
                    "parameters": {},
                    "dependencies": ["star_index"],
                    "outputs": ["results/rna_seq_star/star_align"],
                    "description": "Align paired-end reads with STAR",
                    "profile_name": "multi_thread",
                },
                {
                    "name": "feature_counts",
                    "command": (
                        f"featureCounts -a {gtf} -o results/rna_seq_star/counts/counts.txt "
                        f"-p --countReadPairs -T 8 "
                        f"results/rna_seq_star/star_align/*Aligned.sortedByCoord.out.bam"
                    ),
                    "parameters": {},
                    "dependencies": ["star_align"],
                    "outputs": ["results/rna_seq_star/counts/counts.txt"],
                    "description": "Quantify gene-level counts with featureCounts",
                    "profile_name": "multi_thread",
                },
                {
                    "name": "deseq2",
                    "command": (
                        "Rscript -e 'library(DESeq2); "
                        "cts <- read.table(\"results/rna_seq_star/counts/counts.txt\", header=TRUE, row.names=1, sep=\"\\t\", comment.char=\"#\"); "
                        "cts <- cts[,6:ncol(cts)]; "
                        "coldata <- read.table(\"sample_conditions.tsv\", header=TRUE, row.names=1); "
                        "dds <- DESeqDataSetFromMatrix(countData=cts[,rownames(coldata)], colData=coldata, design=~condition); "
                        "dds <- DESeq(dds); "
                        "write.csv(as.data.frame(results(dds)), \"results/rna_seq_star/deseq2/deseq2_results.csv\")'"
                    ),
                    "parameters": {},
                    "dependencies": ["feature_counts", "build_sample_sheet"],
                    "outputs": ["results/rna_seq_star/deseq2/deseq2_results.csv"],
                    "description": "Differential expression with DESeq2 (sample_conditions.tsv built by build_sample_sheet step)",
                    "profile_name": "default",
                },
                {
                    "name": "multiqc",
                    "command": "multiqc -o results/rna_seq_star/qc results/rna_seq_star/fastqc results/rna_seq_star/star_align results/rna_seq_star/counts",
                    "parameters": {},
                    "dependencies": ["deseq2"],
                    "outputs": ["results/rna_seq_star/qc/multiqc_report.html"],
                    "description": "Generate MultiQC report",
                    "profile_name": "minimal",
                },
            ])
        elif workflow_type == "rna_seq_hisat":
            # Add RNA-seq analysis steps with HISAT2
            # Implementation for other workflow types would go here
            pass
        else:
            # No hardcoded recipe for this workflow_type (chip_seq,
            # single_cell_*, hic, custom, …). Delegate to LLM-driven
            # step generation, using the type's WORKFLOW_TYPES rules
            # as few-shot guidance — that's the architectural intent
            # behind storing example commands in WORKFLOW_TYPES.
            #
            # This mirrors how the local-files path does planning
            # (generate_workflow_plan around line 1060), so a ChIP-seq
            # prompt with a GEO accession produces the same calibre of
            # plan as a ChIP-seq prompt with FASTQs already on disk.
            try:
                rules = workflow_config.get("rules", []) if isinstance(workflow_config, dict) else []
                new_steps = await self._generate_analysis_steps_via_llm(
                    prompt=prompt,
                    geo_accession=geo_accession,
                    workflow_type=workflow_type,
                    rules=rules,
                    existing_steps=workflow_plan["steps"],
                )
                if new_steps:
                    workflow_plan["steps"].extend(new_steps)
                    self.logger.info(
                        "Added %d LLM-generated analysis steps for "
                        "workflow_type=%s", len(new_steps), workflow_type,
                    )
            except Exception as exc:
                self.logger.warning(
                    "LLM-driven analysis-step generation failed for "
                    "workflow_type=%s: %s — emitting download-only plan",
                    workflow_type, exc,
                )

        return workflow_plan

    async def _generate_analysis_steps_via_llm(
        self, *, prompt: str, geo_accession: str, workflow_type: str,
        rules: List[str], existing_steps: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Ask the LLM to produce analysis steps for an arbitrary assay.

        Used by the GEO-download path when the detected workflow_type
        isn't one of the three hardcoded RNA-seq recipes. The LLM gets
        the existing download steps (so it knows the FASTQs land at
        ``raw_data/<SRR>_*.fastq.gz``), the rules from
        ``WORKFLOW_TYPES[workflow_type]`` as exemplars, and the
        original natural-language prompt — and returns the analysis
        steps in the standard step-dict format.
        """
        download_summary = "\n".join(
            f"- {s.get('name', '?')}: {s.get('description', '')[:120]}"
            for s in existing_steps
        )
        rules_block = "\n".join(f"- {r}" for r in (rules or [])) or "(no examples)"
        # FASTQ layout note — depends on whether download_fastq_files
        # produced paired or single. We have no way to know in advance,
        # so explicitly tell the LLM to emit a self-detecting
        # paired-vs-single shell idiom (same trick we use for kallisto).
        system = (
            "You are a senior bioinformatics workflow engineer. Given a "
            "set of GEO-download steps that have already been added to a "
            "workflow plan, you must append the analysis steps required "
            "to fulfill the user's request. Emit ONLY a JSON object with "
            "a single key 'steps' whose value is a list of step dicts. "
            "Each step dict has keys: name, command, parameters, "
            "dependencies, outputs, description, profile_name."
        )
        user = (
            f"Workflow type: {workflow_type}\n"
            f"Original prompt: {prompt}\n\n"
            f"Existing download steps (already in the plan):\n"
            f"{download_summary}\n\n"
            f"FASTQs land at:\n"
            f"  paired-end: raw_data/<SRR>_1.fastq.gz + raw_data/<SRR>_2.fastq.gz\n"
            f"  single-end: raw_data/<SRR>.fastq.gz\n"
            f"Always emit shell idioms that detect both layouts at runtime\n"
            f"(``if [ -f raw_data/${{srr}}_1.fastq.gz ]; then ... else ... fi``)\n"
            f"so the same plan works for either library type.\n\n"
            f"Tool / command exemplars for {workflow_type}:\n"
            f"{rules_block}\n\n"
            f"Each step's ``dependencies`` field must reference one of the "
            f"existing step names above OR a previously-emitted step in "
            f"this list. Place outputs under "
            f"``results/{workflow_type}/<step-specific-subdir>/``.\n"
            f"Append a final ``multiqc`` step that aggregates all per-step "
            f"reports.\n"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        reply = await self._call_openai(messages)
        # Robust JSON extraction — the LLM occasionally wraps in ```json
        cleaned = self._clean_llm_response(reply)
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not m:
            self.logger.warning("LLM analysis-steps reply has no JSON object")
            return []
        try:
            obj = json.loads(self._repair_truncated_json(m.group(0)))
        except json.JSONDecodeError as exc:
            self.logger.warning("LLM analysis-steps JSON parse failed: %s", exc)
            return []
        steps = obj.get("steps", [])
        if not isinstance(steps, list):
            return []
        # Light validation — drop entries missing required keys.
        valid: List[Dict[str, Any]] = []
        for s in steps:
            if not isinstance(s, dict):
                continue
            if not s.get("name") or not s.get("command"):
                continue
            s.setdefault("parameters", {})
            s.setdefault("dependencies", [])
            s.setdefault("outputs", [])
            s.setdefault("description", "")
            s.setdefault("profile_name", "default")
            valid.append(s)
        return valid

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
