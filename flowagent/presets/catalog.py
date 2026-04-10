"""Catalog of pre-built workflow presets.

Each preset is a valid WorkflowPlan dict that can be passed directly to
pipeline generators or the executor without needing an LLM call.
"""

from typing import Any, Dict, List, Optional


PRESET_CATALOG: Dict[str, Dict[str, Any]] = {
    "rnaseq-kallisto": {
        "name": "RNA-seq (Kallisto pseudoalignment)",
        "description": "Standard RNA-seq quantification using FastQC, Kallisto, and MultiQC.",
        "workflow_type": "rna_seq_kallisto",
        "steps": [
            {
                "name": "create_directories",
                "command": "mkdir -p results/fastqc results/kallisto_index results/kallisto_quant results/multiqc",
                "dependencies": [],
                "outputs": ["results/"],
                "resources": {"cpus": 1, "memory": "1G", "time_min": 5},
            },
            {
                "name": "fastqc",
                "command": "fastqc -t 4 data/*.fastq.gz -o results/fastqc",
                "dependencies": ["create_directories"],
                "outputs": ["results/fastqc/*_fastqc.html", "results/fastqc/*_fastqc.zip"],
                "resources": {"cpus": 4, "memory": "2G", "time_min": 30},
            },
            {
                "name": "kallisto_index",
                "command": "kallisto index -i results/kallisto_index/transcripts.idx reference/transcriptome.fa",
                "dependencies": ["create_directories"],
                "outputs": ["results/kallisto_index/transcripts.idx"],
                "resources": {"cpus": 1, "memory": "8G", "time_min": 30},
            },
            {
                "name": "kallisto_quant",
                "command": "for sample in data/*_R1.fastq.gz; do base=$(basename $sample _R1.fastq.gz); kallisto quant -i results/kallisto_index/transcripts.idx -o results/kallisto_quant/${base} -t 4 data/${base}_R1.fastq.gz data/${base}_R2.fastq.gz; done",
                "dependencies": ["kallisto_index"],
                "outputs": ["results/kallisto_quant/*/abundance.tsv"],
                "resources": {"cpus": 4, "memory": "8G", "time_min": 120},
            },
            {
                "name": "multiqc",
                "command": "multiqc results/fastqc results/kallisto_quant -o results/multiqc",
                "dependencies": ["fastqc", "kallisto_quant"],
                "outputs": ["results/multiqc/multiqc_report.html"],
                "resources": {"cpus": 1, "memory": "2G", "time_min": 10},
            },
        ],
    },

    "rnaseq-star": {
        "name": "RNA-seq (STAR + featureCounts)",
        "description": "RNA-seq alignment with STAR and gene counting with featureCounts.",
        "workflow_type": "rna_seq_star",
        "steps": [
            {
                "name": "create_directories",
                "command": "mkdir -p results/fastqc results/star_index results/star_align results/counts results/multiqc",
                "dependencies": [],
                "outputs": ["results/"],
                "resources": {"cpus": 1, "memory": "1G", "time_min": 5},
            },
            {
                "name": "fastqc",
                "command": "fastqc -t 4 data/*.fastq.gz -o results/fastqc",
                "dependencies": ["create_directories"],
                "outputs": ["results/fastqc/*_fastqc.html"],
                "resources": {"cpus": 4, "memory": "2G", "time_min": 30},
            },
            {
                "name": "star_index",
                "command": "STAR --runMode genomeGenerate --genomeDir results/star_index --genomeFastaFiles reference/genome.fa --sjdbGTFfile reference/genes.gtf --runThreadN 8",
                "dependencies": ["create_directories"],
                "outputs": ["results/star_index/SA"],
                "resources": {"cpus": 8, "memory": "32G", "time_min": 60},
            },
            {
                "name": "star_align",
                "command": "for sample in data/*_R1.fastq.gz; do base=$(basename $sample _R1.fastq.gz); STAR --genomeDir results/star_index --readFilesIn data/${base}_R1.fastq.gz data/${base}_R2.fastq.gz --readFilesCommand zcat --outSAMtype BAM SortedByCoordinate --runThreadN 8 --outFileNamePrefix results/star_align/${base}_; done",
                "dependencies": ["star_index"],
                "outputs": ["results/star_align/*_Aligned.sortedByCoord.out.bam"],
                "resources": {"cpus": 8, "memory": "32G", "time_min": 180},
            },
            {
                "name": "featurecounts",
                "command": "featureCounts -a reference/genes.gtf -o results/counts/counts.txt -T 4 -p --countReadPairs results/star_align/*_Aligned.sortedByCoord.out.bam",
                "dependencies": ["star_align"],
                "outputs": ["results/counts/counts.txt"],
                "resources": {"cpus": 4, "memory": "8G", "time_min": 30},
            },
            {
                "name": "multiqc",
                "command": "multiqc results/ -o results/multiqc",
                "dependencies": ["fastqc", "star_align", "featurecounts"],
                "outputs": ["results/multiqc/multiqc_report.html"],
                "resources": {"cpus": 1, "memory": "2G", "time_min": 10},
            },
        ],
    },

    "chipseq": {
        "name": "ChIP-seq (Bowtie2 + MACS2)",
        "description": "ChIP-seq alignment and peak calling.",
        "workflow_type": "chip_seq",
        "steps": [
            {
                "name": "create_directories",
                "command": "mkdir -p results/fastqc results/trimmed results/aligned results/peaks results/multiqc",
                "dependencies": [],
                "outputs": ["results/"],
                "resources": {"cpus": 1, "memory": "1G", "time_min": 5},
            },
            {
                "name": "fastqc",
                "command": "fastqc -t 4 data/*.fastq.gz -o results/fastqc",
                "dependencies": ["create_directories"],
                "outputs": ["results/fastqc/*_fastqc.html"],
                "resources": {"cpus": 4, "memory": "2G", "time_min": 30},
            },
            {
                "name": "trim_galore",
                "command": "for f in data/*.fastq.gz; do trim_galore --cores 4 -o results/trimmed $f; done",
                "dependencies": ["create_directories"],
                "outputs": ["results/trimmed/*_trimmed.fq.gz"],
                "resources": {"cpus": 4, "memory": "4G", "time_min": 60},
            },
            {
                "name": "bowtie2_align",
                "command": "for f in results/trimmed/*_trimmed.fq.gz; do base=$(basename $f _trimmed.fq.gz); bowtie2 -x reference/genome -U $f -S results/aligned/${base}.sam -p 8; samtools sort -@ 4 results/aligned/${base}.sam -o results/aligned/${base}.bam; samtools index results/aligned/${base}.bam; rm results/aligned/${base}.sam; done",
                "dependencies": ["trim_galore"],
                "outputs": ["results/aligned/*.bam"],
                "resources": {"cpus": 8, "memory": "16G", "time_min": 120},
            },
            {
                "name": "macs2_callpeak",
                "command": "macs2 callpeak -t results/aligned/treatment.bam -c results/aligned/control.bam -f BAM -g hs --outdir results/peaks -n experiment",
                "dependencies": ["bowtie2_align"],
                "outputs": ["results/peaks/*_peaks.narrowPeak"],
                "resources": {"cpus": 2, "memory": "8G", "time_min": 30},
            },
            {
                "name": "multiqc",
                "command": "multiqc results/ -o results/multiqc",
                "dependencies": ["fastqc", "bowtie2_align", "macs2_callpeak"],
                "outputs": ["results/multiqc/multiqc_report.html"],
                "resources": {"cpus": 1, "memory": "2G", "time_min": 10},
            },
        ],
    },

    "atacseq": {
        "name": "ATAC-seq (Bowtie2 + MACS2)",
        "description": "ATAC-seq analysis with alignment, filtering, and peak calling.",
        "workflow_type": "atac_seq",
        "steps": [
            {
                "name": "create_directories",
                "command": "mkdir -p results/fastqc results/trimmed results/aligned results/filtered results/peaks results/multiqc",
                "dependencies": [],
                "outputs": ["results/"],
                "resources": {"cpus": 1, "memory": "1G", "time_min": 5},
            },
            {
                "name": "fastqc",
                "command": "fastqc -t 4 data/*.fastq.gz -o results/fastqc",
                "dependencies": ["create_directories"],
                "outputs": ["results/fastqc/*_fastqc.html"],
                "resources": {"cpus": 4, "memory": "2G", "time_min": 30},
            },
            {
                "name": "trim_galore",
                "command": "for f in data/*_R1.fastq.gz; do base=$(basename $f _R1.fastq.gz); trim_galore --cores 4 --paired data/${base}_R1.fastq.gz data/${base}_R2.fastq.gz -o results/trimmed; done",
                "dependencies": ["create_directories"],
                "outputs": ["results/trimmed/*_val_*.fq.gz"],
                "resources": {"cpus": 4, "memory": "4G", "time_min": 60},
            },
            {
                "name": "bowtie2_align",
                "command": "for f in results/trimmed/*_R1_val_1.fq.gz; do base=$(basename $f _R1_val_1.fq.gz); bowtie2 -x reference/genome -1 results/trimmed/${base}_R1_val_1.fq.gz -2 results/trimmed/${base}_R2_val_2.fq.gz --very-sensitive -X 2000 -p 8 | samtools sort -@ 4 -o results/aligned/${base}.bam; samtools index results/aligned/${base}.bam; done",
                "dependencies": ["trim_galore"],
                "outputs": ["results/aligned/*.bam"],
                "resources": {"cpus": 8, "memory": "16G", "time_min": 120},
            },
            {
                "name": "filter_and_dedup",
                "command": "for f in results/aligned/*.bam; do base=$(basename $f .bam); samtools view -b -q 30 -F 1804 -f 2 $f | samtools sort -@ 4 -o results/filtered/${base}.filt.bam; samtools index results/filtered/${base}.filt.bam; done",
                "dependencies": ["bowtie2_align"],
                "outputs": ["results/filtered/*.filt.bam"],
                "resources": {"cpus": 4, "memory": "8G", "time_min": 60},
            },
            {
                "name": "macs2_callpeak",
                "command": "macs2 callpeak -t results/filtered/*.filt.bam -f BAMPE -g hs --nomodel --shift -100 --extsize 200 --outdir results/peaks -n atacseq",
                "dependencies": ["filter_and_dedup"],
                "outputs": ["results/peaks/*_peaks.narrowPeak"],
                "resources": {"cpus": 2, "memory": "8G", "time_min": 30},
            },
            {
                "name": "multiqc",
                "command": "multiqc results/ -o results/multiqc",
                "dependencies": ["fastqc", "bowtie2_align", "filter_and_dedup", "macs2_callpeak"],
                "outputs": ["results/multiqc/multiqc_report.html"],
                "resources": {"cpus": 1, "memory": "2G", "time_min": 10},
            },
        ],
    },
}


def list_presets() -> List[Dict[str, str]]:
    """Return summary of available presets."""
    return [
        {"id": k, "name": v["name"], "description": v["description"]}
        for k, v in PRESET_CATALOG.items()
    ]


def get_preset(preset_id: str) -> Optional[Dict[str, Any]]:
    """Return a preset workflow plan by ID, or None."""
    return PRESET_CATALOG.get(preset_id)
