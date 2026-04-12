"""Pre-pipeline planning: scan the filesystem, detect gaps, ask the user.

``gather_pipeline_context()`` produces a :class:`PipelineContext` that is
passed into ``generate_workflow_plan()`` so the LLM receives concrete
reference paths / download URLs rather than guessing ``reference.fa``.
"""

from __future__ import annotations

import glob
import logging
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .reference_registry import (
    DEFAULT_BUILDS,
    ReferenceFiles,
    default_build,
    list_organisms,
    resolve_organism,
    resolve_references,
)
from .schemas import PipelineContext

logger = logging.getLogger(__name__)

# Extensions we recognise as sequence data
_INPUT_GLOBS = [
    "*.fastq.gz", "*.fq.gz", "*.fastq.bz2", "*.fq.bz2",
    "*.fastq.*.gz", "*.fq.*.gz",
    "*.fastq", "*.fq",
    "*.bam", "*.sam", "*.cram", "*.sra",
]

# Extensions we recognise as reference data
_REF_GLOBS = [
    "reference/*.fa", "reference/*.fa.gz", "reference/*.fasta", "reference/*.fasta.gz",
    "reference/*.idx", "reference/*.cdna.*",
    "*.fa", "*.fa.gz", "*.fasta", "*.fasta.gz",
    "genome/*.fa", "genome/*.fa.gz",
    "refs/*.fa", "refs/*.fa.gz",
]

_GTF_GLOBS = [
    "reference/*.gtf", "reference/*.gtf.gz",
    "reference/*.gff3", "reference/*.gff3.gz",
    "*.gtf", "*.gtf.gz",
    "genome/*.gtf", "genome/*.gtf.gz",
]

# Organism keywords found in prompts
_ORGANISM_KEYWORDS: Dict[str, str] = {
    "human": "human", "homo sapiens": "human", "homo_sapiens": "human",
    "mouse": "mouse", "mus musculus": "mouse", "mus_musculus": "mouse",
    "rat": "rat", "rattus": "rat",
    "zebrafish": "zebrafish", "danio": "zebrafish",
    "drosophila": "drosophila", "fruit fly": "drosophila",
    "c. elegans": "c_elegans", "c elegans": "c_elegans", "worm": "c_elegans",
}

# Workflow-type keywords that indicate a need for genome (not transcriptome)
_GENOME_WORKFLOWS = {"chip_seq", "chipseq", "atac_seq", "atacseq", "variant_calling", "wgs", "wes"}
# Workflows that need a transcriptome / cDNA (kallisto, salmon)
_CDNA_WORKFLOWS = {"rna_seq_kallisto", "rna_seq_salmon"}


def _scan_files(patterns: List[str]) -> List[str]:
    """Glob for files, dedup and sort."""
    found: List[str] = []
    for pat in patterns:
        found.extend(glob.glob(pat))
    return sorted(set(found))


def _detect_pairing(files: List[str]) -> bool:
    """Heuristic: paired-end if we see _R1/_R2 or _1/_2 patterns."""
    names = " ".join(files).lower()
    return bool(re.search(r"[_.]r?[12][_.]", names))


def _detect_organism_from_prompt(prompt: str) -> Optional[str]:
    """Extract organism from the user's free-text prompt."""
    lower = prompt.lower()
    for keyword, org in _ORGANISM_KEYWORDS.items():
        if keyword in lower:
            return org
    return None


def _detect_organism_from_files(files: List[str]) -> Optional[str]:
    """Try to guess organism from reference filenames."""
    joined = " ".join(f.lower() for f in files)
    hints = {
        "homo_sapiens": "human", "grch38": "human", "grch37": "human", "hg38": "human", "hg19": "human",
        "mus_musculus": "mouse", "grcm39": "mouse", "grcm38": "mouse", "mm10": "mouse", "mm39": "mouse",
        "danio_rerio": "zebrafish", "grcz11": "zebrafish",
        "drosophila": "drosophila", "bdgp6": "drosophila",
        "rattus": "rat", "mratbn7": "rat",
    }
    for hint, org in hints.items():
        if hint in joined:
            return org
    return None


def _detect_workflow_type_from_prompt(prompt: str) -> str:
    """Lightweight keyword detection for workflow type."""
    p = prompt.lower()
    if "kallisto" in p:
        return "rna_seq_kallisto"
    if "salmon" in p:
        return "rna_seq_salmon"
    if "star" in p or "hisat" in p:
        return "rna_seq_star"
    if "chip" in p:
        return "chip_seq"
    if "atac" in p:
        return "atac_seq"
    if "variant" in p or "gatk" in p:
        return "variant_calling"
    if "rna" in p:
        return "rna_seq_kallisto"
    return "custom"


def _needs_genome_fasta(workflow_type: str) -> bool:
    """Does this workflow need the full genome FASTA (not just cDNA)?"""
    return workflow_type in _GENOME_WORKFLOWS or workflow_type in {"rna_seq_star"}


def _needs_cdna(workflow_type: str) -> bool:
    return workflow_type in _CDNA_WORKFLOWS


def _needs_gtf(workflow_type: str) -> bool:
    return workflow_type not in _CDNA_WORKFLOWS


# ── Interactive prompting ─────────────────────────────────────

def _cli_ask(prompt_text: str, default: str = "") -> str:
    """Ask on stdin with a default value."""
    suffix = f" [{default}]: " if default else ": "
    try:
        answer = input(prompt_text + suffix).strip()
    except (EOFError, KeyboardInterrupt):
        answer = ""
    return answer or default


async def gather_pipeline_context(
    prompt: str,
    *,
    interactive: bool = True,
    ask_fn: Optional[Callable[[str, str], str]] = None,
    answers: Optional[Dict[str, str]] = None,
) -> PipelineContext:
    """Scan the filesystem and (optionally) ask the user to fill gaps.

    Parameters
    ----------
    prompt
        The user's original natural-language prompt.
    interactive
        If True and running in a TTY, ask questions via stdin.
        If False, use sensible defaults without prompting.
    ask_fn
        Optional override for the question function (for web UI).
        Signature: ``ask_fn(question_text, default) -> answer``.
    answers
        Pre-supplied answers keyed by field name (``organism``,
        ``genome_build``, ``reference_source``).  Skips asking for
        any key already present.
    """
    answers = dict(answers or {})
    if ask_fn is None and interactive and sys.stdin.isatty():
        ask_fn = _cli_ask
    # If not interactive and no ask_fn, we'll just use defaults.

    # ── 1. Discover input files ───────────────────────────────
    input_files = _scan_files(_INPUT_GLOBS)
    paired_end = _detect_pairing(input_files) if input_files else True

    # ── 2. Detect workflow type ───────────────────────────────
    workflow_type = _detect_workflow_type_from_prompt(prompt)

    # ── 3. Scan for existing reference files ──────────────────
    ref_files = _scan_files(_REF_GLOBS)
    gtf_files = _scan_files(_GTF_GLOBS)
    local_fasta = ref_files[0] if ref_files else None
    local_gtf = gtf_files[0] if gtf_files else None

    if local_fasta:
        logger.info("Found local reference: %s", local_fasta)
    if local_gtf:
        logger.info("Found local annotation: %s", local_gtf)

    # ── 4. Detect organism ────────────────────────────────────
    organism = (
        answers.get("organism")
        or _detect_organism_from_prompt(prompt)
        or _detect_organism_from_files(ref_files + gtf_files)
    )

    if not organism and ask_fn:
        organisms_str = ", ".join(list_organisms())
        organism = ask_fn(
            f"Organism? ({organisms_str})",
            "human",
        )
    organism = organism or "human"
    organism = resolve_organism(organism)

    # ── 5. Genome build ───────────────────────────────────────
    build = answers.get("genome_build") or default_build(organism)
    if not build and ask_fn:
        build = ask_fn("Genome build?", "")
    build = build or ""

    # ── 6. Reference source ───────────────────────────────────
    source = answers.get("reference_source") or "ensembl"
    if ask_fn and not answers.get("reference_source"):
        source = ask_fn("Reference source? (ensembl / gencode)", source)

    # ── 7. Resolve download URLs if we need them ──────────────
    ref_url: Optional[str] = None
    gtf_url: Optional[str] = None

    if not local_fasta or (not local_gtf and _needs_gtf(workflow_type)):
        refs = resolve_references(organism, build, source)
        if refs:
            if not local_fasta:
                if _needs_cdna(workflow_type) and refs.cdna_url:
                    ref_url = refs.cdna_url
                elif refs.fasta_url:
                    ref_url = refs.fasta_url
            if not local_gtf and _needs_gtf(workflow_type) and refs.gtf_url:
                gtf_url = refs.gtf_url

    ctx = PipelineContext(
        input_files=input_files,
        paired_end=paired_end,
        organism=organism,
        genome_build=build,
        reference_fasta=local_fasta,
        annotation_gtf=local_gtf,
        reference_source=source,
        reference_url=ref_url,
        annotation_url=gtf_url,
        workflow_type=workflow_type,
    )

    _log_context(ctx)
    return ctx


def _log_context(ctx: PipelineContext) -> None:
    logger.info("Pipeline context: organism=%s build=%s source=%s", ctx.organism, ctx.genome_build, ctx.reference_source)
    logger.info("  Input files: %d found (paired_end=%s)", len(ctx.input_files), ctx.paired_end)
    if ctx.reference_fasta:
        logger.info("  Local reference: %s", ctx.reference_fasta)
    elif ctx.reference_url:
        logger.info("  Will download reference: %s", ctx.reference_url[:100])
    if ctx.annotation_gtf:
        logger.info("  Local annotation: %s", ctx.annotation_gtf)
    elif ctx.annotation_url:
        logger.info("  Will download annotation: %s", ctx.annotation_url[:100])


def build_reference_download_steps(ctx: PipelineContext) -> List[Dict[str, Any]]:
    """Return workflow steps for downloading references, or [] if not needed."""
    steps: List[Dict[str, Any]] = []

    if ctx.reference_url:
        out_name = "reference/transcriptome.fa" if _needs_cdna(ctx.workflow_type) else "reference/genome.fa"
        steps.append({
            "name": "download_reference",
            "command": f"mkdir -p reference && wget -q -O {out_name}.gz {ctx.reference_url} && gunzip -f {out_name}.gz",
            "dependencies": [],
            "outputs": [out_name],
            "description": f"Download {ctx.organism} reference from {ctx.reference_source}",
            "resources": {"cpus": 1, "memory": "2G", "time_min": 30},
        })

    if ctx.annotation_url:
        steps.append({
            "name": "download_annotation",
            "command": f"mkdir -p reference && wget -q -O reference/genes.gtf.gz {ctx.annotation_url} && gunzip -f reference/genes.gtf.gz",
            "dependencies": [],
            "outputs": ["reference/genes.gtf"],
            "description": f"Download {ctx.organism} annotation from {ctx.reference_source}",
            "resources": {"cpus": 1, "memory": "2G", "time_min": 30},
        })

    return steps


def context_to_prompt_supplement(ctx: PipelineContext) -> str:
    """Produce extra text to append to the LLM planning prompt.

    Tells the LLM exactly which reference files are available (either local
    or will be downloaded) so it generates correct commands.
    """
    lines: List[str] = []

    if ctx.reference_fasta:
        lines.append(f"Reference FASTA (local): {ctx.reference_fasta}")
    elif ctx.reference_url:
        if _needs_cdna(ctx.workflow_type):
            lines.append("Reference transcriptome is at: reference/transcriptome.fa")
        else:
            lines.append("Reference genome is at: reference/genome.fa")
        lines.append("DO NOT create a download step -- it is handled externally.")
        lines.append("Steps that need the reference should list 'download_reference' in their dependencies.")

    if ctx.annotation_gtf:
        lines.append(f"Annotation GTF (local): {ctx.annotation_gtf}")
    elif ctx.annotation_url:
        lines.append("Annotation GTF is at: reference/genes.gtf")
        lines.append("DO NOT create a download step -- it is handled externally.")
        lines.append("Steps that need the annotation should list 'download_annotation' in their dependencies.")

    if not ctx.reference_fasta and not ctx.reference_url:
        lines.append("WARNING: No reference file found and no download URL resolved. "
                      "Include a reference download step or ask the user.")

    lines.append(f"Organism: {ctx.organism} ({ctx.genome_build})")
    lines.append(f"Paired-end: {ctx.paired_end}")

    return "\n".join(lines)
