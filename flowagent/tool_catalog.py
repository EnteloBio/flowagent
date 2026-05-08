"""Tool-catalog helpers used at planning *and* scoring time.

Originally lived in :mod:`benchmarks.harness.metrics` (scoring-only). Lifted
here so the planning hot path in :mod:`flowagent.core.llm` can reuse the
same known-tools snapshot, Damerau-Levenshtein typo classifier, and
runtime/shell-token sets — see todo T3 in the FlowAgent architecture
review.

Lives at the package root (``flowagent.tool_catalog``) rather than under
``flowagent.core`` so importing it from the benchmark scorer doesn't drag
in the core package's settings / dotenv loading side-effects, which would
otherwise leak ``.env`` values into test environments.

:mod:`benchmarks.harness.metrics` re-exports every public symbol from this
module so existing scoring code and tests continue to work unchanged.
"""

from __future__ import annotations

import functools
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple


# ── Token sets ──────────────────────────────────────────────────

_SHELL_TOKENS: Set[str] = {
    "mkdir", "cd", "rm", "mv", "cp", "ln", "touch", "test",
    "set", "export", "echo", "source", "bash", "sh", "for", "do",
    "done", "if", "then", "else", "fi", "while", "awk", "sed",
    "grep", "cut", "tr", "sort", "uniq", "head", "tail", "cat",
    "tee", "xargs", "time", "env", "printf", "read",
}

# Extended shell/infrastructure tokens beyond _SHELL_TOKENS — covers common
# runtime-glue commands that legitimately appear as first tokens in pipelines.
_RUNTIME_GLUE_TOKENS: Set[str] = {
    "parallel", "xargs", "find", "gsutil", "gcloud", "aws", "az",
    "kubectl", "helm", "terraform", "sbatch", "srun", "bsub", "qsub",
    "qstat", "squeue", "nohup", "screen", "tmux",
    "jq", "yq", "xmllint", "bc", "date", "hostname",
    "tar", "gzip", "gunzip", "zcat", "pigz", "bzip2", "xz",
    "bgzip", "tabix", "wget", "curl", "aria2", "aria2c", "rsync",
    "scp", "sftp", "cp", "mv", "rm", "ln", "mkdir", "touch",
    "chmod", "chown", "split", "join", "paste", "tee",
    "make", "cmake", "git", "gcc", "java", "docker", "singularity",
    "apptainer", "podman", "conda", "mamba", "micromamba", "pip",
    "pixi", "brew", "nextflow", "snakemake", "cromwell", "toil",
    "cwltool", "python", "python3", "python2", "rscript", "r",
    "julia", "perl", "ruby", "node", "bash", "sh", "zsh",
}

_RUNNER_TOKENS: Set[str] = {
    "rscript", "r", "python", "python3", "python2",
    "julia", "perl", "ruby", "node", "bash", "sh",
}


# Whitelist of real bioinformatics / download / infrastructure tools, used as
# a fallback when the generated known_tools.yaml snapshot isn't on disk.
# Names are normalised (lower-case, ``-`` → ``_``).
_BIOINFO_TOOLS: Set[str] = {
    # QC / trimming
    "fastqc", "multiqc", "trim_galore", "trimgalore", "fastp", "cutadapt",
    "trimmomatic", "bbduk", "atropos", "seqkit", "seqtk", "nanoplot",
    "nanofilt", "pycoqc", "longqc", "filtlong", "porechop",
    # aligners / mappers
    "bwa", "bwa_mem", "bwa_mem2", "bowtie", "bowtie2", "bowtie2_build",
    "star", "starsolo", "star_fusion", "hisat2", "hisat2_build",
    "minimap2", "tophat", "tophat2", "gsnap", "bbmap", "ngmlr", "segemehl",
    # quantification
    "kallisto", "salmon", "kb", "kb_python", "cellranger", "alevin",
    "alevin_fry", "rsem", "stringtie", "cufflinks", "htseq_count",
    "htseq", "featurecounts", "subread", "qualimap",
    # variant calling
    "gatk", "gatk4", "bcftools", "vcftools", "freebayes", "deepvariant",
    "varscan", "strelka", "mutect", "mutect2", "octopus",
    "platypus", "picard", "samtools", "bamtools", "vcf2maf", "snpeff",
    "snpsift", "vep",
    # structural variants / long read
    "manta", "delly", "lumpy", "gridss", "smoove", "svaba", "tiddit",
    "pindel", "medaka", "clair3", "longshot", "nanopolish", "pepper",
    "cnvkit", "qdnaseq", "control_freec", "cnvnator",
    # methylation
    "bismark", "deduplicate_bismark", "bismark_methylation_extractor",
    "bismark_genome_preparation", "methyldackel", "bsmap", "bwa_meth",
    # ChIP / ATAC / peak calling
    "macs", "macs2", "macs3", "homer", "seacr", "genrich", "epic2",
    "chromstar", "chip_r", "diffbind", "chipqc",
    # single-cell / spatial
    "alevin_fry", "starsolo", "cellranger", "spaceranger", "scanpy",
    "anndata", "cellranger_atac", "souporcell", "scvelo", "velocyto",
    # metagenomics
    "kraken", "kraken2", "bracken", "krakenuniq", "metaphlan",
    "centrifuge", "diamond", "mash", "sourmash", "krona", "checkm",
    "gtdbtk", "humann", "phyloflash",
    # assembly / annotation
    "spades", "abyss", "velvet", "trinity", "flye", "hifiasm", "canu",
    "wtdbg2", "miniasm", "unicycler", "quast", "busco", "prokka",
    "bakta", "maker", "augustus", "glimmer", "barrnap", "fastani",
    # R / Python wrappers + packages (via Rscript / python -m)
    "rscript", "python", "python3", "julia", "jupyter", "snakemake",
    "nextflow", "cromwell", "toil",
    "dada2", "tximport", "deseq2", "edger", "limma", "sleuth",
    "chipqc", "diffbind", "qdnaseq",
    # download / conversion / utility
    "wget", "curl", "aria2", "aria2c", "sra_toolkit", "fastq_dump",
    "fasterq_dump", "prefetch", "sratools", "sra_tools", "entrez_direct",
    "bedtools", "bedops", "vcfanno", "ucsc", "liftover",
    # infrastructure / runtime glue
    "tar", "gzip", "gunzip", "zcat", "pigz", "bgzip", "tabix",
    "java", "docker", "singularity", "apptainer", "conda", "mamba", "pip",
    "make", "cmake", "git",
    # imaging / misc
    "cooler", "pairtools", "pairix", "hicexplorer", "juicer", "hicpro",
    "snap_atac", "snaptools",
    # small-RNA
    "mirdeep", "mirdeep2", "srnabench", "mirge3",
    # amplicon
    "qiime", "qiime2", "vsearch", "usearch", "swarm",
    # annotation files
    "gff3sort",
}


# Common bio/data file extensions, used to recognise paths masquerading as
# "first tokens" after label stripping.
_FILE_EXT_RE = re.compile(
    r"\.(?:bam|sam|cram|crai|bai|fa|fasta|fna|ffn|faa|fai|gff|gff3|gtf|"
    r"bed|bedgraph|bw|bigwig|wig|vcf|vcf_gz|bcf|tbi|tsv|csv|txt|log|"
    r"fastq|fq|jsn|json|yaml|yml|html|pdf|png|jpg|idx|mtx|h5|h5ad|"
    r"loom|mcool|cool|pairs|narrowpeak|broadpeak|xls|xlsx|bed_gz)"
    r"(?:\.gz|\.bz2|\.xz)?$"
)


# ── Snapshot loading ────────────────────────────────────────────

# Primary snapshot location. Kept as a module-level constant (rather than
# hidden inside the loader) so tests can ``patch.object`` it to exercise the
# missing-file fallback path. Additional candidates are searched if this
# path doesn't exist.
_HERE = Path(__file__).resolve()
_PKG_ROOT = _HERE.parent  # flowagent/
_REPO_ROOT = _PKG_ROOT.parent
_SNAPSHOT_PATH = _REPO_ROOT / "benchmarks" / "data" / "known_tools.yaml"


def _candidate_snapshot_paths() -> List[Path]:
    """Locations to try, in order, for the known-tools YAML snapshot.

    Allows running both from the repo (where the snapshot lives under
    ``benchmarks/data/``) and from a future installed-package layout
    (``flowagent/data/``) without code changes. ``FLOWAGENT_KNOWN_TOOLS_PATH``
    overrides everything for tests / one-off runs.
    """
    candidates: List[Path] = []
    env = os.environ.get("FLOWAGENT_KNOWN_TOOLS_PATH")
    if env:
        candidates.append(Path(env))
    candidates.append(_SNAPSHOT_PATH)
    candidates.append(_PKG_ROOT / "data" / "known_tools.yaml")
    return candidates


def _normalise_tool_name(name: str) -> str:
    return name.lower().replace("-", "_")


@functools.lru_cache(maxsize=1)
def _load_known_tools() -> Set[str]:
    """Load the checked-in Bioconda/Bioconductor/runtime snapshot.

    Returns a normalised (lower-case, hyphens → underscores) set of known
    tool/package names. Falls back to :data:`_BIOINFO_TOOLS` (the legacy
    hand-curated set) if the snapshot file is absent, so existing test runs
    and CI jobs without the generated file still work correctly.
    """
    for path in _candidate_snapshot_paths():
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
            names: Set[str] = set()
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("- "):
                    names.add(_normalise_tool_name(line[2:].strip()))
            if names:
                return names
        except Exception:
            continue
    return {_normalise_tool_name(t) for t in _BIOINFO_TOOLS}


# ── Token classifier ────────────────────────────────────────────

def _edit_distance(a: str, b: str) -> int:
    """Damerau-Levenshtein distance (pure-Python fallback, O(mn))."""
    la, lb = len(a), len(b)
    prev2 = list(range(lb + 1))
    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)
    for i in range(1, la + 1):
        curr[0] = i
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                curr[j] = min(curr[j], prev2[j - 2] + cost)
        prev2, prev, curr = prev, curr, [0] * (lb + 1)
    return prev[lb]


def _classify_unknown_token(
    token: str,
    known: Optional[Set[str]] = None,
) -> Tuple[str, Optional[str]]:
    """Classify a token that was not found in the known-tools set.

    Returns ``(category, correction)`` where category is one of:

    * ``"filename"``     — token has a path separator or a biodata file
                          extension; not a hallucinated tool name.
    * ``"typo"``         — close Damerau-Levenshtein match to a known tool
                          (distance ≤ 2, both token and match ≥ 5 chars).
    * ``"runtime_glue"`` — common shell/infra word; not a bioinfo tool but
                          also not a hallucination.
    * ``"unknown"``      — genuinely unknown; true hallucination candidate.

    ``known`` defaults to :func:`_load_known_tools()`.
    """
    if known is None:
        known = _load_known_tools()

    n = _normalise_tool_name(token)

    # 1. Filename / path
    if "/" in token or _FILE_EXT_RE.search(n):
        return ("filename", None)

    # 2. Runtime glue
    if n in _RUNTIME_GLUE_TOKENS or n in _SHELL_TOKENS:
        return ("runtime_glue", None)

    # 3. Typo — fuzzy match against known tools (≥5 chars on both sides)
    if len(n) >= 5:
        try:
            from rapidfuzz.distance import DamerauLevenshtein  # type: ignore
            scorer = lambda cand: DamerauLevenshtein.distance(n, cand)
        except ImportError:
            scorer = lambda cand: _edit_distance(n, cand)

        best_match: Optional[str] = None
        best_dist = 3  # threshold: distance ≤ 2 → typo
        for cand in known:
            if len(cand) < 5:
                continue
            d = scorer(cand)
            if d < best_dist:
                best_dist = d
                best_match = cand
        if best_match is not None:
            return ("typo", best_match)

    # 4. Unknown
    return ("unknown", None)


def hallucinated_tools(
    plan_tools: Iterable[str],
) -> List[Tuple[str, str, Optional[str]]]:
    """Return classified hallucination candidates from ``plan_tools``.

    A tool is *recognised* (and therefore excluded from the result) if:
      * its normalised name is in the known-tools snapshot
        (:func:`_load_known_tools`), which covers Bioconda + Bioconductor +
        the curated runtime list, or
      * it is a runner token (``rscript``, ``python``, …), or
      * it is a family-prefix of any known entry (``bwa_mem2`` covers
        ``bwa``; ``hisat2_build`` covers ``hisat2``).

    Each unrecognised token is further *classified* via
    :func:`_classify_unknown_token` into one of four categories:

    * ``"typo"``         — probable misspelling of a known tool
    * ``"filename"``     — looks like a file path or has a biodata extension
    * ``"runtime_glue"`` — common shell/infra command, not a bioinfo tool
    * ``"unknown"``      — genuinely unknown (true hallucination candidate)

    Returns ``List[Tuple[token, category, correction]]`` where ``correction``
    is the closest known tool for typos, ``None`` otherwise.

    Callers that only need the names can use :func:`hallucinated_tool_names`.
    """
    known = _load_known_tools()
    out: List[Tuple[str, str, Optional[str]]] = []
    for t in plan_tools:
        n = _normalise_tool_name(t)
        if not n or n in _RUNNER_TOKENS or n in known:
            continue
        # Family-prefix match against the full snapshot set
        hit = False
        for w in known:
            if n.startswith(w + "_") or w.startswith(n + "_"):
                hit = True
                break
        if not hit and any(n.startswith(w) and len(w) >= 4 for w in known):
            hit = True
        if not hit:
            category, correction = _classify_unknown_token(t, known)
            out.append((t, category, correction))
    return out


def hallucinated_tool_names(plan_tools: Iterable[str]) -> List[str]:
    """Back-compat shim — return just the token strings from
    :func:`hallucinated_tools`.  Use when only the names are needed and
    category/correction metadata is irrelevant.
    """
    return [t for t, _cat, _corr in hallucinated_tools(plan_tools)]


__all__ = [
    "_BIOINFO_TOOLS",
    "_FILE_EXT_RE",
    "_RUNNER_TOKENS",
    "_RUNTIME_GLUE_TOKENS",
    "_SHELL_TOKENS",
    "_SNAPSHOT_PATH",
    "_classify_unknown_token",
    "_edit_distance",
    "_load_known_tools",
    "_normalise_tool_name",
    "hallucinated_tool_names",
    "hallucinated_tools",
]
