"""Scoring functions for benchmark results.

All functions take primitive types (dicts, strings) and return dicts with
scalar fields, so results serialise cleanly to CSV for figure generation.
"""

from __future__ import annotations

import re
import shlex
from typing import Any, Dict, Iterable, List, Optional, Set

try:
    import networkx as nx  # type: ignore
except ImportError:  # pragma: no cover
    nx = None


# ── Basic utilities ──────────────────────────────────────────────

def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Jaccard similarity |A ∩ B| / |A ∪ B|."""
    a_set, b_set = set(a), set(b)
    if not a_set and not b_set:
        return 1.0
    union = a_set | b_set
    return len(a_set & b_set) / len(union) if union else 0.0


def command_token_f1(cmd_a: str, cmd_b: str) -> float:
    """Token-level F1 between two shell commands.

    Uses ``shlex.split`` so quoted args are handled correctly.
    """
    try:
        ta = set(shlex.split(cmd_a or ""))
        tb = set(shlex.split(cmd_b or ""))
    except ValueError:
        # Malformed quoting -- fall back to whitespace split
        ta = set((cmd_a or "").split())
        tb = set((cmd_b or "").split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    tp = len(ta & tb)
    precision = tp / len(ta) if ta else 0.0
    recall = tp / len(tb) if tb else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Plan introspection ───────────────────────────────────────────

_SHELL_TOKENS = {
    "mkdir", "cd", "rm", "mv", "cp", "ln", "touch", "test",
    "set", "export", "echo", "source", "bash", "sh", "for", "do",
    "done", "if", "then", "else", "fi", "while", "awk", "sed",
    "grep", "cut", "tr", "sort", "uniq", "head", "tail", "cat",
    "tee", "xargs", "time", "env", "printf", "read",
}


# Whitelist of real bioinformatics / download / infrastructure tools for the
# hallucination check.  Names are normalised (lower-case, ``-`` → ``_``).
# Keep conservative — a small whitelist will under-credit real tools; a
# bloated one will mask hallucinations. Additions are always welcome.
_BIOINFO_TOOLS = {
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
    # infrastructure / runtime glue (not bioinfo tools but legitimate
    # command-line invocations that can appear as "first tokens")
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


def _normalise_tool_name(name: str) -> str:
    return name.lower().replace("-", "_")


def hallucinated_tools(plan_tools: Iterable[str]) -> List[str]:
    """Return the subset of ``plan_tools`` not recognised as real CLI tools.

    A tool is recognised if:
      * its normalised name is in :data:`_BIOINFO_TOOLS`, or
      * it is a family-prefix of a whitelist entry (``bwa_mem2`` covers
        ``bwa``; ``hisat2_build`` covers ``hisat2``), or
      * it is a runner token (``rscript``, ``python``, …).

    Everything else — typos (``kallsito``), made-up names
    (``super_aligner_pro``), filenames (``raw_data``), etc. — is flagged
    as a hallucination candidate.
    """
    out: List[str] = []
    for t in plan_tools:
        n = _normalise_tool_name(t)
        if not n or n in _RUNNER_TOKENS or n in _BIOINFO_TOOLS:
            continue
        # Family-prefix: any whitelist entry that is a prefix-component match
        hit = False
        for w in _BIOINFO_TOOLS:
            if n == w or n.startswith(w + "_") or w.startswith(n + "_"):
                hit = True
                break
        # Core-family fallback (e.g. ``bowtie2-build`` ≈ ``bowtie2``)
        if not hit and any(
            n.startswith(w) and len(w) >= 4 for w in _BIOINFO_TOOLS
        ):
            hit = True
        if not hit:
            out.append(t)
    return out


# Labels from the FlowAgent system prompt (``"Sort BAM: samtools sort ..."``)
# — some LLMs copy the colon, others drop it. Both variants must be stripped.
#
# The colonless pattern requires ≥2 capitalised words *and* a lowercase / path
# char immediately after, so single capitalised tool names (``GATK``, ``STAR``)
# aren't mistakenly treated as labels.
# Common bio/data file extensions, used to recognise paths masquerading as
# "first tokens" after label stripping.
_FILE_EXT_RE = re.compile(
    r"\.(?:bam|sam|cram|crai|bai|fa|fasta|fna|ffn|faa|fai|gff|gff3|gtf|"
    r"bed|bedgraph|bw|bigwig|wig|vcf|vcf_gz|bcf|tbi|tsv|csv|txt|log|"
    r"fastq|fq|jsn|json|yaml|yml|html|pdf|png|jpg|idx|mtx|h5|h5ad|"
    r"loom|mcool|cool|pairs|narrowpeak|broadpeak|xls|xlsx|bed_gz)"
    r"(?:\.gz|\.bz2|\.xz)?$"
)


_LABEL_PREFIX_RE = re.compile(
    r"^(?:"
    r"[A-Za-z][A-Za-z0-9 _/\\-]{0,50}:\s+"                           # "Sort BAM: …"
    r"|"
    r"[A-Z][A-Za-z0-9]+(?:\s+[A-Za-z0-9][A-Za-z0-9_-]*){1,5}\s+"     # "Sort BAM …"
    r"(?=[a-z/\-])"
    r")"
)


def _strip_label_prefix(seg: str) -> str:
    """Strip a leading ``Label: `` (or colonless ``Label``) prefix.

    FlowAgent's system prompt gives canonical commands like
    ``"FastQC: fastqc file.fastq.gz ..."`` and LLMs often echo the whole
    string verbatim, sometimes with the colon, sometimes without. The label
    is not executable; we drop it to find the real first token.

    Guard rail: don't strip when the first word is itself a known tool, so
    e.g. ``"Bowtie2 index reference.fa ..."`` still extracts ``bowtie2``
    rather than leaving ``reference.fa`` as the "first token".
    """
    m = _LABEL_PREFIX_RE.match(seg)
    if not m:
        return seg
    first = seg.split(maxsplit=1)[0].rstrip(":,")
    if _normalise_tool_name(first) in _BIOINFO_TOOLS:
        return seg
    return seg[m.end():]


def extract_tools_from_plan(plan: Dict[str, Any]) -> Set[str]:
    """Return every CLI tool invoked anywhere in any step command.

    A 'tool' is any non-shell token that appears at the start of a command
    segment (where segments are split by ``;``, ``|``, ``&&``, ``||``, etc.).
    This handles piped commands like ``hisat2 ... | samtools sort | samtools
    index`` — all three invocations are captured, not just the first.

    ``"Label: real_command ..."`` prefixes are stripped before tokenising so
    ``"Sort BAM: samtools sort ..."`` records ``samtools``, not ``sort``.
    """
    tools: Set[str] = set()
    seg_splitter = re.compile(r"(?:\|\||&&|;|\||&)")
    for step in plan.get("steps", []):
        cmd = (step.get("command", "") or "").strip()
        if not cmd:
            continue
        for seg in seg_splitter.split(cmd):
            seg = seg.strip().lstrip("()<> ")
            if not seg:
                continue
            seg = _strip_label_prefix(seg)
            tokens = [t for t in re.split(r"\s+", seg) if t]
            # Strip leading env assignments (FOO=bar) and redirection operators
            while tokens and ("=" in tokens[0] or tokens[0] in {">", "2>", ">>"}):
                tokens = tokens[1:]
            if not tokens:
                continue
            # Strip trailing punctuation (colons, commas) that LLMs sometimes
            # leave on tokens when echoing rule text.
            raw = tokens[0]
            first = raw.split("/")[-1].rstrip(":,;").lower()
            if not first or first in _SHELL_TOKENS:
                continue
            if first.startswith("-") or "=" in first:
                continue
            # Skip file paths and sequencing/biodata filenames — they show up
            # as "first tokens" when an LLM drops a tool name after a label
            # (e.g. "Sort BAM results/x.bam" → no tool, just a path).
            if "/" in raw or _FILE_EXT_RE.search(first):
                continue
            tools.add(first)
    return tools


def tool_covered(expected: str, plan_tools: Set[str],
                 plan: Optional[Dict[str, Any]] = None) -> bool:
    """True iff ``expected`` is covered by any tool in ``plan_tools``.

    Matches are generous by design:
      * exact  — ``bismark`` matches ``bismark``
      * family — ``bismark`` matches ``bismark_methylation_extractor``,
                 ``deduplicate_bismark``, ``bismark-genome-preparation``,
                 since those all come from the Bismark package
      * build/index siblings — ``hisat2`` matches ``hisat2-build``,
                 ``bowtie2`` matches ``bowtie2-build``
      * loose prefix for long names — ``star`` covers ``starsolo``
      * R/Python library fallback — if ``plan`` is supplied and the
        expected name appears as a word anywhere in a step's command or
        name, credit it. This catches ``Rscript dada2_denoise.R`` and
        ``Rscript -e 'library(tximport); ...'`` where the library is the
        real tool but the process-level invocation is ``Rscript``.

    Normalises ``-`` and ``_`` so ``featurecounts`` matches ``featureCounts``.
    """
    def norm(s: str) -> str:
        return s.lower().replace("-", "_")

    e = norm(expected)
    for t in plan_tools:
        n = norm(t)
        if n == e:
            return True
        if n.startswith(e + "_") or n.startswith(e + "build"):
            return True
        parts = n.split("_")
        if e in parts:
            return True
        # Loose prefix match (gated at 4 chars — ``kb`` shouldn't match
        # unrelated tools beginning with ``kb``).
        if len(e) >= 4 and n.startswith(e):
            return True

    # Fallback: library-inside-runner match. Only applies when a plan is
    # supplied and at least one step invokes a script runner, so we don't
    # over-credit tools that appear only in filenames of unrelated plans.
    if plan is not None and len(e) >= 4 and any(
        t in _RUNNER_TOKENS for t in plan_tools
    ):
        # Alphanumeric-only boundaries — underscores count as separators,
        # so ``dada2_denoise_script.R`` matches expected ``dada2``.
        pat = re.compile(
            rf"(?<![a-zA-Z0-9]){re.escape(expected.lower())}(?![a-zA-Z0-9])"
        )
        for step in plan.get("steps", []):
            text = f"{step.get('name', '')} {step.get('command', '')}".lower()
            if pat.search(text):
                return True
    return False


_RUNNER_TOKENS = {
    "rscript", "r", "python", "python3", "python2",
    "julia", "perl", "ruby", "node", "bash", "sh",
}


def build_dag(plan: Dict[str, Any]) -> "nx.DiGraph":
    """Build a NetworkX DiGraph from plan dependencies."""
    if nx is None:
        raise RuntimeError("networkx not installed — install benchmark deps")
    g = nx.DiGraph()
    for step in plan.get("steps", []):
        name = step.get("name", "")
        g.add_node(name)
    for step in plan.get("steps", []):
        name = step.get("name", "")
        for dep in step.get("dependencies", []) or []:
            g.add_edge(dep, name)
    return g


def dag_valid(plan: Dict[str, Any]) -> bool:
    """True iff dependencies form a DAG (no cycles, no dangling references)."""
    if nx is None:
        return True  # Can't verify without networkx; assume valid.
    try:
        g = build_dag(plan)
    except Exception:
        return False
    if not nx.is_directed_acyclic_graph(g):
        return False
    names = {s.get("name") for s in plan.get("steps", [])}
    for step in plan.get("steps", []):
        for dep in step.get("dependencies", []) or []:
            if dep not in names:
                return False
    return True


# ── Schema validation ─────────────────────────────────────────────

# Required keys on a workflow step. Extra keys (resources, parameters,
# tools, profile_name, ...) are allowed because the runtime accepts them.
_REQUIRED_STEP_KEYS = ("name", "command")


def plan_schema_valid(plan: Dict[str, Any]) -> bool:
    """True iff the plan has the structural shape FlowAgent's runtime
    actually requires.

    We deliberately do NOT use ``WorkflowPlanSchema.model_validate`` here
    because that schema declares ``extra: forbid`` — but in practice,
    LLM-generated plans (and the executor) carry extra fields like
    ``resources``, ``parameters``, ``tools``, ``profile_name``, etc.
    Plans with those fields are perfectly executable; rejecting them as
    "invalid" gives a misleading 0% pass rate.

    We require:
      - top-level dict with ``workflow_type`` (str) and ``steps`` (list)
      - each step is a dict with at least ``name`` (str) and ``command`` (str)
      - dependencies, if present, must be a list of strings
    """
    if not isinstance(plan, dict):
        return False
    if not isinstance(plan.get("workflow_type"), str):
        return False
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        return False
    for step in steps:
        if not isinstance(step, dict):
            return False
        for key in _REQUIRED_STEP_KEYS:
            if not isinstance(step.get(key), str) or not step[key]:
                return False
        deps = step.get("dependencies", [])
        if deps is not None and not (isinstance(deps, list) and
                                      all(isinstance(d, str) for d in deps)):
            return False
    return True


def _normalise_type(s: str) -> str:
    """Normalise a workflow_type string for tolerant comparison.

    rna_seq_hisat ≡ rna_seq_hisat2 (LLM sometimes drops trailing digits).
    Strips trailing digits and trailing underscores.
    """
    s = (s or "").strip().lower()
    while s and s[-1].isdigit():
        s = s[:-1]
    return s.rstrip("_")


def type_matches(actual, expected) -> bool:
    """Permissive workflow-type comparison.

    ``expected`` may be a single string or a list of acceptable synonyms.
    If ``custom`` is listed, any non-empty actual type is accepted —
    reflecting that FlowAgent intentionally falls back to ``custom`` for
    niche workflows and some LLMs return a free-text description
    (e.g. ``"Whole-genome bisulfite sequencing (WGBS) analysis"``) rather
    than a canonical slug. That still signals "I know this is a bespoke
    pipeline".
    """
    if isinstance(actual, list):
        actual = actual[0] if actual else ""
    if isinstance(expected, str):
        accept = [expected]
    elif isinstance(expected, (list, tuple)):
        accept = list(expected)
    else:
        return False

    accept_norm = [_normalise_type(e) for e in accept]

    # Wildcard: ``custom`` in expected ⇒ any non-empty actual is OK.
    if "custom" in accept_norm and isinstance(actual, str) and actual.strip():
        return True

    a_norm = _normalise_type(actual)
    return any(e == a_norm for e in accept_norm)


# ── Top-level scoring ─────────────────────────────────────────────

def score_plan(plan: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
    """Compute Benchmark A metrics for one (plan, expectation) pair.

    ``expected`` fields consumed:
      - ``expected_workflow_type``
      - ``expected_tools``        (list[str])
      - ``expected_min_steps``    (int)
      - ``forbidden_tools``       (list[str])
      - ``gold_preset``           (str, optional; preset id for concordance)
    """
    metrics: Dict[str, Any] = {}

    # Structural validity
    metrics["plan_valid"] = plan_schema_valid(plan)
    metrics["dag_valid"] = dag_valid(plan)

    # Type — accept a single expected string OR a list of synonyms.
    # Also tolerant of trailing-digit differences (rna_seq_hisat vs hisat2).
    actual_type = plan.get("workflow_type", "")
    metrics["type_correct"] = type_matches(
        actual_type, expected.get("expected_workflow_type"),
    )
    metrics["actual_workflow_type"] = actual_type

    # Tool coverage
    plan_tools = extract_tools_from_plan(plan)
    metrics["num_tools"] = len(plan_tools)

    # Hallucination check — tools the LLM invoked that we don't recognise
    # as real bioinformatics CLI tools.  Reported as a count and a fraction
    # of the plan's unique tool set (to make it comparable across plan sizes).
    h = hallucinated_tools(plan_tools)
    metrics["num_hallucinated_tools"] = len(h)
    metrics["hallucination_rate"] = (
        len(h) / len(plan_tools) if plan_tools else 0.0
    )
    metrics["hallucinated_tools"] = ";".join(sorted(h)) if h else ""
    expected_tools = [t.lower() for t in (expected.get("expected_tools") or [])]
    forbidden = [t.lower() for t in (expected.get("forbidden_tools") or [])]
    if expected_tools:
        present = [t for t in expected_tools
                   if tool_covered(t, plan_tools, plan=plan)]
        metrics["tools_present_fraction"] = len(present) / len(expected_tools)
    else:
        metrics["tools_present_fraction"] = 1.0
    # Forbidden tools: strict first-token check only (no library fallback)
    # so e.g. ``Rscript -e 'library(wget_wrapper)'`` isn't penalised.
    metrics["no_forbidden_tools"] = not any(
        tool_covered(t, plan_tools) for t in forbidden
    )

    # Step count
    n_steps = len(plan.get("steps", []))
    metrics["num_steps"] = n_steps
    metrics["step_count_ok"] = n_steps >= (expected.get("expected_min_steps") or 0)

    # Gold preset concordance (optional)
    gold_id = expected.get("gold_preset")
    if gold_id:
        try:
            from flowagent.presets.catalog import get_preset
            gold = get_preset(gold_id)
            if gold:
                plan_names = [s.get("name", "") for s in plan.get("steps", [])]
                gold_names = [s.get("name", "") for s in gold.get("steps", [])]
                metrics["preset_name_jaccard"] = jaccard(plan_names, gold_names)
                # Command-level F1 on matching step names
                plan_by_name = {s.get("name"): s.get("command", "")
                                for s in plan.get("steps", [])}
                gold_by_name = {s.get("name"): s.get("command", "")
                                for s in gold.get("steps", [])}
                common = set(plan_by_name) & set(gold_by_name)
                if common:
                    metrics["preset_command_f1"] = sum(
                        command_token_f1(plan_by_name[n], gold_by_name[n])
                        for n in common
                    ) / len(common)
                else:
                    metrics["preset_command_f1"] = 0.0
        except Exception:
            pass  # Gold concordance is best-effort

    # Overall pass/fail (for stacked-bar summary)
    metrics["overall_pass"] = bool(
        metrics["plan_valid"]
        and metrics["dag_valid"]
        and metrics["type_correct"]
        and metrics["tools_present_fraction"] == 1.0
        and metrics["no_forbidden_tools"]
        and metrics["step_count_ok"]
    )
    return metrics


# ── Cost model ────────────────────────────────────────────────────

def cost_usd(prompt_tokens: int, completion_tokens: int,
             model_cfg: Dict[str, Any]) -> float:
    pricing = model_cfg.get("pricing", {}) or {}
    return (
        prompt_tokens * pricing.get("input_per_1k", 0.0) / 1000.0
        + completion_tokens * pricing.get("output_per_1k", 0.0) / 1000.0
    )


def diagnosis_matches(diagnosis: Optional[str], regex: str) -> bool:
    """True if the recovery diagnosis matches the expected-cause regex."""
    if not diagnosis:
        return False
    return bool(re.search(regex, diagnosis))
