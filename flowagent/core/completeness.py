"""Domain-specific structural completeness checks for workflow plans.

The DAG-Plan paper (Gao & Mu et al., 2025, https://arxiv.org/abs/2406.09953)
demonstrates that LLM-emitted DAGs are markedly improved by a "regenerate
if structurally incomplete" reflection loop on top of one-shot generation.
DAG-Plan checks domain rules ("every occupy is paired with a release",
"the graph is fully connected") before accepting the LLM's first draft.

This module is the bioinformatics analogue. It accepts a plan dict (with
``StepKind``-typed steps; if ``kind`` is missing, a heuristic infers it
from name and command) and returns a list of human-readable failures
that can be fed back to the LLM as a reflection prompt.

The rules are intentionally permissive — the goal is to catch *structural*
mistakes (missing index→align edges, dangling downloads, no terminal
sink, orphan branches) that produce a syntactically valid but
semantically broken pipeline. Tool selection and command-flag correctness
are scored elsewhere (``benchmarks/harness/metrics.py``).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import networkx as nx  # type: ignore
except ImportError:  # pragma: no cover
    nx = None

from .schemas import StepKind


# ── Heuristic kind inference ──────────────────────────────────────
#
# The planner LLM is asked (via the ``kind`` field on
# ``WorkflowStepSchema``) to label every step. When that fails — older
# fallback paths, the DAG-blind ablation arm, JSON-repair retries that
# omit fields, etc. — these patterns recover the kind from the
# command's first token and the step name.
#
# Order matters: the first matching pattern wins. More specific tokens
# (``samtools sort``, ``gatk MarkDuplicates``) are listed before generic
# family matches (``samtools``, ``gatk``).
_KIND_PATTERNS: List[Tuple[StepKind, re.Pattern]] = [
    # Downloads / fetch
    (StepKind.DOWNLOAD, re.compile(
        r"\b(?:wget|curl|aria2c?|prefetch|fasterq[-_]dump|fastq[-_]dump|"
        r"sra[-_]?toolkit|datasets|esearch|efetch|geofetch|geoparse|"
        r"download|fetch_)\b", re.I)),
    # Indexing
    (StepKind.INDEX, re.compile(
        r"\b(?:bwa[-_ ]index|bowtie2?[-_]build|hisat2[-_]build|"
        r"samtools[-_ ]faidx|samtools[-_ ]index|star\s+--?runMode\s+genomeGenerate|"
        r"kallisto\s+index|salmon\s+index|bismark[-_]?genome[-_]?preparation|"
        r"gatk[-_ ]?CreateSequenceDictionary|picard[-_ ]?CreateSequenceDictionary|"
        r"\b\w*[-_]?index\b)", re.I)),
    # QC (must come before align/trim because fastqc names sometimes overlap)
    (StepKind.QC, re.compile(
        r"\b(?:fastqc|nanoplot|nanoqc|pycoqc|longqc|seqkit\s+stats|"
        r"qualimap|samtools\s+flagstat|samtools\s+stats|picard\s+CollectInsertSizeMetrics|"
        r"chipqc)\b", re.I)),
    # Read trimming / filtering
    (StepKind.TRIM, re.compile(
        r"\b(?:trim[-_]?galore|trimgalore|cutadapt|fastp|trimmomatic|"
        r"bbduk|atropos|nanofilt|filtlong|porechop|seqtk\s+trimfq)\b", re.I)),
    # Alignment / pseudo-alignment / read-mapping (counts as alignment for
    # the structural rule "needs an index ancestor")
    (StepKind.ALIGN, re.compile(
        r"\b(?:bwa\s+(?:mem|aln|sampe|samse)|bwa-mem2|bowtie2?(?!-build)|"
        r"hisat2(?!-build)|star(?!solo)\b|starsolo|minimap2|tophat2?|gsnap|"
        r"bbmap|ngmlr|segemehl|kallisto\s+quant|salmon\s+quant|bismark\s+(?!genome)|"
        r"alevin|alevin[-_]fry|cellranger\s+count|spaceranger\s+count)\b", re.I)),
    # BAM/CRAM sort
    (StepKind.SORT, re.compile(
        r"\b(?:samtools\s+sort|sambamba\s+sort|picard\s+SortSam)\b", re.I)),
    # Duplicate marking / removal
    (StepKind.DEDUP, re.compile(
        r"\b(?:picard\s+MarkDuplicates|gatk\s+MarkDuplicates|sambamba\s+markdup|"
        r"samtools\s+markdup|deduplicate_bismark|umi[-_]tools\s+dedup)\b", re.I)),
    # Variant / peak / event calling
    (StepKind.CALL, re.compile(
        r"\b(?:gatk\s+(?:HaplotypeCaller|Mutect2|GenotypeGVCFs)|bcftools\s+call|"
        r"freebayes|deepvariant|varscan|strelka2?|mutect2?|octopus|platypus|"
        r"manta|delly|lumpy|gridss|smoove|svaba|tiddit|pindel|medaka|clair3|"
        r"longshot|nanopolish|pepper|cnvkit|qdnaseq|cnvnator|"
        r"macs2?\s+callpeak|macs3\s+callpeak|seacr|genrich|epic2|homer\s+findPeaks)\b",
        re.I)),
    # Quantification / counting
    (StepKind.QUANTIFY, re.compile(
        r"\b(?:featurecounts|featureCounts|htseq[-_]?count|rsem(?!-prepare)|"
        r"stringtie|cufflinks|tximport)\b", re.I)),
    # DE / differential analysis (typically inside Rscript ...)
    (StepKind.DE, re.compile(
        r"\b(?:deseq2|edger|limma|sleuth|dada2|diffbind|chipseeker|"
        r"differential\w*|de_analysis|de_call)\b", re.I)),
    # Reports / aggregations
    (StepKind.REPORT, re.compile(
        r"\b(?:multiqc|krona|busco|quast|scanpy.*report|jupyter\s+nbconvert|"
        r"\w*[-_ ]?report\b|html_summary|render_report)\b", re.I)),
    # mkdir / setup as "other"
    (StepKind.OTHER, re.compile(r"\b(?:mkdir|cd|rm|mv|cp|ln|touch|set|export|echo)\b", re.I)),
]


def infer_step_kind(step: Dict[str, Any]) -> StepKind:
    """Best-effort kind inference from ``command`` / ``name``.

    Used as a fallback when the LLM omits ``kind`` (e.g. on the DAG-blind
    ablation arm or after a JSON-repair retry that drops fields). The
    pattern list is ordered specific → generic.
    """
    haystack = " ".join([
        str(step.get("name", "")),
        str(step.get("command", "")),
    ])
    if not haystack.strip():
        return StepKind.OTHER
    for kind, pat in _KIND_PATTERNS:
        if pat.search(haystack):
            return kind
    return StepKind.OTHER


def fill_missing_kinds(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Mutate ``plan`` so every step has a valid ``kind`` field.

    Steps that already carry a recognised ``StepKind`` value (raw string
    or enum) are left alone. Missing or unrecognised values are filled
    via :func:`infer_step_kind`. Returns the same plan dict (mutated)
    for chaining convenience.
    """
    valid = {k.value for k in StepKind}
    for step in plan.get("steps", []) or []:
        raw = step.get("kind")
        if isinstance(raw, StepKind):
            step["kind"] = raw.value
            continue
        if isinstance(raw, str) and raw in valid:
            continue
        step["kind"] = infer_step_kind(step).value
    return plan


# ── Graph helpers ─────────────────────────────────────────────────

def _build_graph(plan: Dict[str, Any]) -> Optional["nx.DiGraph"]:
    if nx is None:
        return None
    g = nx.DiGraph()
    for step in plan.get("steps", []) or []:
        name = step.get("name")
        if name:
            g.add_node(name, kind=step.get("kind", StepKind.OTHER.value))
    for step in plan.get("steps", []) or []:
        name = step.get("name")
        if not name:
            continue
        for dep in step.get("dependencies", []) or []:
            if dep in g and isinstance(dep, str):
                g.add_edge(dep, name)
    return g


def _kind_of(step: Dict[str, Any]) -> str:
    raw = step.get("kind")
    if isinstance(raw, StepKind):
        return raw.value
    if isinstance(raw, str) and raw:
        return raw
    return StepKind.OTHER.value


# ── Domain rules ──────────────────────────────────────────────────

# Kinds that should have an ``index`` (or ``download`` of a pre-built
# index) ancestor. Pseudo-aligners count too: ``kallisto quant`` needs
# ``kallisto index``.
_NEEDS_INDEX_ANCESTOR = {StepKind.ALIGN.value}

# Kinds that should have a ``report`` descendant (or be a terminal sink
# themselves). The intent is to catch dangling analyses where the LLM
# emits a ``deseq2`` step but never aggregates results.
_NEEDS_REPORT_DESCENDANT = {
    StepKind.QC.value,
    StepKind.QUANTIFY.value,
    StepKind.CALL.value,
    StepKind.DE.value,
}

# Kinds that count as "informative sinks" — DAG leaves whose presence
# satisfies the "needs report descendant" rule even when no explicit
# ``report`` aggregator exists. If the analysis chain ends in a
# ``de`` / ``call`` / ``quantify`` / ``report`` / ``terminal`` step
# that is itself a sink, the result IS the final artefact and the
# rule must not fire. A chain ending in an ``other`` (glue) sink, by
# contrast, is dangling and should fail.
_INFORMATIVE_SINK_KINDS = {
    StepKind.REPORT.value,
    StepKind.TERMINAL.value,
    StepKind.DE.value,
    StepKind.CALL.value,
    StepKind.QUANTIFY.value,
}


def validate_workflow_completeness(
    plan: Dict[str, Any],
    *,
    expected_workflow_type: Optional[str] = None,
    require_terminal_sink: bool = True,
) -> Tuple[bool, List[str]]:
    """Run domain-specific structural checks on a plan.

    Returns ``(ok, failures)``. ``ok`` is True iff ``failures`` is empty.

    The check is non-gating by default in the runtime (``LLMInterface``
    can call this and reflect on failure but still proceed if reflection
    is exhausted). It is wired into the benchmark scorer as a
    ``completeness_pass`` boolean that complements ``dag_valid``.

    Rules (each adds at most one entry to ``failures``):
      - Plan must have at least one step with a recognised ``kind``.
      - Every ``align`` must have an ``index`` ancestor (or share an
        ancestor that downloads a pre-built index).
      - Every ``download`` must have a descendant that consumes it
        (else the file is fetched but never used).
      - ``de`` / ``call`` / ``quantify`` should have a ``report``
        descendant, or themselves be a sink leaf of the DAG.
      - When ``require_terminal_sink`` is True, the DAG must have at
        least one sink (no successors). A graph where every node has a
        successor is cyclic; this is also caught by ``dag_valid`` but
        we surface it here with a more actionable message.
      - The graph is weakly connected (no orphan islands).
    """
    fill_missing_kinds(plan)
    failures: List[str] = []

    steps = plan.get("steps", []) or []
    if not steps:
        return False, ["plan has zero steps"]

    g = _build_graph(plan)
    if g is None:
        # No networkx -- can only do field-level checks.
        return True, []

    by_name: Dict[str, Dict[str, Any]] = {
        s.get("name", ""): s for s in steps if s.get("name")
    }

    if not nx.is_directed_acyclic_graph(g):
        failures.append("dependencies contain a cycle")
        return False, failures

    # Pre-compute ancestor / descendant kind sets for every node.
    anc_kinds: Dict[str, Set[str]] = {}
    desc_kinds: Dict[str, Set[str]] = {}
    for node in g.nodes():
        anc_kinds[node] = {
            _kind_of(by_name[a]) for a in nx.ancestors(g, node) if a in by_name
        }
        desc_kinds[node] = {
            _kind_of(by_name[d]) for d in nx.descendants(g, node) if d in by_name
        }

    # Rule: align needs an index (or download-of-index) ancestor.
    align_missing_index: List[str] = []
    for node in g.nodes():
        step = by_name.get(node, {})
        if _kind_of(step) in _NEEDS_INDEX_ANCESTOR:
            anc = anc_kinds[node]
            if StepKind.INDEX.value not in anc and StepKind.DOWNLOAD.value not in anc:
                align_missing_index.append(node)
    if align_missing_index:
        failures.append(
            f"alignment step(s) lack an index/download ancestor: "
            f"{', '.join(sorted(align_missing_index))}. "
            f"Add an index-build step (e.g. 'bwa index', 'star --runMode genomeGenerate', "
            f"'kallisto index') as a prerequisite."
        )

    # Rule: download must have a descendant.
    dangling_downloads: List[str] = []
    for node in g.nodes():
        step = by_name.get(node, {})
        if _kind_of(step) == StepKind.DOWNLOAD.value:
            if g.out_degree(node) == 0:
                dangling_downloads.append(node)
    if dangling_downloads:
        failures.append(
            f"download step(s) have no consumer downstream: "
            f"{', '.join(sorted(dangling_downloads))}. "
            f"Either remove the download or wire it into an index/align step."
        )

    # Rule: quantify/call/de should reach an informative sink (report,
    # terminal, or another de/call/quantify) OR be sinks themselves.
    # A chain ending in a glue ``other`` sink is dangling and fails.
    missing_report: List[str] = []
    for node in g.nodes():
        step = by_name.get(node, {})
        if _kind_of(step) not in _NEEDS_REPORT_DESCENDANT:
            continue
        if g.out_degree(node) == 0:
            continue  # node is itself the final artefact
        descendants = nx.descendants(g, node)
        sink_kinds = {
            _kind_of(by_name[d])
            for d in descendants
            if d in by_name and g.out_degree(d) == 0
        }
        if sink_kinds & _INFORMATIVE_SINK_KINDS:
            continue
        missing_report.append(node)
    if missing_report:
        failures.append(
            f"analysis step(s) feed downstream nodes but no report aggregator: "
            f"{', '.join(sorted(missing_report))}. "
            f"Add a 'multiqc'/report step that consumes their outputs, or make "
            f"them DAG sinks if they already produce the final artefact."
        )

    # Rule: at least one terminal sink.
    if require_terminal_sink:
        sinks = [n for n in g.nodes() if g.out_degree(n) == 0]
        if not sinks:
            failures.append(
                "no terminal sink in DAG (every node has a successor) -- "
                "the workflow has no final output node"
            )

    # Rule: weakly connected (no orphan branches).
    if g.number_of_nodes() > 1:
        und = g.to_undirected()
        if not nx.is_connected(und):
            comps = list(nx.connected_components(und))
            comp_summaries = [
                f"[{', '.join(sorted(c)[:3])}{'...' if len(c) > 3 else ''}]"
                for c in comps
            ]
            failures.append(
                f"DAG has {len(comps)} disconnected components: "
                f"{' | '.join(comp_summaries)}. "
                f"Wire orphan branches into the main pipeline."
            )

    return (len(failures) == 0), failures


def render_completeness_feedback(failures: List[str]) -> str:
    """Format a failure list as an LLM-facing reflection prompt.

    Used by :func:`flowagent.core.llm.LLMInterface.generate_workflow_plan`
    to ask the LLM to regenerate a plan that addresses the issues.
    """
    if not failures:
        return ""
    bullets = "\n".join(f"  {i+1}. {f}" for i, f in enumerate(failures))
    return (
        "Your previous workflow plan failed structural completeness checks:\n"
        f"{bullets}\n\n"
        "Please regenerate the plan to fix every item above. Keep the same "
        "output JSON schema; only adjust the steps, dependencies, and kinds "
        "to satisfy the structural rules. In particular: every alignment "
        "step needs an index/download ancestor; every download must be "
        "consumed; every analysis step (quantify/call/de) should feed a "
        "report aggregator (e.g. multiqc) or be a DAG sink."
    )
