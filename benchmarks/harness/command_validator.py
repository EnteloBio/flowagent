"""Per-tool command-semantic validators for Benchmark A.

The reviewer of FlowAgent's planning benchmark observed that
``score_plan`` only checks tool *presence* — a plan that says
``kallisto`` somewhere passes whether or not the actual command would
run. This module adds a thin validator: for the bioinformatics tools
the LLM is most likely to invoke wrong, declare the *minimum* flag
combinations every well-formed invocation must carry. The validator
walks each step's command (segment-by-segment, the same way
:func:`extract_tools_from_plan` does) and reports whether the command
is well-formed for the tool it invokes.

Design choices:

* **Conservative**: a tool whose first token does not appear in the
  registry passes by default. The point is to catch *known*
  semantic-violations of *common* tools, not to gate on every
  invocation in the planet.
* **Required-only**: each rule is the minimum set of flags / tokens
  the tool needs to do its job; we do *not* model every flag combo.
  This penalises plans that name a tool but call it with no
  arguments, with the wrong subcommand, or with a flag pair that
  cancels itself out.
* **Sub-command aware**: where it matters (``samtools <subcmd>``,
  ``gatk <subcmd>``, ``kallisto <subcmd>``, ``bwa <subcmd>``), the
  validator dispatches on the second token.

The scoring layer reports ``commands_well_formed_fraction`` (per-plan
fraction of segments that validate). The inference-tier scorer gates
``overall_pass`` on this being 1.0; transcription-tier scoring
reports it but does not gate (so historical results stay comparable).
"""

from __future__ import annotations

import re
import shlex
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ── Tokenisation ─────────────────────────────────────────────────

_SHELL_TOKENS = {
    "mkdir", "cd", "rm", "mv", "cp", "ln", "touch", "test",
    "set", "export", "echo", "source", "bash", "sh", "for", "do",
    "done", "if", "then", "else", "fi", "while", "awk", "sed",
    "grep", "cut", "tr", "sort", "uniq", "head", "tail", "cat",
    "tee", "xargs", "time", "env", "printf", "read", "find",
}

_SEG_SPLIT = re.compile(r"(?:\|\||&&|;|\||&)")


def _segment_tokens(seg: str) -> List[str]:
    """Lex one shell segment into argv tokens.

    Falls back to whitespace splitting on lex errors so a single
    malformed quote in one segment doesn't crash the whole validator.
    """
    seg = seg.strip().lstrip("()<> ")
    if not seg:
        return []
    try:
        return shlex.split(seg, posix=True)
    except ValueError:
        return seg.split()


def _strip_env_assignments(tokens: List[str]) -> List[str]:
    while tokens and (
        ("=" in tokens[0] and not tokens[0].startswith("-"))
        or tokens[0] in {">", "2>", ">>"}
    ):
        tokens = tokens[1:]
    return tokens


def _basename(tok: str) -> str:
    return tok.split("/")[-1].lower()


# ── Per-tool rules ───────────────────────────────────────────────
#
# Each rule is a (predicate, error-message) pair on the tokenised argv
# (with leading binary stripped). Rules return True for "well-formed".


def _has_any_flag(argv: Sequence[str], flags: Sequence[str]) -> bool:
    for tok in argv:
        if tok in flags:
            return True
        # ``--threads=8`` matches the long-form pattern.
        for f in flags:
            if f.startswith("--") and tok.startswith(f + "="):
                return True
    return False


def _has_input_path(argv: Sequence[str]) -> bool:
    """Heuristic: at least one positional argument that isn't a flag."""
    saw = False
    skip_next = False
    for tok in argv:
        if skip_next:
            skip_next = False
            continue
        if tok.startswith("-") and tok not in {"-"}:
            # ``-o foo`` style — eat the value too if not =-form.
            if "=" not in tok and not tok.startswith("--"):
                skip_next = True
            continue
        saw = True
        break
    return saw


def _kallisto_rule(argv: List[str]) -> Optional[str]:
    if not argv:
        return "kallisto invoked with no subcommand"
    sub = argv[0]
    rest = argv[1:]
    if sub == "index":
        if not _has_any_flag(rest, ["-i", "--index"]):
            return "kallisto index needs -i/--index"
        if not _has_input_path(rest):
            return "kallisto index needs a transcriptome FASTA"
        return None
    if sub == "quant":
        if not _has_any_flag(rest, ["-i", "--index"]):
            return "kallisto quant needs -i/--index"
        if not _has_any_flag(rest, ["-o", "--output-dir"]):
            return "kallisto quant needs -o/--output-dir"
        return None
    if sub in {"bus", "pseudo"}:
        return None
    return f"unknown kallisto subcommand: {sub}"


def _salmon_rule(argv: List[str]) -> Optional[str]:
    if not argv:
        return "salmon invoked with no subcommand"
    sub = argv[0]
    rest = argv[1:]
    if sub == "index":
        if not _has_any_flag(rest, ["-t", "--transcripts"]):
            return "salmon index needs -t/--transcripts"
        if not _has_any_flag(rest, ["-i", "--index"]):
            return "salmon index needs -i/--index"
        return None
    if sub == "quant":
        if not _has_any_flag(rest, ["-i", "--index"]):
            return "salmon quant needs -i/--index"
        if not _has_any_flag(rest, ["-o", "--output"]):
            return "salmon quant needs -o/--output"
        if not _has_any_flag(rest, ["-l", "--libType"]):
            return "salmon quant needs -l/--libType"
        return None
    return None


def _star_rule(argv: List[str]) -> Optional[str]:
    is_index = _has_any_flag(argv, ["--runMode"]) and any(
        t == "genomeGenerate" for t in argv
    )
    if is_index:
        if not _has_any_flag(argv, ["--genomeDir"]):
            return "STAR genomeGenerate needs --genomeDir"
        if not _has_any_flag(argv, ["--genomeFastaFiles"]):
            return "STAR genomeGenerate needs --genomeFastaFiles"
        return None
    # Default mode is alignment.
    if not _has_any_flag(argv, ["--genomeDir"]):
        return "STAR alignment needs --genomeDir"
    if not _has_any_flag(argv, ["--readFilesIn"]):
        return "STAR alignment needs --readFilesIn"
    return None


def _hisat2_rule(argv: List[str]) -> Optional[str]:
    if not _has_any_flag(argv, ["-x"]):
        return "hisat2 needs -x <index>"
    if not (_has_any_flag(argv, ["-U"]) or _has_any_flag(argv, ["-1"])):
        return "hisat2 needs -U or -1/-2"
    return None


def _bowtie2_rule(argv: List[str]) -> Optional[str]:
    if not _has_any_flag(argv, ["-x"]):
        return "bowtie2 needs -x <index>"
    if not (_has_any_flag(argv, ["-U"]) or _has_any_flag(argv, ["-1"])):
        return "bowtie2 needs -U or -1/-2"
    return None


def _bwa_rule(argv: List[str]) -> Optional[str]:
    if not argv:
        return "bwa invoked with no subcommand"
    sub = argv[0]
    rest = argv[1:]
    if sub == "index":
        if not _has_input_path(rest):
            return "bwa index needs a reference FASTA"
        return None
    if sub in {"mem", "aln", "bwasw"}:
        # mem needs <reference> <reads...>
        positionals = [t for t in rest if not t.startswith("-")]
        if len(positionals) < 2:
            return f"bwa {sub} needs <reference> <reads>"
        return None
    return None


def _minimap2_rule(argv: List[str]) -> Optional[str]:
    # minimap2 ax preset.fa reads.fq → at least 2 positionals after flags
    positionals = [t for t in argv if not t.startswith("-")]
    if len(positionals) < 2:
        return "minimap2 needs <reference> <reads>"
    return None


def _samtools_rule(argv: List[str]) -> Optional[str]:
    if not argv:
        return "samtools invoked with no subcommand"
    sub = argv[0]
    known = {
        "view", "sort", "index", "merge", "faidx", "fasta", "fastq",
        "stats", "flagstat", "idxstats", "mpileup", "depth", "tview",
        "split", "bedcov", "calmd", "rmdup", "markdup", "addreplacerg",
        "fixmate", "phase", "consensus", "ampliconclip", "reheader",
        "dict", "quickcheck", "head",
    }
    if sub not in known:
        return f"unknown samtools subcommand: {sub}"
    rest = argv[1:]
    if sub in {"sort", "view", "index", "stats", "flagstat", "markdup",
              "fixmate", "merge"} and not _has_input_path(rest):
        return f"samtools {sub} needs an input BAM/SAM/CRAM"
    return None


def _gatk_rule(argv: List[str]) -> Optional[str]:
    if not argv:
        return "gatk invoked with no subcommand"
    sub = argv[0]
    rest = argv[1:]
    walker_needs_R = {
        "HaplotypeCaller", "Mutect2", "BaseRecalibrator",
        "ApplyBQSR", "FilterMutectCalls", "VariantFiltration",
        "CountReads", "AnalyzeCovariates", "SelectVariants",
        "GenomicsDBImport", "CreateSequenceDictionary",
    }
    if sub in walker_needs_R:
        if sub != "CreateSequenceDictionary" and not _has_any_flag(
            rest, ["-R", "--reference"],
        ):
            return f"gatk {sub} needs -R/--reference"
    return None


def _bcftools_rule(argv: List[str]) -> Optional[str]:
    if not argv:
        return "bcftools invoked with no subcommand"
    known = {
        "annotate", "call", "concat", "consensus", "convert",
        "csq", "filter", "gtcheck", "index", "isec", "merge",
        "mpileup", "norm", "plugin", "query", "reheader",
        "roh", "sort", "stats", "view",
    }
    if argv[0] not in known:
        return f"unknown bcftools subcommand: {argv[0]}"
    return None


def _macs_rule(argv: List[str]) -> Optional[str]:
    # accepts macs2/macs3
    if not argv:
        return "macs invoked with no subcommand"
    if argv[0] != "callpeak":
        # other subcommands (bdgcmp, bdgpeakcall, …) are optional
        return None
    rest = argv[1:]
    if not _has_any_flag(rest, ["-t", "--treatment"]):
        return "macs callpeak needs -t/--treatment"
    return None


def _featurecounts_rule(argv: List[str]) -> Optional[str]:
    if not _has_any_flag(argv, ["-a"]):
        return "featureCounts needs -a <annotation>"
    if not _has_any_flag(argv, ["-o"]):
        return "featureCounts needs -o <out>"
    return None


def _htseq_count_rule(argv: List[str]) -> Optional[str]:
    positionals = [t for t in argv if not t.startswith("-")]
    if len(positionals) < 2:
        return "htseq-count needs <bam> <gtf>"
    return None


def _stringtie_rule(argv: List[str]) -> Optional[str]:
    if not _has_input_path(argv):
        return "stringtie needs an input BAM"
    return None


def _trim_galore_rule(argv: List[str]) -> Optional[str]:
    if not _has_input_path(argv):
        return "trim_galore needs FASTQ inputs"
    return None


def _fastp_rule(argv: List[str]) -> Optional[str]:
    if not (_has_any_flag(argv, ["-i", "--in1"])):
        return "fastp needs -i/--in1"
    return None


def _cutadapt_rule(argv: List[str]) -> Optional[str]:
    if not _has_any_flag(argv, ["-a", "-g", "-A", "-G"]):
        return "cutadapt needs an adapter (-a/-g/-A/-G)"
    if not _has_any_flag(argv, ["-o", "--output"]):
        return "cutadapt needs -o/--output"
    return None


def _fastqc_rule(argv: List[str]) -> Optional[str]:
    if not _has_input_path(argv):
        return "fastqc needs at least one FASTQ input"
    return None


def _multiqc_rule(argv: List[str]) -> Optional[str]:
    if not _has_input_path(argv):
        return "multiqc needs an input directory"
    return None


def _picard_rule(argv: List[str]) -> Optional[str]:
    # Picard <Tool> I=... O=... or with -I/-O — accept either.
    if not argv:
        return "picard invoked with no subcommand"
    return None


def _bismark_rule(argv: List[str]) -> Optional[str]:
    if not _has_input_path(argv):
        return "bismark needs an input FASTQ"
    return None


def _kraken2_rule(argv: List[str]) -> Optional[str]:
    if not _has_any_flag(argv, ["--db"]):
        return "kraken2 needs --db <database>"
    return None


def _bracken_rule(argv: List[str]) -> Optional[str]:
    if not _has_any_flag(argv, ["-d", "--db"]):
        return "bracken needs -d/--db"
    return None


def _spades_rule(argv: List[str]) -> Optional[str]:
    if not _has_any_flag(argv, ["-o", "--output"]):
        return "spades needs -o/--output"
    return None


def _hifiasm_rule(argv: List[str]) -> Optional[str]:
    if not _has_any_flag(argv, ["-o"]):
        return "hifiasm needs -o <prefix>"
    return None


def _flye_rule(argv: List[str]) -> Optional[str]:
    if not _has_any_flag(argv, ["--out-dir", "-o"]):
        return "flye needs --out-dir"
    return None


def _busco_rule(argv: List[str]) -> Optional[str]:
    if not _has_any_flag(argv, ["-i", "--in"]):
        return "busco needs -i/--in"
    if not _has_any_flag(argv, ["-l", "--lineage_dataset"]):
        return "busco needs -l/--lineage_dataset"
    return None


def _quast_rule(argv: List[str]) -> Optional[str]:
    if not _has_input_path(argv):
        return "quast needs at least one assembly FASTA"
    return None


def _prokka_rule(argv: List[str]) -> Optional[str]:
    if not _has_input_path(argv):
        return "prokka needs an input contig FASTA"
    return None


def _kb_rule(argv: List[str]) -> Optional[str]:
    if not argv:
        return "kb invoked with no subcommand"
    if argv[0] not in {"ref", "count"}:
        return f"unknown kb subcommand: {argv[0]}"
    return None


def _cellranger_rule(argv: List[str]) -> Optional[str]:
    if not argv:
        return "cellranger invoked with no subcommand"
    if argv[0] not in {"count", "mkref", "multi", "vdj", "aggr"}:
        return f"unknown cellranger subcommand: {argv[0]}"
    return None


_RULES: Dict[str, callable] = {
    "kallisto":      _kallisto_rule,
    "salmon":        _salmon_rule,
    "star":          _star_rule,
    "starsolo":      _star_rule,
    "hisat2":        _hisat2_rule,
    "bowtie2":       _bowtie2_rule,
    "bwa":           _bwa_rule,
    "minimap2":      _minimap2_rule,
    "samtools":      _samtools_rule,
    "gatk":          _gatk_rule,
    "gatk4":         _gatk_rule,
    "bcftools":      _bcftools_rule,
    "macs":          _macs_rule,
    "macs2":         _macs_rule,
    "macs3":         _macs_rule,
    "featurecounts": _featurecounts_rule,
    "htseq-count":   _htseq_count_rule,
    "htseq_count":   _htseq_count_rule,
    "stringtie":     _stringtie_rule,
    "trim_galore":   _trim_galore_rule,
    "fastp":         _fastp_rule,
    "cutadapt":      _cutadapt_rule,
    "fastqc":        _fastqc_rule,
    "multiqc":       _multiqc_rule,
    "picard":        _picard_rule,
    "bismark":       _bismark_rule,
    "kraken2":       _kraken2_rule,
    "bracken":       _bracken_rule,
    "spades":        _spades_rule,
    "hifiasm":       _hifiasm_rule,
    "flye":          _flye_rule,
    "busco":         _busco_rule,
    "quast":         _quast_rule,
    "prokka":        _prokka_rule,
    "kb":            _kb_rule,
    "cellranger":    _cellranger_rule,
}


def validate_command(command: str) -> List[Tuple[str, Optional[str]]]:
    """Walk every segment of ``command`` and return per-tool verdicts.

    Returns a list of ``(tool_first_token, error_or_None)`` pairs. A
    well-formed segment yields ``error_or_None == None``. Segments
    whose first token is shell-builtin or a runner like ``rscript`` /
    ``python`` are silently skipped (no rule fires, no verdict
    returned), matching the behaviour of ``extract_tools_from_plan``.
    """
    out: List[Tuple[str, Optional[str]]] = []
    if not command:
        return out
    for seg in _SEG_SPLIT.split(command):
        tokens = _strip_env_assignments(_segment_tokens(seg))
        if not tokens:
            continue
        first = _basename(tokens[0]).rstrip(":,;")
        if first in _SHELL_TOKENS or first.startswith("-"):
            continue
        # Normalise dashed → underscored variants for lookup.
        key_variants = (first, first.replace("-", "_"))
        rule = None
        for k in key_variants:
            if k in _RULES:
                rule = _RULES[k]
                first = k
                break
        if rule is None:
            continue
        out.append((first, rule(tokens[1:])))
    return out


def score_plan_commands(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Compute command-semantic metrics across every step of a plan.

    Returns a dict with:

    * ``commands_well_formed_fraction`` — fraction of validated segments
      that passed (1.0 when every modelled tool was invoked correctly,
      or when the plan invoked no modelled tools at all).
    * ``num_commands_validated`` — count of segments where a rule fired.
    * ``num_command_errors`` — count of validated segments that failed.
    * ``command_errors`` — joined ``";"``-separated error strings (for
      auditability in the CSV).
    """
    total = 0
    failed = 0
    errors: List[str] = []
    for step in plan.get("steps", []):
        cmd = (step.get("command") or "").strip()
        for tool, err in validate_command(cmd):
            total += 1
            if err is not None:
                failed += 1
                errors.append(f"{tool}: {err}")
    fraction = 1.0 if total == 0 else (total - failed) / total
    return {
        "commands_well_formed_fraction": fraction,
        "num_commands_validated":        total,
        "num_command_errors":            failed,
        "command_errors":                ";".join(errors)[:1000],
    }
