"""Export a FlowAgent workflow + run results as an R Markdown document.

Why both ``.ipynb`` and ``.Rmd``?

For workflows that end in DESeq2 / tximport / edgeR / limma, the analysis
*is* R. An ``.Rmd`` is the native format for that audience: it knits
to HTML/PDF via ``rmarkdown::render()``, opens directly in RStudio with
syntax-highlighted R chunks, and lets the trailing "load results" cells
share one R session (so ``readRDS()``, ``library(DESeq2)``, plotting,
etc. compose as you'd write them by hand).

Output structure mirrors :mod:`flowagent.utils.export_notebook`:
  - YAML front-matter (title, date, output: html_document with TOC).
  - Optional LLM-written assay-aware overview.
  - Workflow-plan summary table.
  - One markdown header + one fenced code chunk per step. Inline R
    invocations of the form ``Rscript -e '<code>'`` are detected and
    extracted into native ``{r}`` chunks; everything else stays as a
    ``{bash}`` chunk so semantics match what FlowAgent actually ran.
  - Run summary, LLM-written interpretation, follow-up suggestions.
  - Trailing ``{r}`` chunks that load standard outputs (``txi.rds``,
    ``deseq2_results.csv``, etc.) where they exist on disk.

Usage:

    from flowagent.utils.export_rmarkdown import export_workflow_to_rmarkdown
    export_workflow_to_rmarkdown(
        workflow_json="flowagent_output/Unnamed_Workflow/workflow.json",
        out_path="flowagent_output/Unnamed_Workflow/notebook.Rmd",
        run_results=...,
        prompt="Download GSE52778 …",
        narrative=...,
    )
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Same probe paths as the .ipynb exporter — kept in sync deliberately so a
# directory containing both formats has parallel "load results" sections.
_RESULT_PROBE = (
    ("DESeq2 results CSV", "results/rna_seq_kallisto/deseq2/deseq2_results.csv", "r-csv"),
    ("Gene counts",        "results/rna_seq_kallisto/deseq2/gene_counts.csv",   "r-csv"),
    ("tximport object",    "results/rna_seq_kallisto/deseq2/txi.rds",           "r-rds"),
    ("MACS2 peaks",        "results/macs2/peaks.narrowPeak",                    "r-bed"),
    ("Filtered VCF",       "results/gatk/filtered.vcf.gz",                      "vcf"),
)

# Detect Rscript -e '...' / R -e '...' so we can pull the embedded R code
# into a native {r} chunk. Handles single OR double quotes; refuses to
# match across multi-command (``&&`` / ``;``) boundaries to avoid
# extracting half-commands.
_RSCRIPT_INLINE_RE = re.compile(
    r"""^\s*(?:Rscript|R\s+--vanilla|R)\s+-e\s+"""
    r"""(?P<q>['"])(?P<code>.*?)(?P=q)\s*(?:#.*)?$""",
    flags=re.DOTALL,
)


def _extract_inline_r(cmd: str) -> Optional[str]:
    """Return the R code inside an ``Rscript -e '...'`` invocation, else None.

    Returns None for any command that is not *entirely* a single Rscript
    invocation (e.g. ``mkdir -p out && Rscript -e ...``); those need to
    remain bash so semantics match what FlowAgent actually executed. The
    regex's ``^\\s*…(?:#.*)?$`` anchoring enforces this — no pre-rejection
    on shell metacharacters because R legitimately uses ``;`` inside its
    own code.
    """
    m = _RSCRIPT_INLINE_RE.match(cmd.strip())
    if m is None:
        return None
    return m.group("code")


_STATUS_BADGE = {
    "completed": "✅ completed",
    "success":   "✅ completed",
    "recovered": "🟡 recovered",
    "failed":    "❌ failed",
    "error":     "❌ failed",
    "skipped":   "⚪ skipped",
    "pending":   "⚪ pending",
}


def _yaml_front_matter(workflow: Dict[str, Any], prompt: Optional[str],
                       metadata: Optional[Dict[str, Any]]) -> str:
    """Standard knit-able header.

    Uses ``html_document`` with TOC + code-folding so the rendered HTML is
    immediately useful; the user can swap in ``pdf_document`` or
    ``word_document`` afterwards without touching anything else.
    """
    name = workflow.get("name") or "FlowAgent run"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    title = f"FlowAgent run — {name}"
    # Escape any double-quotes in title to keep YAML valid.
    title = title.replace('"', "'")
    return (
        "---\n"
        f'title: "{title}"\n'
        f'date: "{today}"\n'
        "output:\n"
        "  html_document:\n"
        "    toc: true\n"
        "    toc_float: true\n"
        "    code_folding: show\n"
        "    theme: cerulean\n"
        "    df_print: paged\n"
        "---\n\n"
        "```{r setup, include=FALSE}\n"
        'knitr::opts_chunk$set(echo = TRUE, eval = FALSE)\n'
        "# Set eval = TRUE on individual chunks (or globally) to actually\n"
        "# re-execute the workflow when knitting.\n"
        "```\n\n"
    )


def _intro_section(workflow: Dict[str, Any], prompt: Optional[str],
                   metadata: Optional[Dict[str, Any]],
                   narrative_intro: Optional[str]) -> str:
    parts: List[str] = []
    n_steps = len(workflow.get("steps", []))
    parts.append(f"**Total steps:** {n_steps}  ")
    if metadata:
        for k in ("model", "provider", "git_sha", "executor"):
            if k in metadata and metadata[k]:
                parts.append(f"**{k}:** `{metadata[k]}`  ")
    if prompt:
        parts.append("")
        parts.append("## Original prompt")
        parts.append("")
        parts.append("> " + prompt.strip().replace("\n", "\n> "))
    if narrative_intro:
        parts.append("")
        parts.append("## Analysis overview")
        parts.append("")
        parts.append(narrative_intro)
    parts.append("")
    parts.append("## Workflow plan")
    parts.append("")
    parts.append("| # | Step | Description | Depends on |")
    parts.append("| --- | --- | --- | --- |")
    for i, s in enumerate(workflow.get("steps", []), 1):
        deps = ", ".join(s.get("dependencies") or []) or "—"
        desc = (s.get("description") or "").replace("|", "\\|")[:120]
        parts.append(f"| {i} | `{s.get('name','?')}` | {desc} | {deps} |")
    parts.append("")
    return "\n".join(parts)


def _section_for_step(step: Dict[str, Any],
                      result: Optional[Dict[str, Any]],
                      narrative: Optional[str]) -> str:
    """Render one step as a markdown header + its code chunk."""
    name = step.get("name", "step")
    desc = step.get("description") or ""
    cmd  = (step.get("command") or "").strip()
    deps = step.get("dependencies") or []

    # Heading + status + meta
    badge = ""
    if result is not None:
        st = str(result.get("status", "")).lower()
        wall = result.get("wall_seconds") or result.get("duration_seconds")
        wall_s = f" · {wall:.1f}s" if isinstance(wall, (int, float)) else ""
        badge = f" — {_STATUS_BADGE.get(st, st)}{wall_s}"

    lines: List[str] = [f"## {name}{badge}", ""]
    if desc:
        lines.append(f"_{desc}_")
        lines.append("")
    if deps:
        lines.append(f"**Depends on:** {', '.join(deps)}")
        lines.append("")
    if narrative:
        lines.append(f"**Why this step:** {narrative}")
        lines.append("")

    # Code chunk
    inline_r = _extract_inline_r(cmd)
    chunk_label = _safe_label(name)
    if inline_r is not None:
        # Native R chunk — runs in the knit session, gets full RStudio
        # syntax highlighting, and shares state with downstream {r} cells.
        lines.append(f"```{{r {chunk_label}, eval=FALSE}}")
        lines.append(inline_r.rstrip())
        lines.append("```")
    else:
        # Plain bash chunk; the `set -euo pipefail` mirrors what the
        # FlowAgent local executor effectively gives you (-o pipefail
        # was already used by several of the GSE52778 plan steps).
        lines.append(f"```{{bash {chunk_label}, eval=FALSE}}")
        lines.append("set -euo pipefail")
        lines.append(cmd)
        lines.append("```")

    # Captured outputs from the actual run, if any. Rmd doesn't have a
    # native pre-populated-output concept like .ipynb, so we render them
    # as a fenced code block immediately after the chunk — visible in the
    # knit output without re-execution.
    if result is not None:
        out = (result.get("stdout") or "").strip()
        err = (result.get("stderr") or "").strip()
        if out or err:
            lines.append("")
            lines.append("**Captured output from the original run:**")
            if out:
                lines.append("```")
                lines.append(_clip(out))
                lines.append("```")
            if err:
                lines.append("")
                lines.append("**stderr:**")
                lines.append("```")
                lines.append(_clip(err))
                lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _safe_label(name: str) -> str:
    """knitr chunk labels must be unique and contain no spaces."""
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower() or "step"


def _clip(text: str, max_chars: int = 4000, max_lines: int = 60) -> str:
    """Trim verbose stdout / stderr so the rendered HTML stays usable."""
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[: max_lines // 2] + ["    [... output truncated ...]"] \
            + lines[-max_lines // 2 :]
    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[: max_chars // 2] + "\n[... output truncated ...]\n" \
            + out[-max_chars // 2 :]
    return out


def _outro_section(workflow: Dict[str, Any], output_root: Path,
                   run_results: Dict[str, Dict[str, Any]],
                   narrative_interpretation: Optional[str],
                   narrative_followup: Optional[str]) -> str:
    parts: List[str] = []
    if run_results:
        n = len(run_results)
        ok = sum(1 for r in run_results.values()
                 if str(r.get("status", "")).lower() in
                 ("completed", "success", "recovered"))
        parts.append("## Run summary")
        parts.append("")
        parts.append(f"- **Steps:** {n}")
        parts.append(f"- **Completed/recovered:** {ok}")
        parts.append(f"- **Failed:** {n - ok}")
        parts.append("")

    if narrative_interpretation:
        parts.append("## Results interpretation")
        parts.append("")
        parts.append(narrative_interpretation)
        parts.append("")

    found = [(label, output_root / rel, kind)
             for (label, rel, kind) in _RESULT_PROBE
             if (output_root / rel).exists()]
    if found:
        parts.append("## Inspect results")
        parts.append("")
        parts.append("The chunks below load the analysis outputs that exist "
                     "on disk. Set `eval=TRUE` on individual chunks (or in "
                     "the document YAML) when you knit.")
        parts.append("")
        for label, path, kind in found:
            try:
                rel = path.relative_to(output_root)
            except ValueError:
                rel = path
            label_id = _safe_label(label)
            if kind == "r-csv":
                parts.append(f"### {label}")
                parts.append("")
                parts.append(f"```{{r {label_id}, eval=FALSE}}")
                parts.append(f'df <- read.csv("{rel}")')
                parts.append("head(df, 20)")
                parts.append("dim(df)")
                parts.append("```")
                parts.append("")
            elif kind == "r-rds":
                parts.append(f"### {label}")
                parts.append("")
                parts.append(f"```{{r {label_id}, eval=FALSE}}")
                parts.append(f'txi <- readRDS("{rel}")')
                parts.append("str(txi, max.level = 1)")
                parts.append("dim(txi$counts)")
                parts.append("```")
                parts.append("")
            elif kind == "r-bed":
                parts.append(f"### {label}")
                parts.append("")
                parts.append(f"```{{r {label_id}, eval=FALSE}}")
                parts.append(f'peaks <- read.table("{rel}", sep="\\t",'
                             ' header=FALSE)')
                parts.append('cat("peaks:", nrow(peaks), "\\n")')
                parts.append("head(peaks, 20)")
                parts.append("```")
                parts.append("")
            elif kind == "vcf":
                parts.append(f"### {label}")
                parts.append("")
                parts.append(f"```{{bash {label_id}, eval=FALSE}}")
                parts.append(f"zcat {rel} | grep -v '^##' | head -50")
                parts.append("```")
                parts.append("")

    if narrative_followup:
        parts.append("## Suggested follow-up analyses")
        parts.append("")
        parts.append(narrative_followup)
        parts.append("")
    return "\n".join(parts)


def export_workflow_to_rmarkdown(
    workflow_json: Path | str,
    out_path: Path | str,
    *,
    run_results: Optional[Dict[str, Dict[str, Any]]] = None,
    prompt: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    narrative: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a knit-able R Markdown capturing a FlowAgent workflow."""
    workflow_json = Path(workflow_json)
    out_path      = Path(out_path)
    workflow      = json.loads(workflow_json.read_text())
    run_results   = run_results or {}
    narrative     = narrative or {}
    step_narr     = narrative.get("step_narrative") or {}

    blocks: List[str] = []
    blocks.append(_yaml_front_matter(workflow, prompt, metadata))
    blocks.append(_intro_section(
        workflow, prompt, metadata,
        narrative_intro=narrative.get("intro"),
    ))
    blocks.append("# Pipeline steps\n")
    for step in workflow.get("steps", []):
        blocks.append(_section_for_step(
            step, run_results.get(step.get("name")),
            narrative=step_narr.get(step.get("name")),
        ))
    blocks.append(_outro_section(
        workflow, output_root=workflow_json.parent.parent,
        run_results=run_results,
        narrative_interpretation=narrative.get("results_interpretation"),
        narrative_followup=narrative.get("followup"),
    ))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(blocks), encoding="utf-8")
    logger.info("Wrote Rmarkdown: %s (%d steps, narrative=%s)",
                out_path, len(workflow.get("steps", [])), bool(narrative))
    return out_path


# ── CLI ──────────────────────────────────────────────────────────

def _cli_main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workflow", required=True)
    ap.add_argument("--results",  default=None,
                    help="Optional JSON of {step_name: {status, stdout, ...}}")
    ap.add_argument("--prompt",   default=None)
    ap.add_argument("--out",      required=True, help="Output .Rmd path")
    args = ap.parse_args()

    run_results = None
    if args.results:
        run_results = json.loads(Path(args.results).read_text())
    export_workflow_to_rmarkdown(
        workflow_json=args.workflow, out_path=args.out,
        run_results=run_results, prompt=args.prompt,
    )


if __name__ == "__main__":
    _cli_main()
