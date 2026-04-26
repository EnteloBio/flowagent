"""Export a FlowAgent workflow + run results as a Jupyter notebook.

A FlowAgent run already emits ``workflow.json`` (the plan) and a DAG image,
but those formats are awkward for the standard "share a reproducible
analysis" workflow that bioinformaticians expect. This module renders the
same content into a ``.ipynb`` capsule:

  - Markdown header with the user prompt and run metadata.
  - One cell per workflow step: a markdown header describing the step
    plus a ``%%bash`` (or ``%%R`` / Python) cell containing the command.
  - When run results are available (stdout / stderr / status), they are
    pre-populated as cell outputs so the notebook is reviewable without
    being re-executed; otherwise the notebook is fresh and ready to run.
  - A trailing markdown cell summarising step statuses, plus optional
    "load results" cells that read DE tables / peak BEDs / VCFs into
    pandas if they exist at the standard paths.

The same module is the BixBench-style capsule format used by Benchmark F
(output fidelity) and Benchmark G (interpretation): a self-contained
notebook + reference data pair that a reviewer can re-execute.

Usage (Python):
    from flowagent.utils.export_notebook import export_workflow_to_notebook
    export_workflow_to_notebook(
        workflow_json="flowagent_output/Unnamed_Workflow/workflow.json",
        run_results={"fastqc": {"status": "completed", "stdout": "...",
                                "stderr": "...", "wall_seconds": 12.4}},
        out_path="flowagent_output/Unnamed_Workflow/notebook.ipynb",
        prompt="Download GSE52778 ...",
    )

Usage (CLI):
    python -m flowagent.utils.export_notebook \\
        --workflow flowagent_output/Unnamed_Workflow/workflow.json \\
        --results  flowagent_output/Unnamed_Workflow/run_results.json \\
        --out      flowagent_output/Unnamed_Workflow/notebook.ipynb
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import nbformat
from nbformat.v4 import (
    new_code_cell, new_markdown_cell, new_notebook, new_output,
)

logger = logging.getLogger(__name__)

# Common output paths for "load results" cells. Order matters: the first
# match seen on disk gets a dedicated preview cell.
_RESULT_PROBE_PATHS = (
    ("DESeq2 results",  "results/rna_seq_kallisto/deseq2/deseq2_results.csv"),
    ("Gene counts",     "results/rna_seq_kallisto/deseq2/gene_counts.csv"),
    ("MACS2 peaks",     "results/macs2/peaks.narrowPeak"),
    ("Filtered VCF",    "results/gatk/filtered.vcf.gz"),
    ("MultiQC report",  "results/rna_seq_kallisto/qc/multiqc_report.html"),
)

# Heuristics for choosing a cell language. We could parse the command but
# a fast prefix check covers the >95% case and avoids dependence on
# shlex tokenisation behaviour for cross-shell commands.
_R_PREFIXES      = ("Rscript", "R -e", "R --vanilla")
_PY_PREFIXES     = ("python ", "python3 ", "python -m", "python3 -m", "py ")


def _cell_for_step(step: Dict[str, Any],
                   result: Optional[Dict[str, Any]],
                   narrative: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return one markdown + one code cell for a single workflow step.

    ``step`` is a workflow.json step entry. ``result`` is the corresponding
    StepResult dict (status, stdout, stderr, wall_seconds, returncode);
    pass ``None`` if the workflow has not yet been executed. ``narrative``
    is optional LLM-generated assay-aware rationale for *this* step;
    when provided it appears as an italicised "Why this step" paragraph
    below the deterministic header.
    """
    name = step.get("name", "<unnamed>")
    desc = step.get("description") or ""
    deps = step.get("dependencies") or []
    cmd  = (step.get("command") or "").strip()

    # ── Markdown header for the step ────────────────────────────────
    status_badge = ""
    if result is not None:
        st = str(result.get("status", "?")).lower()
        glyph = {"completed": "🟢", "success": "🟢", "recovered": "🟡",
                 "failed": "🔴", "error": "🔴",
                 "skipped": "⚪", "pending": "⚪"}.get(st, "⚪")
        wall = result.get("wall_seconds") or result.get("duration_seconds")
        wall_str = f"  ·  {wall:.1f}s" if isinstance(wall, (int, float)) else ""
        status_badge = f" {glyph} **{st}**{wall_str}"

    md_lines = [f"### {name}{status_badge}"]
    if desc:
        md_lines.append(f"_{desc}_")
    if deps:
        md_lines.append(f"**Depends on:** {', '.join(deps)}")
    if narrative:
        md_lines.append(f"**Why this step:** {narrative}")
    md_cell = new_markdown_cell("\n\n".join(md_lines))

    # ── Code cell carrying the command ──────────────────────────────
    if cmd.startswith(_R_PREFIXES):
        code = cmd  # Rscript invocations stay verbatim in a shell cell
        prefix = "%%bash\n"
    elif cmd.startswith(_PY_PREFIXES):
        prefix = "%%bash\n"
        code = cmd
    else:
        # Default: shell. ``%%bash`` is more forgiving than ``!`` for
        # multi-line and pipefail commands, which dominate FlowAgent steps.
        prefix = "%%bash\nset -euo pipefail\n"
        code = cmd

    code_cell = new_code_cell(prefix + code)

    # If we have run results, pre-populate the cell's outputs so the
    # notebook is readable without re-execution.
    if result is not None:
        outputs: List[Any] = []
        stdout = (result.get("stdout") or "").strip()
        stderr = (result.get("stderr") or "").strip()
        rc     = result.get("returncode") or result.get("exit_code")

        if stdout:
            outputs.append(new_output(
                "stream", name="stdout", text=_clip(stdout)))
        if stderr:
            outputs.append(new_output(
                "stream", name="stderr", text=_clip(stderr)))
        if rc and int(rc) != 0:
            outputs.append(new_output(
                "stream", name="stderr",
                text=f"\n[exit code: {rc}]\n"))
        code_cell["outputs"] = outputs
        code_cell["execution_count"] = None  # cell wasn't run *in this kernel*

    return [md_cell, code_cell]


def _clip(text: str, max_chars: int = 6000, max_lines: int = 80) -> str:
    """Trim long step output so notebook stays openable in a browser."""
    lines = text.splitlines()
    clipped = False
    if len(lines) > max_lines:
        lines = lines[: max_lines // 2] + ["    [... output truncated ...]"] \
            + lines[-max_lines // 2 :]
        clipped = True
    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[: max_chars // 2] + "\n[... output truncated ...]\n" \
            + out[-max_chars // 2 :]
        clipped = True
    return out + ("\n" if clipped and not out.endswith("\n") else "")


def _intro_cells(workflow: Dict[str, Any], prompt: Optional[str],
                 metadata: Optional[Dict[str, Any]],
                 narrative_intro: Optional[str] = None
                 ) -> List[Dict[str, Any]]:
    """Header cells: title, prompt, optional LLM-written overview, plan table."""
    name  = workflow.get("name", "FlowAgent workflow")
    steps = workflow.get("steps", [])

    title_md = [f"# FlowAgent run — {name}",
                f"_Generated {datetime.now(timezone.utc).isoformat()}_",
                f"**Total steps:** {len(steps)}"]
    if metadata:
        for k in ("model", "provider", "git_sha", "executor"):
            if k in metadata:
                title_md.append(f"**{k}:** `{metadata[k]}`")

    intro: List[Dict[str, Any]] = [new_markdown_cell("\n\n".join(title_md))]
    if prompt:
        intro.append(new_markdown_cell(
            "## Original prompt\n\n> " + prompt.strip().replace("\n", "\n> ")))
    if narrative_intro:
        intro.append(new_markdown_cell(
            "## Analysis overview\n\n" + narrative_intro))
    intro.append(new_markdown_cell("## Workflow plan\n\n" + _plan_table(steps)))
    return intro


def _plan_table(steps: Sequence[Dict[str, Any]]) -> str:
    """Render the workflow plan as a markdown table for quick scanning."""
    rows = ["| # | Step | Description | Depends on |",
            "| --- | --- | --- | --- |"]
    for i, s in enumerate(steps, 1):
        deps = ", ".join(s.get("dependencies") or []) or "—"
        desc = (s.get("description") or "").replace("|", "\\|")[:120]
        rows.append(f"| {i} | `{s.get('name','?')}` | {desc} | {deps} |")
    return "\n".join(rows)


def _outro_cells(workflow: Dict[str, Any], output_root: Path,
                 run_results: Dict[str, Dict[str, Any]],
                 narrative_interpretation: Optional[str] = None,
                 narrative_followup: Optional[str] = None,
                 ) -> List[Dict[str, Any]]:
    """Trailing summary + interpretation + 'load results' cells."""
    cells: List[Dict[str, Any]] = []

    # Status summary
    if run_results:
        n = len(run_results)
        succ = sum(1 for r in run_results.values()
                   if str(r.get("status", "")).lower() in
                   ("completed", "success", "recovered"))
        cells.append(new_markdown_cell(
            f"## Run summary\n\n"
            f"- **Steps:** {n}\n"
            f"- **Completed/recovered:** {succ}\n"
            f"- **Failed:** {n - succ}"))

    if narrative_interpretation:
        cells.append(new_markdown_cell(
            "## Results interpretation\n\n" + narrative_interpretation))

    # Load-results probe — only emit cells for files that actually exist.
    found = [(label, output_root / rel) for (label, rel) in _RESULT_PROBE_PATHS
             if (output_root / rel).exists()]
    if found:
        cells.append(new_markdown_cell(
            "## Inspect results\n\nThe cells below load the analysis "
            "outputs that exist on disk. They are inert until you run them."))
        for label, path in found:
            rel = path.relative_to(output_root) if path.is_absolute() else path
            if str(path).endswith((".csv", ".tsv", ".tsv.gz")):
                sep = "','" if str(path).endswith(".csv") else "'\\\\t'"
                cells.append(new_code_cell(
                    f"# {label}\n"
                    f"import pandas as pd\n"
                    f"df = pd.read_csv({str(rel)!r}, "
                    f"sep={','.join(sep.split(chr(39))).strip(',') or ','!r})\n"
                    f"display(df.head(20))\n"
                    f"print(f'shape: {{df.shape}}')"))
            elif str(path).endswith(".narrowPeak") or str(path).endswith(".bed"):
                cells.append(new_code_cell(
                    f"# {label}\n"
                    f"import pandas as pd\n"
                    f"peaks = pd.read_csv({str(rel)!r}, sep='\\t', header=None)\n"
                    f"display(peaks.head(20))\n"
                    f"print(f'peaks: {{len(peaks)}}')"))
            elif str(path).endswith(".html"):
                cells.append(new_code_cell(
                    f"# {label}\n"
                    f"from IPython.display import IFrame\n"
                    f"IFrame({str(rel)!r}, width='100%', height=600)"))

    if narrative_followup:
        cells.append(new_markdown_cell(
            "## Suggested follow-up analyses\n\n" + narrative_followup))
    return cells


def export_workflow_to_notebook(
    workflow_json: Path | str,
    out_path: Path | str,
    *,
    run_results: Optional[Dict[str, Dict[str, Any]]] = None,
    prompt: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    narrative: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a Jupyter notebook capturing a FlowAgent workflow.

    Args:
        workflow_json: Path to ``workflow.json`` emitted by FlowAgent.
        out_path:      Destination ``.ipynb``.
        run_results:   Optional ``{step_name: {status, stdout, stderr, ...}}``
                       — when provided, output cells are pre-populated.
        prompt:        Original natural-language prompt.
        metadata:      Run metadata (model, provider, git_sha, executor)
                       to surface in the title cell.
        narrative:     Optional LLM-generated, assay-aware prose with keys
                       ``intro``, ``step_narrative`` (dict keyed by step
                       name), ``results_interpretation``, ``followup``.
                       See :mod:`flowagent.utils.notebook_narrative`.

    Returns the path to the written notebook.
    """
    workflow_json = Path(workflow_json)
    out_path      = Path(out_path)
    workflow      = json.loads(workflow_json.read_text())
    steps         = workflow.get("steps", [])
    run_results   = run_results or {}
    narrative     = narrative or {}
    step_narr     = narrative.get("step_narrative") or {}

    nb = new_notebook()
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "flowagent": {
            "schema_version": 1,
            "workflow_name":  workflow.get("name", ""),
            "n_steps":        len(steps),
            "model":          (metadata or {}).get("model"),
            "git_sha":        (metadata or {}).get("git_sha"),
            "narrative_present": bool(narrative),
        },
    }
    nb.cells.extend(_intro_cells(
        workflow, prompt=prompt, metadata=metadata,
        narrative_intro=narrative.get("intro"),
    ))
    for step in steps:
        nb.cells.extend(_cell_for_step(
            step,
            run_results.get(step.get("name")),
            narrative=step_narr.get(step.get("name")),
        ))
    nb.cells.extend(_outro_cells(
        workflow,
        output_root=workflow_json.parent.parent,
        run_results=run_results,
        narrative_interpretation=narrative.get("results_interpretation"),
        narrative_followup=narrative.get("followup"),
    ))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        nbformat.write(nb, fh)
    logger.info("Wrote notebook: %s (%d cells, narrative=%s)",
                out_path, len(nb.cells), bool(narrative))

    # Best-effort HTML render alongside the .ipynb. Rendering is fast (<1s
    # for notebooks of this size) and makes the artefact viewable from any
    # browser without a Jupyter install. Failures here are non-fatal.
    html_path = render_notebook_to_html(out_path)
    if html_path is not None:
        logger.info("Rendered notebook HTML: %s", html_path)
    return out_path


def render_notebook_to_html(ipynb_path: Path | str,
                            html_path: Optional[Path | str] = None
                            ) -> Optional[Path]:
    """Convert a ``.ipynb`` to a self-contained ``.html`` file.

    Uses ``nbconvert`` (already in our hard deps). Returns the rendered
    HTML path on success, ``None`` if conversion errored — caller should
    treat ``None`` as "skip; the notebook itself is still on disk".
    """
    try:
        from nbconvert import HTMLExporter
    except ImportError:
        logger.warning("nbconvert not available — skipping HTML render")
        return None

    ipynb_path = Path(ipynb_path)
    html_path  = Path(html_path) if html_path else ipynb_path.with_suffix(".html")
    try:
        with ipynb_path.open(encoding="utf-8") as fh:
            nb = nbformat.read(fh, as_version=4)
        exporter = HTMLExporter()
        # ``classic`` template is denser than ``lab`` — better fit for
        # a printable / archival artefact next to the methods Methods.
        exporter.template_name = "classic"
        body, _ = exporter.from_notebook_node(nb)
        html_path.write_text(body, encoding="utf-8")
        return html_path
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to render notebook HTML: %s", exc)
        return None


# ── CLI ──────────────────────────────────────────────────────────

def _cli_main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workflow", required=True,
                    help="Path to workflow.json")
    ap.add_argument("--results", default=None,
                    help="Optional JSON of {step_name: {status, stdout, "
                         "stderr, wall_seconds, ...}}")
    ap.add_argument("--prompt",  default=None)
    ap.add_argument("--out",     required=True,
                    help="Output .ipynb path")
    args = ap.parse_args()

    run_results = None
    if args.results:
        run_results = json.loads(Path(args.results).read_text())

    export_workflow_to_notebook(
        workflow_json=args.workflow,
        out_path=args.out,
        run_results=run_results,
        prompt=args.prompt,
    )


if __name__ == "__main__":
    _cli_main()
