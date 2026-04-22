"""Generate a bespoke run_deseq2.R using the LLM.

Standalone CLI so it can be invoked as a workflow step. Reads real transcript
IDs from kallisto output and the GTF, plus the sample sheet, then asks the LLM
to produce an R script tailored to the data. If the LLM call fails, a
deterministic template is written instead so the pipeline can still proceed.

Usage::

    python -m flowagent.utils.generate_deseq2_script \\
        --quant-dir results/rna_seq_kallisto/kallisto_quant \\
        --sample-sheet sample_conditions.tsv \\
        --gtf raw_data/reference/annotation.gtf \\
        --txi results/rna_seq_kallisto/deseq2/txi.rds \\
        --out-csv results/rna_seq_kallisto/deseq2/deseq2_results.csv \\
        --prompt "Differential expression between Blank and Pamc3 conditions" \\
        --out scripts/run_deseq2.R
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


def _read_abundance_tx_ids(quant_dir: Path, limit: int = 10) -> List[str]:
    for sample in sorted(quant_dir.iterdir()):
        abundance = sample / "abundance.tsv"
        if not abundance.is_file():
            continue
        ids: List[str] = []
        with abundance.open() as f:
            next(f, None)  # header
            for line in f:
                tx = line.split("\t", 1)[0].strip()
                if tx:
                    ids.append(tx)
                if len(ids) >= limit:
                    break
        if ids:
            return ids
    return []


def _read_gtf_tx_ids(gtf: Path, limit: int = 10) -> List[str]:
    pattern = re.compile(r'transcript_id "([^"]+)"')
    seen: List[str] = []
    with gtf.open(errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                continue
            m = pattern.search(line)
            if m and m.group(1) not in seen:
                seen.append(m.group(1))
                if len(seen) >= limit:
                    break
    return seen


def _read_sample_sheet(path: Path, max_rows: int = 20) -> Tuple[List[str], List[List[str]]]:
    with path.open() as f:
        reader_lines = [ln.rstrip("\n") for ln in f]
    if not reader_lines:
        return [], []
    header = reader_lines[0].split("\t")
    rows = [ln.split("\t") for ln in reader_lines[1 : 1 + max_rows] if ln.strip()]
    return header, rows


def _fallback_r_script(txi_path: str, sample_sheet: str, out_csv: str) -> str:
    return (
        "# Deterministic fallback — LLM generation unavailable\n"
        "suppressPackageStartupMessages(library(DESeq2))\n"
        f"txi <- readRDS('{txi_path}')\n"
        f"coldata <- read.table('{sample_sheet}', header=TRUE, row.names=1, sep='\\t')\n"
        "coldata$condition <- factor(coldata$condition)\n"
        "keep <- intersect(colnames(txi$counts), rownames(coldata))\n"
        "stopifnot(length(keep) >= 2)\n"
        "txi$counts    <- txi$counts[, keep]\n"
        "txi$abundance <- txi$abundance[, keep]\n"
        "txi$length    <- txi$length[, keep]\n"
        "dds <- DESeqDataSetFromTximport(txi, colData=coldata[keep,,drop=FALSE], design=~condition)\n"
        "dds <- DESeq(dds)\n"
        f"write.csv(as.data.frame(results(dds)), '{out_csv}')\n"
    )


async def _call_llm(user_prompt: str, system_prompt: str) -> str:
    from flowagent.core.llm import LLMInterface

    llm = LLMInterface()
    return await llm._call_openai(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        timeout=120,
    )


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # drop opening fence (with optional language) and trailing fence
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip() + "\n"


def _build_llm_prompt(
    *,
    user_intent: str,
    quant_tx_ids: List[str],
    gtf_tx_ids: List[str],
    sample_header: List[str],
    sample_rows: List[List[str]],
    txi_path: str,
    sample_sheet: str,
    out_csv: str,
) -> Tuple[str, str]:
    system = (
        "You are a senior RNA-seq analyst. Produce a single, self-contained R "
        "script that runs DESeq2 on a tximport object. The script MUST be "
        "valid R, idempotent, and write results to the CSV path given. "
        "Return ONLY the R script — no markdown, no commentary."
    )
    # Pre-compute strings that contain backslashes so they don't sit inside
    # f-string expression braces — Python < 3.12 rejects `\t` / `\n` literals
    # inside f-string expressions with a SyntaxError at parse time.
    tab = "\t"
    newline = "\n"
    intent = user_intent or "Run DESeq2 differential expression on the available kallisto quantification"
    sample_rows_block = newline.join("  " + tab.join(r) for r in sample_rows)
    user = f"""
Task: write an R script at an agreed path. It will be executed via `Rscript`.

User intent:
  {intent}

Inputs (paths the script must read):
  - tximport RDS: {txi_path}
  - sample sheet (tab-separated, first column is sample_id, has `condition` column): {sample_sheet}

Output:
  - write results to: {out_csv}

Sample sheet header:
  {sample_header}

Sample sheet rows (first {len(sample_rows)}):
{sample_rows_block}

Example transcript IDs from kallisto abundance.tsv (first column):
  {quant_tx_ids}

Example transcript IDs from the reference GTF (transcript_id attribute):
  {gtf_tx_ids}

Requirements:
  1. Use `library(DESeq2)` (suppress startup messages).
  2. `txi <- readRDS(...)`; do NOT rebuild tximport from abundance files.
  3. Read the sample sheet with `sep='\\t'`, `header=TRUE`, `row.names=1`.
  4. Coerce `coldata$condition` to a factor. If the user intent names an
     explicit reference / control (e.g. 'Blank'), use `relevel(..., ref=...)`.
  5. Intersect column names of `txi$counts` with rownames of the sample sheet
     before building the DESeq2 dataset — DO NOT assume they are already aligned.
  6. Handle the kallisto-vs-Ensembl version-suffix mismatch BEFORE calling
     `tximport` would have been relevant — but here we're using a pre-built
     txi, so this concern is already resolved upstream.
  7. `DESeqDataSetFromTximport(txi, colData=..., design=~condition)`.
  8. Run `DESeq()`, write `results(dds)` via `write.csv()` to the output path.
  9. If the user intent names specific levels for the contrast (e.g.
     'Blank vs Pamc3'), pass `contrast=c('condition','Pamc3','Blank')` or
     equivalent to `results()` so log2FC direction matches user intent.
 10. Use `dir.create(dirname(out), recursive=TRUE, showWarnings=FALSE)` to
     ensure the output directory exists before writing.

Return only the R source. No backticks. No explanation.
"""
    return system, user


async def _main_async(args: argparse.Namespace) -> int:
    quant_dir = Path(args.quant_dir)
    sample_sheet = Path(args.sample_sheet)
    gtf = Path(args.gtf)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not quant_dir.is_dir():
        print(f"ERROR: quant-dir not found: {quant_dir}", file=sys.stderr)
        return 2
    if not sample_sheet.is_file():
        print(f"ERROR: sample-sheet not found: {sample_sheet}", file=sys.stderr)
        return 2

    quant_tx_ids = _read_abundance_tx_ids(quant_dir)
    gtf_tx_ids = _read_gtf_tx_ids(gtf) if gtf.is_file() else []
    sample_header, sample_rows = _read_sample_sheet(sample_sheet)

    logger.info(
        "Context gathered: %d abundance tx IDs, %d GTF tx IDs, %d sample rows",
        len(quant_tx_ids), len(gtf_tx_ids), len(sample_rows),
    )

    if args.dry_run:
        script = _fallback_r_script(args.txi, str(sample_sheet), args.out_csv)
        out.write_text(script)
        print(f"Wrote fallback R script to {out} (--dry-run)")
        return 0

    system, user = _build_llm_prompt(
        user_intent=args.prompt or "",
        quant_tx_ids=quant_tx_ids,
        gtf_tx_ids=gtf_tx_ids,
        sample_header=sample_header,
        sample_rows=sample_rows,
        txi_path=args.txi,
        sample_sheet=str(sample_sheet),
        out_csv=args.out_csv,
    )

    try:
        raw = await _call_llm(user, system)
        script = _strip_code_fences(raw)
        if "DESeq" not in script or "readRDS" not in script:
            raise ValueError(
                "LLM response missing expected DESeq2/readRDS anchors; "
                "falling back to template"
            )
    except Exception as exc:
        logger.warning("LLM generation failed: %s — writing deterministic fallback", exc)
        script = _fallback_r_script(args.txi, str(sample_sheet), args.out_csv)

    out.write_text(script)
    print(f"Wrote {out} ({len(script)} bytes)")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--quant-dir", required=True, help="Directory of kallisto per-sample outputs")
    p.add_argument("--sample-sheet", required=True, help="sample_conditions.tsv path")
    p.add_argument("--gtf", required=True, help="Reference GTF (for example tx IDs only)")
    p.add_argument("--txi", required=True, help="Path to tximport RDS (read by the generated script)")
    p.add_argument("--out-csv", required=True, help="Output path the generated script writes results.csv to")
    p.add_argument("--prompt", default="", help="Free-text description of the comparison")
    p.add_argument("--out", required=True, help="Path to write the generated R script")
    p.add_argument("--dry-run", action="store_true", help="Skip LLM, write the deterministic template")
    args = p.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
