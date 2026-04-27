#!/usr/bin/env Rscript
# Reference DE table for Benchmark F case `gse52778_dex_de`.
#
# This is intentionally a frozen recipe: load the canonical `airway`
# Bioconductor dataset (the tidied, version-controlled copy of GSE52778),
# fit a DESeq2 model with `untreated` as the reference level, and write
# the full results table. FlowAgent's candidate output is then scored
# against this with Spearman + top-N Jaccard.
#
# Why use the airway package rather than re-deriving from raw FASTQ:
#   - The airway data bundle is the community-canonical version of
#     GSE52778, sample-sheet-correct and frozen across Bioconductor
#     releases. Comparing against it removes upstream
#     alignment/quantification variance from the fidelity score.
#   - FlowAgent runs use kallisto + tximport; the airway package
#     provides STAR/HTSeq counts. Modest discordance between
#     quantifiers is therefore expected and is itself informative
#     (a high Spearman ρ across genes despite quantifier difference
#     is the signal we care about — that the *biology* matches).
#
# Output: TSV with columns (gene_id, log2FoldChange, padj). The runner's
# de_table comparator strips `.<digits>` Ensembl version suffixes before
# joining, so either bare or versioned IDs are fine here.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L) stop("usage: Rscript make_reference_gse52778.R <out.tsv>")
out_path <- args[[1L]]

suppressPackageStartupMessages({
  library(airway)
  library(DESeq2)
})

data("airway")
airway$dex <- relevel(airway$dex, ref = "untrt")

dds <- DESeqDataSet(airway, design = ~ cell + dex)
dds <- DESeq(dds)
res <- results(dds, contrast = c("dex", "trt", "untrt"))

de <- data.frame(
  gene_id        = rownames(res),
  log2FoldChange = res$log2FoldChange,
  padj           = res$padj,
  stringsAsFactors = FALSE
)
de <- de[!is.na(de$log2FoldChange), ]

dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
write.table(de, out_path, sep = "\t", quote = FALSE, row.names = FALSE)
cat(sprintf("[ok] wrote %d genes → %s\n", nrow(de), out_path))
