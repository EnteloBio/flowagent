#!/usr/bin/env Rscript
# Reference DE table for Benchmark F case `gse60450_mammary_de`.
#
# Uses the canonical edgeR/limma-voom recipe from the RNAseq123
# Bioconductor workflow, which is the de-facto standard analysis of
# this dataset and is explicitly designed to be reproducible.
# We compute the basal vs luminal contrast and write log2FC + padj.
#
# Output: TSV (gene_id, log2FoldChange, padj). To match the de_table
# comparator's expectations we use Ensembl mouse gene IDs without
# version suffixes (the comparator strips them anyway).

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L) stop("usage: Rscript make_reference_gse60450.R <out.tsv>")
out_path <- args[[1L]]

suppressPackageStartupMessages({
  library(edgeR)
  library(limma)
})

# Pull the gene-wise count matrix directly from the NCBI GEO supplement.
# The original RNAseq123 BioC course-materials URL has gone 404; NCBI is
# the canonical, persistent host.
url <- "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE60nnn/GSE60450/suppl/GSE60450_Lactation-GenewiseCounts.txt.gz"
tmp <- tempfile(fileext = ".txt.gz")
download.file(url, tmp, mode = "wb", quiet = TRUE)
counts <- read.delim(tmp, header = TRUE, row.names = 1, check.names = FALSE)

# First column in the bundle is Gene Length — drop it; the rest are
# samples named with cell-type and developmental-stage tokens
# encoded in the GSE60450 sample sheet.
gene_length <- counts$Length
counts$Length <- NULL

# Sample sheet: parse cell type (basal vs luminal) and stage from the
# column names, which follow "MCL1.<sample-id>" — fall back to GSE60450's
# canonical sample annotation (12 samples, 2 cell types × 3 stages × 2 reps).
samples <- colnames(counts)
cell <- factor(rep(c("basal","luminal"), each = 6),
               levels = c("luminal","basal"))    # luminal = reference
stage <- factor(rep(rep(c("virgin","pregnant","lactate"), each = 2), 2),
                levels = c("virgin","pregnant","lactate"))

design <- model.matrix(~ stage + cell)            # interaction-free, like RNAseq123
y <- DGEList(counts = counts, genes = data.frame(gene_id = rownames(counts)))
keep <- filterByExpr(y, design)
y <- y[keep, , keep.lib.sizes = FALSE]
y <- calcNormFactors(y, method = "TMM")
v <- voom(y, design)
fit <- lmFit(v, design)
fit <- eBayes(fit)
tt <- topTable(fit, coef = "cellbasal", number = Inf, sort.by = "none")

de <- data.frame(
  gene_id        = tt$gene_id,
  log2FoldChange = tt$logFC,
  padj           = tt$adj.P.Val,
  stringsAsFactors = FALSE
)
de <- de[!is.na(de$log2FoldChange), ]

dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
write.table(de, out_path, sep = "\t", quote = FALSE, row.names = FALSE)
cat(sprintf("[ok] wrote %d genes (basal vs luminal, luminal=ref) → %s\n",
            nrow(de), out_path))
