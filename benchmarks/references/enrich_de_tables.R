#!/usr/bin/env Rscript
# Add gene_symbol column to Benchmark G/F DE reference tables in-place.
#
# Usage: Rscript references/enrich_de_tables.R

suppressPackageStartupMessages({
  library(org.Hs.eg.db)
  library(org.Mm.eg.db)
  library(AnnotationDbi)
})

here <- if (length(commandArgs(trailingOnly = TRUE)))
  commandArgs(trailingOnly = TRUE)[[1L]] else "references"

add_symbols <- function(path, db, keytype) {
  df <- read.delim(path, stringsAsFactors = FALSE)
  if ("gene_symbol" %in% names(df) && sum(nzchar(df$gene_symbol)) > 100) {
    message("[skip] ", basename(path), ": already enriched")
    return(invisible(NULL))
  }
  keys <- sub("\\..*$", "", df$gene_id)
  sym <- mapIds(db, keys = keys, column = "SYMBOL", keytype = keytype,
                multiVals = "first")
  df$gene_symbol <- unname(sym)
  df$gene_symbol[is.na(df$gene_symbol)] <- ""
  df <- df[, c("gene_id", "gene_symbol", "log2FoldChange", "padj")]
  write.table(df, path, sep = "\t", quote = FALSE, row.names = FALSE)
  message(sprintf("[ok] %s: %d/%d with symbol",
                  basename(path),
                  sum(nzchar(df$gene_symbol)), nrow(df)))
}

add_symbols(file.path(here, "gse52778_himes_DE.tsv"),
            org.Hs.eg.db, "ENSEMBL")
add_symbols(file.path(here, "gse152418_covid_blood_DE.tsv"),
            org.Hs.eg.db, "ENSEMBL")
add_symbols(file.path(here, "gse60450_fu_DE.tsv"),
            org.Mm.eg.db, "ENTREZID")
