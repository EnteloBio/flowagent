#!/usr/bin/env Rscript
# Reference DE table for Benchmark F case `gse152418_covid_blood_de`.
#
# Downloads the GSE152418 raw-counts matrix from NCBI (Ensembl gene IDs
# × samples), pulls the per-sample disease-state labels from the GEO
# accession-display API, and runs DESeq2 with healthy as the reference.
# No recount3 / Bioconductor-data-package dependencies — the only Bioc
# package required is DESeq2 itself.
#
# Output: TSV (gene_id, log2FoldChange, padj). Ensembl gene IDs are
# stripped of version suffixes (the de_table comparator does the same).

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L) stop("usage: Rscript make_reference_gse152418.R <out.tsv>")
out_path <- args[[1L]]

suppressPackageStartupMessages({
  library(DESeq2)
})

# 1. Counts matrix (rows = genes, cols = samples).
counts_url <- paste0(
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE152nnn/GSE152418/suppl/",
  "GSE152418_p20047_Study1_RawCounts.txt.gz"
)
tmp_counts <- tempfile(fileext = ".txt.gz")
download.file(counts_url, tmp_counts, mode = "wb", quiet = TRUE)
counts <- read.delim(gzfile(tmp_counts), header = TRUE, row.names = 1,
                     check.names = FALSE)

# 2. Sample-level disease-state labels from the GEO accession API. We
#    intentionally avoid GEOquery here so the script's only Bioc
#    dependency stays at DESeq2. The text-form response gives an
#    interleaved set of '!Sample_*' lines; we walk it and collect
#    (sample_title, disease_state) pairs.
meta_url <- paste0(
  "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?",
  "targ=gsm&view=brief&form=text&acc=GSE152418"
)
con  <- url(meta_url, open = "rt")
text <- readLines(con); close(con)

title <- ""; ds <- character(0); titles <- character(0); states <- character(0)
for (line in text) {
  if (startsWith(line, "!Sample_title")) {
    title <- sub(".* = ", "", line)
  } else if (startsWith(line, "!Sample_characteristics_ch1") &&
             grepl("disease state:", line, fixed = TRUE)) {
    state <- sub(".*disease state: ", "", line)
    titles <- c(titles, title)
    states <- c(states, state)
  }
}
sheet <- data.frame(sample_title = titles, disease_state = states,
                    stringsAsFactors = FALSE)

# 3. Align counts columns with the sheet. Drop convalescent samples.
keep_states <- c("Healthy", "COVID-19")
sheet <- sheet[sheet$disease_state %in% keep_states, ]
common <- intersect(colnames(counts), sheet$sample_title)
if (length(common) < 10L)
  stop(sprintf("only %d samples align between counts and metadata — check schema",
               length(common)))
counts <- counts[, common, drop = FALSE]
sheet  <- sheet[match(common, sheet$sample_title), ]
sheet$disease_state <- factor(sheet$disease_state, levels = c("Healthy", "COVID-19"))

# 4. DESeq2 with healthy = reference.
dds <- DESeqDataSetFromMatrix(countData = counts,
                              colData = sheet,
                              design = ~ disease_state)
dds <- DESeq(dds)
res <- results(dds, contrast = c("disease_state", "COVID-19", "Healthy"))

de <- data.frame(
  gene_id        = sub("\\..*$", "", rownames(res)),
  log2FoldChange = res$log2FoldChange,
  padj           = res$padj,
  stringsAsFactors = FALSE
)
de <- de[!is.na(de$log2FoldChange), ]

dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
write.table(de, out_path, sep = "\t", quote = FALSE, row.names = FALSE)
cat(sprintf("[ok] wrote %d genes (COVID-19 vs Healthy, Healthy=ref) → %s\n",
            nrow(de), out_path))
