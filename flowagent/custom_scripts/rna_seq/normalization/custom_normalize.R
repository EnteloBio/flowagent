#!/usr/bin/env Rscript

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
args_dict <- list()
for (i in seq(1, length(args), 2)) {
    args_dict[[sub("^--", "", args[i])]] <- args[i + 1]
}

# Load required packages
library(DESeq2)
library(jsonlite)

# Read input data
counts <- read.csv(args_dict$counts_matrix, row.names=1)

# Perform normalization
dds <- DESeqDataSetFromMatrix(
    countData = counts,
    colData = data.frame(condition=factor(colnames(counts))),
    design = ~ 1
)
dds <- estimateSizeFactors(dds)
normalized_counts <- counts(dds, normalized=TRUE)

# Write output
output_file <- "normalized_counts.csv"
write.csv(normalized_counts, output_file)

# Return output paths as JSON
output <- list(
    normalized_counts = output_file
)
cat(toJSON(output))
