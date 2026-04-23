#!/usr/bin/env Rscript
# Panels 5b + 5c — PCA and volcano for a FlowAgent DESeq2 case study.
#
# Reads the run's tximport RDS + DESeq2 results CSV + sample sheet, produces a
# single publication-ready two-panel PDF/SVG using ggplot2 + patchwork.
#
# Usage (from the case-study working directory):
#
#   Rscript benchmarks/case_study/pca_volcano.R \
#     --txi results/rna_seq_kallisto/deseq2/txi.rds \
#     --sample-sheet sample_conditions.tsv \
#     --de-csv results/rna_seq_kallisto/deseq2/deseq2_results.csv \
#     --counts results/rna_seq_kallisto/deseq2/gene_counts.csv \
#     --out figures/fig5bc_biology \
#     --reference untreated \
#     --highlight FKBP5,DUSP1,KLF15,PER1,CRISPLD2,TSC22D3

suppressPackageStartupMessages({
  library(optparse)
  library(DESeq2)
  library(ggplot2)
  library(ggrepel)
  library(patchwork)
})

# ── args ────────────────────────────────────────────────────────────────
op <- OptionParser(option_list = list(
  make_option("--txi",           type="character", default=NULL, help="Path to tximport RDS"),
  make_option("--sample-sheet",  type="character", default="sample_conditions.tsv"),
  make_option("--de-csv",        type="character", help="DESeq2 results CSV"),
  make_option("--counts",        type="character", default=NULL,
              help="Gene-counts CSV (used if --txi missing)"),
  make_option("--out",           type="character", default="fig5bc_biology"),
  make_option("--reference",     type="character", default=NULL,
              help="Reference factor level (e.g. 'untreated')"),
  make_option("--highlight",     type="character", default="",
              help="Comma-separated gene symbols to label on the volcano"),
  make_option("--padj",          type="double", default=0.05),
  make_option("--l2fc",          type="double", default=1.0)
))
args <- parse_args(op)

# ── colours (Okabe-Ito) ─────────────────────────────────────────────────
.C <- list(
  a = "#0072B2",   # blue
  b = "#E69F00",   # amber
  up   = "#b91c1c",
  down = "#0072B2",
  ns   = "#6b7280"
)

# ── panel B: PCA ────────────────────────────────────────────────────────
coldata <- read.table(args$`sample-sheet`, header = TRUE, sep = "\t",
                      row.names = 1, stringsAsFactors = FALSE)
coldata$condition <- factor(coldata$condition)
if (!is.null(args$reference) && args$reference %in% levels(coldata$condition)) {
  coldata$condition <- relevel(coldata$condition, ref = args$reference)
}

if (!is.null(args$txi) && file.exists(args$txi)) {
  txi <- readRDS(args$txi)
  keep <- intersect(colnames(txi$counts), rownames(coldata))
  stopifnot(length(keep) >= 2)
  txi$counts    <- txi$counts[, keep]
  txi$abundance <- txi$abundance[, keep]
  txi$length    <- txi$length[, keep]
  dds <- DESeqDataSetFromTximport(txi, colData = coldata[keep, , drop = FALSE],
                                  design = ~condition)
} else {
  # Fallback: read the gene-counts CSV and pretend it's already rounded integers.
  stopifnot(!is.null(args$counts))
  cts <- as.matrix(read.csv(args$counts, row.names = 1, check.names = FALSE))
  cts <- round(cts)
  mode(cts) <- "integer"
  keep <- intersect(colnames(cts), rownames(coldata))
  stopifnot(length(keep) >= 2)
  dds <- DESeqDataSetFromMatrix(cts[, keep],
                                colData = coldata[keep, , drop = FALSE],
                                design = ~condition)
}
vsd <- vst(dds, blind = TRUE)

# Build PCA manually so the plot matches publication style.
pca <- prcomp(t(assay(vsd)))
pct <- round(100 * (pca$sdev^2) / sum(pca$sdev^2), 1)
pca_df <- data.frame(
  PC1 = pca$x[, 1],
  PC2 = pca$x[, 2],
  sample = rownames(pca$x),
  condition = coldata[rownames(pca$x), "condition"]
)

p_pca <- ggplot(pca_df, aes(PC1, PC2, colour = condition)) +
  geom_point(size = 3.2, alpha = 0.9) +
  geom_text_repel(aes(label = sample), size = 2.6,
                  colour = "#374151", max.overlaps = Inf, min.segment.length = 0) +
  scale_colour_manual(values = c(.C$a, .C$b)) +
  labs(
    x = sprintf("PC1 (%.1f%%)", pct[1]),
    y = sprintf("PC2 (%.1f%%)", pct[2]),
    colour = "Condition",
    subtitle = "b  Sample-level PCA (VST-transformed counts)"
  ) +
  theme_bw(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    plot.subtitle = element_text(face = "bold", hjust = 0, size = 12),
    legend.position = "bottom"
  )

# ── panel C: volcano ────────────────────────────────────────────────────
de <- read.csv(args$`de-csv`, row.names = 1, check.names = FALSE)
stopifnot(all(c("log2FoldChange", "padj") %in% colnames(de)))
de$gene <- rownames(de)
de <- de[!is.na(de$padj) & !is.na(de$log2FoldChange), ]

# Classify
de$class <- "ns"
de$class[de$padj < args$padj & de$log2FoldChange >=  args$l2fc] <- "up"
de$class[de$padj < args$padj & de$log2FoldChange <= -args$l2fc] <- "down"
de$class <- factor(de$class, levels = c("ns", "down", "up"))

# Highlight set
highlight <- strsplit(args$highlight, ",")[[1]]
highlight <- trimws(highlight[nzchar(highlight)])
de$label <- ifelse(toupper(de$gene) %in% toupper(highlight), de$gene, "")

p_vol <- ggplot(de, aes(log2FoldChange, -log10(padj), colour = class)) +
  geom_point(size = 0.9, alpha = 0.55) +
  geom_point(data = subset(de, label != ""),
             colour = "#111827", size = 1.8) +
  geom_text_repel(
    data = subset(de, label != ""),
    aes(label = label), colour = "#111827",
    size = 3.2, fontface = "italic",
    max.overlaps = Inf, min.segment.length = 0, box.padding = 0.6
  ) +
  geom_hline(yintercept = -log10(args$padj), linetype = "dashed", colour = "#9ca3af") +
  geom_vline(xintercept =  c(-args$l2fc, args$l2fc),
             linetype = "dashed", colour = "#9ca3af") +
  scale_colour_manual(
    values = c(ns = .C$ns, down = .C$down, up = .C$up),
    labels = c("Not significant", "Down in treated", "Up in treated"),
    name = NULL
  ) +
  labs(
    x = expression(log[2]~"fold change"),
    y = expression(-log[10]~italic(P)[adj]),
    subtitle = sprintf("c  Differential expression (padj < %.2f, |log2FC| >= %.1f)",
                       args$padj, args$l2fc)
  ) +
  theme_bw(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    plot.subtitle = element_text(face = "bold", hjust = 0, size = 12),
    legend.position = "bottom"
  )

# ── compose + save ──────────────────────────────────────────────────────
combined <- p_pca + p_vol + plot_layout(ncol = 2, widths = c(1, 1.3))

dir.create(dirname(args$out), showWarnings = FALSE, recursive = TRUE)
ggsave(paste0(args$out, ".pdf"),  combined, width = 11, height = 5.2, device = cairo_pdf)
ggsave(paste0(args$out, ".svg"),  combined, width = 11, height = 5.2)
message(sprintf("Wrote %s.pdf and %s.svg", args$out, args$out))

# ── quick sanity summary printed to stdout ──────────────────────────────
n_up   <- sum(de$class == "up")
n_down <- sum(de$class == "down")
n_hl   <- sum(nzchar(de$label))
message(sprintf(
  "DE summary: %d up, %d down (padj<%.2f, |log2FC|>=%.1f); %d highlighted genes labelled.",
  n_up, n_down, args$padj, args$l2fc, n_hl
))
