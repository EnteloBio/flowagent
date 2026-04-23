# Case study — GSE52778 end-to-end (Figure 5 of the manuscript)

Three panels, produced from the artefacts of a single `flowagent prompt` run:

- **5a** — execution trace (Python + matplotlib)
- **5b** — sample PCA on VST-transformed counts (R + DESeq2)
- **5c** — DESeq2 volcano with canonical Dex targets labelled (R + ggplot2)

## Reproducing

Assuming the run finished and your working directory is
`benchmarks/results/realworld_GSE52778/`:

```bash
# 5a — execution trace
python benchmarks/case_study/timeline.py \
    --run-dir benchmarks/results/realworld_GSE52778 \
    --out figures/fig5a_timeline \
    --title "GSE52778 case study — per-step execution trace"

# 5b + 5c — PCA + volcano
cd benchmarks/results/realworld_GSE52778
Rscript ../../case_study/pca_volcano.R \
    --txi results/rna_seq_kallisto/deseq2/txi.rds \
    --sample-sheet sample_conditions.tsv \
    --de-csv results/rna_seq_kallisto/deseq2/deseq2_results.csv \
    --counts results/rna_seq_kallisto/deseq2/gene_counts.csv \
    --out ../../../figures/fig5bc_biology \
    --reference untreated \
    --highlight FKBP5,DUSP1,KLF15,PER1,CRISPLD2,TSC22D3
```

Outputs land in `figures/` as `.pdf` + `.svg`.

## Composing the final Figure 5

Two ways, both quick:

1. **Inkscape / Illustrator / Figma.** Open `fig5a_timeline.svg` and
   `fig5bc_biology.svg`, paste into a single canvas, align. Add the
   figure-level title and panel letters if not already present.

2. **Command-line composition.** Use `pdfjam` or `graphicx` directly in
   the manuscript's LaTeX:

   ```latex
   \begin{figure}
       \includegraphics[width=\textwidth]{figures/fig5a_timeline.pdf}\\
       \includegraphics[width=\textwidth]{figures/fig5bc_biology.pdf}
       \caption{\textbf{End-to-end case study on GSE52778.}
         (a) Per-step execution trace coloured by outcome class.
         (b) Sample PCA on VST-transformed counts; two clean clusters
         separate Dex-treated from untreated libraries.
         (c) DESeq2 differential expression; canonical glucocorticoid
         receptor targets (FKBP5, DUSP1, KLF15, PER1, CRISPLD2, TSC22D3)
         are recovered among the strongest upregulated genes, consistent
         with the original study of Himes et al.}
       \label{fig:case-study}
   \end{figure}
   ```

## R package prerequisites

```r
install.packages(c("optparse", "ggplot2", "ggrepel", "patchwork"))
BiocManager::install(c("DESeq2"))
```

## Expected findings on GSE52778

Strongly upregulated in Dex-treated:

- **FKBP5** — classical GR target, robust across glucocorticoid studies
- **DUSP1** — dual-specificity phosphatase, anti-inflammatory
- **KLF15** — Krüppel-like factor 15, muscle / metabolic target
- **PER1** — circadian gene, well-known GR response
- **CRISPLD2** — noted explicitly in the Himes et al. (2014) paper as a
  novel Dex-induced gene in airway smooth muscle
- **TSC22D3** (GILZ) — glucocorticoid-induced leucine zipper

Seeing these labelled high on the right side of the volcano is the
"it works" sanity check.
