#!/usr/bin/env Rscript
# Install all R / Bioconductor packages used by the Benchmark F
# reference-build scripts. Idempotent — packages already present are
# skipped. Bioconductor packages go through BiocManager so version
# alignment with the current R install is automatic.

required_cran  <- c()  # currently no CRAN-only deps
required_bioc  <- c("airway", "DESeq2",
                    "edgeR", "limma", "Glimma",
                    "SummarizedExperiment")
# Note: recount3 is intentionally NOT required — case 3 was migrated
# to a direct GEO download (GSE152418) so heavy Bioc-data dependencies
# are limited to the airway package (small, vendored counts).

# Bootstrap BiocManager itself if missing.
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager", repos = "https://cloud.r-project.org")
}

missing_cran <- setdiff(required_cran,
                        rownames(installed.packages()))
missing_bioc <- setdiff(required_bioc,
                        rownames(installed.packages()))

if (length(missing_cran) > 0L) {
  cat("Installing CRAN: ", paste(missing_cran, collapse = ", "), "\n")
  install.packages(missing_cran, repos = "https://cloud.r-project.org")
}
if (length(missing_bioc) > 0L) {
  cat("Installing Bioconductor: ", paste(missing_bioc, collapse = ", "), "\n")
  BiocManager::install(missing_bioc, update = FALSE, ask = FALSE)
}

cat("\n[ok] all required R / Bioc packages present:\n")
for (p in c(required_cran, required_bioc)) {
  v <- tryCatch(as.character(packageVersion(p)),
                error = function(e) "MISSING")
  cat(sprintf("  %-22s %s\n", p, v))
}
