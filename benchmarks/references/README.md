# Benchmark F reference outputs

This directory holds the "gold-standard" output files that FlowAgent's
runs are scored against. Reference files are **not committed** —
materialise them on demand:

```bash
# 1. (One-time) install the R / Bioconductor packages used by the
#    DE-table reference builds (cases 1–3).
cd benchmarks
make install-r-deps

# 2. Fetch every reference declared in config/fidelity_cases.yaml.
make references

# Or just one case, e.g. the ER ChIP-seq peak reference:
python references/download_references.py --case gse32222_er_chip
# Skip the R-script cases entirely (CI / no R install):
make references SKIP_R=1
```

Cases use one of two source kinds:

- **`direct_url`** — HTTPS GET, optional gunzip + post-process step.
  The ChIP-seq and ATAC-seq peak references use this path. No R
  install required.
- **`r_script`** — runs an `Rscript` recipe that re-derives the
  reference from a frozen Bioconductor recipe. The three RNA-seq DE
  cases and the PBMC scRNA-seq cell-typing case use this path.

## What each file is and where it comes from

| Case ID | File | Source kind | Notes |
|---|---|---|---|
| `gse52778_dex_de` | `gse52778_himes_DE.tsv` | R script | DESeq2 on the canonical `airway` package; `untreated = ref` |
| `gse60450_mammary_de` | `gse60450_fu_DE.tsv` | R script | edgeR/limma-voom on the NCBI-hosted GSE60450 counts; `luminal = ref` |
| `gse152418_covid_blood_de` | `gse152418_covid_blood_DE.tsv` | R script | DESeq2 on the GEO-deposited GSE152418 raw counts (no `recount3`); `healthy = ref` |
| `gse32222_er_chip` | `gse32222_er_canonical.bed` | direct URL → MACS-txt → BED | One canonical GEO-deposited MACS peak file (GSM798425) |
| `gse74912_atac_immune` | `gse74912_corces_consensus.bed` | direct URL → counts-TSV → BED | First three columns of the deposited counts matrix are the consensus peak coordinates |
| `pbmc10k_celltypes` | `pbmc10k_canonical_markers.tsv` | R script | Seurat/SeuratData canonical PBMC marker genes per cell type |

> An ENCODE SUZ12 ChIP-seq case (`encsr000euq_suz12_h1`) is also
> defined but commented out in `config/fidelity_cases.yaml`; uncomment
> it to enable. There is no longer a GIAB variant-calling case.

## Requirements for re-deriving R references

```r
# DE cases (1–3) — installed by `make install-r-deps`:
BiocManager::install(c(
  "airway", "DESeq2", "edgeR", "limma", "Glimma",
  "org.Mm.eg.db", "SummarizedExperiment"
))
# PBMC scRNA case additionally needs Seurat + SeuratData (CRAN/GitHub).
```

`make install-r-deps` runs this for you. The download orchestrator
prints `[missing-r]` and skips a case if `Rscript` is not on `PATH`.

## Sizes and license notes

- DE TSV files: each ~1–3 MB. Computed locally; no redistribution issue.
- ENCODE narrowPeak: a few MB. Free-use under the ENCODE policy.
- GSE32222 / GSE74912 deposits at NCBI GEO: available without
  restriction; redistribute under the same conditions as the
  underlying publications.
- GIAB v4.2.1 truth VCF: chr20 subset is ~5 MB compressed. NIST policy
  permits redistribution with attribution; full citation should appear
  in any manuscript that uses it.

## Pinning downloads

For each `direct_url` case, set the optional `sha256` field in
`fidelity_cases.yaml` after a clean download to lock provenance:

```yaml
reference_source:
  kind: direct_url
  url: https://www.encodeproject.org/files/ENCFF938PCI/@@download/ENCFF938PCI.bed.gz
  gunzip_to: references/encsr000euq_suz12_idr_peaks.narrowPeak
  sha256: 3d4a...   # paste the hash printed by ``download_references.py``
```

Subsequent runs of `download_references.py --force` will fail loudly
if the upstream file changes, surfacing any silent re-deposit.

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `there is no package called 'DESeq2'` (or `edgeR`/`limma`/`Seurat`) | R/Bioc package missing | `make install-r-deps` (Seurat/SeuratData for the PBMC case must be installed separately) |
| `cannot open URL '...bioconductor.org/...course-materials/...'` | Stale teaching-materials URL | Pull latest of this repo (URLs were updated to NCBI mirrors) |
| `Could not retrieve index file` from bcftools | VCF lacks tabix index | Install `tabix` (`conda install -c bioconda htslib`) — script falls back to streaming zgrep otherwise |
| `HTTP Error 404` from Elsevier ScienceDirect | Publisher-hosted supplements move | Use the YAML's NCBI/GEO alternative (already migrated) |
