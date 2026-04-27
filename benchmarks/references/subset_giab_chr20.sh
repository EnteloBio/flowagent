#!/usr/bin/env bash
# Subset the GIAB NA12878 v4.2.1 truth VCF to chr20.
#
# Input  (positional $1): full-genome benchmark VCF (gzipped).
# Output: same path with chr20 retained — overwrites the input in place
#         so the path declared in fidelity_cases.yaml's ``reference``
#         field stays correct.
#
# Prefers bcftools when available (single-pass, indexed, validated).
# Falls back to a streaming zgrep so the script still runs in
# minimalist images that lack htslib.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: subset_giab_chr20.sh <full_truth.vcf.gz>" >&2
  exit 2
fi
in_vcf="$1"
[[ -f "$in_vcf" ]] || { echo "input not found: $in_vcf" >&2; exit 2; }

out_vcf="${in_vcf%.full}"   # no-op if no .full suffix; we overwrite below
tmp="${in_vcf}.chr20.tmp.vcf.gz"

if command -v bcftools >/dev/null 2>&1 && command -v tabix >/dev/null 2>&1; then
  # bcftools view -r needs a tabix index. The raw GIAB download has no
  # .tbi sidecar, so we build one before the region subset. Both paths
  # are local file ops, so even multi-GB VCFs index in <30s.
  echo "  building tabix index"
  tabix -f -p vcf "$in_vcf"
  echo "  using bcftools view -r chr20"
  bcftools view -r chr20 -Oz -o "$tmp" "$in_vcf"
  rm -f "${in_vcf}.tbi"
else
  echo "  bcftools/tabix not both on PATH — falling back to streaming zgrep"
  # Keep the header (lines starting with #) plus chr20 records.
  ( zgrep -E '^#' "$in_vcf" ; zgrep -E '^chr20\b' "$in_vcf" ) | gzip -c > "$tmp"
fi

mv "$tmp" "$in_vcf"
n_records=$(zgrep -cv '^#' "$in_vcf" || true)
echo "[ok] subset to chr20: $n_records records → $in_vcf"
