"""Maintainer script — regenerate benchmarks/data/known_tools.yaml.

Run this when the Bioconda/Bioconductor snapshot drifts more than a release
cycle (roughly every 6 months, or before a manuscript-grade sweep):

    python benchmarks/refresh_known_tools.py

The generated YAML is checked in; CI and the hallucination detector read the
cached file and do NOT fetch from the network at evaluation time.

Sources
-------
- Bioconda (conda channel, noarch + linux-64 repodata.json)
- Bioconductor (Software + Annotation + Experiment package listing, HTML parse)
- Runtime glue (hand-curated; updated in this file — no network needed)

The union of all three is written as three top-level YAML lists:

    bioconda:     [...] # package names from the channel
    bioconductor: [...] # package names from the package listing
    runtime:      [...] # curated infra / shell / runtime tools

Name normalisation: lower-case, hyphens and dots → underscores.  Duplicates
are removed; entries are sorted for clean diffs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import List, Set

_HERE = Path(__file__).parent
_OUT  = _HERE / "data" / "known_tools.yaml"

# ── Runtime / infrastructure glue (no network) ───────────────────────────────
#
# These are tools that legitimately appear as the first token of a command in a
# bioinformatics pipeline but are NOT bioinformatics tools themselves.  They
# extend the existing _RUNNER_TOKENS / _SHELL_TOKENS sets in metrics.py with
# common system, cloud, and orchestration commands that a pipeline author might
# invoke.

RUNTIME_TOOLS: List[str] = sorted({
    # language runtimes
    "python", "python3", "python2", "rscript", "r", "julia",
    "perl", "ruby", "node", "nodejs", "java", "scala",
    # shell / scripting
    "bash", "sh", "zsh", "dash", "ksh", "fish",
    # workflow managers
    "nextflow", "snakemake", "cromwell", "toil", "cwltool", "wdl",
    # containers / orchestration
    "docker", "singularity", "apptainer", "podman",
    "kubectl", "helm", "terraform",
    # package managers
    "conda", "mamba", "micromamba", "pip", "pixi", "brew",
    # cloud CLIs
    "aws", "gsutil", "gcloud", "az",
    # file system / transfer
    "wget", "curl", "aria2", "aria2c", "rsync", "scp", "sftp",
    "tar", "gzip", "gunzip", "zcat", "pigz", "bzip2", "xz",
    "bgzip", "tabix",
    "cp", "mv", "rm", "ln", "mkdir", "touch", "chmod", "chown",
    "find", "xargs", "parallel", "tee", "split",
    # build / make
    "make", "cmake", "git", "gcc", "g++", "clang",
    # monitoring / misc
    "time", "nohup", "screen", "tmux",
    "echo", "printf", "cat", "head", "tail", "grep", "sed", "awk",
    "cut", "sort", "uniq", "wc", "tr", "paste", "join",
    "jq", "yq", "xmllint",
    # HPC / scheduler
    "sbatch", "srun", "bsub", "qsub", "qstat", "squeue",
})


def _norm(name: str) -> str:
    """Lower-case, replace hyphens and dots with underscores."""
    return re.sub(r"[-.]", "_", name.lower().strip())


def _fetch_bioconda() -> Set[str]:
    """Return the set of package names available in the bioconda channel.

    Parses the conda repodata.json for both noarch and linux-64 sub-dirs.
    The repodata lists packages by filename (``<name>-<ver>-<build>.tar.bz2``
    or ``.conda``); we extract just the name component.
    """
    urls = [
        "https://conda.anaconda.org/bioconda/noarch/repodata.json",
        "https://conda.anaconda.org/bioconda/linux-64/repodata.json",
    ]
    names: Set[str] = set()
    for url in urls:
        print(f"  fetching {url} ...", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = json.loads(resp.read())
        except Exception as exc:
            print(f"  WARNING: could not fetch {url}: {exc}", file=sys.stderr)
            continue

        for key in ("packages", "packages.conda"):
            for fname in (data.get(key) or {}):
                # conda filename format: <name>-<version>-<build>.<ext>
                # Name can contain hyphens; version starts with a digit.
                m = re.match(r"^(.+?)-\d", fname)
                if m:
                    names.add(_norm(m.group(1)))

    print(f"  bioconda: {len(names)} packages", flush=True)
    return names


def _fetch_bioconductor() -> Set[str]:
    """Return package names from the Bioconductor Software + Annotation lists.

    Bioconductor exposes a plain-text PACKAGES file for each view:
        https://bioconductor.org/packages/release/bioc/src/contrib/PACKAGES
    We also grab Annotation and Experiment package lists.
    """
    urls = [
        "https://bioconductor.org/packages/release/bioc/src/contrib/PACKAGES",
        "https://bioconductor.org/packages/release/data/annotation/src/contrib/PACKAGES",
        "https://bioconductor.org/packages/release/data/experiment/src/contrib/PACKAGES",
    ]
    names: Set[str] = set()
    for url in urls:
        print(f"  fetching {url} ...", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            print(f"  WARNING: could not fetch {url}: {exc}", file=sys.stderr)
            continue
        for line in text.splitlines():
            if line.startswith("Package:"):
                pkg = line.split(":", 1)[1].strip()
                if pkg:
                    names.add(_norm(pkg))

    print(f"  bioconductor: {len(names)} packages", flush=True)
    return names


def _write_yaml(bioconda: Set[str], bioconductor: Set[str], runtime: List[str],
                out: Path) -> None:
    """Write the three sorted lists to a simple YAML file.

    Using a hand-rolled writer (no PyYAML dependency in the maintainer
    script) keeps the output diff-stable: one entry per line, no anchors
    or flow-style surprises.
    """
    lines = [
        "# Auto-generated by benchmarks/refresh_known_tools.py — do not edit by hand.",
        "# Sources: Bioconda channel repodata (noarch + linux-64),",
        "#          Bioconductor Software/Annotation/Experiment PACKAGES,",
        "#          hand-curated runtime / infra list.",
        "# Re-generate with: make refresh-tools  (or python benchmarks/refresh_known_tools.py)",
        "",
        "bioconda:",
    ]
    for n in sorted(bioconda):
        lines.append(f"  - {n}")

    lines.append("")
    lines.append("bioconductor:")
    for n in sorted(bioconductor):
        lines.append(f"  - {n}")

    lines.append("")
    lines.append("runtime:")
    for n in sorted(runtime):
        lines.append(f"  - {n}")

    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out} ({out.stat().st_size // 1024} KB, "
          f"{len(bioconda)+len(bioconductor)+len(runtime)} total entries)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(_OUT),
                    help="Output path (default: %(default)s)")
    ap.add_argument("--skip-bioconda", action="store_true",
                    help="Skip Bioconda fetch (useful if network is limited)")
    ap.add_argument("--skip-bioconductor", action="store_true",
                    help="Skip Bioconductor fetch")
    args = ap.parse_args()

    print("Refreshing known-tools snapshot …")

    bioconda:     Set[str] = set()
    bioconductor: Set[str] = set()

    if not args.skip_bioconda:
        bioconda = _fetch_bioconda()
    else:
        print("  bioconda: skipped")

    if not args.skip_bioconductor:
        bioconductor = _fetch_bioconductor()
    else:
        print("  bioconductor: skipped")

    # Normalise runtime list
    runtime_norm = sorted({_norm(t) for t in RUNTIME_TOOLS})

    _write_yaml(bioconda, bioconductor, runtime_norm, Path(args.out))


if __name__ == "__main__":
    main()
