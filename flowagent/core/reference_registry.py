"""Lookup table for reference genome / transcriptome download URLs.

Maps (organism, genome_build, source) to concrete FTP/HTTP URLs for
FASTA, GTF, and pre-built tool indices.  Used by the pipeline planning
phase to inject download steps when no local reference is found.
"""

from dataclasses import dataclass
from typing import Dict, Optional

ENSEMBL_RELEASE = "113"
GENCODE_RELEASE_HUMAN = "46"
GENCODE_RELEASE_MOUSE = "M35"

ENSEMBL_BASE = f"https://ftp.ensembl.org/pub/release-{ENSEMBL_RELEASE}"
GENCODE_BASE = f"https://ftp.ebi.ac.uk/pub/databases/gencode"


@dataclass
class ReferenceFiles:
    """Resolved download URLs for a given organism + build + source."""
    fasta_url: Optional[str] = None
    gtf_url: Optional[str] = None
    cdna_url: Optional[str] = None
    prebuilt_index_url: Optional[str] = None

    @property
    def has_any(self) -> bool:
        return any([self.fasta_url, self.gtf_url, self.cdna_url, self.prebuilt_index_url])


# ── Ensembl references ────────────────────────────────────────

_ENSEMBL: Dict[str, Dict[str, ReferenceFiles]] = {
    "human": {
        "GRCh38": ReferenceFiles(
            fasta_url=f"{ENSEMBL_BASE}/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz",
            gtf_url=f"{ENSEMBL_BASE}/gtf/homo_sapiens/Homo_sapiens.GRCh38.{ENSEMBL_RELEASE}.gtf.gz",
            cdna_url=f"{ENSEMBL_BASE}/fasta/homo_sapiens/cdna/Homo_sapiens.GRCh38.cdna.all.fa.gz",
        ),
    },
    "mouse": {
        "GRCm39": ReferenceFiles(
            fasta_url=f"{ENSEMBL_BASE}/fasta/mus_musculus/dna/Mus_musculus.GRCm39.dna.primary_assembly.fa.gz",
            gtf_url=f"{ENSEMBL_BASE}/gtf/mus_musculus/Mus_musculus.GRCm39.{ENSEMBL_RELEASE}.gtf.gz",
            cdna_url=f"{ENSEMBL_BASE}/fasta/mus_musculus/cdna/Mus_musculus.GRCm39.cdna.all.fa.gz",
        ),
    },
    "rat": {
        "mRatBN7.2": ReferenceFiles(
            fasta_url=f"{ENSEMBL_BASE}/fasta/rattus_norvegicus/dna/Rattus_norvegicus.mRatBN7.2.dna.toplevel.fa.gz",
            gtf_url=f"{ENSEMBL_BASE}/gtf/rattus_norvegicus/Rattus_norvegicus.mRatBN7.2.{ENSEMBL_RELEASE}.gtf.gz",
            cdna_url=f"{ENSEMBL_BASE}/fasta/rattus_norvegicus/cdna/Rattus_norvegicus.mRatBN7.2.cdna.all.fa.gz",
        ),
    },
    "zebrafish": {
        "GRCz11": ReferenceFiles(
            fasta_url=f"{ENSEMBL_BASE}/fasta/danio_rerio/dna/Danio_rerio.GRCz11.dna.primary_assembly.fa.gz",
            gtf_url=f"{ENSEMBL_BASE}/gtf/danio_rerio/Danio_rerio.GRCz11.{ENSEMBL_RELEASE}.gtf.gz",
            cdna_url=f"{ENSEMBL_BASE}/fasta/danio_rerio/cdna/Danio_rerio.GRCz11.cdna.all.fa.gz",
        ),
    },
    "drosophila": {
        "BDGP6.46": ReferenceFiles(
            fasta_url=f"{ENSEMBL_BASE}/fasta/drosophila_melanogaster/dna/Drosophila_melanogaster.BDGP6.46.dna.toplevel.fa.gz",
            gtf_url=f"{ENSEMBL_BASE}/gtf/drosophila_melanogaster/Drosophila_melanogaster.BDGP6.46.{ENSEMBL_RELEASE}.gtf.gz",
            cdna_url=f"{ENSEMBL_BASE}/fasta/drosophila_melanogaster/cdna/Drosophila_melanogaster.BDGP6.46.cdna.all.fa.gz",
        ),
    },
    "c_elegans": {
        "WBcel235": ReferenceFiles(
            fasta_url=f"{ENSEMBL_BASE}/fasta/caenorhabditis_elegans/dna/Caenorhabditis_elegans.WBcel235.dna.toplevel.fa.gz",
            gtf_url=f"{ENSEMBL_BASE}/gtf/caenorhabditis_elegans/Caenorhabditis_elegans.WBcel235.{ENSEMBL_RELEASE}.gtf.gz",
            cdna_url=f"{ENSEMBL_BASE}/fasta/caenorhabditis_elegans/cdna/Caenorhabditis_elegans.WBcel235.cdna.all.fa.gz",
        ),
    },
}


# ── GENCODE references (human + mouse only) ──────────────────

_GENCODE: Dict[str, Dict[str, ReferenceFiles]] = {
    "human": {
        "GRCh38": ReferenceFiles(
            fasta_url=f"{GENCODE_BASE}/Gencode_human/release_{GENCODE_RELEASE_HUMAN}/GRCh38.primary_assembly.genome.fa.gz",
            gtf_url=f"{GENCODE_BASE}/Gencode_human/release_{GENCODE_RELEASE_HUMAN}/gencode.v{GENCODE_RELEASE_HUMAN}.primary_assembly.annotation.gtf.gz",
            cdna_url=f"{GENCODE_BASE}/Gencode_human/release_{GENCODE_RELEASE_HUMAN}/gencode.v{GENCODE_RELEASE_HUMAN}.transcripts.fa.gz",
        ),
    },
    "mouse": {
        "GRCm39": ReferenceFiles(
            fasta_url=f"{GENCODE_BASE}/Gencode_mouse/release_{GENCODE_RELEASE_MOUSE}/GRCm39.primary_assembly.genome.fa.gz",
            gtf_url=f"{GENCODE_BASE}/Gencode_mouse/release_{GENCODE_RELEASE_MOUSE}/gencode.v{GENCODE_RELEASE_MOUSE}.primary_assembly.annotation.gtf.gz",
            cdna_url=f"{GENCODE_BASE}/Gencode_mouse/release_{GENCODE_RELEASE_MOUSE}/gencode.v{GENCODE_RELEASE_MOUSE}.transcripts.fa.gz",
        ),
    },
}


# ── Default genome builds per organism ────────────────────────

DEFAULT_BUILDS: Dict[str, str] = {
    "human": "GRCh38",
    "mouse": "GRCm39",
    "rat": "mRatBN7.2",
    "zebrafish": "GRCz11",
    "drosophila": "BDGP6.46",
    "c_elegans": "WBcel235",
}

# Aliases so users can type common names
ORGANISM_ALIASES: Dict[str, str] = {
    "homo_sapiens": "human",
    "homo sapiens": "human",
    "mus_musculus": "mouse",
    "mus musculus": "mouse",
    "rattus_norvegicus": "rat",
    "rattus norvegicus": "rat",
    "danio_rerio": "zebrafish",
    "danio rerio": "zebrafish",
    "drosophila_melanogaster": "drosophila",
    "drosophila melanogaster": "drosophila",
    "fruit fly": "drosophila",
    "fly": "drosophila",
    "caenorhabditis_elegans": "c_elegans",
    "caenorhabditis elegans": "c_elegans",
    "worm": "c_elegans",
    "c. elegans": "c_elegans",
    "hg38": "human",
    "mm10": "mouse",
    "mm39": "mouse",
}


def resolve_organism(name: str) -> str:
    """Normalise an organism string to a canonical key."""
    key = name.strip().lower()
    return ORGANISM_ALIASES.get(key, key)


def list_organisms() -> list[str]:
    """Return all supported organism keys."""
    return sorted(DEFAULT_BUILDS.keys())


def default_build(organism: str) -> Optional[str]:
    """Return the default genome build for an organism, or None."""
    return DEFAULT_BUILDS.get(resolve_organism(organism))


def resolve_references(
    organism: str,
    build: Optional[str] = None,
    source: str = "ensembl",
) -> Optional[ReferenceFiles]:
    """Look up reference download URLs.

    Parameters
    ----------
    organism : str
        Canonical organism key or alias (e.g. "human", "homo_sapiens").
    build : str, optional
        Genome build (e.g. "GRCh38").  Uses default if omitted.
    source : str
        "ensembl" or "gencode".

    Returns
    -------
    ReferenceFiles or None if the combination is not in the registry.
    """
    org = resolve_organism(organism)
    build = build or DEFAULT_BUILDS.get(org)
    if not build:
        return None

    if source.lower() == "gencode":
        return _GENCODE.get(org, {}).get(build)
    return _ENSEMBL.get(org, {}).get(build)
