"""Download / build reference outputs for Benchmark F (output fidelity).

Reads ``benchmarks/config/fidelity_cases.yaml`` and materialises every
case's ``reference`` file into ``benchmarks/references/`` according to
its ``reference_source`` block. Supports three source kinds:

  ``direct_url``   — HTTPS GET, optional gunzip / post-process. Used for
                     ENCODE peak deposits and the GIAB truth VCF.
  ``r_script``     — Run an R script that re-derives the reference from
                     a frozen Bioconductor recipe (e.g. ``airway`` →
                     DESeq2 DE table). The output path is the YAML's
                     ``reference`` field; the script is responsible for
                     writing exactly that path.
  ``post_process`` — Optional second-stage script (Python or shell)
                     applied after a download (e.g. subset GIAB to
                     chr20, parse a supplementary XLSX into BED).

Skipping behaviour: a reference whose target file already exists is
skipped unless ``--force`` is given. SHA-256 hashes are checked when
declared in the YAML so reviewers can confirm deterministic downloads.

Usage:
    python benchmarks/references/download_references.py
    python benchmarks/references/download_references.py --case gse52778_dex_de
    python benchmarks/references/download_references.py --force
    python benchmarks/references/download_references.py --skip-r-scripts
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_HERE = Path(__file__).parent
_REPO_BENCH = _HERE.parent           # benchmarks/
_DEFAULT_YAML = _REPO_BENCH / "config" / "fidelity_cases.yaml"


# ── Helpers ──────────────────────────────────────────────────────

def _http_get(url: str, dest: Path, *, chunk: int = 1 << 16) -> None:
    """Stream a URL to disk. Uses a UA string because some hosts reject
    the default ``Python-urllib`` agent (notably Elsevier ScienceDirect)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (FlowAgent benchmark fetch)",
    })
    with urllib.request.urlopen(req) as resp, dest.open("wb") as fh:
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            fh.write(buf)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _gunzip(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(src, "rb") as fi, dest.open("wb") as fo:
        shutil.copyfileobj(fi, fo)


def _resolve(p: str | Path, base: Path = _REPO_BENCH) -> Path:
    """Resolve a YAML path relative to ``benchmarks/``."""
    p = Path(p)
    return p if p.is_absolute() else base / p


def _run_post_process(script: Path, *args: str) -> None:
    """Dispatch a post-processing step by file extension.

    .py — invoked with the current Python interpreter.
    .sh / no-extension — invoked through bash.
    .R  — invoked through Rscript.
    """
    suffix = script.suffix.lower()
    if suffix == ".py":
        cmd = [sys.executable, str(script), *args]
    elif suffix == ".r":
        cmd = ["Rscript", str(script), *args]
    else:
        cmd = ["bash", str(script), *args]
    print(f"      → post-process: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ── Per-source-kind handlers ─────────────────────────────────────

def _handle_direct_url(case: Dict[str, Any], *, force: bool) -> bool:
    src = case["reference_source"]
    target = _resolve(case["reference"])
    if target.exists() and not force:
        print(f"  [skip] {case['id']}: {target.name} already exists")
        return False

    url = src["url"]
    # Stage in a temp file alongside the target so an interrupted
    # download leaves the previous good copy intact.
    tmp = target.with_suffix(target.suffix + ".part")
    print(f"  [http] {case['id']}: {url}")
    _http_get(url, tmp)

    # Optional gunzip step (yaml ``gunzip_to`` field — its value is the
    # *final* destination, replacing ``reference``).
    gunzip_to = src.get("gunzip_to")
    if gunzip_to:
        target = _resolve(gunzip_to)
        _gunzip(tmp, target)
        tmp.unlink()
    else:
        tmp.replace(target)

    # Optional post-process (e.g. subset, xlsx → bed).
    post = src.get("post_process")
    if post:
        _run_post_process(_resolve(post), str(target))

    expected = src.get("sha256")
    if expected:
        got = _sha256(target)
        if got != expected:
            raise RuntimeError(
                f"sha256 mismatch for {case['id']}: expected {expected}, got {got}"
            )
        print(f"      sha256 OK ({got[:12]}…)")
    return True


def _handle_r_script(case: Dict[str, Any], *, force: bool) -> bool:
    src = case["reference_source"]
    target = _resolve(case["reference"])
    if target.exists() and not force:
        print(f"  [skip] {case['id']}: {target.name} already exists")
        return False

    script = _resolve(src["script"])
    if not script.exists():
        print(f"  [missing] {case['id']}: R script {script} not found", file=sys.stderr)
        return False
    if shutil.which("Rscript") is None:
        print(f"  [missing-r] {case['id']}: Rscript not on PATH — skipping", file=sys.stderr)
        return False

    bioc = src.get("bioc_packages") or []
    print(f"  [Rscript] {case['id']}: {script.name}  (bioc: {', '.join(bioc) or '—'})")
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["Rscript", str(script), str(target)],
        check=True, cwd=_HERE,
    )
    if not target.exists():
        raise RuntimeError(f"{script.name} ran but {target} was not produced")
    return True


_HANDLERS = {
    "direct_url": _handle_direct_url,
    "r_script":   _handle_r_script,
}


# ── Driver ───────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=str(_DEFAULT_YAML),
                    help="Path to fidelity_cases.yaml")
    ap.add_argument("--case", default=None,
                    help="Only fetch this case ID")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if reference exists")
    ap.add_argument("--skip-r-scripts", action="store_true",
                    help="Skip cases that build references via Rscript "
                         "(useful in CI without an R install)")
    args = ap.parse_args()

    cfg   = yaml.safe_load(Path(args.cases).read_text())
    cases = cfg.get("cases", [])
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            sys.exit(f"case '{args.case}' not in {args.cases}")

    n_done   = 0
    n_failed: List[str] = []
    for case in cases:
        src = case.get("reference_source")
        if not src:
            print(f"  [no-source] {case['id']}: no reference_source declared")
            continue
        kind = src.get("kind")
        if kind == "r_script" and args.skip_r_scripts:
            print(f"  [skip-r] {case['id']}: --skip-r-scripts in effect")
            continue
        handler = _HANDLERS.get(kind)
        if handler is None:
            print(f"  [unknown] {case['id']}: kind={kind!r}", file=sys.stderr)
            n_failed.append(case["id"])
            continue
        try:
            if handler(case, force=args.force):
                n_done += 1
        except Exception as exc:
            print(f"  [error] {case['id']}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            n_failed.append(case["id"])

    print(f"\n[done] {n_done} reference(s) fetched/built. "
          f"{len(n_failed)} failure(s){':' + ', '.join(n_failed) if n_failed else ''}")
    if n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
