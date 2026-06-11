"""Generate manuscript Supplementary Table 3 — software package versions
exercised in the benchmark sweep.

Every harness run writes a ``manifest.json`` recording, among other
provenance, the full set of installed Python package versions (see
``harness/runner.py::write_manifest``). The merged planning sweep's
manifest in turn lists its ``sources`` — the per-model run directories
that were consolidated. This script walks those source manifests,
consolidates the package → version map (flagging any version drift
across runs), and emits:

* ``supp_table3_packages.md``  — a curated table of the key dependencies
  (LLM SDKs, orchestration, agent core, scientific stack) for the
  manuscript supplement, with the Python version / platform / git SHA in
  the caption.
* ``supp_table3_packages.tsv`` — the *complete* environment (every
  installed package and the version(s) observed) for full reviewer
  reproducibility.

By default it reads the latest merged planning run (matching
``make_supp_tables.py`` / ``supp_table_models.py``), but ``--run`` can
point at any run directory: a merged manifest (with ``sources``) is
expanded automatically, while a plain run manifest (with ``packages``)
is used directly.

Usage::

    python make_supp_table3.py
    python make_supp_table3.py --run results/planning/_merged/<TS>
    python make_supp_table3.py --all-key   # full table = key rows only
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).parent


# ── Curated "key dependency" groups for the readable Markdown table ───
# Ordered category → list of distribution names (as reported by
# importlib.metadata, i.e. PyPI names). Packages not present in the
# manifest are silently skipped, so this list can stay aspirational.
_KEY_GROUPS: List[Tuple[str, List[str]]] = [
    ("Agent framework", [
        "flowagent", "bench",
    ]),
    ("LLM provider SDKs", [
        "openai", "anthropic", "google-genai", "google-generativeai",
        "litellm",
    ]),
    ("LLM orchestration", [
        "langchain", "langchain-core", "langchain-community",
        "langchain-openai", "langchain-text-splitters",
        "langgraph", "langgraph-checkpoint", "langgraph-prebuilt",
        "langgraph-sdk", "tiktoken",
    ]),
    ("Validation / config", [
        "pydantic", "pydantic-settings", "pydantic_core", "pydantic-core",
        "PyYAML", "pyyaml",
    ]),
    ("Scientific stack", [
        "numpy", "scipy", "pandas", "matplotlib",
    ]),
    ("HTTP / resilience", [
        "httpx", "httpx-sse", "tenacity",
    ]),
]


# ── Manifest discovery ────────────────────────────────────────────────

def _latest_merged(base: Path) -> Optional[Path]:
    root = base / "planning" / "_merged"
    if not root.exists():
        return None
    runs = sorted([p for p in root.iterdir() if p.is_dir()],
                  key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def _load_manifest(run_dir: Path) -> Optional[Dict[str, Any]]:
    mf = run_dir / "manifest.json"
    if not mf.exists():
        return None
    try:
        return json.loads(mf.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _resolve(base: Path, run_dir: str) -> Path:
    """Source run_dir entries are stored relative to the benchmarks dir."""
    p = Path(run_dir)
    if p.is_absolute():
        return p
    # Stored like "results/planning/<TS>" → relative to _HERE.
    cand = _HERE / p
    if cand.exists():
        return cand
    # Fall back to results-base parent.
    return base.parent / p


def collect_source_manifests(run_dir: Path, base: Path) -> List[Tuple[str, Dict[str, Any]]]:
    """Return [(label, manifest)] for every run contributing packages.

    A merged manifest (``sources``) is expanded into its constituent
    run manifests; a plain run manifest is returned as a single entry.
    """
    top = _load_manifest(run_dir)
    if top is None:
        return []

    out: List[Tuple[str, Dict[str, Any]]] = []
    sources = top.get("sources")
    if isinstance(sources, list) and sources:
        for s in sources:
            sd = _resolve(base, s.get("run_dir", ""))
            man = _load_manifest(sd)
            if man and man.get("packages"):
                out.append((sd.name, man))
    if not out and top.get("packages"):
        # Non-merged run, or merged manifest that itself carries packages.
        out.append((run_dir.name, top))
    return out


# ── Consolidation ─────────────────────────────────────────────────────

def consolidate(manifests: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Build a name → {versions: {ver: n_runs}} map plus env metadata."""
    pkg_versions: Dict[str, Dict[str, int]] = {}
    pythons: Dict[str, int] = {}
    platforms: Dict[str, int] = {}
    git_shas: Dict[str, int] = {}

    for _label, man in manifests:
        for name, ver in (man.get("packages") or {}).items():
            pkg_versions.setdefault(name, {})
            pkg_versions[name][str(ver)] = pkg_versions[name].get(str(ver), 0) + 1
        py = man.get("python")
        if py:
            pythons[str(py)] = pythons.get(str(py), 0) + 1
        plat = man.get("platform")
        if plat:
            platforms[str(plat)] = platforms.get(str(plat), 0) + 1
        sha = man.get("git_sha")
        if sha:
            git_shas[str(sha)] = git_shas.get(str(sha), 0) + 1

    return {
        "n_runs": len(manifests),
        "packages": pkg_versions,
        "pythons": pythons,
        "platforms": platforms,
        "git_shas": git_shas,
    }


def _version_str(versions: Dict[str, int]) -> str:
    """Single version, or '; '-joined list when there's drift across runs."""
    if len(versions) == 1:
        return next(iter(versions))
    # Most-common first.
    ordered = sorted(versions.items(), key=lambda kv: (-kv[1], kv[0]))
    return "; ".join(v for v, _ in ordered)


def _short_py(py: str) -> str:
    """Trim the verbose sys.version string to just the X.Y.Z token."""
    return py.split()[0] if py else py


def _top_key(counter: Dict[str, int]) -> str:
    if not counter:
        return "—"
    return max(counter.items(), key=lambda kv: kv[1])[0]


# ── Rendering ─────────────────────────────────────────────────────────

def _caption(info: Dict[str, Any], n_pkgs: int, source_name: str) -> str:
    py = _short_py(_top_key(info["pythons"]))
    plat = _top_key(info["platforms"])
    sha = _top_key(info["git_shas"])
    sha_short = sha[:12] if sha and sha != "—" else sha
    drift = sum(1 for v in info["packages"].values() if len(v) > 1)
    drift_note = (f" {drift} package(s) varied in version across runs "
                  f"(all versions listed)." if drift else
                  " All runs shared an identical environment.")
    return (
        f"**Supplementary Table 3. Software environment exercised in the "
        f"benchmark sweep.** Package versions were captured from the "
        f"per-run `manifest.json` provenance records of the "
        f"{info['n_runs']} model runs consolidated in the merged planning "
        f"sweep (`{source_name}`). Runs executed under Python {py} on "
        f"{plat} at git revision `{sha_short}`. The full environment "
        f"({n_pkgs} packages) is provided in the accompanying TSV; the key "
        f"dependencies are listed below.{drift_note}\n"
    )


def render_key_md(info: Dict[str, Any], source_name: str) -> str:
    pkgs = info["packages"]
    # Case-insensitive lookup so curated names match manifest casing.
    lower = {k.lower(): k for k in pkgs}

    lines = [_caption(info, len(pkgs), source_name),
             "| Category | Package | Version |",
             "| --- | --- | --- |"]
    seen: set = set()
    for category, names in _KEY_GROUPS:
        first = True
        for want in names:
            key = lower.get(want.lower())
            if key is None or key in seen:
                continue
            seen.add(key)
            ver = _version_str(pkgs[key])
            lines.append("| " + " | ".join([
                category if first else "",
                f"`{key}`",
                ver,
            ]) + " |")
            first = False
    return "\n".join(lines)


def build_full_rows(info: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name in sorted(info["packages"], key=str.lower):
        versions = info["packages"][name]
        rows.append({
            "package": name,
            "version": _version_str(versions),
            "n_distinct_versions": len(versions),
            "n_runs": sum(versions.values()),
        })
    return rows


def _write_tsv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-base", default="results")
    ap.add_argument("--run", default=None,
                    help="Run dir to source manifests from (default: latest "
                         "merged planning run)")
    ap.add_argument("--out-dir", default="results/figures")
    ap.add_argument("--all-key", action="store_true",
                    help="Render the Markdown table from the full package "
                         "list rather than the curated key subset")
    args = ap.parse_args()

    base = (_HERE / args.results_base) if not Path(args.results_base).is_absolute() \
        else Path(args.results_base)
    out_dir = (_HERE / args.out_dir) if not Path(args.out_dir).is_absolute() \
        else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = Path(args.run) if args.run else _latest_merged(base)
    if run_dir is None:
        raise SystemExit(f"No merged planning run found under "
                         f"{base/'planning'/'_merged'}; pass --run explicitly.")
    if not args.run and not run_dir.is_absolute():
        pass  # _latest_merged already returns an absolute path

    manifests = collect_source_manifests(run_dir, base)
    if not manifests:
        raise SystemExit(
            f"No package manifests found via {run_dir/'manifest.json'}. "
            f"Ensure the run (or its sources) wrote manifest.json with a "
            f"'packages' block.")

    info = consolidate(manifests)
    n_pkgs = len(info["packages"])

    # ── Machine-readable full environment (TSV) ──
    full_rows = build_full_rows(info)
    tsv_path = out_dir / "supp_table3_packages.tsv"
    _write_tsv(tsv_path, full_rows)

    # ── Manuscript Markdown ──
    if args.all_key:
        # Render every package as its own row, ungrouped.
        info_for_md = dict(info)
        # Reuse the key renderer by collapsing all packages into one group.
        global _KEY_GROUPS  # noqa: PLW0603 — intentional local override
        saved = _KEY_GROUPS
        _KEY_GROUPS = [("All", sorted(info["packages"], key=str.lower))]
        md = render_key_md(info_for_md, run_dir.name)
        _KEY_GROUPS = saved
    else:
        md = render_key_md(info, run_dir.name)
    md_path = out_dir / "supp_table3_packages.md"
    md_path.write_text(md + "\n")

    drift = sum(1 for v in info["packages"].values() if len(v) > 1)
    print(f"[ok] Table 3 → {md_path} "
          f"({n_pkgs} packages from {info['n_runs']} source runs"
          + (f", {drift} with version drift" if drift else "") + ")")
    print(f"[ok] full environment TSV → {tsv_path}")
    print(f"[info] Python {_short_py(_top_key(info['pythons']))} on "
          f"{_top_key(info['platforms'])}; source {run_dir.name}")


if __name__ == "__main__":
    main()
