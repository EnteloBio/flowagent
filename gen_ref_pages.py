"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

for path in sorted(Path("flowagent").rglob("*.py")):
    # Skip empty paths and special files
    if not path.name or path.name.startswith("_"):
        continue
        
    module_path = path.relative_to("flowagent").with_suffix("")
    doc_path = path.relative_to("flowagent").with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = list(module_path.parts)

    if not parts:  # Skip if parts is empty
        continue

    if parts[-1] == "__init__":
        if len(parts) == 1:  # Skip root __init__.py
            continue
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        print(f"# ::: flowagent.{identifier}", file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

# Write the navigation file only if we have items
if nav:
    with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
        nav_file.writelines(nav.build_literate_nav())
