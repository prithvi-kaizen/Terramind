"""Check committed project structure without importing the ML stack."""

import ast
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def main():
    failures = []
    required = [
        "README.md", "requirements.txt", "requirements-dev.txt", "app.py",
        "infer.py", "endpoint.py", "CONTRIBUTING.md", "SECURITY.md",
        "docs/SETUP.md", "docs/DATA.md", "docs/MODEL_CARD.md",
        "docs/IMPLEMENTATION_PLAN.md", "docs/PROVENANCE.md", "docs/VALIDATION.md",
        ".github/workflows/ci.yml", "docs/artifact_manifest.json",
    ]
    for name in required:
        if not (ROOT / name).is_file():
            failures.append(f"Missing required file: {name}")
    python_files = [ROOT / name for name in ["app.py", "infer.py", "endpoint.py"]]
    for folder in ["src", "scripts", "tests", "deployment"]:
        python_files.extend((ROOT / folder).glob("*.py"))
    for path in python_files:
        try:
            ast.parse(path.read_text(), filename=str(path.relative_to(ROOT)))
        except (SyntaxError, OSError) as exc:
            failures.append(f"Python syntax: {exc}")
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    if len(notebooks) != 5:
        failures.append("Expected the five imported research notebooks")
    for path in notebooks:
        try:
            notebook = json.loads(path.read_text())
            assert notebook["nbformat"] == 4
            for index, cell in enumerate(notebook["cells"]):
                assert cell["cell_type"] in {"code", "markdown", "raw"}
                source = "".join(cell["source"])
                if cell["cell_type"] == "code":
                    # The original alert-writing cell is valid Python after its cell magic.
                    if source.startswith("%%writefile"):
                        source = source.split("\n", 1)[1]
                    ast.parse(source, filename=f"{path.name}:cell {index + 1}")
        except (AssertionError, KeyError, ValueError, SyntaxError) as exc:
            failures.append(f"Notebook {path.name}: {exc}")
    documents = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", ROOT / "SECURITY.md", *list((ROOT / "docs").rglob("*.md"))]
    for path in documents:
        if not path.exists():
            continue
        for target in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", path.read_text()):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            target = unquote(target.split("#", 1)[0])
            if target and not (path.parent / target).exists():
                failures.append(f"Broken local link in {path.relative_to(ROOT)}: {target}")
    manifest_path = ROOT / "docs/artifact_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for artifact in manifest["artifacts"]:
            path = ROOT / artifact["path"]
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
                failures.append(f"Imported artifact missing or changed: {artifact['path']}")
    for path in (ROOT / "docs").glob("*.json"):
        try:
            json.loads(path.read_text())
        except ValueError as exc:
            failures.append(f"Invalid JSON in {path.name}: {exc}")
    for failure in failures:
        print(f"ERROR: {failure}")
    if failures:
        return 1
    print(f"PASS: {len(python_files)} Python files, {len(notebooks)} notebooks, documentation links, JSON, and imported artifact hashes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
