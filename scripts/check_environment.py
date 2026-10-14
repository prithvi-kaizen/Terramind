"""Inspect the local runtime without downloading weights or running inference."""

import importlib
import importlib.metadata
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    print(f"Python: {sys.version.split()[0]}")
    failures = []
    if sys.version_info[:2] != (3, 12):
        print("Note: Python 3.12 is the documented reference environment.")
    for package, module in [
        ("torch", "torch"), ("numpy", "numpy"), ("pandas", "pandas"),
        ("rasterio", "rasterio"), ("matplotlib", "matplotlib"),
        ("terratorch", "terratorch"), ("torchgeo", "torchgeo"),
        ("streamlit", "streamlit"), ("Flask", "flask"),
    ]:
        try:
            importlib.import_module(module)
            print(f"{package}: {importlib.metadata.version(package)}")
        except Exception as exc:
            failures.append(f"{package}: {exc}")
    for path in [ROOT / "terramind_methane_best.pt", *list((ROOT / "sample_input").glob("*.tiff"))]:
        if not path.is_file():
            failures.append(f"Missing artifact: {path.name}")
        else:
            print(f"Artifact: {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")
    if not list((ROOT / "sample_input").glob("*.tiff")):
        failures.append("No sample GeoTIFF found")
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Selected device: {device}")
        try:
            torch.ones(1, device=device).cpu()
        except Exception as exc:
            failures.append(f"Device smoke test: {exc}")
    for failure in failures:
        print(f"ERROR: {failure}")
    if failures:
        print("Install requirements.txt in the active environment; see docs/SETUP.md.")
        return 1
    print("Environment imports and required artifacts: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
