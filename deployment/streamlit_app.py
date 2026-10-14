"""Community Cloud entrypoint using in-process model inference."""

import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["METHANEPULSE_INFERENCE_MODE"] = "direct"
runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
