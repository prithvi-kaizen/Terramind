"""
MethanePulse — Flask inference endpoint
POST /predict   multipart/form-data
  file      : .tiff tile
  threshold : float (optional, default 0.5)
"""

from flask import Flask, request, jsonify
from pathlib import Path
import tempfile, sys, os
import math

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
from infer import run_inference
from rasterio.errors import RasterioIOError

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

CKPT    = BASE_DIR / "terramind_methane_best.pt"
RESULTS = BASE_DIR / "outputs" / "predictions"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "checkpoint_present": CKPT.is_file()})


@app.errorhandler(413)
def upload_too_large(error):
    return jsonify({"error": "Upload exceeds the 64 MiB limit"}), 413


@app.route("/predict", methods=["POST"])
def predict():
    # ── Validate file ─────────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "missing 'file' field in multipart/form-data"}), 400

    uploaded  = request.files["file"]
    if not uploaded.filename or Path(uploaded.filename).suffix.lower() not in {".tif", ".tiff"}:
        return jsonify({"error": "Upload a .tif or .tiff GeoTIFF"}), 400
    try:
        threshold = float(request.form.get("threshold", 0.5))
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"error": "threshold must be a finite number between 0 and 1 (exclusive)"}), 400

    if not CKPT.exists():
        return jsonify({"error": f"Checkpoint not found: {CKPT}"}), 500

    # ── Save upload to a temp file ────────────────────────────────
    tmp = tempfile.NamedTemporaryFile(suffix=".tiff", delete=False)
    try:
        uploaded.save(tmp.name)
        tmp.close()

        # ── Run inference (returns the full alert dict) ───────────
        alert = run_inference(
            tiff_path             = tmp.name,
            checkpoint_path       = str(CKPT),
            output_dir            = str(RESULTS),
            confidence_threshold  = threshold,
        )

        return jsonify(alert), 200

    except (ValueError, RasterioIOError) as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        app.logger.exception("Inference request failed")
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("METHANEPULSE_PORT", "5001")), debug=False)
