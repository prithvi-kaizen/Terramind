# Local setup

Run commands from the repository root. Python 3.12 is the reference interpreter for this setup. Python 3.14 is not the recommended environment for the inherited ML stack.

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py
```

`requirements.txt` pins direct dependencies. Six core versions come from the saved setup notebook: PyTorch 2.11.0, NumPy 2.4.4, Rasterio 1.5.0, pandas 3.0.2, Matplotlib 3.10.9, and TerraTorch 1.2.6. Additional direct dependencies are pinned for the maintained setup. A portable lock of all transitive dependencies remains on the implementation plan.

The model imports TerraTorch only when inference runs. Allow time for dependency imports before expecting a prediction. The maintained CLI builds the encoder architecture without downloading pretrained weights, then loads the complete included checkpoint with strict key matching. No API key or model download is needed for that demo. The original research notebooks still download pretrained weights through Hugging Face.

The original Linux deployment file `packages.txt` lists `gdal-bin` and `libgdal-dev`. Prebuilt Rasterio wheels usually supply the raster-reading runtime; if installation attempts a GDAL source build, use a supported wheel or an isolated conda environment with GDAL. Do not install Linux system packages on macOS or Windows.

## Dashboard

```bash
python -m streamlit run app.py
```

Open http://localhost:8501. The inherited dashboard attempts to start a local Flask process and falls back to in-process inference. A cold request can exceed its ten-second HTTP timeout and trigger a second inference. Use the CLI to test the model independently when troubleshooting. Explicit service lifecycle management and model caching are planned improvements.

The dashboard's model-card tab contains inherited presentation values, including batch size and metric labels that do not match every notebook record. Use [MODEL_CARD.md](MODEL_CARD.md) as the technical reference.

## CLI and output

```bash
python infer.py --input sample_input/0___0_S2B_20240326T165849_R1000_nonorm_bands20m.tiff --output outputs/demo
```

Use a 13-band, 512×512 GeoTIFF in the dataset's band order. The script selects zero-based indices 11 and 12 for B11 and B12. A standalone two-band GeoTIFF is not a valid input to the current loader, even though the model ultimately receives two channels.

`--threshold` sets the segmentation probability cutoff; it is not a calibrated probability of a methane leak. Output coordinates follow the schema described in [MODEL_CARD.md](MODEL_CARD.md). Model outputs can vary across devices and library versions; UUIDs and timestamps also make alert JSON different on every invocation. Dashboard and API outputs go to `outputs/predictions/`, keeping the historical `results/` files intact. Runtime alerts include local artifact paths so the dashboard can display masks returned by the API.

## Flask

```bash
python endpoint.py
```

The local endpoint listens on port 5001. Port 5000 was occupied by an AirTunes service during local testing, so this setup uses 5001. Upload under the multipart field `file`; optionally provide `threshold`. Avoid running a second instance on an occupied port. A health route is available at `GET /health`. Set `METHANEPULSE_PORT` in both service and dashboard environments to use another port. Production hosting, authentication, rate limiting, and long-running job handling are not part of this setup.

## Research notebooks

```bash
python -m pip install -r requirements-dev.txt
python -m ipykernel install --user --name terramind --display-name 'Terramind (Python 3.12)'
python -m jupyterlab
```

Notebook relative paths assume a kernel working directory at the repository root. At the start of each kernel, run:

```python
from pathlib import Path
import os

if Path.cwd().name == "notebooks":
    os.chdir(Path.cwd().parent)
assert Path("infer.py").exists(), "Set the kernel working directory to the repository root"
```

1. `setup.ipynb` checks libraries and creates `data/` and `outputs/` directories.
2. `data.ipynb` records exploration, including superseded band indexing and a rejected sample-ID interpretation. Its final segmentation cells use corrected bands and location splits. Treat earlier cells as historical exploration.
3. `baseline.ipynb` runs the SWIR anomaly baseline. Its original cell 5 references an undefined `train_df`. Before that cell, define `train_df = meta_df[~meta_df['location_id'].isin([8, 9])].reset_index(drop=True)`.
4. `TinyUNet.ipynb` evaluates an existing `outputs/checkpoints/tiny_unet_epoch15.pt`; that checkpoint and a complete training loop are not included.
5. `TerraMind.ipynb` builds and trains the frozen-encoder model, then writes `alert_packet.py` into the working directory. Training needs the full image/label dataset. The included root checkpoint can be copied to `outputs/checkpoints/terramind_methane_best.pt` for the notebook's loading cell.

Keep the root `src/alert_packet.py` helper distinct from the notebook-generated file. The CLI in `infer.py` is the maintained inference entry point.

## Checks

```bash
python scripts/validate_repository.py
python -m pytest
python -m pip check
```

Repository validation checks structure and artifact integrity; tests exercise input validation and API error handling. Neither substitutes for training-data evaluation. See [VALIDATION.md](VALIDATION.md) for the checks actually performed during this setup.
