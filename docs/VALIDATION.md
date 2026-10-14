# Setup validation

Validation date: 14 October 2026. Reference environment: Python 3.12.4 on macOS, with CPU inference. These checks establish that the imported demo can execute; they do not establish new model-quality results.

## Completed checks

| Check | Result |
| --- | --- |
| Install direct dependencies from `requirements.txt` in an isolated virtual environment | Passed after pinning TorchGeo 0.7.1 for TerraTorch 1.2.6 compatibility |
| `python scripts/check_environment.py` | Runtime imports, included assets, and CPU tensor smoke test passed |
| `python -m pip check` | No broken requirements |
| `python -m pytest -q` | 27 tests passed |
| Real CLI sample inference with `HF_HUB_OFFLINE=1` | Loaded all checkpoint keys and produced a mask plus schema 1.1 alert without downloading model weights |
| Predicted-mask comparison with imported sample result | Pixel-for-pixel identical; 1,244 predicted foreground pixels |
| Output raster contract | Single-band uint8, values {0, 1}, same dimensions, transform, and CRS as input |
| Alert output | WGS84 centroid added; unverified hard-coded validation score removed |
| Imported artifact integrity | 13 artifacts byte-compared against upstream and recorded with SHA-256 |
| `python scripts/validate_repository.py` | Eight Python files, five notebooks, documentation links, JSON, and imported artifact hashes passed |
| Real `POST /predict` sample request | HTTP 200, schema 1.1 alert, same 1,244-pixel prediction, and WGS84 coordinates |
| Browser dashboard smoke test | Sample inference displayed the plume overlay and metrics; alert JSON expanded; Alert Log and Model Card tabs loaded |

The sample alert reports 0.4745% coverage and severity class HIGH with softmax confidence approximately 0.3572. These are output values for one included sample, not a new accuracy benchmark or calibrated leak assessment. Its WGS84 centroid is approximately `[-96.469603, 30.209576]` in longitude/latitude order.

## Test coverage

Tests check sample preprocessing, wrong band count and dimensions, missing CRS, non-finite pixels, invalid thresholds, empty and rotated-raster centroid handling, UTM conversion, the health endpoint, missing/unsupported/corrupt uploads, upload limits, threshold forwarding, and temporary-file cleanup after success and validation failure. API unit tests replace model inference with a stub; the real CLI run checks the included weights separately.

The lightweight CI workflow performs structure and artifact checks plus the CPU unit tests. It does not install the complete TerraTorch stack or rerun training. Environment installation and real model inference were checked locally.

## Remaining validation

Full dataset training, common-metric evaluation, independent test-set performance, severity calibration, GPU/MPS execution, and production hosting remain outside this setup. The original notebook logs are preserved as historical records. See [the implementation plan](IMPLEMENTATION_PLAN.md) for the corresponding acceptance checks.

## Documentation review

- Evidence check: PASS. The README distinguishes imported metrics, current behaviour, and planned work; source artifacts are linked.
- Copy check: PASS. New documentation uses concrete technical descriptions and does not add performance, deployment, or security-certification claims.
- Link check: repository validation checks local document and image targets.

The application design is inherited. This setup does not claim a visual, accessibility, or mobile-layout redesign.
