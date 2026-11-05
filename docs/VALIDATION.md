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
| Experiment artifact integrity | 13 artifacts byte-compared against their recorded versions and documented with SHA-256 |
| `python scripts/validate_repository.py` | Eight Python files, five notebooks, documentation links, JSON, and imported artifact hashes passed |
| Real `POST /predict` sample request | HTTP 200, schema 1.1 alert, same 1,244-pixel prediction, and WGS84 coordinates |
| Browser dashboard smoke test | Sample inference displayed the plume overlay and metrics; alert JSON expanded; Alert Log and Model Card tabs loaded |

The sample alert reports 0.4745% coverage and severity class HIGH with softmax confidence approximately 0.3572. These are output values for one included sample, not a new accuracy benchmark or calibrated leak assessment. Its WGS84 centroid is approximately `[-96.469603, 30.209576]` in longitude/latitude order.

## Test coverage

Tests check sample preprocessing, wrong band count and dimensions, missing CRS, non-finite pixels, invalid thresholds, empty and rotated-raster centroid handling, UTM conversion, the health endpoint, missing/unsupported/corrupt uploads, upload limits, threshold forwarding, and temporary-file cleanup after success and validation failure. API unit tests replace model inference with a stub; the real CLI run checks the included weights separately.

The lightweight CI job performs structure and artifact checks plus the CPU unit tests. A separate `cloud-smoke` job installs the complete CPU deployment environment and runs the real sample through `deployment/streamlit_app.py`. Neither job reruns training.

## Cloud preparation

The cloud entrypoint passed a local Streamlit AppTest startup and real sample prediction with model downloads disabled. Its mask contained the expected 1,244 foreground pixels, included WGS84 coordinates, and used a separate output directory for the session. The deployment requirements select Linux CPU-only PyTorch and torchvision. See [DEPLOYMENT.md](DEPLOYMENT.md) for deployment settings; configuration alone is not evidence of a live public URL.

Both jobs passed in [GitHub Actions run 38558347510](https://github.com/prithvi-kaizen/Terramind/actions/runs/38558347510). The hosted app at [terramind-demo.streamlit.app](https://terramind-demo.streamlit.app/) was then verified with a real sample prediction: 1,244 foreground pixels, a visible plume overlay, schema 1.1 JSON with WGS84 coordinates, and working Alert Log and Model Card tabs.

## Remaining validation

Full dataset training, common-metric evaluation, independent test-set performance, severity calibration, GPU/MPS execution, and production service hardening remain outside this setup. The research notebook logs are preserved as historical records. See [the implementation plan](IMPLEMENTATION_PLAN.md) for the corresponding acceptance checks.

## Documentation review

- Evidence check: PASS. The README distinguishes imported metrics, current behaviour, and planned work; source artifacts are linked.
- Copy check: PASS. New documentation uses concrete technical descriptions and does not add performance, deployment, or security-certification claims.
- Link check: repository validation checks local document and image targets.

The application design is inherited. This setup does not claim a visual, accessibility, or mobile-layout redesign.

## Phase 1 verification

Phase 1 adds shared cached inference and reproducible benchmark/evaluation commands. All 42 local tests and both Linux CPU CI jobs passed. See [the Phase 1 report](phase-1-report.md) for exact-mask regression, adapter/cache checks, the measured CPU baseline, and remaining dataset-dependent evaluation.

## External evaluation and deployment follow-up

The fixed real-scene cohort and measured limitations are documented in [EVALUATION.md](EVALUATION.md). It is an out-of-domain transfer test with a declared adapter, not evidence of operational methane detection. The initial real-data artifact is superseded because of a corrected archive-local catalog join bug. Hosted Phase 1 sample inference was verified on 3 November 2026, including model identity and isolated output paths.

The final code passed 51 tests and both Linux CI jobs in [run 41584954483](https://github.com/prithvi-kaizen/Terramind/actions/runs/41584954483). The reference diagnostic was exact (zero changed pixels); bounded near-threshold diagnostics are available for future cross-CPU differences. All 1,000 synthetic inputs were audited, with 999 scored; the corrected real cohort scored 126 of 128 inputs. Raw reports and exclusion details are preserved in `docs/evaluation/`.

## Phase 2 validation

Phase 2 passed 71 unit tests and both Linux jobs in [Actions run 42024939718](https://github.com/prithvi-kaizen/Terramind/actions/runs/42024939718). It adds schema 2.0 and geodesic candidates while preserving raw predictions. The final paired benchmark met the 20% p95 target with 14.1% overhead on the repeated sample. The separate 32-pair validation pilot did not support promoting any removal rule, so all remain informational. The public Streamlit app also passed sample inference, candidate inspection and a parsed four-feature GeoJSON download after deployment. See [the Phase 2 report](phase-2-report.md) for evidence and limitations.

## Phase 3 tooling

The Phase 3 local suite passes 100 tests. Offline real-checkpoint smoke checks verify cached/uncached frozen-feature equivalence, decoder-only gradients, optional flip review and unchanged raw/API/cloud predictions. Existing dashboard smoke checks also pass. The [Phase 3 report](phase-3-report.md) distinguishes software verification from deferred research training and real temporal validation.
