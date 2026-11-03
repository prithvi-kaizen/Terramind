# Phase 1 implementation report

Updated 3 November 2026. Software implementation is complete on `codex/phase-1-inference-baseline`; the full synthetic audit and a limited real-scene stress test have now completed. Deployment-matched independent validation remains open. This report does not establish improved detection quality. Phase 1 was subsequently merged and deployed on 3 November 2026; hosted sample inference and model identity were verified. See [the external evaluation report](EVALUATION.md) for follow-up evidence.

## Changes

- Extracted checkpoint-compatible architecture, preprocessing, artifact writing, and runtime into `methanepulse`. CLI, Flask, and Streamlit use the shared runner; compatibility imports in `infer.py` preserve existing callers.
- Added one bounded model cache per worker with content/configuration invalidation and serialized execution. Request artifacts remain separate from model resources, with unique directories and additive model identity in schema 1.1 alerts.
- Removed automatic local inference after API failure/timeout. Added explicit CPU selection and an operator-run retention command with dry-run default.
- Added CPU benchmarking and binary-mask evaluation commands, content hashes, grouped split-leakage checks, aligned-label validation, explicit metric conventions, and a data inventory.
- Added an unexecuted corrected training-notebook preparation command. Historical notebooks and checkpoint remain unchanged; their artifact hashes still validate. Conflicting training settings are removed in the working copy, and its frozen encoder remains in evaluation mode. No training was run.

## Local verification

The Python 3.12.4 environment has the pinned dependencies installed. Commands run from the repository root:

```bash
python -m pytest -q
python scripts/validate_repository.py
python -m pip check
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 python scripts/smoke_inference.py
HF_HUB_OFFLINE=1 python scripts/benchmark.py --input sample_input/0___0_S2B_20240326T165849_R1000_nonorm_bands20m.tiff --threads 2 --warm-requests 30 --output docs/phase-1-benchmark.json
python scripts/prepare_training_notebook.py
```

The commands use the repository `.venv` locally. Tests: 42 passed, including existing input/API tests and new cache invalidation, concurrent execution, request isolation, manifest leakage, metric calculation, label validation, retention, and notebook-preparation checks. Rasterio emits deprecation warnings for the existing affine multiplication convention. Dependency consistency passed.

The real CPU check passed with network model downloads disabled. The complete sample mask matches the historical reference pixel for pixel; CRS and affine transform match. The shared runner, Flask endpoint, and cloud Streamlit entrypoint produced equivalent detection fields, with one model load and three forward passes in one process. The sample remains 1,244 foreground pixels with WGS84 centroid `[-96.469603, 30.209576]` at threshold 0.5. Streamlit was tested using AppTest; this does not assert that production hosting has updated.

The saved run measured 12.338 seconds from fresh Python process start to first output, 0.184 seconds warm median, 0.207 seconds warm p95, and 982,237,184 bytes peak RSS (about 937 MiB). It loaded the model once for 31 forward passes with zero failed requests. A prior run took 27.447 seconds cold, showing initialization/cache variability; the saved run is a fresh process, not a cold filesystem or host. The benchmark JSON records the complete measurements and hardware description. It measures a single repeated sample on local arm64 macOS, two CPU threads, and 30 warm requests. Startup includes Python imports and model/library initialization; warm requests include checkpoint hashing, preprocessing, a model pass, and artifact writing. Peak RSS is a process-lifetime high-water mark. The exact CPU brand was unavailable through the sandbox, so no specific chip model is claimed. These results are not a comparison against MAPL-EMIT or a deployment capacity benchmark.

## Acceptance status and missing evidence

| Requirement | Status |
| --- | --- |
| Shared adapters, model reuse, cache invalidation, serialized execution | Implemented; tested locally |
| Complete sample mask and georeferencing parity | Passed real CPU regression |
| CPU cold/warm timing and peak memory | Measured; see benchmark JSON |
| Evaluator, split manifests, label validation, calculation tests | Implemented; fixture-tested |
| Full-dataset checkpoint evaluation | Completed 999 scored / 1,000 audited; see follow-up report |
| Verified negative-scene false-positive rates | Measured on 63 provider-curated references; project-side independent review remains open |
| Independent real-plume performance | Limited resized MethaneSET stress test completed; operational validation remains open |
| Historical checkpoint/training-log association | Unknown; no new evidence resolves the disagreement |
| Corrected training configuration | Working copy generated and tested; training not run |
| Linux CPU deployment installation and smoke | Passed both jobs in the linked GitHub Actions run |
| Output retention | Manual cleanup command implemented; no automatic scheduler or disk quota |

Phase 2 software work can proceed from these interfaces. Detection-quality claims and filter promotion must wait for labeled evaluation data. The [data inventory](phase-1-data-inventory.json) lists missing inputs. Additional temporal data and model retraining belong to Phase 3.

## Linux CI and review

Both `validate` and `cloud-smoke` passed in [GitHub Actions run 41579345593](https://github.com/prithvi-kaizen/Terramind/actions/runs/41579345593) for implementation revision `8f09fec`. This includes a clean Linux CPU dependency installation, dependency consistency, repository checks, unit tests, and the real CPU adapter/cache smoke check. Subsequent report-only edits do not change the tested implementation.

The implementation is available in [draft pull request 1](https://github.com/prithvi-kaizen/Terramind/pull/1). Additional local checks verified actual CLI output against the complete reference mask and used a simulated API timeout in Streamlit AppTest to confirm that the dashboard displays an error without calling local inference.

The [external evaluation follow-up](EVALUATION.md) supersedes the original missing-data status: complete synthetic and limited real-scene evidence is now available, and hosted deployment was verified.
