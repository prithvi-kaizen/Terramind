# Phase 1: shared inference and evaluation

The CLI, Flask API, and Streamlit dashboard use `methanepulse.runtime`. A worker holds at most one cached model, keyed by checkpoint content hash, device, architecture, and preprocessing versions. Model construction, prediction, and output writing share a lock. Separate OS processes maintain separate caches. The CLI benefits from reuse only when called repeatedly within a process; separate CLI invocations still start cold.

## Runtime contract

Supported input remains 13 bands at 512 by 512 with the documented ordering and a CRS. The model still uses independently percentile-normalised B11/B12. Named bands accept the sample's zero-padded names and SWIR suffixes. Missing band descriptions mean the documented ordering is assumed, not independently verified. Invalid affine transforms, nodata, nonfinite pixels, constant SWIR bands, and contradictory named-band order fail before loading the model. Supporting nodata with modified normalisation needs a separate evaluated preprocessing version.

Each request writes `<output-root>/<request-uuid>/<tile-stem>_mask.tiff` and the alert JSON. This replaces overwriting files at the output root. Consumers should use the returned `artifacts` paths. Schema 1.1 and detection values are preserved, with additive checkpoint/configuration identity fields. Streamlit histories search only the current session's subtree. Model resources contain no session results. API errors and timeouts stop that attempt; the dashboard does not launch duplicate local inference automatically.

Set `METHANEPULSE_DEVICE=cpu` to force CPU in all adapters, or pass `--device cpu` to the CLI. Default device selection remains MPS, CUDA, then CPU. One active inference request runs per process; this is a research service without a durable queue or cross-process coordination. A failed response can leave completed server output. Retrying manually starts a new request.

Output retention is operator-managed. This command previews completed request directories older than 24 hours outside the newest 100:

```bash
python scripts/cleanup_outputs.py --root outputs/predictions --age-hours 24 --keep 100
```

Add `--apply` to delete those directories. Incomplete directories and uploaded temporary inputs are excluded. No scheduler or hard disk quota is installed; run cleanup periodically and monitor disk usage. Avoid cleanup while someone is inspecting an old result.

## CPU benchmark

```bash
HF_HUB_OFFLINE=1 python scripts/benchmark.py \
  --input sample_input/0___0_S2B_20240326T165849_R1000_nonorm_bands20m.tiff \
  --threads 2 --warm-requests 30 --output outputs/benchmark.json
```

The command starts a fresh Python child, measures startup through first output and 30 warm requests, writes artifacts to temporary directories, and records peak process RSS. It includes checkpoint hashing, preprocessing, and output generation. It exits unsuccessfully on a failed request rather than publishing a successful benchmark. RSS is a lifetime high-water mark, not incremental request memory. Repeated use of the one available sample does not represent diverse scenes or concurrent users. See [the measured report](phase-1-benchmark.json).

## Evaluation manifest

Store a JSON manifest beside the external data, with paths relative to the manifest:

```json
{
  "schema_version": 1,
  "tiles": [
    {
      "id": "unique-tile-id",
      "split": "validation",
      "tile": "tiles/example.tiff",
      "mask": "masks/example.tiff",
      "location": "location-group-id",
      "acquisition": "unique-scene-acquisition-id",
      "plume_id": "simulated-or-observed-plume-group-id",
      "label_source": "Dataset/annotation source and version",
      "verified_negative": false
    }
  ]
}
```

`plume_id` may be omitted when unknown. Use a scene acquisition identifier, not just a calendar date. Keep group identifiers consistent across tiles. All rows, including train and test, are checked for group leakage and duplicate input content before evaluating the requested split. Optional `tile_sha256` and `mask_sha256` fields enforce expected content; actual hashes are always recorded. Labels must be fully valid single-band binary 0/1 rasters aligned with their input. Convert soft labels explicitly, record the conversion, and do not reuse model predictions as labels.

```bash
python scripts/evaluate.py --manifest data/splits.json \
  --split validation --threshold 0.5 --output outputs/evaluation.json
```

Select thresholds on validation data. Use `--split test` only after freezing the configuration. The tool cannot establish whether a user has previously inspected the test set. It reports global pixel IoU/F1 and per-tile scores; both-empty masks score 1. Scene detection means any predicted foreground pixel, so it is not a plume-matching or source-location benchmark. Only explicitly verified negatives contribute to scene false-positive rates and precision. No denominator means `null`. Input hashes, checkpoint, preprocessing, threshold, software versions, and an evaluation ID accompany each report. The checkpoint's historical training-run association remains unknown.

## Preparing future training

```bash
python scripts/prepare_training_notebook.py
```

This creates `outputs/notebooks/TerraMind-working.ipynb`, an unexecuted copy with cleared outputs. It keeps the effective historical AdamW learning rate of 0.001 and scheduler patience of 3 explicit, removes conflicting assignments, and places the frozen encoder in evaluation mode after both training-mode calls. It refuses to overwrite an existing destination. The five historical research notebooks stay unchanged and hash-verifiable. This working copy has not been trained; Phase 3 still requires split validation and a training pipeline.

## Verification

```bash
python scripts/validate_repository.py
python -m pytest -q
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 python scripts/smoke_inference.py
python -m pip check
```

The real-model smoke check verifies the complete historical sample mask, CRS and affine transform, and exact API/cloud prediction equivalence within a worker. Cross-CPU reference comparison allows at most eight probability-boundary pixels within 0.0001 of threshold 0.5; diagnostics report any allowance. It uses one model load and three requests, plus one diagnostic prediction only if the reference differs. CI installs the full Linux CPU deployment environment before this check. See [the Phase 1 report](phase-1-report.md) and [data inventory](phase-1-data-inventory.json).
