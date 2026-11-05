# Phase 2: inspectable plume candidates

Phase 2 adds structured candidate outputs and inexpensive review signals to the existing checkpoint. It does not train a new model, calibrate leak probabilities, or establish operational methane detection. Raw predictions and the default threshold remain unchanged.

## Candidate definition

Each four-connected foreground region receives a request-local ID in scan order. Diagonally touching pixels remain separate. A candidate includes pixel count, physical area, pixel/native bounding boxes, native and WGS84 centroids, probability summaries, a WGS84 polygon with holes, and review flags. Native coordinates carry their source CRS. Pixel bounding boxes use `[column_min, row_min, column_max_exclusive, row_max_exclusive]`; centroids average raster pixel centres using the full affine transform before conversion to longitude/latitude. They are not emission-source estimates and can lie outside a concave region or inside a hole.

Area uses WGS84 ellipsoidal polygon area, with interior-ring areas subtracted. It does not assume native coordinate units are metres: projected feet and geographic-degree inputs have regression tests. Candidate extraction refuses more than 4,096 regions by default before polygon generation; it reports fragmentation as an error instead of silently dropping regions. `CandidateConfig.max_candidates` can be set explicitly within its supported bound for offline work. Regional footprints are supported; geometry that cannot be represented within valid WGS84 coordinates is rejected.

## Review signals and filtering

| Flag | Definition | Default effect |
| --- | --- | --- |
| `small_area` | Geodesic area below the configured review threshold, initially 1,600 m² | Informational |
| `edge_contact` | Component reaches a raster boundary | Informational |
| `threshold_sensitive` | More than 25% local pixel-count change between threshold ±0.05 | Informational |
| `invalid_pixel_overlap` | Raw component overlaps caller-supplied invalid pixels | Review flag; invalid pixels excluded separately from the filtered mask |

Threshold sensitivity uses the existing probability map. The lower-threshold support is the original component plus its one-pixel background neighbourhood, excluding other components; the upper count includes original component pixels surviving the higher threshold. The reported fraction is `1 - upper_count / max(lower_count, 1)`. It is a local sensitivity measure, not whole-image low-threshold component tracking or calibrated uncertainty.

The current inference preprocessing rejects inputs containing nodata before model execution. Standalone candidate extraction also supports an explicit validity mask for testing and future adapters. Validity handling is separate from heuristic filtering. A candidate with a flag is `needs_review`; one without flags is `candidate`. Neither state confirms methane. Temporal corroboration is reserved for Phase 3 and is not emitted.

No heuristic runs as a default removal rule. An explicitly requested experimental filter may remove components carrying `small_area`, `edge_contact`, or `threshold_sensitive`. It always creates a separate filtered mask. The raw mask, raw probability raster, candidate labels, and model-derived severity remain available. Filtered foreground is not used to rewrite the learned severity class.

## Schema 2.0 and artifacts

The machine-readable contract is [alert-schema-2.0.json](alert-schema-2.0.json). Each request directory contains:

- `<stem>_mask.tiff`: raw thresholded prediction, preserving the source CRS and affine.
- `<stem>_probability.tiff`: raw model probabilities, float32, with the same georeferencing.
- `<stem>_candidates.tiff`: int32 component IDs, with zero for background.
- `<stem>_filtered_mask.tiff`: explicit filtered result; identical to raw by default.
- `<stem>_candidates.geojson`: WGS84 polygons and candidate properties with model/configuration provenance.
- `<stem>_alert.json`: schema 2.0 packet with candidates, flags, filter outcome, validity summary, and all artifact paths.

All coordinates and centroids in GeoJSON geometry use longitude/latitude. Native-coordinate properties specify `source_crs`. Candidate IDs are scoped to their alert, not stable physical-source identifiers across dates. Model identity and canonical postprocessing SHA-256 accompany exports. Artifact paths identify files on the worker, not public HTTP download URLs.

Migration from schema 1.1:

| Field | Schema 2.0 behaviour |
| --- | --- |
| `detection`, `severity`, `source`, `model` | Existing fields retained; raw prediction semantics unchanged |
| `artifacts.mask_path` | Still identifies the raw mask |
| `detection.centroid_geographic` | Historical native-CRS compatibility field; use `centroid_wgs84` for longitude/latitude |
| `status: ALERT` | Still means raw foreground above threshold; does not confirm a leak |
| New `candidates` / `candidate_count` | Candidate-level geometry, areas, evidence states and flags |
| New `postprocessing` / `quality` / `filtered` | Configuration hash, review diagnostics and separate filter outcome |
| New artifact keys | Probability, labels, GeoJSON and filtered-mask paths |

Consumers enforcing a schema-version literal must accept 2.0 and regenerate against its schema. Historical sample artifacts remain unchanged at their recorded version. Legacy API responses without candidate fields are not a Phase 2 response.

## Run and inspect

```bash
python infer.py --input sample_input/0___0_S2B_20240326T165849_R1000_nonorm_bands20m.tiff --device cpu
```

Explicit experimental filtering:

```bash
python infer.py --input sample_input/0___0_S2B_20240326T165849_R1000_nonorm_bands20m.tiff \
  --min-area-m2 1600 --filter-flags small_area
```

The Flask multipart API accepts optional `min_area_m2` and comma-separated `filter_flags` in addition to `file` and `threshold`. Invalid settings fail before model execution. For example, `filter_flags=small_area,edge_contact` is explicit opt-in. Omitting the field enables no heuristic filtering.

In Streamlit, use **Run Inference**, inspect the candidate table and individual candidate crop, and expand **Download prediction artifacts** for five downloads. Controls show human-readable review reasons. **Advanced settings** enables experimental filtering; results continue to show their last successful run's settings until another inference is requested. The raw and filtered displays stay separate. Changing the inspector, display choice, or download controls does not run the model again. Empty and error states are tested; uploads are validated before enabling inference.

The Model Card reads checkpoint-specific counts from [model_configuration.json](model_configuration.json), generated by `python scripts/export_model_metadata.py`, and reads recorded evaluation JSON. It warns if an active checkpoint differs from that metadata. Historical notebook settings remain explicitly distinct from verified checkpoint/run provenance.

## Validation evidence

[phase-2-validation.json](phase-2-validation.json) records a deterministic pilot cohort of 32 MethaneSET validation targets and 32 paired clean references, pinned to the same immutable dataset revision as the earlier external study. IDs were selected before inference, and are disjoint from the previously inspected test cohort. All 64 inputs scored, with one model load and exactly 64 forward passes. Every rule is evaluated from the same cached per-scene probability map; no ensemble or extra consistency pass is used.

The adapter is the previously declared 200→512→200 out-of-domain resize, with areas measured on the native 10 m grid. Settings are fixed review defaults, not claimed Google parameters. The comparison is a validation pilot, not a new test benchmark or a cure for the checkpoint's poor transfer.

| Removal rule | Any-foreground recall | Clean-reference false-positive rate | Recall loss vs raw |
| --- | ---: | ---: | ---: |
| None (raw) | 31/32, 96.875% | 32/32, 100% | 0 |
| Small area | 29/32, 90.625% | 31/32, 96.875% | 6.25 percentage points |
| Edge contact | 31/32, 96.875% | 32/32, 100% | 0 |
| Threshold sensitivity | 30/32, 93.75% | 32/32, 100% | 3.125 percentage points |
| Combined | 28/32, 87.5% | 30/32, 93.75% | 9.375 percentage points |

None meets the predeclared requirement of fewer false-positive scenes with at most two percentage points of recall loss. Edge removal changes neither scene rate. Paired bootstrap intervals cluster by 16 sites and remain wide; they do not justify promotion either. All removal rules stay off by default. The checkpoint's original real-test results are preserved and have not been used for tuning.

Reproduce the pilot with `python scripts/validate_phase2_rules.py`. The script pins the source revision, records the cohort, uses archive-aware parent/child joins, validates binary masks and alignment, and saves per-input counts, hashes and paired comparisons. Data stays under ignored `data/raw/phase2-validation/`.

## Performance and checks

[phase-2-benchmark.json](phase-2-benchmark.json) compares a fresh Phase 1 Git revision with Phase 2 on identical inputs, hardware and thread settings. Run `python scripts/benchmark_phase2.py --baseline-ref 74f0238` to reproduce. The recorded ratio, memory and workload limitations are authoritative; a single repeated sample is not a deployment-capacity benchmark.

```bash
python scripts/validate_repository.py
python -m pytest -q
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 python scripts/smoke_inference.py
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 python scripts/smoke_phase2_ui.py
```

The CPU smoke validates schema 2.0, raw mask regression, probability/filtered rasters, and exact CLI-runner/API/cloud adapter agreement. UI smoke uses real inference for candidate selection, downloads, opt-in filtering and no extra inspection passes; separate, isolated fixtures cover empty/error presentation. The default remains one model pass per request. Candidate geometry, holes, rotated transforms, geographic/projected CRS, feet units, nodata flags, fragmentation, configuration hashes and explicit filtering have unit coverage.
