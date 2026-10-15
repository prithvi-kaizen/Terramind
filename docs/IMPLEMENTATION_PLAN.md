# Terramind implementation plan: three phases

Status: proposed work, 15 October 2026. The tasks below are not implemented capabilities.

## Objective and starting point

Improve the reliability and efficiency of the existing Sentinel-2 methane-plume prototype while retaining a CPU-compatible demo. Adapt candidate vetting, structured plume outputs, and repeated-observation checks from MAPL-EMIT to the data and resources available here.

The repository already includes a trained checkpoint, sample GeoTIFF, CLI, local Flask API, a [public Streamlit demo](https://terramind-demo.streamlit.app/), 27 runtime tests, and a Linux CPU inference smoke check. See [VALIDATION.md](VALIDATION.md). Full training data and an independent real-plume benchmark are not included. Historical experiment logs disagree, and the checkpoint's training-run association remains unresolved; see [MODEL_CARD.md](MODEL_CARD.md).

This plan replaces the previous five-milestone roadmap. It retains its requirements for checkpoint traceability, consistent evaluation, shared inference, accurate documentation, and release validation within three implementation phases.

## Research basis and adaptation boundaries

Google's [MAPL inference library](https://github.com/google-research/mapl) separates candidate detections from vetted results and exports plume geometries and detection metrics. Its [dataset documentation](https://developers.google.com/earth-engine/datasets/catalog/projects_climate-and-sustainability_assets_ghg_emit_mapl_emit_plumes_v1_0) describes spectral checks and repeated observations as evidence for assessing candidates. These motivate our proposed checks; their effectiveness on Sentinel-2 must be measured separately.

Keep the current two-band TerraMind input contract initially. EMIT-specific hyperspectral fitting cannot be transferred directly to B11/B12. A plume centroid is not an emission-source estimate; area-based severity is not methane enhancement or an emission rate. Connected components can separate disconnected regions, but do not disentangle overlapping plumes. Repeated detections are corroboration, not proof of methane.

## Compute and evaluation rules

| Decision | Initial constraint |
| --- | --- |
| Default inference | CPU supported, batch size 1, one forward pass per tile, one active inference request per worker |
| Model lifecycle | One cached model per configured checkpoint/device in a worker; bounded cache and explicit invalidation |
| Quality checks | Reuse one probability map; avoid ensembles and dense overlapping inference by default |
| Optional consistency check | At most one additional forward pass for a flagged tile; disabled by default |
| Temporal checks | At most three explicitly supplied dates per location in the first version; reuse compatible stored predictions |
| Training | Frozen encoder first; bounded head-training experiments; no large synthetic-data generation campaign |
| Cost claims | Record hardware, wall time, memory, forward-pass count, and training resources; no numerical savings claim against Google without comparable measurements |

Measure cold process start through first output separately from warm requests. Use identical tiles, thread settings, output generation, and hardware for comparisons. Record peak process memory, checkpoint/cache disk sizes, median and p95 latency, and failed requests. Include data loading, preprocessing, and output writing in end-to-end timings. Synchronise accelerator work if benchmarking a GPU or MPS device.

Use location/acquisition/plume-identity split manifests with leakage checks. Select thresholds, quality rules, and model configurations using training/validation data. Keep the official test set untouched until the configuration for a release is frozen; do not repeatedly use it to choose changes. Report global pixel IoU/F1, per-tile distributions, scene-level detection precision/recall, and false-positive scenes divided by verified negative scenes. Define empty-mask handling and plume-matching rules explicitly. Scene-level metrics require genuinely negative scenes.

## Phase 1: share inference and establish a measured baseline

**Outcome:** one reusable inference pipeline, repeatable evaluation, and measured CPU performance before changing predictions.

Estimated effort: 3–5 working days, excluding dataset recovery and downloads. No retraining is required for the inference refactor.

### Tasks

1. Create a `methanepulse` package for preprocessing, model construction/loading, prediction, and artifact writing. Route `infer.py`, `endpoint.py`, and `app.py` through it. Preserve the current checkpoint, default threshold, direct cloud mode, and coordinate behaviour during this refactor.
2. Separate model loading from prediction. Cache a model in evaluation mode by checkpoint hash, device, and architecture configuration. Keep Streamlit model resources shared without sharing session output paths. Serialize model execution initially, prevent duplicate work after API timeouts, and bound output retention.
3. Validate input metadata and valid-pixel masks before inference: band order, shape, CRS, affine transform, nodata, finite values, and constant bands. Reject unsupported inputs with explicit errors. Introduce prediction-changing preprocessing fixes as separate evaluated changes.
4. Add a benchmark command that saves machine-readable cold/warm timing and memory results. Run one cold request and at least 30 warm requests on a fixed tile set; identify repeated use of the single sample as a limited workload if more data is unavailable.
5. Add an evaluator and split manifest. Evaluate the current checkpoint under a new, traceable evaluation ID with checkpoint hash, preprocessing, threshold, dependencies, and split. Investigate the historical log disagreement; retain an unknown training-run association if it cannot be established.
6. Create a data inventory covering the available positive tiles, required negative scenes, and a small independent real-plume evaluation set. Record acquisition dates, label sources, access terms, and uncertainty. Fix conflicting notebook optimizer/scheduler settings and keep the frozen encoder in evaluation mode before future training.

### Deliverables

- Shared inference package and thin CLI/API/dashboard entrypoints.
- Benchmark command, baseline JSON report, and documented reference CPU configuration.
- Evaluation command, split manifest format, and checkpoint evaluation manifest.
- Updated setup and model-card documentation distinguishing measured results from unresolved history.

### Acceptance gate

- Existing tests and real CPU cloud smoke check pass. The included sample retains its 1,244-pixel mask at threshold 0.5; verify the full mask against the existing reference, along with CRS, transform, and coordinates.
- Repeated requests load the model once per worker; a checkpoint/configuration change invalidates the cache. CLI, API, and dashboard return equivalent predictions without output collisions or session-data leakage.
- Benchmarks include cold and warm results and memory. Any missing dataset or real-scene evidence is explicitly marked unavailable; it is not replaced with a synthetic-data generalisation claim.
- Evaluation can run from a saved configuration once data is supplied. Phase 2 software work may proceed with the sample and test fixtures, but claims of improved detection quality wait for evaluation data.

## Phase 2: add plume candidates and inexpensive quality checks

**Outcome:** inspectable candidate-level outputs and transparent review signals, using the existing model and one default forward pass.

Estimated effort: 4–6 working days. Depends on the Phase 1 inference interfaces and benchmark.

### Tasks

1. Extract connected components with a documented connectivity rule. Give each candidate an ID, pixel count, physical area, bounding box, polygon, projected centroid, WGS84 centroid, and probability summary. Calculate area in an appropriate metric CRS or geodesically; never assume coordinate units are metres.
2. Preserve the raw probability map and raw mask separately from optional filtered outputs. Export GeoJSON and a versioned JSON alert with explicit coordinate conventions, checkpoint/configuration hashes, candidate fields, and quality reasons. Provide a migration note for schema 1.1 consumers.
3. Add flags for invalid-pixel overlap, tile-edge contact, small area, and mask sensitivity to nearby thresholds. Derive threshold sensitivity from the existing probability map. Keep nodata exclusion distinct from heuristic filtering and preserve a record of affected pixels/candidates.
4. Show `candidate`, `needs review`, and later `temporally corroborated` as evidence states. Keep quality flags, model probabilities, and area-based severity separate. Do not present an uncalibrated rule score as a probability that a leak is real.
5. Tune any area filter and threshold settings on validation data. Compare raw output, each individual rule, and the combined rules. Measure lost recall as well as false positives; tiny detections may be genuine. Leave unsupported rules as informational flags.
6. Add candidate inspection and mask/JSON/GeoJSON downloads to Streamlit. Generate its model-card values from verified metadata, replacing historical unsupported values. Retain useful loading, empty, error, and retry states and check keyboard/mobile usability.

### Deliverables

- Candidate extraction and quality-check modules, with raw and filtered artifacts.
- Versioned alert schema, GeoJSON export, and documented evidence states.
- Dashboard candidate view and downloads.
- Validation comparison of each rule's effect on quality and latency.

### Acceptance gate

- Tests cover empty masks, disconnected regions, nodata boundaries, edge contact, rotated affine transforms, and geographic/projected CRS. Exported geometries align with the raster.
- Default inference performs exactly one forward pass. No heuristic silently replaces the raw prediction.
- Target end-to-end warm p95 latency is no more than 20% above the cached Phase 1 baseline on the same workload. This is an engineering target to test, not a measured result; profile or disable costly defaults if missed.
- Promote a filter only if validation shows fewer false-positive scenes with no more than two percentage points of absolute scene-recall loss. This is a proposed acceptance tolerance, fixed before the experiment. Report uncertainty and defer promotion when the evaluation set is too small to support the decision.

## Phase 3: improve reliability through targeted training and corroboration

**Outcome:** a measured model update and optional extra evidence, with bounded computation and a documented release decision.

Estimated effort: 5–8 working days plus data collection and training. Training and reliability claims require the Phase 1 data inventory to be fulfilled.

### Tasks

1. Collect and review hard negatives, including difficult surfaces and cloud/shadow cases. Keep locations and acquisitions disjoint across splits. Use analyst labels or independently supported observations for a real-scene test; absence of a Google detection does not establish a negative, and its model outputs are not ground truth.
2. Cache frozen-encoder embeddings for fixed preprocessing. Key caches by input content, encoder checkpoint, band order, preprocessing configuration, and augmentation. Keep the encoder in evaluation mode. Image augmentations require separately computed embeddings; never reuse an incompatible cache entry.
3. Bound the initial search to three head-training configurations using a fixed seed. Select on validation data, then repeat the selected configuration with three seeds if resources permit. Record epochs, CPU/GPU hours, cache creation time/storage, and the effect of adding negatives. Compare severity against simple area/majority baselines and investigate the zero-accuracy log.
4. Compare the updated model with the current checkpoint and a consistent SWIR baseline using the same evaluator. Recover or retrain a two-band TinyUNet only as a separately budgeted baseline; its historical aggregate metrics are not a controlled comparison. Test the L1C/S2L2A and normalisation mismatch before adopting any change.
5. Add an optional workflow for up to three dated, co-registered tiles per location. Validate band scaling, valid coverage, resolution, and registration. Match candidates using a documented spatial tolerance and record dates, observation count, and displacement. Account for plume motion; do not treat centroid agreement as source identification or a single missed date as proof of absence. If compatible dates cannot be obtained, defer this feature explicitly.
6. Add an optional consistency check for flagged candidates: one flipped-input pass, invert the output transform, and report mask agreement. Evaluate this as a review signal, not calibrated uncertainty. Keep the default off; an automatic mode may check at most 10% of tiles in a batch and must log its routing decisions.
7. Publish an evaluation and compute report with per-feature comparisons, errors, limitations, and a model/checkpoint manifest. Update README, model card, demo, CI, and release notes only for accepted changes. Confirm code/checkpoint/data terms before assigning a project license or redistributing new data.

### Deliverables

- Reviewed data/split manifests, embedding-cache tooling, and bounded head-training command.
- Updated checkpoint only if accepted, with reproducible configuration and evaluation evidence.
- Optional temporal and consistency-check modules with explicit acquisition and pass budgets.
- Versioned release, quality-versus-cost report, and updated demo/documentation.

### Acceptance gate

- Select one operating threshold on validation data before test evaluation. Target fewer false-positive scenes at the same two-percentage-point absolute recall-loss tolerance used in Phase 2; disclose pixel-level tradeoffs and uncertainty. Keep the existing checkpoint if evidence does not support the new one.
- Cached and uncached fixed-input training features agree within a documented numerical tolerance. No training or threshold selection uses held-out test labels.
- For a single date, if fraction `q` receives one extra consistency pass, model-pass count is `N × (1 + q)` for `N` new tiles. With automatic routing capped at 10%, it is at most `1.1 × N`; this does not imply a matching wall-time or energy ratio. Report temporal-date passes separately.
- Result-cache keys include input content, checkpoint, preprocessing, threshold, and postprocessing versions. Repeated temporal checks reuse only compatible outputs. Tests cover missing dates, non-overlapping coverage, registration failures, empty masks, and changed configuration.
- Publish measured CPU latency, memory, and training resources alongside quality results. Avoid numerical cost or accuracy comparisons with MAPL-EMIT across different sensors and workloads.

## Sequence, dependencies, and deferred work

Implement Phase 1, then Phase 2, then Phase 3. The estimates total 12–19 developer working days, excluding data access, annotation, licensing discussions, and training queues. They are planning estimates, not delivery commitments. Preserve the current demo while phases are developed; release each accepted phase after tests and a hosted sample check.

Real-plume and negative-scene evidence is a dependency for detection-quality claims, not for implementing the Phase 1 refactor or Phase 2 output schema. If that evidence is unavailable, ship the software improvements with the limitation stated and leave training/corroboration tasks incomplete.

Defer physical emission-rate estimation, learned source localisation, EMIT hyperspectral ingestion, large synthetic training campaigns, and orbital deployment. A compact student model, quantisation, container packaging, and accelerator-specific tuning are follow-up work only if the measured CPU cost still exceeds the deployment budget; validate accuracy and hardware-specific speed before claiming a benefit.
