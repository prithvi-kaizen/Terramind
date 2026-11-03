# Terramind implementation plan: three phases

Status: updated 3 November 2026. Phase 1 software is implemented; full-data and real-plume evaluation remain pending. Phases 2 and 3 are proposed work. See [the Phase 1 report](phase-1-report.md).

## Objective and starting point

Improve the reliability and efficiency of the existing Sentinel-2 methane-plume prototype while retaining a CPU-compatible demo. Adapt candidate vetting, structured plume outputs, and repeated-observation checks from MAPL-EMIT to the data and resources available here.

The repository already includes a trained checkpoint, sample GeoTIFF, CLI, local Flask API, a [public Streamlit demo](https://terramind-demo.streamlit.app/), 42 runtime tests, and a Linux CPU inference smoke check. See [VALIDATION.md](VALIDATION.md). Full training data and an independent real-plume benchmark are not included. Historical experiment logs disagree, and the checkpoint's training-run association remains unresolved; see [MODEL_CARD.md](MODEL_CARD.md).

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

## Execute the phases with Codex

Open this repository as the Codex workspace. Run the prompts below in order, one phase at a time. Each prompt refers to the tasks and acceptance gate above. Before starting the next phase, inspect the previous phase's diff and evidence report. Keep each phase on its own `codex/` branch; use smaller commits for refactoring, feature changes, and documentation so changes remain reviewable. Publishing changes to the hosted demo should follow explicit deployment instructions.

For every phase, Codex should inspect repository instructions, current files, and uncommitted changes first. Preserve unrelated work and the existing checkpoint, sample input, and historical experiment artifacts. Record actual commands and results in `docs/phase-N-report.md`, replacing `N` with the phase number. Separate implemented software, completed experiments, failed checks, and unavailable evidence. Run the existing repository validator and relevant tests; use the full Linux CPU smoke check for inference/dependency changes. Do not fabricate benchmark results, labels, or training history.

### Codex prompt for Phase 1

```text
Implement Phase 1 of docs/IMPLEMENTATION_PLAN.md in this repository.
Read the complete plan and follow its compute constraints and acceptance gate.
Create or use branch codex/phase-1-inference-baseline after checking current work.

Start with a shared methanepulse package and thin CLI, Flask, and Streamlit
adapters. Separate model loading from prediction, add a bounded model cache
with checkpoint/configuration invalidation, and serialize model execution.
Preserve per-session output isolation and prevent duplicate inference retries.
Keep preprocessing changes separate from the behavior-preserving refactor.

Add reproducible CPU benchmarking, split/evaluation manifest support, and a
data inventory. Record hardware, thread settings, cold/warm timings, memory,
checkpoint hash, and forward-pass counts. Run the sample benchmark locally.
Compare the complete sample mask with the existing reference, not just its
1,244-pixel foreground count. Verify georeferencing and adapter equivalence.

Run meaningful cache, concurrency, input-validation, and regression tests,
the repository validator, and the real CPU cloud-entrypoint smoke check.
Do not retrain or claim model-quality improvements during this refactor.
If full evaluation data is missing, implement and test the evaluator using
fixtures, identify the missing evidence, and continue independent work.
Write docs/phase-1-report.md with changes, commands, measurements, limitations,
and the acceptance-gate outcome. Finish with reviewable changes; do not begin
Phase 2 or publish a deployment automatically.
```

### Codex prompt for Phase 2

```text
Implement Phase 2 of docs/IMPLEMENTATION_PLAN.md using the accepted Phase 1
interfaces. Inspect docs/phase-1-report.md and verify the required software
checks before proceeding. Create or use branch codex/phase-2-plume-candidates.

Extract disconnected plume candidates and export their geometry, area,
centroids, probability summaries, checkpoint identity, and quality flags.
Use correct CRS-aware areas and WGS84 GeoJSON. Version the alert schema and
document compatibility with existing schema 1.1 consumers. Preserve the raw
mask and probability map alongside any optional filtered result.

Add edge, small-area, invalid-pixel, and threshold-sensitivity flags using
the existing probability map. Keep them informational unless validation
supports filtering. Keep evidence states separate from probabilities and
area-based severity. Add candidate inspection and artifact downloads to the
dashboard and replace unsupported model-card values with verified metadata.

Test empty and multi-component masks, nodata, rotated transforms, projected
and geographic CRS, schema compatibility, exports, and the actual dashboard
workflow. Verify one default model pass per tile and compare warm p95 timing
against Phase 1, targeting no more than 20 percent additional latency.
Run the repository validator, relevant tests, and real CPU smoke check.
If labeled evaluation data is available, compare each quality rule separately
and together using the plan's recall-loss tolerance. Otherwise keep filters
off by default and report that detection-quality improvement is unverified.
Write docs/phase-2-report.md with evidence and unresolved dependencies.
Finish with reviewable changes; do not begin Phase 3 or deploy automatically.
```

### Codex prompt for Phase 3

```text
Implement Phase 3 of docs/IMPLEMENTATION_PLAN.md after reviewing the earlier
phase reports. Create or use branch codex/phase-3-reliability. Inventory actual
data, labels, aligned dates, hardware, and storage before running experiments.
Do not assume the full training corpus or a real-plume benchmark is present.

Build reviewed split manifests, frozen-encoder feature caching, and bounded
head-training/evaluation commands. Validate cache equivalence and invalidation.
Keep the encoder in evaluation mode and account for augmentations in cache
keys. Limit the initial search to three configurations; report cache creation
cost, training resources, and validation outcomes. Do not launch paid compute
or large downloads without authorization. When local data is insufficient,
complete runnable tooling and tests and list the exact missing inputs.

Add optional corroboration for up to three supplied, aligned dates, rejecting
incompatible coverage or registration. Add an optional flipped-input agreement
check with at most one extra pass, disabled by default; cap automatic batch
routing at 10 percent. Track date-specific passes separately and reuse only
compatible cached predictions. Do not infer source location or emission rate.

Use held-out evaluation and the plan's false-positive/recall acceptance gate
to decide whether a trained checkpoint should replace the existing one.
Report uncertainty and retain the current checkpoint if evidence is inadequate.
Run targeted training/cache/temporal tests, repository validation, and CPU
smoke checks. Write docs/phase-3-report.md and a quality-versus-compute report.
Update documentation only for capabilities actually verified. Prepare a
release checklist and rollback instructions; do not publish automatically.
```

When a phase ends, ask Codex to explain any unmet acceptance criteria before commissioning the next phase. Missing evaluation data can leave research tasks pending while tested software work proceeds; it cannot establish improved real-world detection.
