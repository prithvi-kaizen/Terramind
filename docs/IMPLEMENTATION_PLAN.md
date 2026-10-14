# Implementation plan

## Objective

Turn the existing MethanePulse prototype into a reproducible geospatial ML project with inspectable experiments, reliable local inference, and a demonstrable application. Preserve the original research evidence and distinguish implemented behaviour from planned capabilities.

The current application, notebooks, sample tile, checkpoint, and saved outputs provide a useful starting point. The highest-value work is resolving experiment provenance and making evaluation reproducible. Adding more model complexity before those steps would make performance claims harder to assess.

## Repository foundation

The application lives at the repository root, duplicate notebooks are consolidated by byte comparison, and documentation records the architecture, dataset, setup, and known limitations. Direct dependencies are pinned; runtime input checks, coordinate conversion, tests, and CI are added. Actual execution evidence is recorded in [VALIDATION.md](VALIDATION.md).

The original five notebooks, checkpoint, sample raster, experiment JSON, sample prediction, and presentation remain available. The historical README is archived rather than used as the current technical description.

## Milestone 1: establish a reproducible experiment baseline

Estimated effort: 2–3 working days, assuming dataset access.

1. Identify which run produced `terramind_methane_best.pt`. Compare the checkpoint against both imported TerraMind records and rerun evaluation on the exact saved validation subset. Record its SHA-256, preprocessing configuration, severity thresholds, seed, split, and environment together.
2. Recover or retrain TinyUNet; its evaluation notebook references a checkpoint that is absent. Add the missing train loop.
3. Freeze a split manifest based on acquisition, location, and simulated plume identity. Keep the official test set untouched. Document the relationship between the custom 619/185 split and the original dataset split.
4. Replace historical notebooks' hidden state with imports from shared modules. Fix `baseline.ipynb`'s undefined `train_df` and remove superseded sample-ID labels from the runnable path.
5. Replace conflicting scheduler and learning-rate assignments with a single config. Keep the frozen encoder in evaluation mode even when decoder heads train.
6. Lock transitive dependencies for the supported platforms and record model-library revisions.

Acceptance: a clean environment can run data validation, training, and evaluation with explicit commands; every result points to a checkpoint and split manifest. A saved evaluation report reproduces its listed metrics within a documented tolerance. No result is attributed to an unidentified run.

## Milestone 2: make the comparison scientifically defensible

Estimated effort: 3–4 working days, plus training time.

1. Implement one evaluator used by the SWIR baseline, TinyUNet, and TerraMind. Report global pixel IoU/F1 and per-tile distributions separately. State treatment of empty masks.
2. Select thresholds on validation data; evaluate the fixed threshold once on the held-out test set. The inherited physics threshold sweep currently selects on validation and reports that same set.
3. Evaluate by held-out location and acquisition, with leakage checks across simulated plume identities. Report multiple seeds where feasible.
4. Add true negative scenes before reporting scene-level detection precision, recall, or false-alarm rates. The notebook reports no zero-plume training tiles.
5. Resolve zero severity accuracy in `docs/terramind_training_history.json`. Compare severity against majority-class and area-based baselines; report macro-F1, confusion matrix, and class distribution.
6. Test L1C/S2L2A modality consistency and normalisation alternatives. Evaluate whether the four-channel TinyUNet and two-channel TerraMind inputs make the comparison fair; include a two-channel TinyUNet ablation.
7. Replace dashboard and README comparison values with generated evaluation summaries from the same run.

Acceptance: an evaluation command generates a traceable report and qualitative examples, with a documented split, aggregation rule, baseline, and uncertainty. Any superiority or generalisation claim must be supported by this report. If TerraMind performs worse, document that result and use it to guide the next experiment.

## Milestone 3: consolidate the inference pipeline

Estimated effort: 2–3 working days.

1. Move preprocessing, architecture, checkpoint loading, and alert construction into a `methanepulse` package. The CLI, Flask service, dashboard, and notebook helper currently duplicate parts of the pipeline.
2. Cache one validated model per device; avoid loading the encoder and checkpoint for every request. The maintained CLI already avoids a redundant pretrained-weight download by using the complete included checkpoint; carry that behaviour into the shared package.
3. Add nodata masks, explicit band metadata, dtype and CRS checks, constant-band handling, and predictable error responses. Extend shape support only with tests for georeferencing and output alignment.
4. Version and validate the alert schema. Keep projected and WGS84 coordinates explicit. Attach checkpoint hash and model configuration; never attach a hard-coded validation score.
5. Add minimum plume area and connected-component postprocessing as configured experiments, not assumed accuracy improvements. Separate predicted class from triage rules.
6. Store each request under a unique ID; avoid output overwrites and collisions. Add retention controls and a documented concurrency policy.

Acceptance: CLI and API produce matching masks from the same input/config; invalid inputs fail before model execution; output dimensions, CRS, transform, and schema are validated. Repeated requests reuse the model and cannot overwrite one another's artifacts.

## Milestone 4: make the application easy to inspect

Estimated effort: 2–3 working days.

1. Replace implicit Flask subprocess startup with an explicit service mode or direct model service. The dashboard's ten-second HTTP timeout can trigger duplicate inference.
2. Update historical model-card copy from verified configuration and results. Remove orbital deployment language until a device benchmark exists.
3. Add direct mask/JSON downloads, clear coordinate labels, and an explanation of area-based severity. Preserve empty, loading, error, and retry states.
4. Check mobile layout, keyboard navigation, focus visibility, and colour contrast. Verify unsupported upload feedback through the actual interface.
5. Create a recorded-results demo mode using clearly labelled stored outputs, so reviewers can inspect the workflow without a heavy model download.
6. Capture real application screenshots and a short walkthrough that identifies the input, mask, alert, limitations, and reproducible commands.

Acceptance: a reviewer can inspect the workflow in a few minutes; all displayed metrics are tied to source artifacts; every control has tested behaviour; the recorded-output mode is visibly distinct from a fresh prediction.

## Milestone 5: release a documented version

Estimated effort: 2–3 working days.

1. Add a container with a verified startup command and explicit CPU/GPU expectations. Benchmark cold start, warm inference, peak memory, and checkpoint size on named hardware.
2. Extend CI with package installation, CPU model-shape tests, alert-schema tests, and one sample inference smoke test using verified weights. Keep model-quality evaluation separate from fast unit tests.
3. Decide code, checkpoint, and included-data licensing with the relevant owners and source terms. Add a project license once that decision is established.
4. Package the checkpoint with a manifest and checksum, preferably as a release asset or documented model download instead of growing ordinary Git history.
5. Prepare a versioned release, changelog, reproducibility report, screenshots, and demo link if hosting is ready. Document service resource limits and failure recovery before public hosting.

Acceptance: installation works from a clean clone; CI passes; the release has verified artifacts, traceable metrics, a tested demo path, and clear terms. Public hosting is optional for the first reproducible release.

## Later research

Temporal multi-acquisition comparison, calibrated emission-rate estimation, human-annotated or independently observed plume evaluation, and INT8/embedded deployment are later experiments. Each needs its own data, methodology, and validation. Freezing a foundation encoder alone does not establish deployability or geographic generalisation.

## Suggested order

Complete milestones 1 and 2 before using performance claims. Milestone 3 supplies shared code for milestone 4. Milestone 5 packages the tested result. The estimates assume one developer and exclude data-download delays, training queues, and licensing discussions; they are planning estimates, not commitments.
