# MethanePulse model card

## Purpose and scope

The prototype segments simulated methane-plume patterns in Sentinel-2 tiles and assigns a relative mask-area class. It supports research and demonstrations. It has not been validated for regulatory reporting, automated mitigation decisions, continuous monitoring, physical emission quantification, or spacecraft deployment.

## Architecture

The frozen encoder is `terramind_v1_small`, with modality `S2L2A` and the `SWIR_1`/`SWIR_2` band subset, loaded through TerraTorch. See the [upstream model card](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-small) for the foundation model's provenance and terms.

Inputs to the file loader have 13 channels at 512×512. B11 and B12 receive independent 2nd/98th percentile normalisation. The model resizes them to 224×224, uses the last encoder layer's 196×384 patch embeddings, and sends those embeddings to two heads:

- Segmentation: four stride-2 transposed convolutions, batch normalisation and ReLU, followed by a 1×1 convolution and interpolation to 512×512.
- Severity: mean pooling over patch tokens, Linear(384, 128), ReLU, Dropout(0.3), and Linear(128, 3).

The saved notebook output reports 21,481,728 frozen encoder parameters and 616,388 trainable head parameters. The severity branch does not use a CLS token. A frozen encoder does not guarantee identical output across hardware or guarantee good geographic generalisation.

## Training record

The notebook uses a location-based 619/185 split, batch size 2, ten epochs of severity warmup, then twenty joint epochs. Joint loss is 0.7 times segmentation loss plus 0.3 times class-weighted severity cross-entropy. Segmentation loss is an equal mixture of BCE-with-logits and soft Dice loss.

The instantiated AdamW optimizer uses learning rate 1e-3 and weight decay 1e-4. A later unused `lr=5e-4` assignment does not change that optimizer. A second scheduler assignment overrides the first: effective ReduceLROnPlateau patience is 3, not 5. The notebook also calls `model.train()`, which switches its frozen encoder into training mode; the Phase 1 notebook preparation command creates an unexecuted corrected copy that keeps the encoder in evaluation mode.

The dataset uses L1C imagery while the configured model modality is S2L2A. The impact of this mismatch and the per-tile normalisation on pretrained features has not been evaluated.

## Results and disagreements

| Source | Observation |
| --- | --- |
| `notebooks/TerraMind.ipynb`, training cell | Best IoU 0.3298 at epoch 8; severity accuracy 0.395 at that epoch |
| `docs/terramind_training_history.json` | Best IoU 0.3257 at epoch 6; F1 0.5110 there; severity accuracy 0.0 in all 20 records |
| Same JSON log | Best F1 0.5112 at epoch 4; this is a different epoch from best IoU |
| Imported checkpoint and sample alert | No embedded run manifest establishes which log produced the weights; the historical alert hard-codes IoU 0.3298 |

The baseline notebook averages per-tile scores. TinyUNet aggregates global pixel counts. The TerraMind notebook averages per-batch IoU, including an uneven final batch. The separate log has no accompanying evaluator in the imported application. Do not use the inherited 3.2× improvement claim as a controlled benchmark.

TinyUNet's saved global pixel accuracy is 0.9969, but foreground occupies little of each image; that score is unsuitable as a headline detection metric. No independent test-set result, repeated-seed variance, negative-scene evaluation, or calibrated confidence result is included.

## Alert semantics

The maintained CLI uses schema version 2.0, retaining the raw detection fields from 1.1:

- `centroid_geographic` retains projected `[x, y]` coordinates in `source.crs` for compatibility. Its name is historical; it is not necessarily longitude/latitude.
- `centroid_wgs84` provides `[longitude, latitude]` in EPSG:4326, using pixel-centre coordinates and the full raster affine transform.
- Probability values are model outputs, not calibrated leak probabilities.
- `LOW`, `MEDIUM`, and `HIGH` describe a learned area proxy; they are not kg/h classes or independently measured hazard levels.
- `CLEAR` means no pixel crossed the selected threshold. It does not certify absence of methane.
- Runtime alerts omit a validation score because the checkpoint-to-run mapping is unresolved.
- `artifacts` identifies mask and alert files on the local server filesystem; these are not public download URLs.

The original sample alert and `src/alert_packet.py` remain schema 1.0 historical artifacts. They do not contain the maintained coordinate correction.

## Full synthetic checkpoint audit

All 1,000 published GeoCH4PlumeNet inputs were attempted at threshold 0.5. The published test split scored 195 of 196 images, with global pixel IoU 0.42906 and F1 0.60048; one matching label was unavailable to the archive adapter. Published splits share locations and plume identifiers, and checkpoint training membership is unresolved. The score is a descriptive artifact audit, not evidence of independent generalisation. The combined 999-image score includes training images. See [EVALUATION.md](EVALUATION.md) for all splits and raw counts.

## External real-scene stress test

The existing checkpoint at threshold 0.5 was evaluated on a deterministic MethaneSET test subset: 64 targets and 64 provider-curated clean references. One pair had invalid pixels and was rejected, leaving 63 positives and 63 negatives. An explicitly declared offline resize adapter returned probability maps to the native label grid. Pixel IoU was 0.01742 and F1 was 0.03425; 62 of 63 clean references triggered predictions. Only 33 of 63 positive scenes had any overlap with their annotated plume.

These results demonstrate poor real-scene transfer under this adapter. They do not establish operational methane detection. The subset is small and includes correlated sites; references are provider-labeled rather than independently reviewed by this project. The initial external run had an archive-local catalog ID join bug and is superseded by a corrected rerun. See [the evaluation protocol and reports](EVALUATION.md).

## Candidate review (Phase 2)

Four-connected regions receive physical area, GeoJSON geometry, pixel/CRS-aware centroids, and informational quality flags. Area is WGS84 ellipsoidal area, including polygon holes. Candidate flags and evidence states are uncalibrated; neither an unflagged candidate nor high model probability confirms methane. Default output preserves all raw predictions. Explicit experimental filtering creates a separate artifact and does not rewrite severity. The 32-pair validation pilot failed the two-percentage-point recall-loss promotion criterion for rules that reduced false positives. All removal rules remain off by default. See [PHASE_2.md](PHASE_2.md).

## Known limitations

There is one acquisition per inference, no temporal comparison, no atmospheric correction pipeline, no emission-rate estimation, and no deployment-matched negative-scene benchmark. Input support is restricted to the documented shape and band order. The service caches one model per worker, serializes inference, and writes isolated request outputs. Output cleanup is operator-managed. The dashboard uses direct inference by default; its optional local API mode stops on timeout without launching duplicate local work. See [Phase 1](PHASE_1.md).

The hackathon presentation discusses orbital and INT8 deployment. This repository contains no device-specific benchmark, quantised artifact, radiation-tolerance evidence, or flight-software qualification. Those are prospective research directions.

## Artifact traceability

The checkpoint is preserved unchanged from the hackathon experiment. [PROVENANCE.md](PROVENANCE.md) records its SHA-256. A new experiment should save its split manifest, seed, dependency versions, thresholds, model configuration, evaluation convention, and checkpoint hash together.
