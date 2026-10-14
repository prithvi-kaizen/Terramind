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

The instantiated AdamW optimizer uses learning rate 1e-3 and weight decay 1e-4. A later unused `lr=5e-4` assignment does not change that optimizer. A second scheduler assignment overrides the first: effective ReduceLROnPlateau patience is 3, not 5. The notebook also calls `model.train()`, which switches its frozen encoder into training mode; keeping that encoder in evaluation mode is a reproducibility fix on the plan.

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

The maintained CLI uses schema version 1.1:

- `centroid_geographic` retains projected `[x, y]` coordinates in `source.crs` for compatibility. Its name is historical; it is not necessarily longitude/latitude.
- `centroid_wgs84` provides `[longitude, latitude]` in EPSG:4326, using pixel-centre coordinates and the full raster affine transform.
- Probability values are model outputs, not calibrated leak probabilities.
- `LOW`, `MEDIUM`, and `HIGH` describe a learned area proxy; they are not kg/h classes or independently measured hazard levels.
- `CLEAR` means no pixel crossed the selected threshold. It does not certify absence of methane.
- Runtime alerts omit a validation score because the checkpoint-to-run mapping is unresolved.
- `artifacts` identifies mask and alert files on the local server filesystem; these are not public download URLs.

The original sample alert and `src/alert_packet.py` remain schema 1.0 historical artifacts. They do not contain the maintained coordinate correction.

## Known limitations

There is one acquisition per inference, no temporal comparison, no atmospheric correction pipeline, no emission-rate estimation, and no negative-scene benchmark. Input support is restricted to the documented shape and band order. The service reloads its model for each request and retains outputs on disk. The dashboard can trigger duplicate work after an HTTP timeout.

The upstream presentation discusses orbital and INT8 deployment. This repository contains no device-specific benchmark, quantised artifact, radiation-tolerance evidence, or flight-software qualification. Those are prospective research directions.

## Artifact traceability

The checkpoint is imported unchanged from upstream `master`. [PROVENANCE.md](PROVENANCE.md) records its SHA-256. A new experiment should save its split manifest, seed, dependency versions, thresholds, model configuration, evaluation convention, and checkpoint hash together.
