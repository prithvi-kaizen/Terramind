# Phase 3: reliability research tooling

Phase 3 software supports frozen-encoder experiments and optional review signals. Research training and real multi-date evaluation remain deferred because reviewed data is unavailable. The existing checkpoint and public demo remain the accepted release. No accuracy improvement is claimed.

## Data required before training

Use the schema 1 split manifest described in [Phase 1](PHASE_1.md). Every row additionally needs `review_status: "accepted"`, `reviewer` and `review_evidence`. Keep `label_source`, `location`, `acquisition`, `plume_id` when available, and `verified_negative`. Training and validation must each include positive masks and independently verified negative scenes. Use difficult surfaces/clouds/shadows as reviewed negatives where the input contract supports them. Model predictions do not supply negative ground truth.

Tile and binary label GeoTIFFs must align exactly. The maintained input remains 13 bands, 512×512, fully valid, with B11/B12 percentile normalisation. The manifest loader rejects duplicate inputs and leakage of location, acquisition or plume identifiers across splits. It checks file hashes for all splits; training reads labels only for train and validation. An accepted review field is an attestation from the manifest author, not automatic scientific verification.

The current downloads contain prior resized stress-test and validation cohorts, not a new independent training benchmark. Reusing them under new split names does not remove leakage. Resolve L1C/L2A and normalisation differences before a new model-quality experiment. The saved [inventory](phase-3-inventory.json) identifies what is actually local.

## Bounded decoder training

```bash
python scripts/phase3.py train \
  --manifest /path/to/reviewed-splits.json \
  --directory outputs/phase3/experiment-01 \
  --epochs 3 --max-seconds 1800
```

The command uses CPU, two threads, batch size one and seed 42. It starts every configuration from the existing checkpoint. The search contains exactly three configurations: BCE at learning rates 0.0001 and 0.0003, and BCE plus 0.5 Dice loss at 0.0001. The encoder stays frozen and in evaluation mode. Only the segmentation decoder trains. Severity weights remain unchanged because independent severity labels are unavailable.

The command permits 1–10 epochs and 1–7200 seconds. Wall-time checks occur between operations; they do not interrupt a running tensor operation. An exhausted budget records its status and never promotes a checkpoint. Use a fresh output directory for each run. `--repeat-selected` adds seeds 43 and 44 for the configuration selected by validation global IoU, for five runs in total. `--without-negatives` supports a separately budgeted ablation with identical manifests, seeds and configurations. It removes negatives from training only; validation negatives remain.

Features are content-addressed compressed float32 token tensors, one file per fixed input/augmentation. Keys include input hash, actual encoder-state hash, complete source checkpoint hash, preprocessing/architecture versions, B11/B12 order, augmentation, Torch version and device. `identity` and horizontal `flip-x` are the only cache augmentations. The training command currently uses identity features only. Decode-time training dropout is not cached. The cache rejects invalid tensors and changed checkpoint files. Cache directories are local research artifacts; remove obsolete entries explicitly when storage is needed.

The output contains complete candidate checkpoints, `completed-runs.json`, and a resource/evaluation `report.json`. Validation compares the current checkpoint, trained decoder and a fixed SWIR difference baseline using the same binary-count conventions. Scene detection means any foreground pixel, so overlap recall is also reported. The SWIR baseline is absolute independently normalised B11/B12 difference greater than 0.2; it is not methane-specific. Site-bootstrap intervals report paired scene-rate differences. A point-estimate gate requires fewer false-positive scenes and at most 0.02 absolute recall loss. It does not approve release.

Freeze the chosen checkpoint/configuration/threshold in an evaluation manifest before using the existing evaluator on independent test data:

```bash
python scripts/evaluate.py --manifest /path/to/reviewed-splits.json \
  --checkpoint outputs/phase3/experiment-01/CHOSEN-CHECKPOINT.pt \
  --split test --threshold 0.5 --output outputs/phase3/held-out.json
```

The example checkpoint name is a placeholder for the frozen selection. Threshold 0.5 is fixed before training. Report site-level uncertainty and inspect errors before promotion. Historical TinyUNet aggregates and severity logs are not controlled comparisons; retraining TinyUNet and independent severity evaluation require separately reviewed data and budgets.

## Optional consistency review

```bash
python scripts/phase3.py consistency --tile /path/to/tile.tiff \
  --output outputs/review-off.json
python scripts/phase3.py consistency --tile /path/to/tile.tiff \
  --enable --output outputs/review-flip.json
```

Without `--enable`, this performs one base pass. Enabled review adds exactly one horizontal-flip pass and inverts the output before comparing masks and probabilities. It reports mask IoU, both-empty status and mean absolute probability difference. Empty/empty IoU is one. These signals are not calibrated uncertainty or evidence of methane. Raw predictions are unchanged.

For automatic batch routing, provide JSON with `tiles`, each containing `tile` and a boolean `flagged`. Paths resolve relative to the manifest. Flags are supplied from prior review, not inferred ground truth.

```bash
python scripts/phase3.py batch-consistency --manifest /path/to/batch.json \
  --mode automatic --output outputs/batch-review.json
```

The default mode is `off`. Automatic mode chooses flagged rows in manifest order up to `floor(N/10)`, logs every routing decision and executes at most `N + floor(N/10)` passes. Fewer than ten rows receive no automatic extra pass. Explicit single-tile review remains available. Temporal-date passes are accounted separately.

## Optional temporal review

Supply one to three distinct dated observations from one location. Missing dates are allowed, but one observation cannot establish corroboration. A minimal observation entry is:

```json
{
  "date": "2024-03-26",
  "tile": "/path/to/tile.tiff",
  "registration_residual_pixels": 0.1,
  "registration_evidence": "path or reference to reviewed alignment measurements",
  "radiometry": {
    "processing_level": "L2A",
    "scale": 0.0001,
    "offset": 0,
    "band_order_verified": true,
    "evidence": "reference to band/scaling metadata"
  }
}
```

Put entries in `observations` and provide a nonempty top-level `location`:

```bash
python scripts/phase3.py temporal --manifest /path/to/dates.json \
  --cache outputs/phase3/predictions --tolerance-m 200 \
  --output outputs/phase3/temporal.json
```

All inputs must pass maintained preprocessing, use the same processing level/scaling, and have identical CRS, affine transform, shape, resolution and full valid coverage. The reviewer must supply alignment evidence with a residual no greater than 0.25 pixels. Grid equality does not measure image-content registration. Duplicate image content and duplicate dates are rejected before prediction. No resampling, atmospheric correction or image registration happens automatically.

The result cache includes input content, checkpoint, architecture, preprocessing, threshold, postprocessing/version, Torch and device. Compatible predictions are reused. New dates each cost one pass; temporal matching adds no model passes. All date pairs use one-to-one minimum-distance centroid matching within an explicit 0–1000 m tolerance, with a pair-count budget. Direct matches contribute dates, counts and displacements to each candidate's separate `temporal_evidence`. Two matched dates yield `temporally_corroborated`; the original quality flags/evidence state remain intact. This is pairwise evidence, not a transitive track or an emission-source estimate. Plume movement can break matches, and a missed observation does not prove absence.

## Verification and release

```bash
python -m pytest -q
python scripts/validate_repository.py
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 python scripts/smoke_inference.py
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 python scripts/smoke_phase3.py
```

See [the implementation report](phase-3-report.md), [quality versus compute](phase-3-quality-compute.md), and [release checklist](PHASE_3_RELEASE.md). Phase 3 outputs are separate research reports; the deployed schema 2.0 alert contract does not change. The hosted dashboard does not expose these experiments.
