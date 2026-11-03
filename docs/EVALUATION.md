# Public-data evaluation protocol

This protocol evaluates the existing checkpoint at threshold 0.5 without retraining, threshold searches, or changing the live app's preprocessing. The two experiments answer different questions.

## Complete synthetic dataset audit

Download all three archives from [GeoCH4PlumeNet v1](https://zenodo.org/records/16813369), verify the publisher's MD5 checksums, and evaluate every included image with its matching ground-truth mask. Read TIFFs directly from each archive to avoid maintaining an additional extracted copy. Convert labels with the historical rule `mask > 0.5`; check image/label shape, CRS, and transform before counting pixels. Record failed inputs rather than substituting predictions or ignoring failures silently.

Report global pixel IoU/F1, mean tile IoU, scene counts, and any-mask-overlap counts by published split. Also audit location/plume identifiers shared between published splits and report the historical location-8/9 subset. These are descriptive measurements of a preserved checkpoint: its training-run association is unknown, and evaluating training images does not establish held-out performance. Keep the combined full-corpus score separate from the published test split. Report scene precision as unavailable when there are no negative scenes.

## Real-plume cohort

Use [MethaneSET s2-finetune](https://huggingface.co/datasets/tacofoundation/methaneset), which contains MARS plume masks and provider-curated clean references, under CC BY-NC-SA 4.0. Resolve and save the immutable Hugging Face revision before acquiring data. Select 64 entries from its test split, restricted to catalog sensors `S2A` and `S2B`, in ascending SHA-256 order of `terramind-real-v1:` plus the sample ID. Save the cohort before model predictions. Evaluate each target and its `bg0` reference, with the reference treated as a provider-labeled negative. Do not infer negatives from missing detections.

The subdataset README contains copied Landsat descriptions. The actual catalog and a downloaded archive identify 13-band Sentinel-2 rasters with B11/B12 at positions 12/13. The evaluator restricts the catalog sensor, checks the TIFF band count, shape and valid pixels, and requires alignment with the plume mask. It uses the catalog's TIFF byte offsets to fetch only target, reference, and label data; no full multi-gigabyte country archive is needed. Exact HTTP range responses and source hashes are verified and recorded.

These real chips cover 200 by 200 pixels at 10 m, while the demo accepts 512 by 512 tiles. The offline adapter preserves the geographic footprint, bilinearly resizes all bands to 512 by 512, applies the existing model preprocessing, then bilinearly returns predicted probabilities to the original grid before thresholding at 0.5. Ground-truth masks stay on the native grid. This is an explicitly out-of-domain stress test with different spatial context and effective resolution, not an operational Sentinel-2 validation or a reproduction of the dataset authors' benchmark. No adapter setting is selected using its test results.

Report positive and negative scene counts, scene recall/precision, false-positive scene rate, global pixel IoU/F1 and individual outcomes. A detection anywhere in the scene is weaker evidence than overlap with the annotated plume; report both. Correlated observations/sites and the modest subset size limit statistical independence. Provider-curated references have not been independently reviewed by this project.

## Reproduce

Install the normal CPU deployment requirements, then run:

```bash
HF_HUB_OFFLINE=1 python scripts/evaluate_public_data.py synthetic \
  --data data/raw/evaluation --output outputs/public-evaluation --remove-archives
HF_HUB_OFFLINE=1 python scripts/evaluate_public_data.py real \
  --data data/raw/evaluation --output outputs/public-evaluation --real-samples 64
```

The synthetic task downloads about 5.8 GB cumulatively and deletes each verified archive after its evaluation when `--remove-archives` is supplied. Real-data acquisition uses two small catalogs and selected TIFF ranges. Neither command uploads images, trains weights, or invokes paid compute services. Outputs include sample hashes, input counts, failed cases, threshold, checkpoint identity, source revision and metric conventions. Archive downloads and evaluation can take substantial time depending on bandwidth and CPU.

The manually dispatched `Public dataset evaluation` GitHub Actions workflow can run either experiment or both in separate CPU jobs and retains JSON reports as artifacts for 30 days. Download and preserve the numeric reports in the repository after completion; raw third-party images and masks stay outside Git. Regular repository CI remains a separate check of software correctness and sample prediction parity.

## Real-scene results

The [fixed cohort](evaluation/real-cohort.json) contains 64 test targets and 64 paired clean references across ten countries. The [corrected results](evaluation/real-results.json) evaluate 63 positive scenes and 63 references. One pair contains invalid pixels and is rejected by the supported input contract. The checkpoint and threshold are unchanged.

| Measure | Corrected result |
| --- | ---: |
| Inputs evaluated / selected | 126 / 128 |
| Global pixel IoU | 0.01742 |
| Global pixel F1 | 0.03425 |
| Clean references with any predicted foreground | 62 / 63 (98.4%) |
| Positive scenes with any predicted foreground | 61 / 63 (96.8%) |
| Positive scenes with any annotated-mask overlap | 33 / 63 |

The high any-foreground recall accompanies an unacceptable false-positive rate and weak spatial overlap. It does not establish useful methane detection. The evaluation is a small, correlated, out-of-domain cohort with a declared resize adapter. Future training must address real-data transfer and negatives before operational claims are warranted.

The initial Actions real-data artifact from run 41582495788 is invalid: its catalog join used archive-local parent IDs without the source filename. The corrected code joins on both fields and requires exactly one target/reference/mask. A regression test covers collisions across country archives. The fixed cohort IDs and selection method were retained; thresholds and model weights were not tuned using these outcomes. The corrected report supersedes the initial artifact.

The corrected real run used the exact evaluator source whose SHA-256 is recorded in `real-results.json`. A later download-only optimization adds four bounded HTTP range workers for synthetic archives; it does not change predictions or the real-scene method.
