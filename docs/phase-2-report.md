# Phase 2 implementation report

Updated 5 November 2026. Candidate extraction, review signals, schema 2.0 exports, explicit experimental filtering and the dashboard workflow are implemented. Phase 3 training and corroboration are not implemented. No new model weights or detection-accuracy improvement is claimed.

## Delivered

- Four-connected regions with request-local IDs, geodesic area, native/WGS84 centroids, bounds, polygons with holes, and uncalibrated probability summaries.
- Informational small-area, edge-contact, threshold-sensitivity and validity flags, with candidate/needs-review evidence states. Fragmentation is bounded before polygon generation.
- Raw probability/mask preservation, label raster, separate filtered mask, GeoJSON provenance, schema 2.0 JSON and a machine-readable migration contract.
- CLI/API configuration, candidate inspection/crops, five downloads, raw/filtered display selection, pre-inference upload validation and data-driven model-card metadata.
- A deterministic validation-only pilot, individual/combined rule comparisons and paired site-bootstrap intervals. None of the rules qualifies for default promotion.

## Verification

The local suite passed 71 tests. It covers masks with no candidates, diagonal/disconnected regions, holes, rotated transforms, geographic and projected CRS, projected feet, nodata boundaries, review settings, configuration hashes, fragmentation limits, explicit filtering and preservation of raw artifacts. Original artifact hashes and notebook structure still validate.

Real offline CPU smoke passes with the included checkpoint. The saved sample retains 1,244 foreground pixels and four candidates. Schema 2.0 validates against the committed JSON Schema. Probability and mask rasters preserve the input grid. Default filtered output equals raw. The shared runner, Flask API and cloud entrypoint agree, with one cached model and one forward pass per request.

Streamlit AppTest verified real candidate inspection, all five downloads, explicit filtering, raw preservation, one model load, and no additional forward pass while changing inspection/display controls. Isolated, clearly synthetic fixtures verified empty/error states. The actual browser produced a downloadable GeoJSON FeatureCollection with four Polygon features, checkpoint provenance and no default filter rules. A synthetic three-band, 16×16 upload displayed the supported-input error and disabled Run Inference.

At 390×844, document width and scroll width were both 390 pixels; all five download buttons were 44 pixels high and fitted their column. At width 900, document and scroll widths were both 900. The temporary viewport override was reset. The existing focus accent has measured contrast 5.72:1 on the background. Keyboard and hosted checks are recorded below when complete; this is not a claim of a full independent accessibility audit.

## Computation

The final [paired benchmark](phase-2-benchmark.json) uses Phase 1 revision `74f0238`, the same checkpoint/tile, two CPU threads, separate fresh processes and 30 warm requests per variant:

| Measure | Fresh Phase 1 baseline | Phase 2 |
| --- | ---: | ---: |
| Warm median | 0.16765 s | 0.18466 s |
| Warm p95 | 0.17975 s | 0.20508 s |
| Peak process RSS | 1,018,642,432 bytes | 1,116,520,448 bytes |
| Model loads | 1 | 1 |
| Forward passes, including warm-up | 31 | 31 |

Warm p95 increased 14.1%, meeting the proposed maximum of 20% for this workload. An earlier development run measured 6.8%; the final saved run is used here. Host activity and cache state cause timing variability. The extra rasters/geometry increase storage and memory, and this single repeated tile does not establish a general latency guarantee across fragmented masks, concurrent users or different hardware.

## Rule validation and decision

The [cohort](phase-2-validation-cohort.json) contains 32 validation targets and 32 clean references, disjoint from the earlier test cohort. All 64 scored with exactly 64 forward passes. The [report](phase-2-validation.json) includes source revision, input/label hashes, rule settings, per-input counts and paired bootstrap intervals grouped over 16 sites.

The small-area rule reduced clean-reference detections from 32/32 to 31/32 but lost 6.25 percentage points of any-foreground recall. Combined rules reduced them to 30/32 but lost 9.375 points. Threshold-sensitivity removal lost 3.125 points without reducing the scene false-positive rate. Edge removal changed neither scene rate. These fail the two-point recall-loss/fewer-false-positive promotion criterion, and uncertainty intervals do not support promotion. All rules remain informational by default. The adapted validation cohort does not establish operational performance or supersede the poor real-test results.

## Acceptance status

| Gate | Outcome |
| --- | --- |
| Candidate geometry, CRS/units, holes, validity, empty/edge/rotated tests | Passed locally |
| Raw artifacts preserved; schema 2.0/GeoJSON export | Passed locally |
| One default forward pass; inspection does not rerun model | Passed locally |
| Warm p95 at most 20% above fresh Phase 1 | Passed on the recorded single-sample workload |
| Individual/combined rule evaluation | Completed; no filter promoted |
| Real downloads, invalid-upload feedback, narrow/tablet layout | Verified locally in browser |
| Linux full environment, adapter/schema/UI smoke | Pending remote CI evidence |
| Hosted Phase 2 prediction, candidates and downloads | Pending publication verification |

See [PHASE_2.md](PHASE_2.md) for reproduction commands and schema migration. Raw data and local output rasters stay outside Git; the reports preserve measurement evidence. Existing model licensing and training-run traceability limitations remain unchanged.
