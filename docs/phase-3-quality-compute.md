# Phase 3 quality versus compute

The [machine-readable CPU report](phase-3-compute.json) measures the included sample on macOS ARM, Python 3.12.4, Torch 2.11.0, two CPU threads. This is a software workload, not a real-plume benchmark.

| Measurement | Recorded result |
| --- | ---: |
| Cache creation, first feature tensor | 0.197 s |
| Compatible feature-cache hit | 0.039 s |
| Compressed feature tensor | 281,493 bytes |
| Cached versus uncached feature maximum absolute difference | 0 |
| Default review-off median / p95, 30 warm requests | 0.100 / 0.107 s |
| Flip-review median / p95, 30 warm requests | 0.162 / 0.169 s |
| Flip-review mask IoU with original prediction | 0.3572 |
| Mean absolute probability difference | 0.004064 |
| Peak process RSS across the complete smoke workload | 1,161,592,832 bytes |
| Total smoke wall time | 13.79 s |

The final recorded run followed the test runs, without another agent-started compute job running alongside it. Host scheduling and cache state can still vary. The timing includes preprocessing and inference but excludes schema 2.0 raster/geometry artifact writing. It cannot be compared directly with Phase 2 end-to-end output timings. The review benchmark used one warm-up enabled request (two passes), 30 disabled requests (30 passes) and 30 enabled requests (60 passes): 92 review passes and one runtime model load. Feature-equivalence checks and one disposable decoder gradient step happened earlier in the same process and are separate from that runtime counter.

A 0.3572 flip mask IoU demonstrates prediction instability on the sample. It is not a false-positive reduction result or a calibrated probability. Agreement, even when high, would not confirm methane. Automatic routing limits extra passes to `floor(N/10)`; this does not establish an equivalent wall-time, energy or monetary saving.

The smoke's decoder gradient step used an artificial all-zero label solely to check frozen-encoder gradients. It did not create, select or save a research checkpoint. No research training was performed, so there are no training-quality results, negative-ablation results, independent severity results or training-hour claims. Bounded commands now record those resources when reviewed data is supplied.

| Research acceptance question | Current decision |
| --- | --- |
| Does targeted training reduce false positives within 0.02 recall loss? | Unmeasured; retain existing checkpoint |
| Does cached training preserve fixed-input features? | Verified on the real included checkpoint and sample |
| Do real aligned dates corroborate candidates? | Unmeasured; reviewed observations unavailable |
| Is flip agreement useful for real-scene vetting? | Unmeasured; optional review only |
| Is an EMIT/Google numerical cost or quality comparison supported? | No comparable sensor/workload experiment |

The existing poor real-scene transfer findings still apply. See [the Phase 2 report](phase-2-report.md) and [evaluation protocol](EVALUATION.md).
