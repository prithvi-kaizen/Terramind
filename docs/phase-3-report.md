# Phase 3 implementation report

Phase 3 software is implemented for review. Scientific training, a replacement checkpoint and real temporal validation remain deferred. The current checkpoint is retained. The public demo continues to use the accepted Phase 2 workflow.

## Implemented software

- Separate encoder/decode methods preserve the original forward path and keep the frozen encoder in evaluation mode.
- Content-addressed feature caches include actual encoder state, source checkpoint, input, band order, preprocessing, augmentation, Torch and device. Fixed-input equivalence and invalidation are tested.
- Reviewed split-manifest checks and batch-one CPU decoder training support three fixed configurations, validation selection, optional two additional seeds, negative ablation, fixed SWIR comparison, paired site intervals and cooperative wall-time budgets. Test labels are not read for training or selection. Candidate weights never replace the accepted checkpoint automatically.
- Optional horizontal-flip review adds exactly one pass. Batch routing defaults off; automatic routing logs decisions and caps extra passes at `floor(N/10)`.
- Optional one-to-three-date review rejects incompatible grids, radiometry, invalid coverage, missing registration evidence, duplicate dates and duplicate image content before inference. Compatible stored predictions are reused; date-specific passes are separate. Pairwise one-to-one centroid matches report dates, counts and displacement without changing original quality flags.
- CLI commands, CI smoke coverage, a reproduction guide, compute report, release conditions and rollback instructions are included. No dashboard interface or default inference setting changes.

## Actual data and experiment decision

The [inventory](phase-3-inventory.json) records about 104 MB of local downloads, mostly earlier real-scene stress-test and validation inputs plus partial synthetic archives. It records about 10.9 GB free at inspection. These do not supply a reviewed independent Phase 3 training/test manifest, native deployment-matched labels, or reviewed co-registered dates. No additional data download or paid compute was started.

Remaining research inputs are: reviewed positive and difficult negative scenes in the supported input grid/radiometry; independent location/acquisition/plume-disjoint split assignments; label evidence and access terms; and one to three distinct dated observations with band/scaling and measured image-registration evidence. Independent severity labels and a controlled TinyUNet baseline are also unavailable. Existing historical severity-log disagreement remains unresolved.

## Verification evidence

The real offline CPU smoke verified exact cached/uncached feature agreement, cache reuse, decoder-only gradients and one extra enabled review pass. Its disposable gradient check uses an explicitly artificial mask and saves no experimental model. The existing cloud/API smoke retains the full raw reference mask exactly: 1,244 foreground pixels and zero changed pixels, equivalent adapters, one runtime load and three requests/passes. Original checkpoint/notebook artifacts remain unchanged.

The [compute report](phase-3-compute.json) contains 30 warm requests per review mode and process memory. See [quality versus compute](phase-3-quality-compute.md) for measurement scope. No numerical comparison with Google is supported.

Commands used:

```bash
python -m pytest -q
python scripts/validate_repository.py
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MPLCONFIGDIR=/tmp/terramind-mpl python scripts/smoke_inference.py
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MPLCONFIGDIR=/tmp/terramind-mpl python scripts/smoke_phase3.py --benchmark-runs 30 --output docs/phase-3-compute.json
```

## Acceptance status

| Gate | Outcome |
| --- | --- |
| Frozen-feature equivalence and compatible cache reuse | Passed with real checkpoint/sample and fixtures |
| Cache invalidation and no held-out training labels | Covered by tests |
| Three-configuration decoder command and resource reporting | Implemented; software tested with synthetic fixtures |
| Automatic extra-pass budget and routing logs | Covered by tests |
| Temporal validation, missing dates, empty masks and compatible cache reuse | Covered by fixtures; no real multi-date result |
| Targeted training quality and accepted replacement weights | Deferred; reviewed independent data unavailable |
| Domain-mismatch, negative-ablation, severity and controlled TinyUNet experiments | Deferred; research inputs/budgets required |
| Full suite, repository validator, real inference/UI smoke | 100 local tests, repository validation, real API/cloud regression, dashboard smoke and Phase 3 CPU smoke passed |
| Hosted Phase 3 deployment | Not requested by the phase release plan; accepted demo retained |

See [PHASE_3.md](PHASE_3.md) for commands and input contracts, and [release conditions](PHASE_3_RELEASE.md) before promoting model changes.

Local verification: 100 tests passed; repository validation checked 38 Python files, five notebooks, documentation links, JSON and original artifact hashes. Both Linux jobs passed for implementation commit `3a96eb3` in [Actions run 42055595340](https://github.com/prithvi-kaizen/Terramind/actions/runs/42055595340), including all tests, complete CPU dependency installation, raw/API/cloud regression, dashboard checks and the Phase 3 frozen-feature/gradient/flip smoke. [PR 3](https://github.com/prithvi-kaizen/Terramind/pull/3) contains the reviewable changes. It is not merged or deployed.

Documentation review PASS: numerical claims trace to the saved compute/inventory reports; artificial fixtures and missing research evidence are explicitly identified. No interface or visual assets were added.
