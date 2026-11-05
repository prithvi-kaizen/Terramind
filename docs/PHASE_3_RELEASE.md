# Phase 3 release checklist

Phase 3 tooling is reviewable on its own branch. It does not approve a new checkpoint or a hosted deployment.

## Software checks

- Run repository validation, complete unit tests and offline CPU inference regression.
- Run the real frozen-feature/cache/gradient smoke and existing dashboard smoke.
- Require Linux CI for the exact proposed commit before merge.
- Confirm default sample predictions, raw artifacts and schema 2.0 remain compatible.
- Preserve the accepted checkpoint and imported artifact hashes.

## Research conditions before model promotion

- Review deployment-matched positives and hard negatives; record label evidence and access terms.
- Freeze location/acquisition/plume-disjoint manifests; keep test labels out of training and selection.
- Resolve radiometry/normalisation domain differences; document severity labels or keep severity unchanged.
- Run at most three initial configurations, then selected-seed repeats within a stated resource budget.
- Compare baseline, selected model, fixed SWIR baseline and a separately budgeted negative ablation.
- Freeze the chosen checkpoint and threshold before independent test evaluation.
- Require fewer false-positive scenes with no more than 0.02 absolute scene-recall loss; inspect overlap/pixel tradeoffs and site-level uncertainty.
- Record model/input/cache hashes, errors, compute, dependency versions and licensing before redistribution.
- For temporal release, obtain distinct dated tiles with reviewed radiometry/registration evidence and evaluate actual matches/errors.

## Deployment and rollback

The accepted checkpoint hash is `923aed2dac32cf632e5157e392bb7f20db24c7081d2908cb464a9ef3b2e561d0`. Existing raw masks and schema 2.0 alert fields remain the compatibility reference. Phase 2 hosted deployment was verified at merge `3a78f6b`; documentation evidence followed in `3d248f2`.

After software review and explicit deployment instruction, merge only accepted changes. Do not point the demo at an experimental checkpoint based on a validation point estimate. Run the hosted sample and verify candidate inspection/downloads after any deployment.

If a merged change fails, revert that release commit through a reviewed Git revert, restore the accepted checkpoint/configuration, restart the worker to clear model resources and rerun the raw-mask/schema smoke. Result/feature caches are local derived artifacts; use a new cache directory when restoring a configuration rather than reusing unknown entries. Rollback must preserve user inputs and original research artifacts. No destructive Git reset is required.
