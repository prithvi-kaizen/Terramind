# Project provenance

Terramind is developed and maintained by [prithvi-kaizen](https://github.com/prithvi-kaizen).

The five research notebooks are organised under `notebooks/`. The application lives at the repository root, and the hackathon presentation is available at `docs/hackathon/Slide.pdf`. The historical README remains in `docs/hackathon/README.original.md`, with a notice explaining its discrepancies with the code and logs.

The maintained README, setup/data guides, model card, implementation plan, validation scripts, tests, CI, and contribution files are added during this setup. `infer.py` and `endpoint.py` receive input checks, safer checkpoint loading, corrected centroids, and local-server validation. The imported dashboard and notebook source remain recognisable; their remaining issues are documented rather than represented as already fixed.

## Artifact integrity

[artifact_manifest.json](artifact_manifest.json) records historical file paths, file sizes, and SHA-256 values for 13 preserved experiment artifacts. Repository validation checks these hashes.

Checkpoint:

```text
terramind_methane_best.pt
size: 88804452 bytes
sha256: 923aed2dac32cf632e5157e392bb7f20db24c7081d2908cb464a9ef3b2e561d0
```

The included sample alert retains its historical machine-local path and score fields as evidence. It is not a newly executed result. New alerts use the maintained CLI's schema and omit the unverified checkpoint validation score.

## Third-party components

The project uses the TerraMind foundation model via TerraTorch and the GeoCH4PlumeNet dataset. Their publishers remain the sources of model and data attribution. Links and interpretation are documented in [MODEL_CARD.md](MODEL_CARD.md) and [DATA.md](DATA.md).

## License status

The project does not contain a project-level `LICENSE` file. This setup does not assign a new license to the code or redistribute the full external dataset. The foundation model's card lists Apache-2.0; that does not determine the license of this application, its adapted checkpoint, or its sample data. A project-level licensing decision is tracked in the implementation plan.
