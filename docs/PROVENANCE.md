# Project provenance

Terramind is maintained by [prithvi-kaizen](https://github.com/prithvi-kaizen). The developer identifies this as their hackathon project. The import preserves the original commit authors and team identifiers rather than replacing their history.

Source: [Zy1r9t7ec/methane-tracker](https://github.com/Zy1r9t7ec/methane-tracker).

| Source branch | Imported revision | Contents |
| --- | --- | --- |
| `main` | `bce7a7dacc9341accebcc6e69617e22385a97850` | Five research notebooks |
| `master` | `291525986212f964ac9d546cd2eee950f455a255` | Application, inference, endpoint, checkpoint, sample, logs, notebooks, and presentation |

The branches originally have separate histories. A merge joins them without rewriting authorship. Identical notebooks from `main` and `master` are consolidated under `notebooks/`. The application moves from `submissions/MethanePulse/` to the root; the slide deck moves to `docs/hackathon/Slide.pdf`. The original README remains in `docs/hackathon/README.original.md`, with a historical-record notice.

The maintained README, setup/data guides, model card, implementation plan, validation scripts, tests, CI, and contribution files are added during this setup. `infer.py` and `endpoint.py` receive input checks, safer checkpoint loading, corrected centroids, and local-server validation. The imported dashboard and notebook source remain recognisable; their remaining issues are documented rather than represented as already fixed.

## Artifact integrity

[artifact_manifest.json](artifact_manifest.json) records the original source path, file size, and SHA-256 for 13 unchanged imported artifacts. Repository validation checks these hashes.

Checkpoint:

```text
terramind_methane_best.pt
size: 88804452 bytes
sha256: 923aed2dac32cf632e5157e392bb7f20db24c7081d2908cb464a9ef3b2e561d0
```

The included sample alert retains its historical machine-local path and score fields as evidence. It is not a newly executed result. New alerts use the maintained CLI's schema and omit the unverified checkpoint validation score.

## Third-party components

The project uses the TerraMind foundation model via TerraTorch and data identified by the upstream README as GeoCH4PlumeNet. Their original publishers remain the sources of model and data attribution. Links and interpretation are documented in [MODEL_CARD.md](MODEL_CARD.md) and [DATA.md](DATA.md).

## License status

Neither imported source branch contains a project-level `LICENSE` file. This setup does not assign a new license to inherited code or redistribute the full external dataset. The foundation model's card lists Apache-2.0; that does not determine the license of this application, its adapted checkpoint, or its sample data. A project-level licensing decision is tracked in the implementation plan.
