# Contributing

Use Python 3.12 and follow [the setup guide](docs/SETUP.md). Install `requirements-dev.txt` for tests and notebooks.

Before proposing a change, run:

```bash
python scripts/validate_repository.py
python -m pytest
python -m pip check
```

Describe the behaviour changed and how you checked it. Changes to spectral-band order, normalisation, data splits, thresholds, or metric aggregation should include the experimental configuration and relevant result artifacts. Keep raw training data, caches, credentials, and runtime outputs out of commits.

Use `infer.py` for maintained inference. Keep original notebooks and hackathon material recognisable; label historical records instead of silently rewriting their scores. Model-quality improvements require common evaluation on a fixed held-out split. Unit-test success alone does not establish higher accuracy.

Open a bug report with the command, Python/package versions, device, and error text. For an enhancement, explain the use case and proposed acceptance check. Licensing of inherited artifacts remains documented in [project provenance](docs/PROVENANCE.md).
