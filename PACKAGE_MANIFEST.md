# Package manifest

**Frozen:** 2026-10-02
**Package version:** 1.0
**Research status:** smoke-approved UCPA; full preregistered MNIST/CIFAR-10 validation not yet executed.

## Contents

- `.gitignore`: 1 files
- `README.md`: 1 files
- `aggregate_results.py`: 1 files
- `configs`: 28 files
- `evaluate_preregistered_gates.py`: 1 files
- `pyproject.toml`: 1 files
- `pytest.ini`: 1 files
- `requirements.txt`: 1 files
- `research`: 9 files
- `research_history`: 27 files
- `run_experiment.py`: 1 files
- `scripts`: 5 files
- `tests`: 2 files
- `ucpa_fl`: 16 files
- `verification`: 42 files

## Confirmatory study

- 26 frozen full-study configs in `configs/suite_manifest.yaml`.
- Fresh confirmatory seeds: 6101–6105.
- Tier A: core evidence; Tier B: stress/robustness; Tier C: sensitivity/communication.
- Full results are intentionally absent. Engineering-only verification runs are under `verification/`.

## Integrity

`PACKAGE_INTEGRITY.sha256` hashes every file in the frozen directory except itself. The outer ZIP SHA-256 is distributed alongside the archive.
