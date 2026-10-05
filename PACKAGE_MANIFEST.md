# Package manifest

**Frozen:** 2026-10-04
**Package version:** 2.1
**Research status:** NC-SACPA Phase-VIII smoke-test approved; full preregistered MNIST/CIFAR-10 validation intentionally unrun.

## Frozen research state

- Proposed method: Neighborhood-Corroborated Scale-Adaptive Peer Alignment (NC-SACPA).
- Smoke confirmation: all 7 Phase-VIII gates passed on untouched seeds 801–805.
- Full confirmatory seeds: 9101–9105, verified absent from packaged result-bearing files.
- Full suite: 28 configs (6 Tier A / 8 Tier B / 14 Tier C).
- Frozen NC-SACPA: p=2, beta=.8, q=3, min_support=2.

## Important contents

- `README.md`: entry point and run commands.
- `research/PREREGISTRATION.md`: frozen full-study hypotheses/gates.
- `research/METHOD.md`: exact method.
- `research/NOVELTY_POSITIONING.md`: 2026-10-04 prior-art positioning.
- `evaluate_ncsacpa_gates.py`: executable seven-gate full-study decision.
- `research_history/phase6_ucpa_full_negative/`: killed UCPA full-study evidence.
- `research_history/phase7_sacpa_rejected/`: preregistered failed predecessor plus raw confirmation.
- `research_history/phase8_ncsacpa_smoke_approved/`: approved smoke preregistration, raw seeds 801–805, evaluator, and gate report.
- `verification/FINAL_VERIFICATION.md`: final engineering/integrity audit.

## File counts before integrity manifest
- `.gitignore`: 1 files
- `PACKAGE_MANIFEST.md`: 1 files
- `README.md`: 1 files
- `aggregate_results.py`: 1 files
- `configs`: 30 files
- `evaluate_ncsacpa_gates.py`: 1 files
- `pyproject.toml`: 1 files
- `pytest.ini`: 1 files
- `requirements.txt`: 1 files
- `research`: 12 files
- `research_history`: 54 files
- `run_experiment.py`: 1 files
- `scripts`: 5 files
- `tests`: 2 files
- `ucpa_fl`: 16 files
- `verification`: 78 files

## Integrity

`PACKAGE_INTEGRITY.sha256` hashes every distributable file except itself. The outer ZIP checksum is supplied beside the archive.
