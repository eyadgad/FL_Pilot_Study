# Final engineering verification — RWSSA research pipeline v1.0

The package was assembled from the verified RWSSA smoke scripts plus the previously tested FL engine; it was **not** taken from an official xFedAlign or iFLASH implementation.

## Checks completed before archive freeze

- Unit tests: **26 passed** (including gain weighting, sparse encoding, no silent Dirichlet-to-IID fallback, five-way split and inherited baseline/risk tests).
- Python compilation: pass.
- Frozen suite YAML: **23 configs** parse (3 pilot, 8 core confirmatory, 7 stress, 4 exploratory, 1 sanity).
- Pilot seeds **2401–2403**, main seeds **2501–2510**; neither used in the distributed engineering output files.
- Independent synthetic clean end-to-end runner: pass, including RWSSA, sparse variant, quality-aware baseline and ablations; JSONL outputs/metrics generated; run hashes verify.
- Independent synthetic **artifact-attack + four-component calibration** end-to-end runner: pass; clean and attacked scenarios both produced; run hashes verify.
- Aggregation regenerates `runs.csv`, `summary.csv`, paired comparison CSV and Markdown.
- Pilot gate with sanity-only data: correctly returns **PENDING** (not a false approval).
- Config hashes, pip-freeze, code version, per-run output and checkpoint integrity supported.

## What was *not* verified or proven

- Real MNIST/FashionMNIST/CIFAR-10 dataset download/training was **not executed here**. No novel real-data results or conference-level evidence are claimed.
- **No formal DP/privacy guarantee** for the transmitted client gain vectors. The full dense channel is unusually expensive for CIFAR and is included in the reported communication budget.
- The smoke's two-risk selection criterion is reproduced, not upgraded without evidence to four-metric CRC. Deletion/insertion risk remains an empirical held-out endpoint. Four-metric calibration is a separate stress setting.
- xFedAlign-style and iFLASH-inspired baselines are clearly tagged reproductions/proxies, not author-equivalent experiments.
- Cross-GPU strict bitwise determinism is not guaranteed. Smoke runs and tests only verify local execution and syntax/integrity.
- A pass/fail gate is a resource-allocation aid, not publication/acceptance prediction.

## Handoff invariant

Run Tier P and inspect the gate **before** proceeding to Tier A/B/C. Preserve all raw run directories/checkpoints and negative results. Archived scripts must not be changed after the first confirmatory run.
