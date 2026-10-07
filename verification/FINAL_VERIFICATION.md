# CFBA-CRC final verification report

## Freeze status
This report records the final pre-handoff verification. No confirmatory full-study run (seeds 1701–1705) has been executed.

## Static/package checks
- editable local installation with `--no-build-isolation`: **PASS**
- Python compile check: **PASS**
- pytest: **21/21 PASS**
- suite configs: **22/22 parse successfully**
- tier counts: **A=6, B=8, C=8**
- every full-study config uses methods `[local, fedattr_mean, xfedalign_median, global_crc, cfba_crc]`: **PASS**
- every full-study config uses seeds 1701–1705: **PASS**
- actual bundled run manifests using seeds 1701–1705: **0**

## Regression issues found and fixed before freeze
1. The first vectorized deletion/insertion implementation integrated over unit-spaced perturbation steps, allowing AUC > 1. All such smoke runs were discarded; the implementation now integrates over a normalized `[0,1]` perturbation axis and has a bounded-AUC test.
2. A compact smoke task reached only ~41% accuracy. Those runs were discarded rather than interpreted.
3. An initial Monte-Carlo unit test incorrectly treated CRC as a high-probability per-calibration-draw guarantee. It was replaced with an expected-risk sanity check consistent with the actual CRC claim.
4. The inherited shared runner evaluated `local` per-sample explanations with the default beta=.2. This was caught during final end-to-end verification. Local-XAI is now forced to beta=0 exactly; a regression test verifies zero excess risk when local and oracle match.

## Data-separation check
Each client uses five disjoint splits: task training, surrogate fitting, artifact estimation, private conformal calibration, and final evaluation. Unit tests check pairwise disjointness and calibration/evaluation separation.

## Clean end-to-end sanity
- seed: 1699 (non-confirmatory)
- synthetic IID task, client task accuracy mean: **0.9958**
- per-run SHA-256 verification: **PASS**
- Local-XAI beta: **0.0**, excess risk approximately 0
- frozen full-study gate on sanity-only evidence: **PENDING**, as required

## Attacked-prior recalibration sanity
- seed: 1698 (non-confirmatory)
- client task accuracy mean: **1.0**
- per-run SHA-256 verification: **PASS**
- clean certified client betas: `[0.0, 0.0, 0.0, 0.1]`
- attacked certified client betas: `[0.0, 0.0, 0.0, 0.0]`
- attacked CFBA excess risk: approximately 0

This confirms that attack evaluation recalibrates against the attacked prior instead of reusing clean certificates.

## Smoke-evidence integrity
Preserved valid CFBA smoke evidence under `research_history/phase14_cfba_crc_smoke/`:
- five rotation seeds 1601–1605;
- five patch seeds 1601–1605;
- erasing stress seed 1601;
- risk–consistency frontier seed 1606.

Run-level integrity verification:
- primary smoke runs: **10/10 PASS**
- erasing stress: **1/1 PASS**
- frontier: **1/1 PASS**

## Fresh-tree verification
A separate copy of the frozen tree was created before package hashing and independently passed:
- compile / package verification;
- 21/21 tests;
- clean and attack run SHA-256 checks;
- all 22 frozen configs;
- zero actual confirmatory-seed run records.

## Scientific smoke decision
The final direction-readiness score is **79/100**, interpreted only as permission to spend the full validation budget, not as a conference acceptance probability. The principal remaining risk is originality because CRC, conformal property alignment, conformal explanation methods, and site-conditional federated CRC are prior art.
