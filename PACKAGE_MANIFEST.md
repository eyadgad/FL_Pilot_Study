# CFBA-CRC full research package manifest — v4

## Status
**Approved for full validation after adversarial smoke testing.** The direction score is 79/100 on the project's pre-full-study readiness rubric; this is not an acceptance probability.

## Frozen confirmatory study
- method: CFBA-CRC (Conformal Fidelity-Budgeted Alignment)
- fresh seeds: 1701–1705
- 22 configs total: 6 Tier A, 8 Tier B, 8 Tier C
- alpha: .05 for confirmatory/stress study
- beta grid: `[0,.1,.2,.4,.6,.8]`
- five disjoint client partitions
- methods in every full config: Local-XAI, FedAttr mean, xFedAlign-style median, global CRC, client-specific CFBA
- seven frozen gates: `evaluate_cfba_gates.py`

## Pre-full-study smoke evidence
- valid five-seed rotation and patch smoke: seeds 1601–1605
- erasing stress: seed 1601
- risk-consistency frontier: seed 1606
- raw evidence and integrity files: `research_history/phase14_cfba_crc_smoke/`
- readiness score and interpretation: `research/READINESS_SCORE.md`

## Verification included
- unit/regression tests for CRC selector, AUC normalization, data-split disjointness, and baseline identity
- clean end-to-end synthetic CFBA sanity seed 1699
- attacked-prior recalibration sanity seed 1698
- per-run SHA-256 verification
- frozen gate returns `PENDING` when real confirmatory evidence is absent

## Critical implementation safeguards
1. Local-XAI is evaluated with beta=0 exactly.
2. Calibration and final evaluation examples are disjoint.
3. Attacked priors are recalibrated; clean certificates are not reused.
4. CRC controls the monotonized bounded four-component loss; documentation does not reinterpret it as a per-sample high-probability guarantee.
5. Tier C cannot redefine Tier A/B success.

## Known external-dependency limitation
The package is self-contained for its xFedAlign-style reproduction, but a paper-stage comparison should additionally run the authors' official xFedAlign implementation when accessible.
