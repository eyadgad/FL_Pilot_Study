# Final package verification — NC-SACPA full pipeline

**Freeze date:** 2026-10-04  
**Suite version:** 2.1

## Research-state verification

- Phase-VIII NC-SACPA smoke status: `SMOKE_TEST_APPROVED_FOR_CONTINUATION`.
- Frozen smoke confirmation seeds: 801–805; raw JSONL, preregistration hash, evaluator, and gate report are preserved in `research_history/phase8_ncsacpa_smoke_approved/`.
- UCPA v1 full negative result and Phase-VII SACPA rejection are preserved in `research_history/`.
- Full-study seeds are 9101–9105.
- Search of all packaged result-bearing `manifest.json`, `events.jsonl`, `metrics.jsonl`, and `final_results.json` files found **zero** occurrences of seeds 9101–9105. The full confirmation block is untouched.

## Static/package checks

`python scripts/verify_package.py` passed:

- Python compilation: PASS
- unit/regression tests: **10/10 PASS**
- every YAML experiment config parses through the strict dataclass loader
- suite manifest parses successfully

The NC-SACPA tests include:

- self-tuned-kernel behavior under distance rescaling;
- neighborhood corroboration blocking unsupported foreign sparse support;
- a regression distinguishing NC-SACPA from the rejected global-count predecessor.

## End-to-end engineering check

A fresh synthetic federated sanity run was executed after the final NC-SACPA code integration. The run completed and produced task metrics, explanation artifacts, all coordination methods, logs, final results, checkpoint, and a per-run SHA-256 manifest.

`python scripts/verify_runs.py --outputs verification/ncsacpa_sanity/output` verified **2/2** packaged synthetic sanity runs with no integrity failures (the earlier and final rerun).

The final sanity aggregation regenerated successfully. Running `evaluate_ncsacpa_gates.py` on sanity-only evidence returns **PENDING**, confirming that the evaluator does not falsely pass when the required full-study families/seeds are absent.

## Frozen suite audit

- total configs in `configs/suite_manifest.yaml`: **28**
- Tier A: **6** core confirmatory configs
- Tier B: **8** stress/robustness configs
- Tier C: **14** exploratory sensitivity/communication configs
- every config uses seeds 9101–9105
- all Tier-A/Tier-B configs retain frozen NC-SACPA values: `p=2`, `beta=.8`, `q=3`, `min_support=2`
- Tier-C NC-SACPA parameter variations are explicitly labeled exploratory and are not allowed to redefine Tier-A claims

## Environment limitations of this verification

The sandbox can execute the synthetic study but cannot verify external torchvision downloads or run the official xFedAlign GitHub repository in its intended external environment. Those are intentionally runtime/external checks for the user's machine. The package is self-contained for its internal baselines; official xFedAlign is an optional provenance cross-check, not a hidden dependency.

## Freeze rule

No code/config/document changes should be made after `PACKAGE_INTEGRITY.sha256` is generated without regenerating that manifest and changing the package checksum.
