# Phase V approved-candidate manifest

**Decision:** `SMOKE_TEST_APPROVED_FOR_CONTINUATION`

Core files:

- `phase5_approved_smoke_test.md` — full result and limitations.
- `phase5_preregistered_plan.md` — gate frozen before seeds 501–505.
- `phase5_novelty_audit.md` — closest-work audit and novelty risk.
- `ucpa.py` — standalone UCPA alignment operator.
- `test_ucpa.py` — basic operator tests.
- `phase5_confirm_chunk.py` — final patch/rotation confirmation runner.
- `phase5_evaluate.py` — exact pre-registered gate evaluator.
- `reproduce_phase5.sh` — deterministic Phase-V reproduction entry point.
- `results/P5_confirmation_summary.json` — machine-readable final gate.
- `results/P5_integrity_sha256.txt` — hashes of frozen plan and final results.
- `results/P5_confirm_patch_501.jsonl` … `505.jsonl` — fresh patch raw results.
- `results/P5_confirm_rot_501.jsonl` … `505.jsonl` — fresh rotation raw results.

The earlier project history is intentionally preserved, including killed A′/B,
Phase-III identifiability work, and failed Phase-IV/V precursor methods.
