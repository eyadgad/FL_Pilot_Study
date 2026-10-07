# Verified RWSSA Smoke Package

This is a repaired, standalone version of the multi-proposal smoke study.

## What changed from the original evidence ZIP

- The original 61 evidence files were hash-verified before modification.
- The required `ucpa_fl` source is bundled under `cfba_core/`.
- Smoke scripts now resolve dependencies relative to this directory rather than hard-coded `/mnt/data/...` paths.
- Two additional untouched confirmation seeds (1864, 1865) are included.
- Two additional attack seeds (1873, 1874) are included.
- `FINAL_VERIFICATION.md` and `FINAL_VERIFICATION.json` record the final audit.

The original `SMOKE_STUDY_REPORT.md` is preserved unchanged as the historical selection report.

## Verify evidence

```bash
python verify_evidence.py
```

## Reproduce a stored confirmation exactly

```bash
python risk_weighted_confirm.py \
  --seeds 1861 \
  --shifts rotation \
  --out ./rerun_check
```

The resulting `rerun_check/rotation_1861.json` should match the stored `risk_weighted_confirm/rotation_1861.json` exactly in a deterministic environment.

## Reproduce the no-shift control

```bash
python risk_weighted_confirm.py \
  --seeds 1881,1882 \
  --shifts none \
  --out ./no_shift_rerun
```

## Reproduce attack runs

```bash
python risk_weighted_attack.py \
  --seeds 1871,1872,1873,1874 \
  --out ./attack_rerun
```

See `FINAL_VERIFICATION.md` for the final scientific interpretation and next gate.
