# Final package verification

**Performed:** 2026-10-02/03 UTC-equivalent execution environment, after the final UCPA regression correction and before archive creation.

## Passed checks

1. Python compile of all `ucpa_fl` modules and entry points.
2. **7/7 pytest tests**, including a numerical regression test against an independent transcription of the smoke-approved UCPA operator.
3. All frozen YAML configs parsed successfully: sanity + suite manifest + 26 full-study configs.
4. Offline editable-install metadata verified with `pip install -e . --no-deps --no-build-isolation`. Standard build isolation could not be tested in the sandbox because outbound package-index access is disabled; this is an environment-network limitation, not a package-code failure.
5. Clean end-to-end synthetic run using the **linear-surrogate** explanation path completed, aggregated, and passed `integrity.sha256` verification.
6. Independent end-to-end synthetic run using **direct task-model integrated gradients plus random-support attribution poisoning** completed, produced both clean/attacked results, aggregated, and passed run-integrity verification.
7. Aggregation produced paired UCPA-vs-xFedAlign-style tables. The preregistered gate script correctly returned `PENDING` on sanity-only results rather than treating missing core experiments as a pass.
8. Final config parser check reported all configs valid after the verification runs.

## Important scope

The full MNIST/CIFAR-10 Tier-A/B/C study was **not** run during packaging. That is intentional: seeds 6101–6105 remain untouched confirmatory evidence for the user's run. The verification runs use synthetic data/seeds 101 and 909 and are engineering checks only.

The package cannot verify external dataset download or official xFedAlign GitHub execution inside this sandbox because outbound git/package access is blocked. The self-contained experiment does not depend on that checkout; official-code comparison remains an explicit pre-publication step.
