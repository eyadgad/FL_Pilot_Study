# Final Verification — Multi-Proposal FL Explanation Smoke Study

## Final decision

**RWSSA remains approved for a medium-cost real-data pilot.**

**Verified paper-direction readiness: 78/100.**

This is a research go/no-go score, not a conference acceptance probability. The score is reduced from the original 80/100 because the expanded fresh-seed audit found one reproducible outlier seed and attack robustness is favorable on average but not uniform seed-by-seed.

**Do not jump directly to the days-long conference-scale suite.** The next study should be a small real-data pilot on MNIST/Fashion-MNIST plus CIFAR-10 using the actual sparse-artifact pipeline and stronger quality-aware baselines.

## Integrity verification

- Original archive SHA-256: `540231d2d53a8a5aad0415981a98e43fd0e53421dcdf1ccfda4f99c23429e536`
- Original internal evidence manifest: **61/61 files verified with zero mismatches** before source-path repair.
- Original confirmation seed 1861 / rotation rerun: **bit-exact reproduction**.
- Original attack seed 1871 rerun: **bit-exact reproduction**.
- Original no-shift seed 1881 rerun via the generic confirmation runner: **bit-exact reproduction**.
- Standalone repaired package seed 1861 / rotation rerun: **bit-exact reproduction**.

The original evidence archive was numerically trustworthy. The original scripts were not standalone because they imported `/mnt/data/cfba_src/CFBA_full_pipeline`. This verified package includes the required `ucpa_fl` source under `cfba_core/` and patches the smoke scripts to use relative paths.

## Expanded untouched confirmation

The original confirmation used seeds 1861–1863. Final verification added untouched seeds **1864 and 1865** without changing `rho=0.30` or any method rule.

Total: **5 independent training seeds × 3 heterogeneity families = 15 runs**.

Minimum client task accuracy over the 15 runs: **0.9863**.

### Aggregate across all 15 runs

| Method | Oracle JSD ↓ | EDI ↓ | Deletion ↓ | Insertion ↑ | Top-k ↑ | Excess risk ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Local | 0.138024 | 0.137612 | 0.412926 | 0.800931 | 0.430348 | 0.000000 |
| xFedAlign-style | 0.131092 | 0.094488 | 0.322831 | 0.841359 | 0.471676 | 0.006486 |
| **RWSSA** | **0.122434** | **0.076460** | **0.301352** | **0.853859** | **0.522112** | **0.002167** |

### RWSSA wins versus xFedAlign across 15 runs

- Oracle JSD: **15/15**
- EDI: **12/15**
- Deletion AUC: **15/15**
- Insertion AUC: **11/15**
- Top-k overlap: **15/15**
- Excess risk: **14/15**

Seed **1865** is a genuine failure mode: RWSSA has worse EDI and insertion AUC than xFedAlign in rotation, patch, and erasing for that seed, although it still improves JSD, deletion AUC, and top-k overlap. This must remain visible in any future paper.

## Expanded attribution-poisoning verification

The original attack smoke used seeds 1871–1872. Final verification added untouched seeds **1873–1874**, giving four independent seeds.

### Artifact-shift attack — four-seed mean

| Method | JSD ↓ | EDI ↓ | Deletion ↓ | Insertion ↑ | Top-k ↑ | Risk ↓ |
|---|---:|---:|---:|---:|---:|---:|
| xFedAlign | 0.142846 | 0.098490 | 0.329383 | 0.818803 | 0.419915 | 0.013699 |
| **RWSSA** | **0.134650** | **0.082081** | **0.317757** | **0.834751** | **0.466912** | **0.005182** |

Seed wins for RWSSA: JSD 3/4, EDI 3/4, deletion 3/4, insertion 3/4, top-k 4/4, risk 4/4.

### Random-support attack — four-seed mean

| Method | JSD ↓ | EDI ↓ | Deletion ↓ | Insertion ↑ | Top-k ↑ | Risk ↓ |
|---|---:|---:|---:|---:|---:|---:|
| xFedAlign | 0.142793 | 0.098975 | 0.335211 | 0.820108 | 0.429020 | 0.014740 |
| **RWSSA** | **0.135692** | **0.083393** | **0.316747** | **0.832098** | **0.473015** | **0.005450** |

Seed wins for RWSSA: JSD 3/4, EDI 3/4, deletion 3/4, insertion 3/4, top-k 4/4, risk 4/4.

Therefore the correct robustness claim is **favorable average robustness with one mixed seed**, not uniform robustness.

## No-shift control

The original two fresh no-explicit-shift runs are reproducible from the generic confirmation runner (`--shifts none`). They favor RWSSA over xFedAlign on JSD, EDI, deletion, insertion, and top-k. This remains useful evidence that the mechanism is not restricted to one synthetic heterogeneity pattern.

## Methodological verification

The smoke code uses three disjoint evaluation-record partitions per client:

1. feature-benefit / policy-selection split,
2. calibration split for client scale selection,
3. untouched test split.

Task training, surrogate fitting, and artifact estimation are already separate in the inherited FL pipeline. For the real-data pilot, preserve these separations explicitly and log sample IDs / partition hashes.

Important limitation: the smoke calibration function uses a **cheap excess JSD + top-k loss** to select the client scale. It does not provide a formal CRC guarantee for deletion/insertion. The real-data pilot must either:

- implement the full bounded multi-metric risk used in the final claim, or
- avoid claiming a conformal/CRC guarantee.

## Novelty status

**Medium risk.** The exact RWSSA mechanism was not found in the targeted search, but several ingredients are established:

- xFedAlign coordinates federated explanations using a Global Explanation Prior.
- iFLASH performs faithfulness-aware federated explanation aggregation.
- recent metric-guided attribution-fusion work explicitly optimizes multiple explanation-quality objectives.

The only defensible novelty claim is narrow:

> Client-private task-model-fidelity estimates are aggregated **feature-wise** to identify a shared explanation subspace, and each attribution coordinate is aligned toward the federated prior in proportion to evidence that doing so improves fidelity while maintaining cross-client consistency.

Do not claim novelty for quality-aware aggregation, explanation weighting, shared priors, or feature attribution themselves.

## Verified readiness score

| Dimension | Weight | Verified score |
|---|---:|---:|
| Problem significance | 15 | 13 |
| Originality / literature separation | 20 | 12 |
| Technical mechanism | 20 | 17 |
| Expanded smoke evidence | 20 | 17 |
| Current baseline strength | 10 | 8 |
| Theory path | 10 | 7 |
| Reproducibility / practicality | 5 | 4 |
| **Total** | **100** | **78** |

**78/100 > requested 75/100 threshold.**

## Required next gate before any days-long study

Run a medium-cost real-data pilot only. It must include at least:

1. MNIST or Fashion-MNIST with actual sparse artifacts;
2. CIFAR-10 with a CNN/ResNet-level task model;
3. Local-XAI;
4. xFedAlign-style prior alignment;
5. FedAttr/global mean;
6. an iFLASH-style faithfulness/quality-aware aggregation baseline;
7. a metric-guided / quality-aware fusion baseline if implementable;
8. RWSSA binary-mask and no-residual ablations;
9. random feature weights and residual-only controls;
10. communication and privacy cost of feature-benefit maps.

Kill RWSSA before a full suite if it fails to beat the strongest quality-aware baseline on **at least four of five** primary metrics (JSD, EDI, deletion, insertion, top-k), or if feature-benefit leakage/communication is disproportionate.

## Final status

**RWSSA: VERIFIED FOR MEDIUM-COST REAL-DATA PILOT.**

**RWSSA: NOT YET VERIFIED FOR A CONFERENCE-SCALE MULTI-DAY RUN.**
