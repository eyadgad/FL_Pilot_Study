# Phase VII pre-registration: SACPA smoke test

Frozen before running seeds **701--705**.

## Candidate

**SACPA -- Scale-Adaptive Corroborated Peer Alignment** replaces a single global explanation prior with a personalized peer-only prior constructed from the same sparse attribution means/support masks used by the xFedAlign-style pipeline.

For client `i` and peer `k`, let `d_ik = JSD(E_i,E_k)`. Define a local scale

`σ_i = median_{r != i} d_ir`

and a symmetric self-tuning peer weight

`g_ik = exp(-(d_ik / sqrt(σ_i σ_k))^2)`.

For coordinate `j`, the peer prior is the `g_ik`-weighted mean over peers that actually reported `j`. The target client is excluded from construction of its peer prior. If the target did not report coordinate `j`, that coordinate may enter the prior only when **at least two peers report it**. If no peer supports a coordinate, SACPA falls back to the target's own value at that coordinate.

The final aligned explanation is

`A_i = normalize(0.4 E_i + 0.6 P_i)`.

Frozen parameters/rules:
- squared self-tuning kernel power `p=2`;
- `β=0.6` peer-prior shrinkage;
- median local JSD scale;
- missing-coordinate corroboration threshold = 2 peers;
- top-k = 128;
- artifact noise in the smoke test = Gaussian `dp_sigma=0.10`, matching the full-pipeline scale used in Phase VI;
- no variance channel is used.

These choices were selected only from development seeds 601 and 602 and the previously returned Phase-VI artifacts. No result from seeds 701--705 may alter the method before the gate is evaluated.

## Why this revision exists

UCPA v1 failed the full confirmatory pipeline because its absolute JSD bandwidth collapsed peer weights when sparse/noised artifact JSD changed scale across datasets. SACPA makes the whole-explanation kernel scale-adaptive and removes the strict support-intersection rule. A coordinate absent from the target is treated as unknown, but is imported only with multi-peer corroboration.

## Data and heterogeneity

Binary MNIST (3 vs 8), same controlled smoke-test framework used in Phase V, but explanations are sanitized to **top-128 sparse artifacts** before coordination.

Families:
1. patch shared;
2. patch grouped, strong and moderate strengths;
3. patch personal, strong and moderate strengths;
4. rotation shared;
5. rotation grouped;
6. rotation personal.

Explanation sample counts: `{8,16,32,64}`.

## Fresh confirmation seeds

`701, 702, 703, 704, 705`.

## Baselines

All methods receive exactly the same sanitized explanation artifacts.

- Local / self-prior.
- Oracle-tuned GlobalMedian over beta `{0.1,0.2,0.4,0.6,0.8,1.0}`.
- Oracle-tuned Cluster over `k={2,3,4,6}` and the same beta grid.
- SACPA-no-corroboration: identical to SACPA but missing-coordinate support threshold = 1. This is the direct support-rule ablation.

GlobalMedian and Cluster are intentionally allowed oracle tuning on confirmation data; SACPA remains frozen.

## Primary metric

Mean JSD to an independent high-sample oracle explanation (`oracle_jsd`), lower is better.

## Approval gate

SACPA is **SMOKE-TEST APPROVED FOR CONTINUATION** only if all criteria pass.

### A. Task sanity
Every fresh seed/scenario has task test accuracy >= 0.90.

### B. Homogeneous denoising
For both patch-shared and rotation-shared:
- SACPA improves `oracle_jsd` over Local by >= 50%; and
- SACPA is no more than 10% worse than oracle-tuned GlobalMedian.

### C. Heterogeneous aggregate superiority
Across the four heterogeneous families with equal family weighting, SACPA must improve mean `oracle_jsd` by:
- >= 20% versus Local;
- >= 8% versus oracle-tuned GlobalMedian; and
- >= 8% versus oracle-tuned Cluster.

### D. No hidden family collapse
For each heterogeneous family separately, SACPA must be no more than 2% worse than oracle-tuned GlobalMedian, and it must strictly beat GlobalMedian in at least 3 of the 4 families.

For patch grouped/personal, the 2% non-inferiority condition must also hold separately at strong and moderate patch strengths.

### E. Fresh-seed robustness
On the equal-family heterogeneous average:
- SACPA beats Local in 5/5 seeds;
- SACPA beats oracle-tuned GlobalMedian in at least 4/5 seeds;
- SACPA beats oracle-tuned Cluster in at least 4/5 seeds.

### F. Corroboration mechanism
On personal-patch heterogeneity, SACPA must improve over the no-corroboration ablation by >= 20% in mean `oracle_jsd`.

Across the other heterogeneous families combined, corroboration may worsen `oracle_jsd` by at most 7% relative to the no-corroboration ablation.

### G. Peer mechanism is active
Across heterogeneous confirmation cases, the mean peer mass (sum of peer weights before beta mixing) must be >= 1.5. This prevents approval of a method that is effectively Local-XAI.

## Interpretation

Passing this gate justifies integration into the full MNIST/CIFAR-10 pipeline and a broader theory/novelty evaluation. It does not imply publication readiness.

## Failure rule

Any failed gate means SACPA is not approved. No retuning on seeds 701--705 is permitted.
