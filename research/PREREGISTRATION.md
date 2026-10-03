# UCPA full-study preregistration (frozen before full runs)

**Date frozen:** 2026-10-02  
**Suite:** 1.0  
**Confirmatory seeds:** 6101, 6102, 6103, 6104, 6105. These seeds were not used in the Phase-5 smoke-test development/confirmation.

## Question

Can a personalized explanation prior that borrows only from **globally similar peers** and only on **coordinates whose disagreement is compatible with estimation uncertainty** preserve client-local explanation fidelity better than a single global prior, without giving up the consistency benefits of federated explanation coordination?

## Frozen UCPA

For client `i`, peer `k`, class `c`, feature `j`:

`g_ikc = exp(-JSD(E_ic, E_kc) / h)`

`q_ikcj = exp(-0.5 * ((E_icj-E_kcj) / (z*sqrt(V_icj+V_kcj)))^2)`

`w_ikcj = g_ikc * q_ikcj`, with sparse support intersection gating and self weight 1.

Frozen values: `z=2`, `h=0.048`, `beta=0.2`, variance-floor fraction `0.05`. No core-run result may be used to change these values.

## Methods in every core run

- Local-XAI: no shared prior.
- FedAttr-Agg reproduction: server weighted mean attribution template.
- xFedAlign-style reproduction: sparse/noised artifacts + coordinatewise median global prior + beta=0.2 local/global mixing.
- UCPA.
- UCPA whole-explanation-only ablation.
- UCPA coordinate-uncertainty-only ablation.
- Continuous-vector clustering comparator with cluster-wise prior.

All methods receive the **same trained FedAvg task model, same client partitions, same surrogate-fit data, same artifact-estimation data, and same final evaluation data**. Task accuracy is therefore not a method-dependent quantity in the explanation-coordination comparison.

The xFedAlign comparator is a transparent reproduction of the published mechanism, **not the authors' official code**. The package records the official ICML 2026 repository separately and includes an optional fetch script.

## Data separation

Within each client's training shard:

- 65% task-model training
- 15% surrogate fitting
- 10% explanation-artifact estimation
- 10% untouched final explanation evaluation

These sets are disjoint.

## Tier-A core experiments

1. MNIST, Dirichlet label skew alpha=0.1, no extra shift.
2. MNIST + client-specific rotations.
3. MNIST + client-specific sparse patch correlations.
4. CIFAR-10, Dirichlet label skew alpha=0.1, no extra shift.
5. CIFAR-10 + client-specific rotations.
6. CIFAR-10 + client-specific photometric shift.

The two no-extra-shift experiments are controls. The four shifted experiments are the primary heterogeneity tests.

## Primary outcome

`artifact_fidelity_jsd`: mean class-conditional JSD between each method's coordinated client explanation summary and a disjoint high-fidelity direct-integrated-gradients summary computed from the final client evaluation split. Lower is better.

This metric is primary because a global-consistency metric alone mechanically rewards a single global prior and can hide genuine client-specific semantics.

## Secondary outcomes

- `sample_fidelity_jsd` to direct task-model IG.
- Pairwise EDI (method-independent cross-client drift).
- Method-reference EDI (reported but not used as sole evidence across methods).
- Deletion AUC (lower better).
- Insertion AUC (higher better).
- Top-k overlap with direct task-model IG.
- Communication bytes per client artifact.
- Mean and worst client task accuracy.
- Artifact-membership AUC using mean-only vs mean+variance signals.

## Confirmatory gates

The full study earns a **CONTINUE** decision only if all of the following hold on Tier A:

1. **Local-fidelity gain:** UCPA has lower `artifact_fidelity_jsd` than xFedAlign-style median prior in at least 3 of the 4 shifted core experiment families, and wins at least 4/5 paired seeds in each of those successful families.
2. **Not just local isolation:** UCPA has lower pairwise EDI than Local-XAI in at least 3 of the 4 shifted core families.
3. **Functional fidelity:** pooled across shifted core families, UCPA is not materially worse than xFedAlign-style reproduction by more than +0.01 absolute deletion AUC or -0.01 absolute insertion AUC.
4. **Two-scale mechanism:** pooled over shifted core + unequal-size stress experiments, full UCPA's primary outcome is better than both the whole-only and coordinate-only ablations. If either ablation matches it, the two-scale novelty claim must be revised.
5. **Deep-model relevance:** both CIFAR-10 shifted experiments complete with mean client task accuracy >=0.35. If the task model itself fails, explanation comparisons on that condition are considered non-informative rather than negative evidence for UCPA.
6. **No hidden privacy claim:** if mean+variance artifact membership AUC exceeds mean-only AUC by >0.05 or exceeds 0.65 absolute AUC, the paper must explicitly report a privacy concern and cannot claim privacy parity without a mitigation experiment.

The study can still be scientifically useful if a gate fails, but the corresponding claim must be narrowed or killed.

## Statistics

- Five independent seeds for every config.
- Report mean, SD, and paired bootstrap 95% CI.
- UCPA-vs-xFedAlign comparisons are paired by seed and include Wilcoxon signed-rank p-values where defined.
- Primary claims are based on effect direction, confidence intervals, and per-seed consistency; secondary metric p-values are descriptive and are not used for post-hoc claim fishing.
- Tier-C sensitivity experiments are exploratory and cannot change the confirmatory decision on the frozen parameters.

## Stress tests (Tier B)

- Unequal client sample sizes (log-normal subsampling) on MNIST and CIFAR rotation.
- 50% task-client participation on MNIST rotation. Scope limitation: explanation artifacts are evaluated at a final snapshot from all clients; this is not a dynamic per-round explanation-participation experiment.
- Gradually increasing client-specific rotation during FL task training, with final-snapshot explanation evaluation.
- Attribution artifact shift attack and random-support attack.

## Sensitivity (Tier C)

- Top-k = 32, 64, 128, 256 on MNIST rotation.
- Gaussian artifact noise sigma = 0, .05, .1, .2.
- UCPA z = 1 or 4 at h=.048; h=.024 or .096 at z=2.

## Stopping / revision rules

Do not alter frozen parameters after seeing Tier-A outcomes. If the two-scale method fails but an ablation succeeds, revise the contribution to the simpler mechanism. If xFedAlign-style global alignment is consistently better under both local fidelity and functional fidelity, kill UCPA. If UCPA only succeeds on MNIST but not CIFAR-10, do not make a general deep-vision claim.
