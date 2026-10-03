# Phase IV-B pre-registration: Heterogeneity-Gated Explanation Alignment (HGEA)

Registered: 2026-10-02 (America/Toronto), **after** Phase-IV REPA failed and **before** any Phase-IV-B confirmation seed was run.

## Why this phase exists

Phase IV falsified the smooth random-effects REPA candidate. A simpler coordinate-wise variance gate was substantially stronger on development data. Phase IV-B therefore treats all seeds 101–205 and all Phase-IV/P4b diagnostics as **development evidence** and tests a fixed, simpler mechanism on untouched seeds.

## Candidate

Working name: **HGEA — Heterogeneity-Gated Explanation Alignment**.

For each explanation coordinate/group `j`, client `i` sends:

- local mean attribution `e_ij`, and
- sampling variance of that mean `v_ij` (estimated from the client's local explanation samples).

The server computes

`R_j = Var_i(e_ij) / (mean_i(v_ij) + eps)`.

If `R_j <= c`, disagreement is treated as sampling-noise dominated and the coordinate is aligned to the cross-client mean. If `R_j > c`, the coordinate is treated as genuinely heterogeneous and each client keeps its local value. The vector is then renormalized.

The fixed threshold is **c = 4**. It was selected because it was the best single threshold across the two development stress strengths (0.90/0.10 and 0.75/0.25 patch-label correlations). It will not be changed after confirmation begins.

This is deliberately simpler than REPA. The claim is not that variance ratios or hard thresholding are new statistics. The possible contribution is a federated-XAI coordination rule that uses *within-client attribution uncertainty* to decide which explanation dimensions should be globally aligned and which should remain personalized.

## Untouched confirmation data

Fresh seeds: **301, 302, 303, 304, 305, 306, 307**.

Two pre-registered stress strengths:

- strong: P(patch | positive)=0.90, P(patch | negative)=0.10
- moderate: 0.75 / 0.25

Three heterogeneity structures:

- shared: one common explanatory patch across all clients
- grouped: four client groups with different explanatory patches
- personal: every client has a distinct explanatory patch

Explanation sample sizes: 8, 16, 32, 64 positive examples/client.

## Frozen comparison methods

1. Local explanations (no alignment).
2. GlobalMedian: xFedAlign-style single robust prior, with Phase-IV discovery beta frozen per scenario.
3. GlobalMean: strongest single-mean soft-alignment baseline selected on development:
   - shared beta=1.0
   - grouped beta=0.1
   - personal beta=0.4
4. Cluster: strongest fixed-k explanation clustering selected on development:
   - shared k=2, beta=0.8
   - grouped k=4, beta=0.8
   - personal k=2, beta=0.6
5. QGate: Cochran-Q/FDR heterogeneity-testing gate, q=0.01, FDR correction, mean prior. This is a principled statistical alternative discovered after REPA failed.
6. HGEA: fixed ratio gate c=4, full alignment on accepted-homogeneous dimensions.

No baseline hyperparameter will be changed using seeds 301–307.

## Metrics

Primary: mean JSD to a large-sample oracle local explanation.

Secondary:
- L1 to oracle
- core JSD after removing controlled patch pixels
- own-patch attribution-mass error
- foreign-patch mass
- pairwise explanation JSD (reported, but not interpreted as universally better when lower)
- task-model accuracy
- gate sensitivity on truly heterogeneous patch dimensions and specificity on shared/core dimensions

## Approval criteria

HGEA is **SMOKE-TEST APPROVED** only if ALL central criteria below hold on the complete untouched confirmation grid.

### A. Heterogeneous fidelity
In both grouped and personal settings, pooled over both signal strengths and all sample sizes:
- HGEA must have lower oracle-JSD than every frozen baseline, and
- improve oracle-JSD by at least **15%** over the strongest non-HGEA baseline.

### B. Homogeneous behavior
In shared settings:
- HGEA must be within **3%** oracle-JSD of the best global baseline, and
- improve over Local by at least **25%**.

### C. Seed robustness
For grouped and personal settings separately, HGEA must beat the strongest non-HGEA baseline in oracle-JSD on at least **6 of 7 seeds** when averaged over both signal strengths/sample sizes.

### D. Mechanism
Under grouped/personal heterogeneity:
- the gate must preserve (not globally align) at least **75%** of true heterogeneous patch coordinates on average,
- while globally aligning at least **75%** of non-patch/core coordinates on average.
Under shared semantics, it must globally align at least **90%** of patch coordinates.

### E. Explanation-specific value
Relative to the best single-global baseline under grouped/personal heterogeneity:
- HGEA must reduce own-patch attribution-mass error by at least **25%**, and
- must also improve core JSD over Local by at least **25%**.
This prevents approval based solely on preserving a synthetic patch while failing to denoise shared explanatory structure.

### F. Novelty collision
A current-literature audit must find no direct prior that already performs coordinate/group-level federated explanation alignment by comparing between-client explanation dispersion to within-client explanation-estimation uncertainty. UncertainXFL is a relevant uncertainty-aware XFL neighbor and must be explicitly distinguished; xFedAlign is the direct global-prior baseline.

If A–E pass but F fails, mark `NOVELTY_COLLISION`, not approved.

## Scope of approval

Passing this phase means only: **one approach is approved for continued research / larger validation**. It is not a claim of top-conference readiness. The smoke test is a controlled linear-model MNIST stress test and must later be reproduced with xFedAlign's full surrogate alignment, nonlinear models, naturally heterogeneous data, and official baselines.
