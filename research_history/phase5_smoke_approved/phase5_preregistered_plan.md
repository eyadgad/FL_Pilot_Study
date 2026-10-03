# Phase V final pre-registration: UCPA cross-heterogeneity smoke test

Date frozen: 2026-10-02, **before final seeds 501--505 are run**.

## Candidate

**UCPA -- Uncertainty-Compatible Peer Alignment** coordinates federated explanation summaries without forcing either one global explanation prior or a discrete client clustering.

For client attribution mean vectors `E_i` and estimated variances of those means `V_i`, UCPA uses two peer-compatibility factors:

1. **whole-explanation compatibility**

   `g_ik = exp(-JSD(E_i,E_k) / h)`

2. **coordinate uncertainty compatibility**

   `c_ikj = exp(-0.5 * ( |E_ij-E_kj| / (z*sqrt(V_ij+V_kj+floor)) )^2 )`

The peer weight is `w_ikj = g_ik*c_ikj`.  Each client/coordinate is replaced by the normalized weighted peer mean. Self weight is fixed to one. The variance floor is `0.05 * median(V[V>0])`.

The parameters were selected **only from development data** spanning both sparse-patch and coherent-rotation heterogeneity and are now frozen:

- `z = 2.0`
- `h = 0.048`
- `lambda = 0.0` (no uniform floor on whole-explanation weights)
- variance-floor fraction = `0.05`

A post-development homogeneous-coordinate fallback was tested and is **not part of the frozen candidate**: although it improved one development average, it degraded rotation heterogeneity and introduced another threshold. No result from seeds 501--505 may be used to change UCPA before this gate is evaluated.

## Fresh confirmation data

Untouched seeds: **501, 502, 503, 504, 505**.

Two independent heterogeneity generators are required:

1. **Patch heterogeneity** on binary MNIST (3 vs 8), with both previously defined strengths:
   - strong: `(P(marker|positive), P(marker|negative)) = (0.90, 0.10)`
   - moderate: `(0.75, 0.25)`
   - scenarios: `shared`, `grouped`, `personal`
2. **Rotation heterogeneity** on the same binary MNIST task:
   - shared: all 0 degrees
   - grouped: four coherent angle groups `[-20,-7,+7,+20]`
   - personal: 12 angles spanning -24 to +24 degrees

Explanation sample counts remain `[8,16,32,64]` exactly as in development.

## Baselines

All baselines use exactly the same sampled explanation summaries as UCPA.

- `Local`: no cross-client alignment.
- `GlobalMean`: single global mean prior, beta grid `{0.1,0.2,0.4,0.6,0.8,1.0}`.
- `GlobalMedian`: single global median prior, same beta grid.
- `Cluster`: K-means explanation clustering with `k in {2,3,4,6}` and beta grid above.
- `QGate`: fixed Cochran/BH coordinate gate (`q=.01`, FDR, mean prior).
- `HGEA`: fixed variance-ratio coordinate gate (`c=4`).
- `CoordPeer`: coordinate-uncertainty peer smoothing without the whole-explanation kernel (`z=1.5`), an ablation of UCPA.

For `GlobalMean`, `GlobalMedian`, and `Cluster`, the final report may choose the best grid value **after seeing the confirmation oracle metric** for each domain/scenario. This intentionally advantages those baseline families and is an upper-bound comparison; UCPA remains fixed and receives no such oracle tuning.

## Primary metric

Mean Jensen-Shannon divergence (`oracle_jsd`) between the aligned client explanation and an independent high-sample oracle explanation. Lower is better.

The oracle is used only for evaluation, never by UCPA.

## Approval gate

UCPA is labelled **SMOKE-TEST APPROVED FOR CONTINUATION** only if *all* conditions below pass.

### A. Task sanity

Across every fresh seed/scenario/domain, downstream task test accuracy must be at least **0.90**. This prevents an explanation result caused by a collapsed classifier.

### B. Homogeneous denoising

Separately for patch-shared and rotation-shared:

- UCPA mean `oracle_jsd <= 0.003`, and
- UCPA improves over `Local` by at least **60%**.

UCPA is not required to beat full global averaging when all explanations are genuinely shared.

### C. Heterogeneous fidelity against local and global priors

For each of the four heterogeneous families:

- patch-grouped (pooled over both strengths),
- patch-personal (pooled over both strengths),
- rotation-grouped,
- rotation-personal,

UCPA must improve mean `oracle_jsd` by at least **10%** versus `Local` **and** versus the best oracle-tuned single-global baseline (`GlobalMean` or `GlobalMedian`).

For the patch domain this must also hold at **>=5% versus the best global baseline separately at each patch strength**, so pooling cannot hide a failure.

### D. Fresh-seed robustness

For each of the four heterogeneous families:

- UCPA must beat `Local` on **5/5** seed-level averages; and
- UCPA must beat the selected best global baseline on at least **4/5** seed-level averages.

### E. Unified adaptive-baseline test

Pool all four heterogeneous families, weighting each family equally. UCPA's mean `oracle_jsd` must be lower than each of:

- fixed `HGEA`,
- fixed `QGate`,
- fixed `CoordPeer`, and
- the **oracle-tuned Cluster family** (cluster parameters may be selected separately by domain/scenario, advantaging Cluster).

This criterion tests the proposed value of one continuous mechanism across sparse feature heterogeneity and coherent client-level heterogeneity, rather than winning only one favorable stress test.

### F. Mechanism sanity

For both patch and rotation domains, averaged over fresh seeds/sample counts:

- the mean whole-explanation peer weight in `shared` must exceed the larger of `grouped` and `personal` by at least **0.10**; and
- the mean effective peer count in `shared` must exceed the larger of `grouped` and `personal` by at least **0.75**.

This checks that UCPA actually borrows more broadly when explanations are shared and narrows its peer neighborhood under heterogeneity.

## Interpretation of a pass

A pass means only: **the approach has survived a pre-registered CPU-scale smoke test strongly enough to justify next-stage work** (official xFedAlign comparison, nonlinear CNNs, additional datasets, theory, and a broader novelty audit).

It does **not** mean the approach is publication-ready or likely to be accepted at a top conference.

## Failure rule

If any criterion fails, the candidate is not approved. Failures must be reported without retuning UCPA on seeds 501--505.
