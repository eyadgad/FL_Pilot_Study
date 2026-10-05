# Method: Uncertainty-Compatible Peer Alignment (UCPA)

## Motivation

A single global explanation prior can reduce cross-client explanation drift, but under genuine client heterogeneity it can also average away client-specific semantics. Purely local explanations preserve those semantics but can be noisy when a client has few attribution samples. UCPA treats explanation coordination as **heteroskedastic peer smoothing**, not global consensus.

For client `i`, class `c`, let `E_ic` be the transmitted nonnegative normalized attribution summary and `V_ic` the estimated coordinatewise variance of that summary's sampling mean.

### Whole-explanation relevance

`g_ikc = exp(-JSD(E_ic, E_kc)/h)`.

This suppresses borrowing across clients whose explanation distributions differ coherently across many features.

### Coordinate uncertainty compatibility

`q_ikcj = exp(-0.5 * ((E_icj-E_kcj)/(z*sqrt(V_icj+V_kcj+epsilon)))^2)`.

This suppresses borrowing on a feature when the observed discrepancy is large relative to the amount expected from estimation uncertainty.

### Personalized prior

`w_ikcj = g_ikc q_ikcj`.

The client keeps self weight 1. For sparse artifacts, peer borrowing on a coordinate is allowed only where both clients transmitted that coordinate. The personalized prior is

`P_icj = (E_icj + sum_{k != i} w_ikcj E_kcj) / (1 + sum_{k != i} w_ikcj)`

followed by simplex normalization. Final aligned explanation summaries use the same frozen soft alignment strength `beta=0.2` as the xFedAlign-style comparator.

## What is and is not claimed as novel

Not individually novel: Jensen-Shannon similarity, kernel smoothing, uncertainty weighting, personalized FL, clustering, sparse top-k communication, or attribution aggregation.

Candidate novelty: the **two-scale explanation-space peer rule** that combines whole-explanation semantic relevance with coordinate-level *sampling-uncertainty compatibility* to construct a client-specific federated explanation prior. The novelty audit remains medium risk until the full study and a final pre-submission search are complete.

## Mechanistic predictions

1. Under homogeneous explanation semantics, whole and coordinate compatibility should be high and UCPA should denoise toward pooled estimates.
2. Under coherent shifts such as rotation, whole-explanation relevance should contract the peer set rather than force one global prior.
3. Under sparse feature-specific heterogeneity, coordinate compatibility should block inappropriate borrowing only on the affected features.
4. With unequal client sample sizes, the uncertainty term should help low-sample clients borrow more while protecting well-estimated genuine differences.
5. If whole-only or coordinate-only matches full UCPA, the full two-scale mechanism is unnecessary and the contribution must be simplified.

## Relation to xFedAlign

xFedAlign (ICML 2026) distills local surrogates, sends sparse/noised top-k per-class attribution artifacts, robustly aggregates them into one Global Explanation Prior (coordinatewise median or trimmed mean), and softly aligns local explanations to that prior. UCPA retains the same privacy/communication artifact concept but replaces one global prior with a continuous client- and coordinate-specific peer prior.

This package's `xfedalign_median` baseline is a self-contained mechanism reproduction. It is not the authors' official implementation; see `BASELINE_PROVENANCE.md`.
