# Method: Neighborhood-Corroborated Scale-Adaptive Peer Alignment (NC-SACPA)

## Problem

A single federated explanation prior can denoise client explanations but can also erase real client-specific semantics. Purely local explanations preserve heterogeneity but do not use evidence from related clients. Earlier UCPA attempted continuous peer borrowing with an absolute JSD bandwidth and coordinate uncertainty; full validation showed that the fixed bandwidth collapsed peer mass to approximately one client at realistic sparse-artifact scales. Phase VII SACPA fixed scale collapse but allowed a distant group to authorize a feature merely because enough distant clients reported it.

NC-SACPA addresses both failures without transmitting an uncertainty/variance channel.

## Inputs

For client `i` and class `c`:

- `E_ic`: nonnegative, simplex-normalized sparse attribution summary.
- `M_ic`: transmitted top-k support mask.

All clients use the same artifact construction and task model in a run.

## 1. Scale-adaptive whole-explanation similarity

Let

`d_ikc = JSD(E_ic, E_kc)`.

For each client/class, estimate a local distance scale

`σ_ic = median_{r != i} d_irc`.

The symmetric peer weight is

`g_ikc = exp(-(d_ikc / sqrt(σ_ic σ_kc))^p)`

with frozen `p=2`.

This removes the absolute-distance scale that caused UCPA v1 to collapse when explanation JSDs changed across datasets, sparsity levels, or explainers.

## 2. Neighborhood corroboration for missing sparse coordinates

For each target client/class, rank peers by `g_ikc` and take the frozen top `q=3` peers.

- If feature `j` is already reported by the target (`M_icj=1`), all peers that report `j` may contribute, weighted by `g_ikc`.
- If the target did not report `j`, the feature may enter the peer prior only if at least `m=2` of the target's `q=3` nearest peers independently report it.
- For an imported missing feature, only those local-neighborhood peers contribute.
- If no eligible peer reports a coordinate, the target value is retained rather than treating missing support as a confident zero.

This rule was introduced because Phase VII global-count corroboration failed on strong grouped sparse patches: a numerically larger but semantically distant group could authorize a foreign feature.

## 3. Personalized peer prior

For eligible coordinates,

`P_icj = sum_{k != i} g_ikc M_kcj E_kcj / sum_{k != i} g_ikc M_kcj`,

with the neighborhood restriction above for coordinates missing from the target. The prior is normalized to the simplex.

The target client is excluded from the peer prior; this avoids counting the local explanation twice.

## 4. Final aligned explanation

`A_ic = normalize((1-β) E_ic + β P_ic)`

with frozen `β=0.8`.

Frozen smoke-approved values:

- kernel power `p=2`
- alignment `β=0.8`
- local corroboration neighborhood `q=3`
- minimum local support `m=2`

The full confirmatory suite must not change these values using Tier-A results.

## Communication

NC-SACPA uses only the sparse attribution mean/support artifact. It does **not** require UCPA v1's variance channel. In this package its communication accounting is therefore matched to the xFedAlign-style mean artifact at the same top-k setting.

## What is not claimed as novel

Jensen-Shannon divergence, self-tuning kernels, nearest-neighbor graphs, sparse top-k explanations, peer averaging, personalized FL, and support voting are established ideas.

The candidate contribution is narrower: a federated explanation-coordination rule that combines **self-tuned explanation-space collaboration** with **local-neighborhood corroboration of sparse explanation support** to create a client-specific prior that can move continuously between pooled and local behavior without a fixed global prior or fixed number of client clusters.

## Frozen negative controls

The package keeps:

- `ucpa`: fixed-bandwidth uncertainty-compatible UCPA v1 — killed after full confirmation.
- `sacpa_global_count`: scale-adaptive Phase-VII predecessor — rejected because global support counts failed strong grouped sparse heterogeneity.
- `ucpa_whole_only`, `ucpa_coord_only`: UCPA mechanism ablations.

These are historical/negative controls, not co-primary proposed methods.
