# CFBA-CRC: Conformal Fidelity-Budgeted Alignment

## Research question
Federated explanation alignment can reduce cross-client explanation drift, but stronger alignment can also damage the explanation's fidelity to the deployed task model. A single global alignment strength is especially problematic under non-IID clients: one difficult client can make a globally safe policy overly conservative for everybody else.

CFBA-CRC asks: **can each client align as strongly as its own private evidence permits while controlling a bounded multi-metric excess-fidelity risk?**

## Shared prior
The server constructs the same coordinate-wise median global explanation prior used by the package's xFedAlign-style reproduction. No new raw data or calibration examples are transmitted.

For client `i`, local explanation `E_i`, shared prior `G`, and alignment strength `beta`:

`A_i(beta) = Normalize((1-beta) E_i + beta G)`.

The frozen grid is `[0, 0.1, 0.2, 0.4, 0.6, 0.8]`.

## Private calibration target
Each client has a calibration split disjoint from:

1. task-model training;
2. surrogate fitting;
3. transmitted artifact estimation;
4. final evaluation.

On a calibration example, the direct task-model Integrated Gradients (IG) explanation is used as a private higher-fidelity reference. For every candidate beta, define four *excess harms relative to Local-XAI*:

- deletion harm: `max(0, deletion_beta - deletion_local)`;
- insertion harm: `max(0, insertion_local - insertion_beta)`;
- support harm: `max(0, topk_local - topk_beta)`;
- distributional harm: `max(0, JSD_beta_to_IG - JSD_local_to_IG)`.

The per-example loss is their maximum:

`L(x,beta) = max(0, deletion_harm, insertion_harm, support_harm, distributional_harm)`.

All terms lie in `[0,1]`, so `L in [0,1]`. Controlling expected `L` controls each constituent expected excess harm by the same budget.

## Monotone envelope
CRC requires a nested/monotone family. The observed loss over increasing alignment is therefore monotonized client-side:

`L_tilde(x,beta_g) = max_{h <= g} L(x,beta_h)`.

This is a conservative envelope: once a stronger alignment has demonstrated a harm, even stronger actions inherit at least that harm.

## Conformal Risk Control selector
For `n` calibration examples and bounded loss `B=1`, the implemented finite-sample CRC upper empirical risk is

`U_n(beta) = n/(n+1) * mean_j L_tilde(x_j,beta) + 1/(n+1)`.

The client chooses the largest beta satisfying

`U_n(beta) <= alpha`,

with frozen `alpha=0.05`.

`beta=0` is the Local-XAI fallback. If the calibration sample is too small to certify even the fallback under the finite-sample correction, the client returns beta=0 but marks the action uncertified; the confirmatory gate requires complete certification.

## Global-CRC control
To test whether client-specific calibration actually adds value, the strongest single globally safe action is

`beta_global = min_i beta_i^max`.

Every client then uses the same `beta_global`. CFBA only earns a paper-level claim if the client-specific policy improves explanation coordination over this globally certified control under heterogeneous clients.

## Attack protocol
For attribution-artifact poisoning experiments, certification is recomputed against the **attacked prior**. A certificate from the clean prior is never reused for an attacked prior.

## Communication
Calibration examples, IG explanations, losses, and risk curves remain client-local. The server receives the same sparse mean/support artifact family as the xFedAlign-style baseline. CFBA therefore has equal explanation-artifact communication in this pipeline; implementation metadata such as selected beta is logged for research audit but is not needed as a high-dimensional payload.

## Scope of guarantee
The CRC result is a finite-sample **expected-risk** guarantee for the defined bounded loss under its exchangeability assumptions. It is not:

- a high-probability guarantee that every calibration draw or every future example has loss <= alpha;
- a guarantee of human-perceived interpretability;
- a guarantee under arbitrary temporal distribution shift;
- a claim that direct IG is ground truth.

The full study tests empirical held-out risk, but the theory and the empirical gate must not be conflated.
