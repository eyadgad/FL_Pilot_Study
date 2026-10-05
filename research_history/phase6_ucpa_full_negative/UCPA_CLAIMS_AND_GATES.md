# Claims that the full pipeline is allowed to test

This file prevents post-hoc inflation of the contribution.

## Claim C1 — personalized prior under genuine explanation heterogeneity

Candidate claim: UCPA preserves client-local explanation fidelity better than a single global prior when clients have genuinely different feature semantics, while still reducing drift relative to purely local explanations.

Evidence required: Tier-A rotation/patch/color experiments; primary outcome plus pairwise EDI and perturbation fidelity. MNIST alone is insufficient.

## Claim C2 — both scales matter

Candidate claim: whole-explanation similarity and coordinate-level uncertainty solve complementary failure modes.

Evidence required: full UCPA beats `ucpa_whole_only` and `ucpa_coord_only` when pooled over heterogeneous and unequal-sample conditions. If not, simplify the method and claim.

## Claim C3 — uncertainty helps with unequal estimation quality

Candidate claim: UCPA handles heteroskedastic explanation estimates better than similarity-only alignment.

Evidence required: unequal-size stress experiments, with full UCPA outperforming whole-only alignment on primary fidelity.

## Claim C4 — communication/privacy trade-off is acceptable

Not automatically granted. UCPA sends variance as well as attribution means, so it incurs additional bytes and may expose membership information. The package reports both. A paper must report any privacy degradation and should add a mitigation if it is material.

## Claims NOT permitted from this package alone

- Formal differential privacy: Gaussian noise is an experimental hardening mechanism here; no privacy accountant or formal epsilon guarantee is implemented.
- Byzantine robustness: the poisoning stress is empirical, not a proof.
- Dynamic partial-participation explanation alignment: current partial-participation config applies to task training; final explanation artifacts include all clients.
- Generality to text/tabular or real federated deployments: not tested in this package.
- Exact reproduction of official xFedAlign numbers: official code is external.
- Conference acceptance likelihood.
