# Claims allowed by this study

This file prevents post-hoc claim inflation.

## C1 — personalized explanation coordination

Candidate claim: NC-SACPA can preserve client-local explanation fidelity better than a single global explanation prior under multiple forms of client-specific explanation heterogeneity while still obtaining a measurable coordination benefit over Local-XAI.

Required evidence: Gates G1 and G2 plus Tier-A paired results. MNIST alone is insufficient.

## C2 — scale-adaptive collaboration avoids UCPA v1 collapse

Candidate claim: self-tuned explanation-space similarity avoids the fixed-bandwidth peer-collapse observed in UCPA v1 across dataset/explainer scales.

Required evidence: mechanism logs/peer mass across MNIST and CIFAR plus the full Tier-A primary result. Do not claim this from the Phase-VIII smoke test alone.

## C3 — neighborhood corroboration addresses sparse support contamination

Candidate claim: local-neighborhood corroboration improves robustness to grouped sparse heterogeneity relative to the Phase-VII global-count predecessor without unacceptable degradation elsewhere.

Required evidence: G4 and the patch-family analysis.

## C4 — the practical cost is competitive

Candidate claim: NC-SACPA does not require a variance channel and can match xFedAlign-style sparse mean/support communication at matched top-k.

Required evidence: G6 and actual byte logs.

## C5 — robustness is not materially worse than the closest global-prior comparator

Required evidence: G7. Passing this is non-inferiority, not a claim of Byzantine robustness.

## Claims not permitted from this package alone

- formal differential privacy;
- Byzantine-robust guarantees;
- exact reproduction of official xFedAlign results;
- generality to text/tabular modalities or natural cross-silo deployments;
- superiority to all personalized FL methods;
- formal optimality of the self-tuned kernel or neighborhood size;
- conference acceptance probability.
