# Phase IX preregistration — Rank-Preserving Global Alignment (RPGA)

Frozen before confirmation seeds 1201–1205.

## Hypothesis
A federated explanation can move attribution **magnitudes** toward a shared robust global prior without changing the client's feature **ranking**. Projecting the global prior onto the client's local ranking cone should reduce cross-client explanation drift while preserving ranking-based functional fidelity by construction.

## Method
For local sample attribution `e` and xFedAlign-style class prior `g`:

1. obtain the complete descending local coordinate order `pi`;
2. reorder `g` by `pi`;
3. solve the Euclidean isotonic projection onto the non-increasing cone using PAVA;
4. break projection ties infinitesimally in the exact local sort order;
5. map the projected magnitudes back to the original coordinates.

No beta, uncertainty channel, extra communication, or client clustering is used.

## Comparators
- Local-XAI
- xFedAlign-style median prior with beta=0.2
- FedAttr/global-mean prior (full prior; diagnostic oversmoothing baseline)

## Fresh confirmation
Seeds: 1201, 1202, 1203, 1204, 1205.
Families: client-specific rotations and sparse patches on the controlled synthetic FL task.
Development seeds 1101–1104 are excluded from confirmation.

## Frozen gates
All gates must pass.

### G1 — Ranking/functional invariance
For both families and every seed:
- local top-k retention = 1.0 (tolerance 1e-12);
- |RPGA deletion AUC - Local deletion AUC| <= 1e-4;
- |RPGA insertion AUC - Local insertion AUC| <= 1e-4.

### G2 — Coordination gain
For each family:
- mean RPGA pairwise EDI <= 0.90 * mean Local pairwise EDI;
- RPGA pairwise EDI < Local in at least 4/5 seeds.

### G3 — Oracle-fidelity non-inferiority
Pooled across both families:
- mean RPGA oracle JSD <= mean Local oracle JSD + 0.003;
- mean RPGA oracle JSD <= 0.95 * mean xFedAlign(beta=.2) oracle JSD.

### G4 — Pareto behavior
Across all 10 seed-family pairs, at least 8/10 must have:
- RPGA EDI < Local EDI; and
- RPGA oracle JSD <= Local oracle JSD + 0.01.

## Decision
- `APPROVE_SMOKE`: all gates pass.
- `REJECT`: any gate fails.

Approval means only that RPGA deserves integration into the full MNIST/CIFAR pipeline. It is not evidence of conference readiness.
