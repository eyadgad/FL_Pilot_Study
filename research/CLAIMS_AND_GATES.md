# Claims and falsification gates

## Candidate claim 1 — heterogeneous safe alignment
A single globally safe alignment strength can be overly conservative when clients have different fidelity tolerances. CFBA should improve pairwise EDI over a globally certified fixed beta in at least 3/4 primary heterogeneous families while obeying the same risk budget.

**Killed if:** Gate 4 fails.

## Candidate claim 2 — bounded fidelity cost
CFBA's improvement must not be purchased through material degradation of the deployed task-model explanation.

**Killed/revised if:** held-out risk or pooled Local-XAI non-inferiority gates fail.

## Candidate claim 3 — no added high-dimensional communication
CFBA calibration remains private and reuses the xFedAlign-style sparse explanation artifact.

**Killed/revised if:** communication exceeds the matched xFedAlign artifact in the implementation.

## Candidate claim 4 — attack-aware certification
When the shared prior is attacked, the client must recalibrate rather than reuse a clean certificate.

**Killed/revised if:** attacked-prior risk or fidelity gates fail.

## Claims explicitly not made
- CFBA does not claim to beat unconstrained xFedAlign on raw EDI in every regime.
- CFBA does not claim CRC, conformal prediction, IG, median aggregation, or per-client calibration are individually novel.
- The CRC guarantee is not a guarantee of human interpretability.
