# Baseline provenance

- **Local-XAI:** no federated explanation alignment; per-sample local explanation is unchanged.
- **FedAttr mean:** global mean attribution prior; simple oversmoothing control.
- **xFedAlign-style median:** self-contained reproduction of the published idea of constructing a robust Global Explanation Prior from compact client attribution artifacts and softly aligning local explanations toward it. This package does not claim to be the authors' official implementation.
- **Global CRC:** same prior, beta grid, client calibration rule and risk budget as CFBA, but one beta is imposed on every client: the minimum certified cap. This is the critical control for the claim that client-specific risk budgets matter.
- **CFBA-CRC:** same shared prior, but each client privately uses its own maximal certified beta.

Paper-stage positioning must also compare conceptually against conformal/risk-controlled model-property alignment, conformal explanation methods, and site-conditional federated CRC. See `NOVELTY_POSITIONING.md`.
