# Data partitions and metrics

Each client's local data is split into **five disjoint partitions**:
1. task training (50%);
2. surrogate fitting (15%);
3. transmitted explanation artifact estimation (10%);
4. private CFBA calibration (15%);
5. final held-out evaluation (10%).

This separation prevents the conformal selector from being evaluated on its calibration records and prevents the explanation artifact from being judged on the examples used to estimate it.

Primary metrics:
- `pairwise_edi`: cross-client explanation drift (lower is better);
- `sample_fidelity_jsd`: JSD to direct task-model IG on final evaluation examples (lower is better);
- deletion AUC (lower is better);
- insertion AUC (higher is better);
- oracle top-k overlap (higher is better);
- `excess_fidelity_risk`: the held-out four-component bounded CFBA loss (lower is better);
- communication bytes per client artifact.

`artifact_fidelity_jsd` is retained for continuity with earlier phases but is secondary to per-sample held-out fidelity in the CFBA claim.
