# Known limitations before the full NC-SACPA run

1. **Vision only.** The self-contained suite covers MNIST and CIFAR-10, not xFedAlign's text/tabular modalities.
2. **Controlled shifts.** Rotation, sparse patch, and photometric transformations are mechanism stressors, not substitutes for naturally occurring federated explanation heterogeneity.
3. **Final-snapshot coordination.** Explanation methods are evaluated after task training to isolate coordination. This package does not yet implement an evolving per-round NC-SACPA prior.
4. **xFedAlign comparator is a mechanism reproduction.** The official authors' code must be cross-checked before publication claims.
5. **Linear-surrogate primary track.** Direct task-model IG is used as an independent oracle and as a Tier-B explainer-source stress, but the package does not reproduce every modality-specific surrogate in xFedAlign.
6. **No formal privacy guarantee.** Artifact perturbation and membership audits are empirical; there is no privacy accountant.
7. **No Byzantine theorem.** Two poisoning tests are empirical only.
8. **No theorem yet.** The self-tuned kernel and local corroboration are mechanistically motivated. Full results should determine whether a formal analysis is worth developing.
9. **Five confirmatory seeds.** Paired analysis is used; tiny effects should not be overinterpreted even if statistically consistent.
10. **Smoke-test predecessor is sometimes numerically better.** Phase-VII SACPA has a lower overall mean in some smoke families but fails a preregistered strong grouped-patch condition. NC-SACPA is chosen for robustness across families, not because it dominates its predecessor everywhere.
