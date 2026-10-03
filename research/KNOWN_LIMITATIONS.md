# Known limitations before the full run

1. **Vision only.** The self-contained suite covers MNIST and CIFAR-10, not xFedAlign's text/tabular modalities.
2. **Synthetic client shifts.** Rotation, patch, erasing/color transformations are controlled stressors. They are useful for mechanism identification but are not substitutes for naturally occurring federated explanation heterogeneity.
3. **Final-snapshot coordination.** The explanation comparison is deliberately decoupled from task training and evaluated at the final model snapshot. It does not implement an evolving per-round UCPA prior.
4. **xFedAlign comparator is a reproduction.** Official authors' code should be run separately before publication.
5. **Linear surrogate track.** The self-contained surrogate is sparse linear in input space. Direct task-model IG is used as the independent fidelity oracle. A future reproduction may use the authors' modality-specific surrogate architecture.
6. **No formal privacy guarantee.** Variance sharing is audited for empirical membership distinguishability but no DP accountant is provided.
7. **No theorem yet.** UCPA is motivated as heteroskedastic product-kernel smoothing. The full results should determine whether a formal consistency/bias-variance analysis is worth developing.
8. **Five seeds.** This matches the closest published baseline's reporting convention and supports paired analysis, but small effects should not be overinterpreted.
