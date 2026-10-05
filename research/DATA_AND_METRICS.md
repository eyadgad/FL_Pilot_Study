# Data, shifts, artifacts, and metrics

## Datasets

- MNIST: efficient mechanism and stress testing.
- CIFAR-10: required nonlinear/deeper-vision holdout.

Torchvision downloads the official train/test sets. Within each federated training shard, samples are split 65/15/10/10 into task-model training, surrogate fitting, artifact estimation, and untouched final explanation evaluation.

## Client heterogeneity

- Dirichlet alpha=.1 label skew is present in full core configurations.
- Rotation creates coherent input/explanation shifts.
- Patch creates sparse client-specific feature heterogeneity on MNIST.
- Photometric color shift creates client-specific CIFAR-10 appearance shift.
- Lognormal client-size variation tests unequal local data quantity / explanation precision.
- Drift rotation changes the task-training environment over rounds; explanation evaluation is final-snapshot only.

## Explanation artifact

Per-class attributions are nonnegative and normalized. The shared artifact uses top-k coordinates, clipping, 8-bit mean quantization, and configurable Gaussian perturbation.

- xFedAlign-style and NC-SACPA coordination use sparse mean/support information.
- UCPA v1 also uses variance-of-the-mean values and is retained only as a negative control.

Communication is estimated from coordinate indices, quantized means, and any method-specific auxiliary values.

## Fidelity oracle

Direct integrated gradients of the trained global task model on the **disjoint client evaluation split** is the local functional explanation reference. It is separate from surrogate fitting and artifact estimation.

## Metrics

- `artifact_fidelity_jsd`: JSD from coordinated class summary to direct-IG client summary. **Primary.**
- `sample_fidelity_jsd`: per-example JSD to direct IG.
- `pairwise_edi`: average pairwise explanation divergence for shared classes; measures coordination using the same definition for all methods.
- `reference_edi`: method-reference consistency in the style of a global-prior evaluation; secondary because references differ by method.
- deletion/insertion AUC.
- top-k oracle overlap.
- task and per-client task accuracy sanity checks.
- communication bytes per client artifact.
- artifact membership AUC as an empirical privacy exposure audit, not a DP guarantee.
