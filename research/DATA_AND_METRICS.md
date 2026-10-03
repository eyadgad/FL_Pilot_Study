# Data, shifts, and metrics

## Datasets

- MNIST: mechanism replication / efficient stress tests.
- CIFAR-10: required nonlinear/deeper-vision holdout.

Torchvision downloads the official train/test sets. Each federated train shard is subsequently split 65/15/10/10 into task/surrogate/artifact/evaluation sets.

## Client heterogeneity

- Dirichlet alpha=.1 label skew is present in all full core configs.
- Rotation creates coherent input-space explanation shifts.
- Patch creates sparse client/label-correlated feature shifts on MNIST.
- Photometric color shift creates client-specific CIFAR-10 appearance shift.
- Lognormal subsampling creates unequal explanation-estimation precision.
- Drift rotation increases over task-training rounds; explanation evaluation is final-snapshot only.

## Explanation artifact

Per-class attributions are nonnegative and normalized. The artifact uses top-k coordinates, L2 clipping, 8-bit mean quantization, and configurable Gaussian perturbation. UCPA additionally transmits variance-of-the-mean estimates on transmitted coordinates. Communication bytes are estimated from indices, quantized means, and float32 variance values.

## Fidelity oracle

Direct integrated gradients of the trained global task model on the **disjoint client evaluation split** is the local functional explanation reference. This is intentionally separate from the surrogate and from artifact estimation.

## Metrics

- `artifact_fidelity_jsd`: JSD from coordinated class summary to direct-IG client summary. Primary.
- `sample_fidelity_jsd`: per-example JSD to direct IG.
- `pairwise_edi`: average pairwise JSD among client explanation summaries for classes represented by both clients. Unlike method-reference EDI, it uses the same definition for every method.
- `reference_edi`: xFedAlign-style client-to-method-reference consistency. Secondary because different methods induce different references.
- deletion/insertion AUC: target-class probability as top-ranked input coordinates are removed/inserted.
- top-k oracle overlap.
- communication bytes.
- artifact membership AUC: empirical distinguishability of examples used vs not used to estimate a transmitted artifact, using mean-only and mean+variance scores.
