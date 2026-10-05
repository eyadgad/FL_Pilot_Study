# NC-SACPA full-study preregistration

**Frozen package date:** 2026-10-04  
**Suite:** 2.1  
**Confirmatory seeds:** 9101, 9102, 9103, 9104, 9105

These seeds are reserved for the full study and were not used in Phase-VII/Phase-VIII method development or smoke confirmation.

## Research question

Can NC-SACPA improve **client-local explanation fidelity** under genuine federated explanation heterogeneity while still reducing cross-client explanation drift, matching the communication cost of a sparse global-prior method, and avoiding the scale-collapse and support-contamination failures observed in UCPA v1 and SACPA v1?

## Frozen method

NC-SACPA uses:

- self-tuned whole-explanation JSD kernel, power `p=2`;
- peer-only personalized prior;
- alignment strength `beta=0.8`;
- three nearest explanation peers for support corroboration;
- at least two of those three peers must report a feature before a feature absent from the target may be imported.

No Tier-A result may change these values.

## Common experimental controls

Every explanation-coordination method in a run receives:

- the identical trained FedAvg task model;
- identical client partitions;
- identical surrogate-fit samples;
- identical artifact-estimation samples;
- identical final explanation-evaluation samples;
- identical local attribution artifact samples.

Each client's training shard is split 65/15/10/10 into task training, surrogate fitting, artifact estimation, and untouched final explanation evaluation. This isolates explanation coordination from task-model training.

## Core methods

- `local`: Local-XAI / no coordination.
- `fedattr_mean`: global mean attribution template reproduction.
- `xfedalign_median`: xFedAlign-style sparse median Global Explanation Prior reproduction.
- `cluster`: strong continuous-vector clustering comparator.
- `ucpa`: killed fixed-bandwidth predecessor, retained as negative control.
- `sacpa_global_count`: rejected scale-adaptive/global-count predecessor.
- `nc_sacpa`: proposed frozen method.

UCPA mechanism ablations are retained for historical diagnostics but are not required to justify NC-SACPA.

## Tier A — required core evidence

Six five-seed experiments:

1. MNIST, Dirichlet label skew control.
2. MNIST + client-specific rotation.
3. MNIST + sparse client-specific patch heterogeneity.
4. CIFAR-10, Dirichlet label skew control.
5. CIFAR-10 + client-specific rotation.
6. CIFAR-10 + client-specific photometric shift.

The four shifted experiments are the primary heterogeneity families.

## Tier B — required stress/robustness for the final gate

- unequal client sample sizes on MNIST/CIFAR-10 rotation;
- partial task participation;
- time-varying rotation shift;
- direct task-model IG as the communicated explanation source on MNIST/CIFAR-10;
- artifact-shift poisoning;
- random-support poisoning.

## Tier C — exploratory sensitivity/communication

- top-k = 32/64/128/256;
- artifact Gaussian-noise sweep;
- NC-SACPA beta = .6/.9;
- self-tuned kernel power = 1/3;
- local corroboration neighborhood size = 2/4.

Tier C is exploratory and cannot be used to retroactively tune Tier A.

## Primary metric

`artifact_fidelity_jsd`: class-conditional JSD between a method's coordinated explanation summary and a high-fidelity direct-integrated-gradients summary computed on the **disjoint final client evaluation split**. Lower is better.

This is primary because consistency alone mechanically favors a single shared explanation even when client-specific semantics are real.

## Secondary metrics

- `sample_fidelity_jsd`
- `pairwise_edi`
- xFedAlign-style `reference_edi`
- deletion AUC (lower is favorable under this implementation's convention)
- insertion AUC (higher is favorable)
- top-k oracle overlap
- task/client accuracy sanity checks
- per-client artifact communication bytes
- poisoning robustness
- empirical artifact membership audit

## Frozen full-study decision gate

`evaluate_ncsacpa_gates.py` is the executable source of truth.

### G1 — primary local fidelity

Across MNIST rotation, MNIST patch, CIFAR-10 rotation, and CIFAR-10 color, NC-SACPA must beat `xfedalign_median` on the seed-paired primary metric in at least **3 of 4 families**. A successful family requires:

- all five seed pairs present;
- at least 4/5 seed wins;
- negative mean paired JSD difference.

### G2 — coordination benefit

NC-SACPA must reduce `pairwise_edi` relative to Local-XAI in at least **3 of 4** heterogeneous core families.

### G3 — functional non-inferiority

Pooled across the four heterogeneous core families versus xFedAlign-style median:

- deletion AUC mean delta <= +0.01;
- insertion AUC mean delta >= -0.01;
- top-k oracle-overlap mean delta >= -0.05.

### G4 — revision mechanism

On MNIST patch, NC-SACPA must improve the primary metric over the rejected `sacpa_global_count` predecessor. Across the other three heterogeneous core families its mean primary-metric degradation relative to that predecessor must be <= 0.02.

### G5 — CIFAR task validity

Mean client task accuracy must be >= 0.35 for both CIFAR heterogeneous core families. This is a validity floor, not a performance claim for the explanation method.

### G6 — communication

NC-SACPA's per-client explanation-artifact communication must not exceed the xFedAlign-style mean-artifact comparator at matched top-k.

### G7 — poisoning robustness

Pooled over both attacked MNIST rotation experiments, NC-SACPA's attacked primary fidelity may be at most +0.01 JSD worse than xFedAlign-style median.

## Overall decision

- `CONTINUE`: all seven gates are complete and pass.
- `REVISE_OR_KILL`: all are complete and at least one fails.
- `PENDING`: required evidence is incomplete.

A `CONTINUE` result means the method earned a paper/theory phase; it does **not** imply conference acceptance or publication readiness.
