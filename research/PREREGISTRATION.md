# CFBA-CRC full-study preregistration

This file freezes the confirmatory study **before seeds 1701–1705 are run**.

## Primary claim
Under heterogeneous federated explanation shifts, client-private Conformal Risk Control can permit useful explanation alignment for clients that can safely accept it, while a single globally certified alignment strength is forced to obey the most restrictive client.

The claim is **not** that CFBA always has lower EDI than unconstrained xFedAlign. The comparison that tests the adaptive mechanism is CFBA versus `global_crc`, because both obey the same fidelity-risk budget and use the same prior/grid/calibration information.

## Frozen method
- risk budget `alpha = 0.05`;
- beta grid `[0, .1, .2, .4, .6, .8]`;
- bounded four-component excess-fidelity loss defined in `METHOD.md`;
- monotone cumulative-max envelope over increasing beta;
- finite-sample CRC correction `n/(n+1)*empirical + 1/(n+1)`;
- client-private calibration using direct task-model IG;
- global prior: coordinate-wise median xFedAlign-style prior;
- attacks: recalibrate against the attacked prior.

## Evidence separation
Development/smoke seeds 1601–1606 are permanently excluded from the full confirmatory run. Full confirmation uses fresh seeds **1701–1705**. Tier C is exploratory and cannot be used to rewrite Tier A/B success criteria.

## Tier A
Six core experiments:
- MNIST label-skew control;
- MNIST rotation;
- MNIST sparse patch;
- CIFAR-10 label-skew control;
- CIFAR-10 rotation;
- CIFAR-10 color/photometric shift.

The primary heterogeneous families are MNIST rotation, MNIST patch, CIFAR-10 rotation, and CIFAR-10 color.

## Tier B
Required stress tests include unequal client sizes, partial task participation, rotation drift, direct task-model IG as explainer source, and two attribution-poisoning attacks.

## Frozen gates
Executable in `evaluate_cfba_gates.py`.

1. **Task validity.** Mean client task accuracy >= .80 for MNIST core families and >= .35 for CIFAR-10 core families.
2. **Certification completeness.** `certified_fraction == 1.0` for CFBA in every core family and seed.
3. **Held-out composite risk.** Mean CFBA held-out `excess_fidelity_risk <= .05` in every primary heterogeneous family.
4. **Client-specific value.** CFBA must improve mean pairwise EDI over `global_crc` by >=5% and win >=4/5 paired seeds in at least 3/4 heterogeneous families.
5. **Fidelity non-inferiority to Local-XAI.** Pooled across the four heterogeneous families: sample JSD <= Local + .02; deletion AUC <= Local + .01; insertion AUC >= Local - .01; oracle top-k overlap >= Local - .05.
6. **Communication parity.** Explanation-artifact bytes must equal the xFedAlign-style mean/support baseline.
7. **Attack recalibration safety.** Under each attack, recalibrated CFBA must have mean held-out excess risk <=.05, sample-JSD degradation <=.02 versus Local, pairwise EDI no worse than Local, and complete certification.

All seven gates must pass for `CONTINUE`. Missing evidence returns `PENDING`; any failed gate returns `REVISE_OR_KILL`.

## Statistics
Five paired seeds are sufficient for the frozen research gate, not for a final significance claim. If the method survives, the paper-stage core comparisons should be expanded to at least 10 independent seeds or otherwise use a prespecified power/precision analysis. Bootstrap intervals and paired tests in this package are descriptive at the five-seed gate stage.
