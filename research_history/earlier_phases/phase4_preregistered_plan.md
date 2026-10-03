# Phase IV pre-registered smoke test: uncertainty-aware personalized explanation alignment

Date registered: 2026-10-02 (America/Toronto), before Phase-IV experiment execution.

## Working hypothesis

A single global explanation prior can reduce sampling noise, but it can also erase legitimate client-specific explanatory structure under genuine semantic heterogeneity. A server can separate these two sources of disagreement by combining (i) between-client dispersion of local explanation summaries with (ii) the within-client sampling uncertainty of those summaries. An empirical-Bayes/random-effects shrinkage target should therefore borrow strongly from the global prior where disagreement is explainable by sampling noise, and preserve local explanatory structure where between-client heterogeneity exceeds sampling uncertainty.

Working name: **REPA** — Random-Effects Personalized Attribution alignment. This is only a working name; random-effects estimation itself is classical and is not claimed as novel.

## Controlled MNIST smoke-test setting

Binary MNIST task: digit 3 vs 8. A FedAvg logistic task model is trained over 12 clients. Explanations use exact zero-baseline logit integrated-gradients for this linear task model (`w * x`) and are normalized into attribution distributions.

We stress the assumption behind a single global explanation prior using label-predictive border patches that are part of the task model's learned decision function:

- **shared**: all clients have the same patch location (semantically aligned explanations),
- **grouped**: clients belong to four patch-location groups,
- **personal**: every client has a distinct patch location (continuous/client-specific heterogeneity analogue).

The patch appears for the positive class with 90% probability and for the negative class with 10% probability. It is intentionally a controlled mechanism, not a claim of real-world naturalness.

Each client estimates its explanation summary from small held-out explanation samples. A large transformed MNIST test pool gives an oracle local explanation target for evaluation only.

## Methods compared

1. **Local**: no alignment.
2. **Global**: soft alignment toward one coordinate-median global explanation prior (xFedAlign-style principle; not a reproduction of the full xFedAlign surrogate-training algorithm).
3. **Cluster**: k-means client clustering in explanation-summary space, with a cluster prior and the same soft blend.
4. **Variance gate**: coordinate-wise hard gate that aligns a coordinate only when between-client variance is not substantially larger than estimated within-client sampling variance.
5. **REPA**: coordinate-wise empirical-Bayes/random-effects shrinkage using estimated between-client semantic variance `tau^2 = max(0, s_between^2 - mean(v_within))`, with posterior weight on the global mean `B_i = v_i / (v_i + tau^2 + eps)`.

Global/cluster blend strength and the variance-gate threshold will be tuned only on discovery seeds, then frozen.

## Discovery and confirmatory split

- Discovery seeds: 101, 102, 103.
- Confirmatory seeds: 201, 202, 203, 204, 205.
- Explanation sample sizes: 8, 16, 32, 64 positive examples/client when available.

No Phase-I/II/III seed is reused for Phase-IV confirmation.

## Primary metrics

- Mean Jensen-Shannon divergence (JSD) from each method's adjusted client explanation to its oracle local explanation (**primary fidelity metric**).
- Mean L1 error to oracle.
- False attribution mass assigned to other clients' / other groups' patch coordinates.
- JSD on the shared image core after removing all patch pixels (tests whether shared structure is denoised).
- Pairwise explanation JSD (reported as consistency only; lower is not automatically better under true heterogeneity).

## Approval gate (fixed before experiments)

The approach is **APPROVED FOR CONTINUATION** only if all of the following hold on the five confirmatory seeds:

1. **Heterogeneous fidelity:** In both grouped and personal settings, REPA has lower mean oracle-JSD than Local and Global at at least 3 of 4 explanation sample sizes, and its aggregate mean improvement over the better of Local/Global is at least 5%.
2. **Homogeneous denoising:** In the shared setting, REPA improves oracle-JSD over Local by at least 10% in aggregate and is no worse than Global by more than 3%.
3. **Not a trivial hard gate:** Across all settings/sample sizes, REPA improves aggregate oracle-JSD over the tuned Variance-Gate baseline by at least 2%, OR provides a clear Pareto improvement in fidelity and false-patch mass. If the hard gate matches REPA within 2% without a trade-off, the method is revised/killed as unnecessarily complex.
4. **Not reducible to clustering:** In the personal setting, REPA must improve aggregate oracle-JSD over the best fixed-k cluster baseline by at least 3%. (Grouped clients may appropriately favor clustering.)
5. **Mechanistic prediction:** Estimated `tau^2` on patch coordinates must be materially larger in grouped/personal than shared settings, while core-coordinate shrinkage remains stronger than patch-coordinate shrinkage under heterogeneity. This must hold in all five confirmatory seeds in aggregate.
6. **No cherry picking:** Results are aggregated over all pre-registered confirmation seeds and sample sizes; failures are reported.

If any central criterion fails, mark **REVISE** or **KILL**, do not call the approach approved.

## Novelty gate

Before final approval, search current literature for federated XAI methods that already perform uncertainty-aware/random-effects or empirical-Bayes personalization of explanation priors. If a direct prior is found, downgrade the claim regardless of experiment performance.

The novelty claim, if it survives, is deliberately narrow: *using within-client explanation uncertainty to statistically separate sampling disagreement from genuine client-specific explanation heterogeneity and construct personalized federated explanation priors.* Classical random-effects/empirical-Bayes methodology itself is prior art.
