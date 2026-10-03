# Phase V result: UCPA is smoke-test approved for continuation

## Decision

**SMOKE-TEST APPROVED FOR CONTINUATION**

This decision is based on a pre-registered gate frozen before the final untouched seeds 501–505 were run. All six pre-registered criteria passed without retuning UCPA.

This is a research-continuation decision, **not** a claim that UCPA is publication-ready or that it would be accepted at NeurIPS/ICML/ICLR.

## The approved candidate

### Uncertainty-Compatible Peer Alignment (UCPA)

For each client `i`, let `E_i` be its normalized estimated attribution vector and `V_i` the estimated sampling variance of the mean attribution at each coordinate.

UCPA uses two compatibility terms for borrowing explanation information from client `k`:

1. **Whole-explanation compatibility**

\[
g_{ik}=\exp\{-\operatorname{JSD}(E_i,E_k)/h\}.
\]

2. **Coordinate uncertainty compatibility**

\[
c_{ikj}=\exp\left[-\frac12\left(
\frac{|E_{ij}-E_{kj}|}
{z\sqrt{V_{ij}+V_{kj}+\epsilon_v}}
\right)^2\right].
\]

The coordinate-specific peer weight is

\[
w_{ikj}=g_{ik}c_{ikj}.
\]

The aligned explanation is the normalized weighted peer mean. Self-weight is fixed to one.

Frozen parameters from development only:

- `z = 2.0`
- `h = 0.048`
- whole-explanation weight floor `lambda = 0`
- variance-floor fraction `0.05`

The standalone implementation is in `ucpa.py`.

## Why this candidate emerged

The project did not jump directly to UCPA.

1. The original noisy-label A′/B direction was killed after novelty and mechanism audits.
2. The identifiability pivot was also closed because the broad theoretical result had strong prior collisions.
3. In explanation alignment, a random-effects shrinkage candidate failed because a much simpler feature-wise variance gate was better.
4. That gate (HGEA) passed sparse patch heterogeneity but failed an independent rotation stress: coherent explanation shifts require cross-feature/client structure.
5. Coordinate-only peer smoothing improved rotation behavior but did not fully capture coherent client structure.
6. UCPA was then developed to combine **whole-explanation neighborhood structure** with **coordinate-level uncertainty compatibility**.
7. One parameter set was selected on development seeds spanning both patch and rotation heterogeneity, frozen, and then tested on untouched seeds.

The failed predecessors remain part of the evidence because they identify what each scale of UCPA contributes.

## Final confirmation protocol

Fresh seeds: `501, 502, 503, 504, 505`.

Two qualitatively different heterogeneity generators:

- **Patch:** client-specific predictive image patches, with strong and moderate strengths; shared/grouped/personal cases.
- **Rotation:** coherent client-specific image rotations; shared/grouped/personal cases.

Explanation sample counts: `8, 16, 32, 64`.

The classifier is a small federated logistic model on binary MNIST (3 vs 8). Exact zero-baseline integrated-gradient magnitudes are used as the attribution representation. A disjoint higher-sample explanation set defines the evaluation oracle.

Important: this is a controlled mechanism smoke test, not the full xFedAlign pipeline.

## Main confirmation results

| Heterogeneous family | UCPA JSD | Local JSD | Best oracle-tuned global | Global JSD | UCPA vs Local | UCPA vs global |
|---|---:|---:|---|---:|---:|---:|
| Patch — grouped | 0.002690 | 0.010442 | GlobalMedian | 0.006521 | 74.2% lower | 58.7% lower |
| Patch — personal | 0.002488 | 0.011155 | GlobalMedian | 0.005046 | 77.7% lower | 50.7% lower |
| Rotation — grouped | 0.004408 | 0.008642 | GlobalMean | 0.006303 | 49.0% lower | 30.1% lower |
| Rotation — personal | 0.004808 | 0.009134 | GlobalMean | 0.006639 | 47.4% lower | 27.6% lower |

The global baseline families were allowed to choose their best beta **using the final oracle metric**, which intentionally advantages them. UCPA remained frozen.

### Fresh-seed robustness

UCPA beat both Local and the selected global baseline on **5/5 untouched seeds in all four heterogeneous families**.

### Shared explanations

When client explanations were genuinely shared:

- patch: UCPA JSD `0.001839`, an **81.4%** reduction relative to Local;
- rotation: UCPA JSD `0.002130`, an **80.8%** reduction relative to Local.

The gate did not require UCPA to beat full global averaging in the truly homogeneous case.

### Unified adaptive-baseline test

Equal-weight mean JSD across the four heterogeneous families:

| Method | Mean JSD |
|---|---:|
| **UCPA** | **0.003599** |
| Coordinate-only peer smoothing | 0.004328 |
| HGEA variance gate | 0.005552 |
| QGate | 0.005764 |
| Oracle-tuned clustering | 0.005819 |

Relative to these pooled heterogeneous results, UCPA was:

- **16.9%** lower than coordinate-only peer smoothing;
- **35.2%** lower than HGEA;
- **37.6%** lower than QGate;
- **38.2%** lower than oracle-tuned clustering.

This is the most important smoke-test result: specialized mechanisms were stronger in some individual settings (for example HGEA on patch heterogeneity), but one frozen UCPA rule was substantially more robust across both sparse feature-specific and coherent client-level heterogeneity.

### Mechanism sanity

UCPA behaved as intended rather than using an effectively fixed neighborhood.

- Patch: mean whole-explanation peer weight was `0.748` for shared clients versus at most `0.570` under heterogeneity; effective peers were `11.49` versus at most `10.67`.
- Rotation: corresponding weights were `0.727` versus at most `0.489`; effective peers were `11.39` versus at most `8.47`.

Thus UCPA borrows broadly when clients are explanation-compatible and contracts its peer neighborhood when client explanations diverge.

## Reproducibility / integrity

- The pre-registration is `phase5_preregistered_plan.md`.
- A scan confirmed seeds 501–505 did not occur in any earlier JSONL result file.
- Final raw output contains 7,380 confirmation rows.
- SHA-256 hashes of the pre-registration, all ten final result files, and the summary are stored in `results/P5_integrity_sha256.txt`.
- The final evaluator is `phase5_evaluate.py`.
- The standalone UCPA operator has three basic unit tests in `test_ucpa.py`; all pass.

## Novelty position after the final audit

The closest direct method is **xFedAlign (ICML 2026)**, which robustly aggregates attribution artifacts into one Global Explanation Prior and softly aligns local explanations toward it. The supplied xFedAlign paper itself notes that ambiguous groups or drifting feature semantics may require adaptive grouping or drift-aware updates.

**UncertainXFL (2025)** explicitly models uncertainty in federated explanations, but it uses uncertainty of concept/logical rules to rank/select rules and influence model aggregation. It does not construct sampling-uncertainty-aware, client-specific attribution priors.

Similarity-aware collaboration, kernel smoothing, and uncertainty weighting are established ideas in broader ML and personalized FL. Therefore the defensible novelty claim is narrow:

> **a two-scale federated explanation-coordination mechanism that forms a continuous personalized prior by combining whole-explanation client compatibility with coordinate-level sampling uncertainty.**

No direct collision was found in the targeted current-literature audit, but novelty risk remains **medium** until official-code and citation-network comparisons are completed. See `phase5_novelty_audit.md`.

## What is NOT established yet

The smoke test deliberately leaves major publication requirements unresolved:

1. **No nonlinear deep model yet.** The current classifier is linear/logistic.
2. **No official xFedAlign run yet.** GlobalMean/Median are mechanism-level baselines, not a reproduction of xFedAlign.
3. **No CIFAR/text/tabular experiments yet.** Only binary MNIST is used in this controlled test.
4. **No sparse top-k communication test.** The current experiment uses full 784-dimensional attribution summaries.
5. **No privacy/poisoning analysis.** Sharing attribution variance may create additional leakage surface.
6. **No convergence/MSE theorem.** UCPA currently has a strong empirical mechanism story but not a formal statistical guarantee.
7. **No explanation-faithfulness training loop.** UCPA is tested as an alignment estimator, not yet injected as a regularizer into a nonlinear local surrogate/task model.
8. **Parameter transfer is untested.** `z=2, h=.048` may not transfer unchanged across dimensions/explainers/datasets.

## Next research gate

Do not invent another method now. The next phase should attempt to kill UCPA at realistic scale.

Priority order:

1. **Integrate UCPA into official xFedAlign code** as a personalized replacement for the single Global Explanation Prior while preserving xFedAlign's task/alignment loss and communication accounting.
2. Run nonlinear vision experiments on MNIST/CIFAR-10 using xFedAlign's exact partitions, explainer/surrogate protocol, deletion/insertion AUC, EDI, Top-k overlap, and task accuracy.
3. Add a second modality (tabular or text) only after the vision mechanism survives.
4. Compare against official xFedAlign plus Local-XAI, FedAttr-Agg, Fed-XAI, and UncertainXFL where technically comparable.
5. Test sparse/quantized top-k means **and variances** and measure the communication delta.
6. Stress continuous concept drift, minority clients, unequal client sample sizes, partial participation, and explanation poisoning.
7. Derive a statistical view of UCPA as heteroskedastic two-scale kernel regression on a client explanation manifold; seek an MSE/bias-variance result explaining when peer borrowing beats global, local, and discrete clustering.
8. Re-run the novelty audit immediately before paper framing.

## Conference relevance

ICML 2026 explicitly asks whether empirical claims follow from well-designed experiments and whether originality/significance are clear relative to current work. NeurIPS 2026 similarly emphasizes quality, significance, originality, and reproducibility. The present phase meets only an **early evidence gate** under those dimensions; the nonlinear official-baseline validation and theory/analysis above are still necessary for a credible top-conference submission.

## Key sources used in the final audit

- Wasif et al., xFedAlign, ICML 2026: https://proceedings.mlr.press/v306/wasif26a.html
- Zhang & Yu, UncertainXFL: https://arxiv.org/abs/2503.05194
- Hoefler et al., FedXDS: https://arxiv.org/abs/2606.31742
- Gholizade et al., Federated XAI review: https://arxiv.org/abs/2607.13045
- ICML 2026 Reviewer Instructions: https://icml.cc/Conferences/2026/ReviewerInstructions
- NeurIPS 2026 Reviewer Guidelines: https://neurips.cc/Conferences/2026/ReviewerGuidelines
