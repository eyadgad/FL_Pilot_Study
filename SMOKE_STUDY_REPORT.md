# Multi-Proposal Federated Explanation Alignment Smoke Study

## Decision

**APPROVE FOR MEDIUM-COST / REAL-DATA PILOT:** Risk-Weighted Shared-Subspace Alignment (provisional name: **RWSSA**).

**Paper-direction readiness:** **80/100**. This is a research-readiness score, not an acceptance probability.

The approved candidate is the only proposal in this search that beat the xFedAlign-style fixed global-prior baseline on *all measured dimensions* across every fresh confirmation run: oracle explanation JSD, pairwise explanation drift (EDI), deletion AUC, insertion AUC, top-k oracle overlap, and bounded excess fidelity risk.

## Common smoke protocol

All proposal families were evaluated using the same federated synthetic image task and, within a given run, the same trained model, client partition, direct task-model Integrated Gradients reference explanations, and xFedAlign-style global prior. This isolates the explanation-coordination mechanism from task-model training.

The smoke search used separate proposal-development seeds and later untouched confirmation seeds. The frozen winner used residual alignment floor `rho = 0.30` before confirmation.

### Primary baselines

- **Local-XAI:** no federated explanation alignment.
- **xFedAlign-style fixed alignment:** soft interpolation toward a robust global attribution prior with `beta = 0.2`.

The full future pilot must additionally implement the closest quality-aware aggregation baselines such as iFLASH, rather than treating this smoke comparison as a complete SOTA evaluation.

## Proposal search

| Family | Core idea | Smoke outcome | Decision |
|---|---|---|---|
| Sample-Gated CRC | Align only samples already compatible with global prior | Some functional improvement, but oracle JSD worse than Local and weaker than xFedAlign | KILL |
| Class-Policy CRC | Different alignment shape by predicted class, globally certified | No consistent advantage over xFedAlign/global policy | KILL |
| Importance-Weighted CRC | Protect locally important coordinates, align low-importance coordinates | Too little useful coordination; weak fidelity/EDI trade-off | KILL |
| Hybrid Importance × Disagreement | Align low-importance, low-conflict coordinates | Worse drift and no competitive global result | KILL |
| Uncertainty-Compatible Global Alignment | Preserve statistically significant local deviations | Mostly collapses toward Local behavior | KILL |
| Private Feature-Risk Masks | Each client aligns coordinates where global prior is closer to task-model IG | Strong fidelity improvement but **EDI gets worse** because clients learn different masks | KILL AS FINAL; useful mechanism result |
| Quorum-Safe Masks | Align coordinates judged safe by enough clients | Fidelity improves, but coordination remains weak | KILL |
| Shared Safe Subspace | All clients align one server-aggregated safe feature set | First strong candidate; substantially improves fidelity and some EDI | REVISE |
| Risk-Weighted Shared Subspace | Continuous global feature-safety weights + common residual alignment + private client calibration | Dominates xFedAlign on fresh confirmation and survives attacks/no-shift control | **APPROVE** |

## Approved candidate: RWSSA

For client `i`, class `c`, feature `j`, estimate privately on a held-out policy-selection split whether the global explanation prior is closer to the direct task-model explanation than the client's local explainer:

\[
 b_{icj}=\mathbb{E}_x\left[|E^{local}_{icj}(x)-E^{oracle}_{icj}(x)|-|G_{cj}-E^{oracle}_{icj}(x)|\right].
\]

Positive `b_icj` means that, for this coordinate, moving toward the shared prior tends to improve fidelity to the deployed task model.

The server aggregates these client feature-benefit maps:

\[
 g_{cj}=\frac{1}{N}\sum_i b_{icj}.
\]

Positive aggregated benefits are converted to a class/feature safety weight `S_cj in [0,1]`. The frozen smoke version uses a continuous positive-gain scaling and residual alignment floor `rho = 0.30`:

\[
 A_{cj}=\rho+(1-\rho)S_{cj}.
\]

Each client then chooses a private overall policy scale `lambda_i` on an independent calibration split from a fixed grid. Its final explanation is

\[
 E'_i(x)=\operatorname{Normalize}\left((1-\lambda_i A_c)\odot E_i(x)+\lambda_i A_c\odot G_c\right).
\]

The central mechanism is **not** just another global beta. Different attribution coordinates receive different alignment strengths based on cross-client evidence that the global prior improves fidelity to the deployed task model.

## Frozen fresh confirmation

Frozen candidate: `rho = 0.30`.

Untouched seeds: **1861, 1862, 1863**.

Unseen heterogeneity families: **rotation, patch, erasing** (9 total runs).

All task models were valid (~near-perfect accuracy in this controlled smoke setting).

### Aggregate across 9 fresh runs

| Method | Oracle JSD ↓ | EDI ↓ | Deletion AUC ↓ | Insertion AUC ↑ | Top-k overlap ↑ | Excess risk ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Local | 0.139732 | 0.137851 | 0.397315 | 0.811031 | 0.433214 | 0.000000 |
| xFedAlign-style | 0.133370 | 0.094653 | 0.320711 | 0.845196 | 0.469672 | 0.007471 |
| **RWSSA** | **0.123099** | **0.068158** | **0.297444** | **0.861581** | **0.526855** | **0.002717** |

RWSSA beat xFedAlign-style alignment in:

- Oracle JSD: **9/9** fresh runs
- EDI: **9/9**
- Deletion AUC: **9/9**
- Insertion AUC: **9/9**
- Top-k overlap: **9/9**
- Excess risk: **8/9**

Mean RWSSA minus xFedAlign differences:

- JSD: **-0.01027**
- EDI: **-0.02650**
- deletion AUC: **-0.02327**
- insertion AUC: **+0.01638**
- top-k overlap: **+0.05718**
- excess risk: **-0.00475**

### Confirmation by heterogeneity family

**Rotation**

- xFedAlign: JSD ~0.1404, EDI ~0.1005, deletion ~0.3381, insertion ~0.8337, top-k ~0.4308.
- RWSSA: JSD ~0.1291, EDI ~0.0728, deletion ~0.3092, insertion ~0.8536, top-k ~0.4876.

**Patch**

- xFedAlign: JSD ~0.1283, EDI ~0.0948, deletion ~0.3085, insertion ~0.8402, top-k ~0.4886.
- RWSSA: JSD ~0.1189, EDI ~0.0674, deletion ~0.2933, insertion ~0.8623, top-k ~0.5376.

**Erasing**

- xFedAlign: JSD ~0.1315, EDI ~0.0886, deletion ~0.3156, insertion ~0.8617, top-k ~0.4896.
- RWSSA: JSD ~0.1212, EDI ~0.0642, deletion ~0.2898, insertion ~0.8689, top-k ~0.5553.

## Attribution-poisoning smoke

Fresh seeds: **1871, 1872**.

Attacks: 25% malicious attribution artifacts using **artifact shift** and **random support**.

### Artifact-shift attack

| Method | JSD ↓ | EDI ↓ | Deletion ↓ | Insertion ↑ | Top-k ↑ | Risk ↓ |
|---|---:|---:|---:|---:|---:|---:|
| xFedAlign | 0.138529 | 0.100301 | 0.325755 | 0.802022 | 0.401353 | 0.018637 |
| **RWSSA** | **0.135532** | **0.096911** | **0.323913** | **0.813150** | **0.441636** | **0.007076** |

### Random-support attack

| Method | JSD ↓ | EDI ↓ | Deletion ↓ | Insertion ↑ | Top-k ↑ | Risk ↓ |
|---|---:|---:|---:|---:|---:|---:|
| xFedAlign | 0.140208 | 0.100880 | 0.337418 | 0.800745 | 0.406379 | 0.019584 |
| **RWSSA** | **0.138395** | **0.097892** | **0.331359** | **0.805834** | **0.450425** | **0.009365** |

On seed 1872, the attacked policy calibration automatically reduced one client's overall alignment scale to zero, while the other clients remained active. This is promising safety behavior but is not yet a complete Byzantine/privacy theorem.

## No-explicit-shift control

Fresh seeds: **1881, 1882**.

| Method | JSD ↓ | EDI ↓ | Deletion ↓ | Insertion ↑ | Top-k ↑ |
|---|---:|---:|---:|---:|---:|
| Local | 0.124557 | 0.117692 | 0.378502 | 0.857963 | 0.464111 |
| xFedAlign | 0.120265 | 0.079971 | 0.303291 | 0.866286 | 0.501953 |
| **RWSSA** | **0.111702** | **0.057694** | **0.271104** | **0.871355** | **0.561523** |

Thus the smoke improvement is not restricted to an explicitly injected heterogeneity pattern.

## Paper-direction readiness score

This score is a go/no-go research score, **not an acceptance probability**.

| Dimension | Weight | Score |
|---|---:|---:|
| Problem significance | 15 | 13 |
| Originality / current-literature separation | 20 | 12 |
| Technical mechanism | 20 | 17 |
| Fresh-seed smoke evidence | 20 | 19 |
| Strength of current baselines | 10 | 8 |
| Theory path | 10 | 7 |
| Reproducibility / practicality | 5 | 4 |
| **Total** | **100** | **80** |

**Decision: clears the user's required >=75/100 paper-direction bar.**

## Why the novelty score is not higher

The components are not individually novel:

- xFedAlign already coordinates local explanations through a shared global attribution prior.
- iFLASH already performs quality-/faithfulness-aware aggregation of federated SHAP explanations.
- Recent metric-guided attribution-fusion work explicitly optimizes multiple explanation-quality metrics.
- Feature-level federated attribution methods exist for purposes such as drift analysis.

The candidate novelty must therefore be stated narrowly: **client-private, task-model-fidelity estimates are aggregated feature-wise to identify a shared explanation subspace, and each attribution coordinate is aligned toward the federated prior in proportion to evidence that doing so improves fidelity while still improving cross-client consistency.**

A targeted search did not identify that exact mechanism, but this is a **medium-risk novelty position**, not a novelty guarantee.

## Fatal issues that must be resolved before a full conference-scale run

1. **Real-data pilot first.** The current approval is based on a controlled synthetic image task. Do not jump directly to a multi-day full suite. The next gate should be one MNIST/Fashion-MNIST and one CIFAR-10 pilot seed using the actual sparse-artifact pipeline.
2. **Implement iFLASH and metric-guided/quality-aware competitors.** xFedAlign alone is no longer sufficient for the final baseline set.
3. **Communication/privacy.** Benefit maps expose more coordinate-level quality information unless quantized, sparsified, secure-aggregated, or otherwise protected. Measure communication and leakage.
4. **Separate policy-selection and certification data.** The final pipeline must preserve disjoint task-training, explainer fitting, artifact estimation, feature-benefit estimation, calibration/certification, and final evaluation data.
5. **Theory.** A useful theory target is concentration / sign-recovery of the aggregated feature-benefit score and how benefit-estimation error affects the alignment-vs-fidelity frontier. Do not claim CRC guarantees from the current cheap smoke selector unless the final selector is formulated correctly.
6. **Ablations.** Required: binary safe mask vs continuous weighting; no residual floor; residual-only; no private calibration; client-specific masks; random feature weights; global mean / median prior; iFLASH-style client-level quality weighting.

## Next gate

**APPROVE a medium-cost real-data pilot, not yet the days-long final study.**

The real-data pilot should be killed immediately if RWSSA fails to beat xFedAlign and quality-aware aggregation on at least four of the five metrics (JSD, EDI, deletion, insertion, top-k), or if its benefit-map communication/privacy cost is disproportionate to the gains.
