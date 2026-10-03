# Federated Learning on MNIST: Ten ICML 2026 Papers

Ten federated-learning papers published at the **43rd International Conference on Machine Learning (ICML 2026)**, Seoul, PMLR volume 306. Each camera-ready PDF evaluates the proposed method on the **MNIST** handwritten-digit dataset (LeCun et al.), not only on Fashion-MNIST, FEMNIST, or MNIST-M. Local copies are in [`papers/`](papers/).

These ten were chosen from 95 ICML 2026 papers whose titles contain “federated,” after checking the PDFs for a real MNIST experiment. They cover clustered personalization, heterogeneous architectures, HPC scheduling, Byzantine clients, performative bilevel learning, fairness, explainability, data skew, parameter-free compression, and adversarial generalization.

## Short introduction

All ten papers train a model across clients that keep their data local, and all of them test that idea on MNIST. They do not attack the same failure of federated learning. CLoVE groups clients that share a data distribution by clustering the losses of several candidate models, instead of assigning everyone to the single lowest-loss model. LEGO-FL lets clients use different network architectures by splitting models into blocks, picking a shared assembly, and grafting it onto each local model. FedQueue is for HPC sites whose jobs wait in a batch queue: it predicts the wait, drops updates that arrive too late, and down-weights stale ones. DeMoA keeps federated training robust when only some clients participate and some of them are Byzantine, by mixing fresh momentum from the sampled clients with cached momentum from the rest. Federated bilevel performative prediction handles the case where deploying the model changes the data clients will see next, and looks for a stable upper-level and lower-level solution under that feedback. FedEBA+ makes client accuracies more even without giving up average accuracy, using closed-form aggregation weights from a maximum-entropy objective. xFedAlign makes explanations agree across clients by sharing only a few top attributions and pulling local explanations toward a global prior, while the task model still trains as usual. FedVeer estimates how skewed each client is from model updates alone, choosing the neighborhood size so majority clients do not dominate the estimate. ParFreFL removes learning-rate tuning and cuts the extra communication of earlier parameter-free FL; its compressed variant sends momentum and error feedback as one vector. The last paper studies generalization of federated adversarial training: stronger attacks make SGD less stable, a Moreau envelope removes that dependence on the attack size, and zeroth-order updates cover clients that cannot backpropagate.

| # | Paper | MNIST setup in the paper | PDF |
| --- | --- | --- | --- |
| 1 | CLoVE | Supervised accuracy under label skew and other non-IID mixes, next to CIFAR-10 and FMNIST | [01_CLoVE_ICML2026.pdf](papers/01_CLoVE_ICML2026.pdf) |
| 2 | LEGO-FL | IID and non-IID accuracy, including a 100-client setting, next to CIFAR-10 and SVHN | [02_LEGO-FL_ICML2026.pdf](papers/02_LEGO-FL_ICML2026.pdf) |
| 3 | FedQueue | 4 clients, non-IID Dirichlet partitions, CNN, cross-entropy | [03_FedQueue_ICML2026.pdf](papers/03_FedQueue_ICML2026.pdf) |
| 4 | DeMoA | ConvNet, IID and non-IID, partial participation, Byzantine attacks | [04_DeMoA_ICML2026.pdf](papers/04_DeMoA_ICML2026.pdf) |
| 5 | Federated bilevel performative prediction | CNN, 10 clients, uniform weights, Dirichlet non-IID | [05_FedBiPP_ICML2026.pdf](papers/05_FedBiPP_ICML2026.pdf) |
| 6 | FedEBA+ | 2-hidden-layer MLP; shard and Dirichlet partitions; accuracy and client-variance plots | [06_FedEBA+_ICML2026.pdf](papers/06_FedEBA+_ICML2026.pdf) |
| 7 | xFedAlign | Vision benchmark; also the dataset used for explanation drift, membership inference, and attribution poisoning | [07_xFedAlign_ICML2026.pdf](papers/07_xFedAlign_ICML2026.pdf) |
| 8 | FedVeer | Compared with Fashion-MNIST, FEMNIST, and CIFAR-10, including noisy-defense and ablation tables | [08_FedVeer_ICML2026.pdf](papers/08_FedVeer_ICML2026.pdf) |
| 9 | ParFreFL / ComParFreFL | IID and non-IID accuracy vs. rounds and bits, CNN-scale workload | [09_ParFreFL_ICML2026.pdf](papers/09_ParFreFL_ICML2026.pdf) |
| 10 | FAL stability | 100 clients, IID partition, 50% participation; generalization gap vs. attack radius, local epochs, and client count | [10_FAL-stability_ICML2026.pdf](papers/10_FAL-stability_ICML2026.pdf) |

---

## 1. CLoVE: Clustering of Loss Vector Embeddings

- **Authors:** Randeep Bhatia, Nikos Papadis, Murali Kodialam, TV Lakshman, Sayak Chakrabarty
- **Venue:** ICML 2026, PMLR 306. [Proceedings page](https://proceedings.mlr.press/v306/bhatia26a.html)
- **PDF:** [papers/01_CLoVE_ICML2026.pdf](papers/01_CLoVE_ICML2026.pdf)

**Problem.** Personalized federated learning often assumes clients fall into unknown clusters that share a data distribution. Standard clustered FL either needs well-separated raw data (which MNIST digits do not have: centralized k-means only reaches about 50% adjusted Rand index) or, like IFCA, assigns every client to the single model with the lowest loss. On MNIST that assignment collapses: several true clusters all prefer the same initial model, the other models stop receiving updates, and the algorithm never recovers the clusters. IFCA also needs each initial model to start close to its cluster optimum, which is hard to guarantee.

**Approach.** CLoVE keeps several candidate models and, for each client, builds a **loss-vector embedding**: the vector of losses of those models on the client’s private data. Clients in the same cluster produce similar loss patterns even when their raw images overlap; clients in different clusters do not. The server clusters these vectors, aggregates updates inside each cluster, and repeats, so clustering and model fitting refine each other. The method applies to supervised and unsupervised tasks and does not need a near-optimal initialization. In a linear setting the authors prove high-probability cluster recovery in a single round and exponential convergence to the cluster optima.

---

## 2. LEGO-FL: Heterogeneous Models as a LEGO Assembly

- **Authors:** Zeqi Leng, Chunxu Zhang, Guodong Long, Bo Yang
- **Venue:** ICML 2026, PMLR 306. [Proceedings page](https://proceedings.mlr.press/v306/leng26b.html)
- **PDF:** [papers/02_LEGO-FL_ICML2026.pdf](papers/02_LEGO-FL_ICML2026.pdf)

**Problem.** Edge clients cannot all train the same network: compute and memory differ, so federated learning has to train **heterogeneous architectures**. Distillation, shared heads, and prototypes exchange predictions or features but waste structural diversity. Subnet methods (pruning a shared supernet) force every client model to be a slice of one template, so the server cannot really tailor a different topology to each device. Direct averaging is impossible once the parameter tensors no longer match.

**Approach.** LEGO-FL treats blocks of neurons as interchangeable pieces. It has three stages. (1) **Candidate generation:** group blocks into equivalence sets by representation similarity, then run a center-anchored tree search under non-strict ordering, with dynamic block completion so every path is a valid network. (2) **Training-free consensus:** pick one global architecture from that pool without a full extra training loop. (3) **Personalized grafting:** stitch the consensus network onto each client’s local model and distill the result. Clients therefore collaborate on modular pieces without sharing one fixed backbone. On MNIST, CIFAR-10, and SVHN, under IID and non-IID partitions and mixed CNN “model zoos,” LEGO-FL beats homogeneous FL baselines and prior heterogeneous methods, including at 100 clients.

---

## 3. FedQueue: Queue-Aware Federated Learning for HPC

- **Authors:** Yijiang Li, Emon Dey, Zilinghan Li, Krishnan Raghavan, Ravi Madduri, Kibaek Kim
- **Venue:** ICML 2026, PMLR 306. [Proceedings page](https://proceedings.mlr.press/v306/li26ae.html)
- **PDF:** [papers/03_FedQueue_ICML2026.pdf](papers/03_FedQueue_ICML2026.pdf)

**Problem.** When each “client” is an HPC facility, the batch scheduler (Slurm, PBS, and similar) often dominates wall-clock time: a job can wait minutes to hours before it starts, and the wait is stochastic. Synchronous FedAvg stalls on the slowest queue. Fully asynchronous FL keeps moving but aggregates updates computed from very old models when queues spike. Throughput-profiling methods such as FedCompass budget local steps from compute speed, but they assume availability is stable and do not control scheduler admission delay.

**Approach.** FedQueue treats predicted queue delay as an input to the protocol. Each facility runs an online predictor (an exponentially weighted moving average in the implementation; the analysis does not depend on that choice) and uses it to budget wall-clock time and the number of local steps. A **cutoff admission** rule buffers updates that arrive too late, so staleness stays bounded with high probability when prediction error is sub-Gaussian. Accepted updates are aggregated with staleness-aware, inverse learning-rate scaling. For non-convex objectives the convergence rate is \(O(1/\sqrt{R})\) under that staleness bound. A cross-facility deployment improves loss by 20.5% over baselines. On non-IID MNIST (4 clients, CNN), controlled queue simulations cut time-to-target accuracy by up to 60% when queue variance is high.

---

## 4. DeMoA: Delayed Momentum Aggregation

- **Authors:** Kaoru Otsuka, Yuki Takezawa, Makoto Yamada
- **Venue:** ICML 2026, PMLR 306. [Proceedings page](https://proceedings.mlr.press/v306/otsuka26a.html)
- **PDF:** [papers/04_DeMoA_ICML2026.pdf](papers/04_DeMoA_ICML2026.pdf)

**Problem.** Byzantine-robust federated learning usually assumes every client participates in every round, and that fewer than half of them are faulty. Partial participation breaks that assumption: the clients sampled in one round can be a Byzantine majority even if Byzantines are a minority of the whole population. Median-style aggregators then no longer approximate the honest average, and robust methods diverge. The few partial-participation defenses either need huge minibatches or do not handle these Byzantine-majority rounds.

**Approach.** DeMoA (Delayed Momentum Aggregation) changes what the server averages. Honest clients keep a local momentum buffer, which is what makes small time-coupled attacks (such as ALIE) separable from gradient noise. In a round, the server aggregates **fresh** momentum from the sampled clients together with the **most recently cached** momentum from clients that were not sampled. From the server’s point of view, Byzantines therefore stay a minority of the aggregated set whenever they are a minority of all clients, even if they dominate the sample. There is no extra communication. The paper proves convergence under standard smoothness, variance, and heterogeneity assumptions. On a ConvNet trained on MNIST (and ResNet-18 on CIFAR-10), with a 20% Byzantine ratio and only 10% participation, DeMoA remains accurate where FedAvg, FedCM, and prior robust methods collapse within a few epochs.

---

## 5. Federated Bilevel Performative Prediction

- **Authors:** Liangxin Qian, Chang Liu, Xuanyu Cao, Jun Zhao, Kwok-Yan Lam
- **Venue:** ICML 2026, PMLR 306. [Proceedings page](https://proceedings.mlr.press/v306/qian26b.html)
- **PDF:** [papers/05_FedBiPP_ICML2026.pdf](papers/05_FedBiPP_ICML2026.pdf)

**Problem.** Federated bilevel optimization (hyperparameter tuning, meta-learning, representation learning) assumes each client’s data distribution is fixed. **Performativity** violates that: deploying a decision changes user behavior and therefore the data the client will see next. Single-level performative FedAvg exists, and centralized bilevel performative prediction exists, but not the combination. Decision-dependent shift biases both gradients and hypergradients; clients differ in data and in how they respond to the deployed model; and the communication schedule itself changes the distributions, because redeployment is what induces the shift.

**Approach.** The paper defines a **federated bilevel performatively stable (FBPS)** point: a pair of upper-level and lower-level solutions that is stable when both risks are evaluated under the client-specific distributions induced by those decisions, rather than under a frozen dataset. It gives conditions for existence and uniqueness. Two algorithms compute that point. **FBi-RRM** is a repeated-retraining method with linear convergence when a contraction condition holds. **FBi-SGD** estimates the federated hypergradient with stochastic updates and converges under diminishing steps when the performative sensitivities are small enough. Strategic regression and meta strategic classification match the predicted stability thresholds. A non-convex check trains a CNN on MNIST split across 10 clients with Dirichlet heterogeneity and shows the performative methods improve over bilevel training that ignores the feedback loop.

---

## 6. FedEBA+: Fair Aggregation from Maximum Entropy

- **Authors:** Zhichao Wang, Lin Wang, Ye Shi, Sai Praneeth Karimireddy, Xiaoying Tang
- **Venue:** ICML 2026, PMLR 306. [Proceedings page](https://proceedings.mlr.press/v306/wang26ii.html). Code: [T-Lab-CUHKSZ/FedEBA-Plus](https://github.com/T-Lab-CUHKSZ/FedEBA-Plus)
- **PDF:** [papers/06_FedEBA+_ICML2026.pdf](papers/06_FedEBA+_ICML2026.pdf)

**Problem.** A single global model trained by FedAvg is accurate on average and poor for clients whose data differ from that average. Fairness here means client accuracies should not spread too widely. Existing fair FL methods split into two unsatisfactory groups: methods such as q-FFL and agnostic FL improve the worst client by giving up global accuracy, while methods that only implicitly reweight clients never directly control that variance, so the fairness they reach is weak.

**Approach.** FedEBA+ puts the maximum-entropy principle on the **aggregation distribution**, not on a uniform resource split. Entropy is maximized subject to a constraint that ties the aggregate to a fairness-aware objective, which yields a closed-form weight for each client proportional to its loss. High-entropy weights pull client accuracies toward each other without a hand-tuned fairness exponent. A stepwise alignment step then matches gradient directions across clients so that heterogeneous local updates do not undo the fair aggregate. A practical variant, Prac-FedEBA+, keeps communication comparable to FedAvg. Convergence holds for non-convex losses, and the variance (fairness) analysis extends beyond the usual strongly convex special cases. On MNIST the model is a two-hidden-layer MLP; the paper plots both accuracy and cross-client variance, with the same pattern on Fashion-MNIST, CIFAR-10, and CIFAR-100.

---

## 7. xFedAlign: Global–Local Attribution Alignment

- **Authors:** Dawood Wasif, Terrence J. Moore, Chang-Tien Lu, Jin-Hee Cho
- **Venue:** ICML 2026, PMLR 306. [Proceedings page](https://proceedings.mlr.press/v306/wasif26a.html)
- **PDF:** [papers/07_xFedAlign_ICML2026.pdf](papers/07_xFedAlign_ICML2026.pdf)

**Problem.** Federated models in healthcare, finance, and similar settings need explanations, but the server cannot see inputs. The same prediction can depend on different features at different clients (**explanation drift**). Purely local explanations never form a system-level account. Forcing one global explanation misrepresents client-specific boundaries. Sharing gradients, logits, or dense attribution maps is both a privacy risk and too large to ship every round. Replacing the task model with an interpretable surrogate usually costs accuracy.

**Approach.** xFedAlign separates task training from explanation coordination. Each client fits a small surrogate that mimics the deployed model and emits per-class attributions in a normalized, modality-agnostic group space. Those attributions are reduced to a private top-\(k\) artifact (clipping, quantization, light noise) of a few kilobytes. The server robustly aggregates them into a **Global Explanation Prior** and sends the prior back. Clients add a light alignment penalty so local explanations move toward the prior without changing the task-loss optimization. The task model can stay a normal network and can keep using FedAvg. On MNIST, xFedAlign matches FedAvg accuracy (about 0.986 IID and 0.930 non-IID) while lowering explanation disagreement and improving deletion/insertion AUC versus local XAI, aggregated attributions, and prior federated XAI. The same MNIST setup is used for membership-inference and attribution-poisoning tests.

---

## 8. FedVeer: Self-Adaptive Skew Estimation

- **Authors:** Yun Xin, Bangqi Pan, Jianfeng Lu, Shuqin Cao, Gang Li, Guanghui Wen
- **Venue:** ICML 2026, PMLR 306. [Proceedings page](https://proceedings.mlr.press/v306/xin26b.html)
- **PDF:** [papers/08_FedVeer_ICML2026.pdf](papers/08_FedVeer_ICML2026.pdf)

**Problem.** Label skew and feature skew make FedAvg converge to a model that fits majority clients and fails on the rest. Uploading prototypes or label histograms would quantify that skew but leaks information and costs bandwidth. Kernel density estimates on neighboring model updates avoid raw data, yet they use a **fixed** neighborhood. When most clients are skewed, that neighborhood is dominated by the majority, and balanced clients look like outliers. Stochastic updates and partial participation also perturb the kernel geometry, so the skew estimate jitters from round to round.

**Approach.** FedVeer estimates skew from model updates, but the neighborhood size \(k\) is not fixed. A max-margin objective picks \(k\) each round so the density estimate is not swallowed by the skewed majority; the authors prove that margin problem has a unique solution. Aggregation weights then come from the resulting kernel density. For noisy updates, a Kalman filter with an optimal gain tracks the margin, and a high-probability bound shows that filtering strictly shrinks the chance of a large deviation from the noise-free margin. No features or label counts are uploaded. Across MNIST, Fashion-MNIST, FEMNIST, and CIFAR-10, FedVeer improves accuracy by up to 6.36% over four baselines and cuts the accuracy drop under noisy updates by up to 6.01%.

---

## 9. ParFreFL: Parameter-Free Federated Learning under Compression

- **Authors:** Wenjing Yan, Xiangyu Zhong, Ying-Jun Angela Zhang
- **Venue:** ICML 2026, PMLR 306. [Proceedings page](https://proceedings.mlr.press/v306/yan26x.html)
- **PDF:** [papers/09_ParFreFL_ICML2026.pdf](papers/09_ParFreFL_ICML2026.pdf)

**Problem.** Learning rates, momentum, and compression ratios are painful to tune in federated learning because clients differ in data and network conditions, and a rate that works on one partition fails on another. PAdaMFed removes that tuning but **doubles** communication: every round, both sides exchange two model-sized tensors in each direction. Naively compressing both tensors couples their errors and can destabilize the parameter-free update. The paper’s target is a parameter-free method whose communication matches ordinary FL and whose guarantees do not depend on how hard the link is compressed.

**Approach.** **ParFreFL** keeps the parameter-free property and cuts PAdaMFed’s communication in half by no longer shipping both full tensors. **ComParFreFL** goes further: it folds the momentum increment and the compression error-feedback into one transmitted vector, so biased compressors are allowed without a second message. The convergence statement does not depend on the compression ratio, which the authors describe as the first such guarantee for compressed FL. The analysis allows arbitrary heterogeneity and partial participation, with linear speedup in the number of local steps and in the number of participating clients. On IID and non-IID MNIST (and FMNIST), the untuned methods match or beat tuned FedAvg, PAdaMFed, and compressed baselines while using fewer communicated bits.

---

## 10. Generalization of Federated Adversarial Learning

- **Authors:** Yongkang Yang, Chang Cao, Ke Zhang, Han Li, Hong Chen, Rushi Lan
- **Venue:** ICML 2026, PMLR 306. [Proceedings page](https://proceedings.mlr.press/v306/yang26f.html)
- **PDF:** [papers/10_FAL-stability_ICML2026.pdf](papers/10_FAL-stability_ICML2026.pdf)

**Problem.** Federated adversarial learning (FAL) trains a shared model against client-side adversarial perturbations, written as a min–max problem. Prior FAL theory is mostly about convergence. Empirically, robust **training** accuracy rises while robust **test** accuracy falls as training continues (robust overfitting). The inner attack is non-smooth, and its effective regularity depends on the attack radius \(\delta\). That dependence makes SGD less algorithmically stable, so generalization bounds get worse as attacks get stronger or as the number of rounds grows. Smoothing the adversarial loss still leaves a \(\delta\)-dependent term and needs gradients. Many clients cannot backpropagate at all: they only have black-box query access.

**Approach.** The paper analyzes FAL through **uniform algorithmic stability** rather than convergence alone. For standard federated SGD with a diminishing step size, it derives a perturbation-dependent generalization bound: larger \(\delta\) loosens the bound, matching the overfitting observation. It then replaces the adversarial objective with a **Moreau envelope**, whose stability bound no longer depends on \(\delta\), so robustness and generalization can improve together. The same argument is extended to **zeroth-order** updates, which estimate gradients from function queries and therefore cover clients without local backpropagation. Experiments on MNIST (100 clients, IID, half participating each round, 5 local epochs), plus CIFAR-10, SVHN, and large LIBSVM sets, plot the generalization gap against attack strength, local epochs, and the number of clients.

---

## Scope

All ten papers are main-conference ICML 2026 publications (PMLR 306, published 2026). ICLR 2026, CVPR 2026, and AAAI 2026 also have federated-learning papers, but many OpenReview PDFs are still marked “under review,” and several published papers that mention MNIST only evaluate Fashion-MNIST, FEMNIST, or MNIST-M. Those were not included. MNIST numbers above are taken from the camera-ready PDFs linked in this note.
