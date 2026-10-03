# Literature matrix

Status legend: **[read]** = PDF text inspected in this project; **[web]** = abstract/HTML page inspected via web fetch; **[search]** = only seen in search results (title/snippet), not verified in detail.

**Current audit:** Phase III (2026-10-02), §H below, supersedes earlier inspection-status/novelty statements for the works re-examined there. Earlier sections remain the historical record. Full assumptions, source-access limitations and search queries are in `phase3_central_literature.md` and `phase3_federated_literature.md`.

## A. Seed papers (all ICML 2026, PMLR 306; PDFs in `papers/`, text extracted and read in this project)

| # | Paper (as titled in PDF) | Setting / protocol | Core mechanism | Info sent beyond FedAvg | Theory | MNIST setup (from PDF) | Key unacknowledged assumptions that matter in deployment | Open questions exposed |
|---|---|---|---|---|---|---|---|---|
| 1 | CLoVE: Personalized FL through Clustering of Loss Vector Embeddings (Bhatia, Papadis, Kodialam, Lakshman, Chakrabarty) **[read]** | Clustered FL; K̂ models broadcast until assignment stabilises; fraction ρ participates | Each client reports vector of losses of all K̂ models on its data; server k-means on loss vectors + min-cost matching cluster→model | K̂ scalars/client + K̂× download until stability | One-shot cluster recovery + exponential convergence in mixed linear regression (needs ∆/σ≫1) | CNN, 20–1000 clients, 500 pts/client, label skew / rotations / label-permutation concept shift, 3 seeds | Raw (unnormalised) losses clustered: any per-client additive loss component unrelated to cluster identity (e.g. heterogeneous label noise) enters the embedding; membership fixed after "early stopping"; honest loss reports; privacy of loss vectors "underexplored" (App. C.2) | Behaviour of loss-vector geometry under heterogeneous data quality; drift after stability |
| 2 | LEGO-FL: Learning Heterogeneous Federated Models as a LEGO Assembly Games (Leng, Zhang, Long, Yang) **[read]** | Model-heterogeneous FL, 50–100 clients, 10% participation | CKA block grouping (needs server public data = 10% of train set), tree search, NASWOT-style training-free proxy, grafting + KD | Full local models / blocks to server | none | 50 clients, IID / 2-class pathological, model zoos of CNN/VGG | Server-side public dataset; server sees full client models; no FedAvg baseline in heterogeneous tables | Accuracy *decreases* with participation rate (Fig. 3a) — unexplained |
| 3 | FedQueue: Queue-Aware FL for Cross-Facility HPC Training (Li, Dey, Li, Raghavan, Madduri, Kim) **[read]** | 4–12 HPC facilities, batch-scheduler queue delays | EWMA queue prediction → per-facility time/step budgets; cutoff admission + buffering; harmonic staleness decay φ(τ)=1/(1+βτ); inverse-LR scaling η_k∝E_min/E_k | observed queue delay | O(1/√R) non-convex with bounded staleness; staleness bound w.h.p. under sub-Gaussian prediction error | 4 clients, Dirichlet α=0.5 (0.1 in App.), CNN, Adam, 50 rounds, seed 42 | Queue delay independent of data; global test set only (no per-facility metric); failed jobs not modelled | Persistent down-weighting of slow facilities (objective shift) not measured |
| 4 | Delayed Momentum Aggregation (DeMoA) (Otsuka, Takezawa, Yamada) **[read]** | Byzantine-robust FL with Bernoulli(p) partial participation | Server caches each client's last momentum, decays non-sampled ones by (1−α p), robust-aggregates all n | none (server stores n momenta) | Convergence to O(cδζ²) neighbourhood for any p | ConvNet, n=25, δ=0.2, p∈{0.5,0.1}, IID/non-IID, 5 attacks, seeds {0,1,2} | Homogeneous, exogenous participation probability p (same for all clients); per-client server state; individual vectors visible to server (no SecAgg) | Behaviour under heterogeneous / data-correlated p_i; adaptive attacks on cache (left as future work) |
| 5 | Federated Bilevel Performative Prediction (Qian, Liu, Cao, Zhao, Lam) **[read]** | Bilevel FL where UL/LL distributions depend on deployed decision | FBPS fixed point; FBi-RRM (contraction), FBi-SGD (hypergradient via auxiliary v) | gradients of 3 variables | existence/uniqueness, linear (RRM) and O(1/r) (SGD) under strong convexity + small sensitivities | CNN, 10 clients, Dirichlet 0.3, participation 0.5, ε_c=0.1, ε_d=0.05 | Distribution maps known/simulated; immediate synchronous redeployment; strong convexity | Lagged/heterogeneous deployment; fairness of feedback (App. K) |
| 6 | FedEBA+: Towards Fair and Effective FL via Entropy-Based Model (Wang, Wang, Shi, Karimireddy, Tang) **[read]** | Client-level performance fairness | Max-entropy aggregation p_i ∝ exp(F_i/τ) over sampled set + step-wise gradient alignment toward entropy-weighted "fair gradient" | per-client loss (and 1-step gradient in full version) | non-convex convergence; variance ≤ FedAvg in GLM / strongly convex-with-outlier | 2-hidden-layer MLP, 100 clients, 10/round, 2 shards/client & Dirichlet; K=10 local steps | **Loss is a faithful proxy for "under-served"**; App. E.1 toy example argues up-weighting a client whose data are pure noise *reduces* disparity; noisy-label experiment corrupts *all* clients (ε=0.5), never a mix of noisy and clean-minority clients; honest loss reports. App. E (verified in text 2026-10-02): optional robust variant for noisy labels — weights exp(F_i^r/τ) with F_i^r = CE + γ·self-distillation loss on augmented data (Local Self-Regularization, after Jiang et al. 2022); still a CE-level statistic; **not implemented in this project — required baseline** | Allocation when high loss is irreducible (noise) vs reducible (under-representation) |
| 7 | Explainable FL via Global–Local Attribution Alignment (xFedAlign) (Wasif, Moore, Lu, Cho) **[read]** | XAI coordination, 8 clients, 15 rounds | surrogate distillation, group-space IG attributions, top-k noisy artifacts, coordinate-median Global Explanation Prior, JSD alignment of *surrogate* | top-k attribution artifacts (KB) | dispersion contraction (App. A) | small CNN, 8 clients, Dirichlet 0.1, seed 1337 | Alignment acts on surrogate, not task model; EDI measured against each method's own reference | Whether aligned explanations stay faithful under genuine heterogeneity |
| 8 | FedVeer: Self-Adaptive Skew Estimation (Xin, Pan, Lu, Cao, Li, Wen) **[read]** | Skew-aware aggregation from model updates | kNN kernel density of updates with max-margin choice of k; Kalman filter on margin | none | uniqueness of max-margin k; McDiarmid bound on margin deviation | 60 clients (20 non-skewed in noisy test), MNIST/FMNIST/FEMNIST/CIFAR-10 + 34 CEKA tabular sets | Server knows each client's *data source* s (indicator 1[n∈s] in Eq. 10); "good" clients assumed to form the densest region | Majority-skew regimes where density ≠ quality |
| 9 | Robust Federated Learning Against Adaptive Compression (ParFreFL / ComParFreFL) (Yan, Zhong, Zhang) **[read]** | Parameter-free FL with compression | normalised local momentum steps; SCAFFOLD-style per-client control variates c_i; compress δ_i = m_i − c_i (momentum increment + EF in one vector) | 1 model-sized vector | O((∆+L+σ)/(SKT)^{1/4}) for ParFreFL; compression only in lower-order terms | MNIST/FMNIST IID/non-IID, Top-k 0.1/0.05 | Step sizes depend on horizon T, S, K ("parameter-free" ≠ horizon-free); uniform sampling; persistent per-client server state | Behaviour under unknown T, churn, rare participation |
| 10 | Towards Understanding Generalization of Federated Adversarial Learning: Perspective of Algorithmic Stability (Yang, Cao, Zhang, Li, Chen, Lan) **[read]** | FAL generalisation | uniform-stability bounds; Moreau envelope (FalME) removes δ dependence; zeroth-order (FalZO) | none | O(((nN)^{-1}L + L_zδ)T^{1/2}log T) for SGD-FAL; δ-free for FalME | 100 clients, IID, 50% participation, 5 local epochs | IID partitions; sample-level (not client-level/participation-gap) generalisation | Generalisation to unseen clients under adversarial training |

## B. External literature relevant to the selected problem area (noise-vs-under-representation confounding in loss-driven FL)

| Work | Status | Mechanism | Relation to our candidates |
|---|---|---|---|
| Ghosh, Kumar, Sastry, "Robust Loss Functions under Label Noise for Deep Neural Networks", AAAI 2017 (arXiv 1712.09482) | [web] | Thm 1: for symmetric losses (Σ_i L(f(x),i)=C), R^η(f) = Cη/(k−1) + (1 − ηk/(k−1))·R(f), η<(k−1)/k | **The affine identity used by A′ and B is this known theorem specialised to 0-1 error (C=k−1).** Our contribution can only be its use for FL client statistics (allocation, clustering embeddings, η estimation), not the identity |
| Manwani & Sastry, "Noise Tolerance Under Risk Minimization", IEEE Trans. Cybernetics 2013 | [search] | 0-1 loss risk minimisation is tolerant to uniform noise | precursor of the above |
| Agarwal & Zhang, "Minimax Regret Optimization for Robust Machine Learning under Distribution Shift", COLT 2022 (PMLR v178) | [web] | DRO over total risk is dominated by high-noise distributions (their Prop. 1); minimise worst-case *regret* R_P(f) − inf_f' R_P(f') | **Centralised precedent of the normative idea behind A′ (equalise excess, not total, risk).** No FL, no per-client estimator, generic loss |
| Zhang, Bai, Tu, Yang, Hu, "Efficient Stochastic Approximation of Minimax Excess Risk Optimization" (MERO), ICML 2024 | [web] | stochastic estimation of each distribution's minimal risk, then biased convex-concave SA | centralised MERO algorithm; A′ can be read as a cheap *federated, decision-level* regret estimator |
| q-FFL — Li, Sanjabi, Beirami, Smith, "Fair Resource Allocation in Federated Learning", ICLR 2020 | [search] (OpenReview PDF listed) | weights ∝ F_i^q | loss-driven allocation (failure target) |
| AFL — Mohri, Sivek, Suresh, "Agnostic Federated Learning", ICML 2019 | known; not re-inspected | minimax over client mixture | loss-driven allocation |
| TERM — Li, Beirami, Sanjabi, Smith, "Tilted Empirical Risk Minimization", ICLR 2021 | known; not re-inspected | positive tilt (fairness) / negative tilt (robustness); hierarchical tilt needs groups | closest centralised analogue of "fair + robust" |
| Power-of-Choice — Cho, Wang, Joshi, AISTATS 2022 "Towards Understanding Biased Client Selection in FL" | [search] (PMLR v151 PDF listed) | select highest-loss clients among candidates | loss-driven selection (failure target) |
| RHO-LOSS — Mindermann et al., "Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt", ICML 2022 (PMLR v162) | [search] | sample-level reducible holdout loss using an irreducible-loss model | centralised principle that Candidate A lifts to the client level |
| FedPCA — Wu, Yan, Sang, Yu, Chen, "FedPCA: Noise-Robust Fair Federated Learning via Performance-Capacity Analysis", arXiv 2503.10567 (2025) | [web] (HTML v1 re-checked 2026-10-02) | 3-component GMM on per-client (loss, feature-dispersion score) pairs to split mislabeled / rare / common clients (dispersion computed over feature clusters; the paper does not say explicitly whether clusters follow given or predicted labels); weights w = N̂·r·exp(−q·S) / Σ (N̂ = reliable-sample count, r = label reliability, S = dispersion); scores and weights smoothed over time; SAM in local training; 10-round FedAvg warm-up; q per dataset (CIFAR-10 3, RSNA ICH 1, ISIC 2019 0.4). Datasets: CIFAR-10 (50 clients), RSNA ICH and ISIC 2019 (20 clients each). Rare clients = Gaussian noise added to the images of 10 % (CIFAR-10) / 20 % (RSNA, ISIC) of clients, giving 8:2 / 9:1 data-scale imbalance across distributions; mislabeled clients = random label flips in a subset (ρ, η) of common clients | **Closest prior for Candidate A — same three-group problem as ours** (common, covariate-shifted rare, mislabeled; "rare and mislabeled data clients exhibit similar loss trends"). Difference: signal (CE loss + feature dispersion + population GMM + per-dataset q) vs our held-out decision-level reducibility of the client's own update. *Correction 2026-10-02: an earlier version of this row said covariate-shift rarity was not tested — wrong; FedPCA's rare clients are covariate-shifted.* |
| DQFed — Usman, Bernardi, Cimitile, "Introducing a Quality-Driven Approach for Federated Learning", Sensors 25(10):3083, 2025 | [web] | label-entropy weights (needs label histogram) + supervised-VAE noise penalty | distinguishes imbalance vs noise with reported label statistics |
| DQA client selection — Song, Li, Wan, Fu, Jiang, "Data Quality-Aware Client Selection in Heterogeneous FL", Mathematics 12(20):3229, 2024 | [web] | rate of *training*-loss reduction during local epochs as quality | training-loss (not held-out) learnability signal; no fairness |
| Wilhelm, Yilmaz, Kao, "Noise-aware Client Selection for carbon-efficient FL via Gradient Norm Thresholding", arXiv 2603.04194 | [web] | probing-round gradient-norm utility threshold | notes loss-based selection picks noisy clients; does not examine rare clients |
| FedNoRo (IJCAI 2023), FedCorr (CVPR 2022), FedFixer, FedELC, FedSIR (2026), "Masked optimization" (2506.02079) | [search] | noisy-client detection mostly by population-relative outlier statistics (loss GMM, LID, spectra) | risk of flagging rare clean clients; assume clean majority |
| RCC-PFL — "Robust Client Clustering under Noisy Labels in Personalized FL", arXiv 2503.19886 | [web] | label-agnostic clustering from HOG-feature covariance eigen-structure; shows IFCA degrades under label noise | **Closest prior for Candidate B** (noise-robust clustered FL), different signal (features, not losses) |
| Scott, Luo, Ho, "Different Corruptions, Different Signals: Uncertainty and Loss in Federated Data Quality", arXiv 2609.31454 (25 Sep 2026) | [web] (abstract, 2026-10-02) | compares input-conditional uncertainty (aleatoric variance, MC-dropout, entropy) with prediction-label loss as per-sample corruption detectors in FL. Loss detects persistent label flips (AUC 0.85 CIFAR-10, 0.95 SVHN) where uncertainty is at chance; entropy detects image noise (0.66–0.67). ResNet-20, Dirichlet non-IID | **Closest neighbour of our cross-family diagnosis** (signal–corruption matching). Different question: per-sample detection, not loss-driven allocation / selection / clustering; no calibration confound of reducibility; no fairness. Must be cited |
| FedCova, "Robust Federated Covariance Learning Against Noisy Labels", arXiv 2603.04062 | [search] | noisy-label FL method (title/snippet only) | not inspected; check before submission |
| FedLAW (Li et al., ICML 2023 "Revisiting weighted aggregation in FL with neural networks") | [search] | learns aggregation weights on server proxy data | validation-driven weights (needs server data) |
| FedHAW — Nakai-Kasai & Wadayama, arXiv 2605.00458 (2026) | [web] | hypergradient online update of aggregation coefficients without validation data | learned aggregation weights; no noise/fairness |
| Fairis — Trivedi et al., arXiv 2608.06469 | [web] | absolute-score fairness weights with bounded influence vs fairness poisoning in FairFed | manipulation of reported statistics in fairness-aware aggregation |
| BaFFLe / client-feedback validation | [search] | clients evaluate global model to detect backdoors | client-side evaluation of candidate models (Candidate C precedent for the *signal*, not the purpose) |

## C. External literature checked for discarded directions (see research_state.md §Discarded)

| Work | Status | Why it matters |
|---|---|---|
| FedStale — Angelo Rodio & Giovanni Neglia, "FedStale: leveraging stale client updates in federated learning", arXiv 2405.04171 (May 2024) | [web] | fixed β convex combination of fresh/stale updates; "stale updates help with homogeneous participation + high data heterogeneity" |
| FedSteer — Zhang, Pereira, Siew, Liu, Joe-Wong, El-Azouzi, arXiv 2606.10124 | [web] | corrective subspace projection for stale cached gradients, heterogeneous p_i |
| FedAU — Wang & Ji, ICLR 2024, "A Lightweight Method for Tackling Unknown Participation Statistics in Federated Averaging" | [search] | online participation-interval estimation and reweighting |
| Who Trains Matters (FedIPW), arXiv 2604.26604 | [web] | inverse-probability weighting for enrollment + participation bias |
| Behfar & Mortier, "Cumulative Utility Parity for Fair FL under Intermittent Client Participation", arXiv 2602.13651 (Feb 2026) | [web] | exogenous intermittent participation; surrogate updates from cached models with decaying confidence |
| Wen et al., "Rethinking Byzantine Robustness in Dynamic FL: Participation-Aware Threats and a Dynamic-Aware Defense", IEEE TDSC 2026 | [web] | participation-driven fluctuations of malicious ratio |
| Hashimoto, Srivastava, Namkoong, Liang, ICML 2018 "Fairness Without Demographics in Repeated Loss Minimization" | known; not re-inspected | retention-driven disparity amplification, DRO remedy (centralised) |
| Chen, Li, Zhang, "Asymptotically Fair Participation in ML Models: an Optimal Control Perspective", arXiv 2311.10223 | [web] | churn feedback loop, optimal control (centralised, not FL) |
| Performative FL — Jin et al., AAAI 2024 | [search] | single-level performative FedAvg |

## D. Venue review criteria (official pages, fetched 2026-10-02)

| Venue | Page | What reviewers score |
|---|---|---|
| NeurIPS 2026 | https://neurips.cc/Conferences/2026/ReviewerGuidelines [web] | Quality, Clarity, Significance, Originality |
| ICML 2026 | https://icml.cc/Conferences/2026/ReviewerInstructions [web] | Soundness, Presentation, Significance, Originality (each 1–4); Overall 1–6; Confidence 1–5; checks that claims are well supported, that authors are honest about weaknesses, and that limitations / negative societal impact are discussed |
| ICLR 2026 | https://iclr.cc/Conferences/2026/ReviewerGuide [web] | Four questions: the specific problem tackled; motivation and placement in the literature; whether the claims are supported (correct, scientifically rigorous); significance (new knowledge of value to the community) |

## E. Validation datasets named in the plan (not re-inspected this session unless marked)

| Dataset | Status | Why |
|---|---|---|
| CIFAR-10N — Wei et al., "Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations", ICLR 2022 | known; not re-inspected | real annotator label sets with different noise rates → natural per-client label-quality heterogeneity |
| ISIC 2019 and RSNA ICH as used by FedPCA (20 clients each) | [web] via FedPCA | allows a head-to-head in FedPCA's own setting |
| FLamby Fed-ISIC2019 (natural multi-centre split) | known; not re-inspected | real site heterogeneity |
| FEMNIST (LEAF) | known; not re-inspected | natural writer clients, small n_i |

## F. Searches for the conflict signature T1′ (2026-10-02)

| Query | Result |
|---|---|
| "federated learning detect label flipping clients error rate worse than random chance symmetric label noise distinguish" | noisy-client detection (FedSIR, FedELC, masked optimisation, FedNoRo-style) and the Scott et al. paper above; no use of above-chance error as a routing signal |
| "clustered federated learning concept shift label permutation noisy labels distinguish systematic relabeling from random label noise" | FedSIR, FedDiv, FedCova, reviews; no clustered-FL routing by above-chance error |
| Caveat | label-flipping *attack* detection (e.g. update-space PCA) is a related literature, not searched exhaustively here |

## G. Phase II novelty audit (2026-10-02)

Searches were run by three literature sub-agents (WebSearch / WebFetch); the closest priors were then re-fetched by hand.

**Tags:**
- **[web-verified]**: re-fetched in the main session, and the stated mechanism was read on the page;
- **[agent-read]**: read by a sub-agent through a summarising fetch tool, not re-checked by hand. Spot-check formulas before citing;
- **[abstract]**: abstract only.

### G1. The statistical principle (noisy vs clean 0–1 risk; CE vs bounded statistics)

| Work | Tag | Relevant content | Bearing on our claim |
|---|---|---|---|
| Ghosh, Kumar, Sastry, "Robust Loss Functions under Label Noise for Deep Neural Networks", AAAI 2017 (arXiv 1712.09482) | [web] (session 1) | Thm 1: for symmetric losses, R^η = Cη/(K−1) + (1 − ηK/(K−1))R, η < (K−1)/K. The 0–1 loss (C = K−1) and MAE are symmetric; CE is not | The affine identity and threshold are **known** |
| Toner & Storkey, "Noisy Early Stopping for Noisy Labels", arXiv 2409.06830 (Sep 2024; no venue found) | [web-verified] | Fact 1: R^η(q) = R(q)(1 − cη/(c−1)) + η under uniform symmetric class-preserving noise, so model selection on noisy accuracy picks the clean-best model. Fact 2: this is the *only* noise model for which the noisy and clean minimisers coincide for every estimator set and distribution. Pairwise noise breaks it (decision-tree depth chosen on noisy accuracy is 18 % worse) | Our Prop. 1–2 (affine identity; exact proportionality only under symmetric noise) are **known** in the model-selection setting. Closest overlap with the "statistical principle" |
| Chen, Ye, Chen, Zhao, Heng, "Robustness of Accuracy Metric and its Inspirations in Learning with Noisy Labels", AAAI 2021 (arXiv 2012.04193) | [web-verified] | Assumption 2 (diagonal dominance). App. A Eq. 12: max noisy accuracy − noisy accuracy(h) = Σ_{i≠j} Pr[Y=i](T_ii − T_ij)C_ij(h). Eq. 13 bounds clean error by that gap / min margin. A noisy validation set is reliable | Our Lemma 1 (margin form) is the pairwise version of their Eq. 12, so it is **known** in substance |
| Natarajan, Dhillon, Ravikumar, Tewari, NeurIPS 2013 | [agent-read] | Binary class-conditional noise: the noisy α-weighted 0–1 risk is affine in clean risk (Thm 9) | Binary precursor |
| Menon, van Rooyen, Ong, Williamson, ICML 2015 | [agent-read] | Under class-conditional corruption, balanced error and AUC are affine in their clean versions; plain error is not | Precursor for class-conditional noise |
| Lam & Stork, IJCAI 2003, "Evaluating Classifiers by Means of Test Data with Noisy Labels" | [agent-read] | Correcting the error rate on a noisy test set; for validation "just their ordering" matters | Evaluation-side precursor |
| Olmin & Lindsten, AISTATS 2022 | [agent-read] | Strictly proper losses on noisy labels converge to the noisy posterior: same boundary, miscalibrated for clean labels | Background for the CE/calibration point |
| Patrini et al., CVPR 2017; Ma et al., ICML 2020; Zhang & Sabuncu, NeurIPS 2018; Lukasik et al., ICML 2020 | [agent-read] | Loss correction, normalised losses (CE not robust), GCE, label smoothing ↔ noise | Training-loss literature. Our statistics use these losses as *valuation statistics of client updates* |
| TransTS (Sci. China Inf. Sci. 2026); "Attended Temperature Scaling" (arXiv 1810.11586) | [agent-read] [abstract] | Temperature scaling is distorted when the calibration labels are noisy | Background for TS-local vs TS-global |
| Scott, Luo & Ho, "Different Corruptions, Different Signals", arXiv 2609.31454 (Sep 2026) | [web] (session 2) | In FL, per-sample loss detects label flips; uncertainty detects image noise | Per-sample detection, not client-level valuation |

**Verdict on (1).**
- Known: the identity, the threshold, the ranking preservation, the uniqueness of symmetric noise and the margin form.
- Not found in these searches:
  - the federated *client-statistic* framing: per-client offsets cancel in before/after differences, attenuated by a(η_i);
  - the dissociation that CE/Brier *reductions* misrank clients while TS-local CE fixes only symmetric noise;
  - the finding that every *level* statistic fails.
- Novelty risk **high** for the theory, **medium** for the federated empirical characterisation.

### G2. A′ (held-out before/after improvement as aggregation weight)

| Work | Tag | Mechanism | Overlap with A′ | Difference |
|---|---|---|---|---|
| Zhang, Xu, Yao, Zhang, Tian, Wang, "Federated Domain Generalization With Generalization Adjustment" (FedGA), CVPR 2023 | [web-verified] | Gap = loss of the global model − loss of the client's previous local model, **on the client's training data**. Weights are moved toward larger gaps with a decaying step (a_i ← a_i + (G_i − μ)d_r/max(G − μ)) and normalised | **High**: shifts global aggregation weight toward clients whose loss the global model leaves most reducible | Training data, not a holdout; CE loss; additive momentum update; domain generalisation; no label noise |
| Zhang, Sapra, Fidler, Yeung, Alvarez, "Personalized Federated Learning with First Order Model Optimization" (FedFomo), ICLR 2021 | [web-verified] | w_n = (L_i(θ_i^{t−1}) − L_i(θ_n^t))/‖θ_n − θ_i‖ on **a held-out validation split of client i's own data**, then max(·, 0) and normalised | **High** for the *held-out improvement → weight* mechanism | Personalised (scores other clients' models for client i); loss; no noise |
| Fang & Ye, "Robust Federated Learning with Noisy and Heterogeneous Clients" (RHFL), CVPR 2022 | [web-verified] | Client confidence = (1 / symmetric-loss level) × (drop of SL loss between rounds) on the client's private noisy data; softmax-weighted | **Medium-high**: loss *drop* as a weight, explicitly against noisy clients | Training data, round-to-round; × 1/loss (which would also penalise hard clean clients); distillation setting |
| Graves et al., "Automated Curriculum Learning for Neural Networks", ICML 2017 | [agent-read] | Held-out "prediction gain" L(θ) − L(θ′) as the reward of an Exp3.S bandit over tasks | Exponential weights from held-out learning progress | Single learner, task sampling |
| DoReMi (Xie et al., NeurIPS 2023); RHO-LOSS (Mindermann et al., ICML 2022) | [agent-read] | Clipped excess loss over a reference model → exponentiated-gradient domain weights / per-example priority | Reducible vs irreducible loss | Needs a reference model; not federated |
| Zeno / Zeno++ (Xie et al., ICML 2019 / 2020); FLTrust (NDSS 2021); FedVal (USENIX Sec 2023); ClipFL; GTG-Shapley; FedCE (CVPR 2023); ARFL | [agent-read] | Server-side validation-loss drop or accuracy, trust scores, contribution valuation | Validation-based scoring of updates | Server or clean data; robustness or valuation goals |
| FedPCA (arXiv 2503.10567) | [web] | Same rare-vs-mislabelled problem; GMM on (loss, feature dispersion) | Same goal | Different signal |
| FedEBA+ (arXiv 2301.12407) | [web] (seed paper) | p_i ∝ exp(F_i/τ) | Same weight form | Loss level |

**Verdict on (2).** A′'s mechanism is the published "weight clients by validation improvement" family (FedGA, FedFomo, RHFL), with a held-out split of the client's own data and a bounded statistic. The only remaining difference is *which statistic*. Stage 7 shows that choice is not specific to 0–1 (MAE and clipped CE do as well). Novelty risk **high**.

### G3. B (affine-invariant error-vector clustering)

| Work | Tag | Mechanism | Overlap | Difference |
|---|---|---|---|---|
| CLoVE, arXiv 2506.22427 (ICML 2026 seed paper) | [web-verified] (abstract and method) | k-means on per-client vectors of losses under the K cluster models + matching; theory with √loss; no label noise | **High**: identical pipeline | z-normalised 0–1 vectors, which are invariant to per-client symmetric noise in expectation |
| Gu, Chen, Wen, Cai, Han, "Novel clustered federated learning based on local loss" (LCFL), arXiv 2407.09360 | [web-verified] | Client distance built from differences of local losses of locally trained models | Within-client loss *differences* (removes a client's own offset) | CE, no noise model, no normalisation of scale |
| IFCA (Ghosh et al., NeurIPS 2020) | [agent-read] | argmin-loss assignment | Same signal family | Argmin, no embedding; argmin of 0–1 is noise-invariant by T1 → tested in Stage 11 |
| RCC-PFL (arXiv 2503.19886); FB-NLL (arXiv 2604.19729) | [agent-read] | Label-free clustering from feature covariance spectra under noisy labels; FB-NLL also uses a clean server set | Same problem | Label-free. [interp] cannot separate clusters that differ only in labelling function (our "perm" clusters, relabelled clients) |
| "Selective collaboration": Xu et al., "A two-stage federated learning method for personalization via selective collaboration" (FedSC), Computer Communications 232, 2025; Bao, Wang, Wu, He, "Optimizing the Collaboration Structure in Cross-Silo Federated Learning" (FedCollab), ICML 2023 | [web-verified] (abstracts) | Choose collaborators from data-subspace principal vectors (FedSC) or from distribution distance and data quantity (FedCollab) | Who should share a model with whom | No loss or error statistics; no label noise. Label-quality heterogeneity would not enter their similarity at all |
| FedNoRo / FedELC (per-class loss vectors, min-max normalised); FedCorr, FedDiv, FedGR, FedA3I (per-client noise fractions from loss GMMs) | [agent-read] | Normalised client loss vectors to detect noisy clients; per-client noise-rate estimates | Normalised loss vectors; η̂ | Clean vs noisy detection, not cluster discovery; η̂ from GMM fractions, not a 0–1 offset |

**Verdict on (3).** B is CLoVE with a different statistic and per-client normalisation. Its genuine differences are:
- invariance of the representation to per-client symmetric noise;
- an η̂ by-product;
- the ability, unlike label-free clustering, to separate labelling functions.

Novelty risk **medium-high**, conditional on Stage 11 (does argmin-0–1 already achieve the same?).

**Could not verify** (from the sub-agents):
- CLoVE's ICML 2026 acceptance outside our seed set;
- the venues of FedGR, FedSIR, FedRG, FedCova, LCFL and FB-NLL;
- FedClean's aggregation rule;
- which "Selective Collaboration" paper the brief meant: two candidates were found and verified (FedSC 2025, FedCollab ICML 2023; row above).

## H. Phase III — identifiability and minimal information (2026-10-02)

Question: **what clean client-update utility is identifiable from the permitted federated observations?** Main target is `R_Q(theta)-R_Q(theta+a Delta_i)`, not an unspecified quality score. Threat A (honest corruption) is separate from strategic reporting. The new report's observation model and target control all comparisons.

### H1. Centralized identification, crowdsourcing and decisions

| Question | Closest prior result | Assumptions | Observables | Identifiable quantity | Limitation relevant to FL |
|---|---|---|---|---|---|
| Unknown transition from one label? | [Liu, Cheng & Zhang, ICML2023](https://proceedings.mlr.press/v202/liu23g/liu23g.pdf) [read] | Recovery uses informative independent views/structure | Joint noisy labels at an instance | Channel/posterior up to permutation | Repeated probes are not independent annotations; semantic utility is not permutation invariant |
| Asymmetric decomposition? | [Scott, Blanchard & Handy, COLT2013](https://proceedings.mlr.press/v30/Scott13.pdf) [read] | Irreducibility and contamination restrictions | Noisy class-conditional distributions | Distinguished latent decomposition | Low forward rates alone do not imply anchors/irreducibility |
| Need a whole channel for a decision? | [Menon et al., ICML2015](https://proceedings.mlr.press/v37/menon15.pdf) [read] | Binary contamination with positive scale | Corrupted distributions | Balanced-error/AUC ordering | Not arbitrary clean global risk; unknown per-client scales can reorder gains |
| Clean loss correction? | [Natarajan et al., NeurIPS2013](https://proceedings.neurips.cc/paper_files/paper/2013/file/3871bd64012152bfb53fdf04b401193f-Paper.pdf) [read] | Known nonsingular binary channel | Noisy labels and model predictions | Unbiased clean expected losses | Update-contrast correction is an immediate consequence |
| Partial labels and loss operators? | [Cid-Sueiro, NeurIPS2012](https://papers.neurips.cc/paper/4789-proper-losses-for-learning-from-partial-labels.pdf) [read] | Appropriate mixing matrix/loss transformation | Weak labels | Proper expected losses | Target-specific column-space recovery is elementary operator algebra |
| General decision impossibility? | [van Rooyen & Williamson, JMLR2018](https://jmlr.org/papers/volume18/16-315/16-315.pdf) [read] | Statistical decision/corruption experiments | Noisy observations, multiple corrupted sources | Loss recovery and lower bounds | Same-input adaptive transcript theorem is a data-processing corollary |
| Annotator reliability? | [Dawid–Skene1979](https://doi.org/10.2307/2346806) [abstract]; [Zhang et al., NeurIPS2014](https://proceedings.neurips.cc/paper_files/paper/2014/file/700c82b01b20a5e3e160dab2e511c48a-Paper.pdf) [read] | Informative independent groups, rank, positive priors and orientation for recovery | Overlapping item responses | Confusion matrices/latent labels | Private data disjointness is not shared-item annotation overlap |
| Multi-view identification? | [Allman, Matias & Rhodes2009](https://www.cs.uaf.edu/~jrhodes/papers/Latent.pdf) [read] | CI views and Kruskal-rank conditions | Joint multi-view law | Factors up to permutation | Orientation and prediction dependence remain |
| Agreement moments? | [Fu et al., ICML2020](https://proceedings.mlr.press/v119/fu20a/fu20a.pdf) [read] | Independent nondegenerate triplets/sign resolution | Weak-source moments | Accuracy magnitudes/model parameters | Square-root agreement inversion is not new |
| Peer-only evaluation? | [Liu & Helmbold, AISTATS2020](https://proceedings.mlr.press/v108/liu20d/liu20d.pdf); [Liu, Wang & Chen, EC2020](https://arxiv.org/pdf/1802.09158) [read] | Informative reference/score calibration; the latter uses orientation and task assumptions | Peer reports over shared tasks | Scores/regret relative to hidden truth | One-bit orientation and relabelled counterfactual worlds are explicit prior art |
| Complementary/noisy labels? | [Ishida et al., ICML2019](https://proceedings.mlr.press/v97/ishida19a/ishida19a.pdf) [read] | Known complementary-label design | Systematically wrong labels | Arbitrary-loss risk/validation | High corruption is not inherently non-identifiable |
| Recent IDN/CCN alternatives? | [Causal IDN2025](https://arxiv.org/html/2412.13516v2); [multinomial mixture2025 version](https://arxiv.org/html/2301.01405v3); [CI mixture2026](https://arxiv.org/html/2604.07191v1); [MIND2026](https://arxiv.org/pdf/2605.16081) [relevant theory read] | Causal factors/repeated labels/feature CI/anchors, respectively | Structured features and labels | Model-specific channels/mixtures | No unrestricted single-noisy-view FL recovery; see memo for theorem qualifications |

### H2. Federated reliability — newly inspected mechanisms

| Question | Closest prior result | Assumptions | Observables | Identifiable quantity / estimator | Limitation relevant to FL |
|---|---|---|---|---|---|
| Is federated reliability identifiability already explicit? | [FedDS2026](https://discovery.ucl.ac.uk/id/eprint/10223584/1/1-s2.0-S0020025526003567-main.pdf) [method/theory read] | DS CI, fixed confusion matrices, factorization/orientation conditions | Uploaded models on common unlabeled set | Prediction confusion matrices; mean diagonal weights | Direct novelty collision. Unknown-existence DD wording admits permutation counterexample; not marginal update utility |
| Can clean calibration recover noise rate? | [FNRE2026](https://file.techscience.com/files/onlinefirst/2026/4.8/TSP_CMC_75102/TSP_CMC_75102.pdf) [method read] | Calibration transfer; uniform wrong-label/agreement approximation | Clean counts and local model-label agreement | Client corruption estimates | Pooled accuracy need not transfer to rare clients; dependence matters |
| Do class-subspace signals identify noisy clients? | [FedSIR2026](https://arxiv.org/html/2604.20825v1) [method read] | Distinct clean geometry, useful references | Class-subspace descriptors and models | Relative noisy-client detection/correction | Descriptors invariant to a class-name permutation at fixed features; not clean-utility identification |
| Does shared proxy similarity handle structured corruption? | [FedSNC2026](https://doi.org/10.1016/j.neucom.2026.133647) [abstract/introduction only] | Proxy structure and group interpretation | Proxy features/client similarities | Groups/corrected labels | Pairwise-noise experiments overlap proposed mechanism; full method/guarantees remain unavailable |
| Can covariance replace losses? | [FedCova2026](https://arxiv.org/html/2603.04062v1) [method read] | Gaussian-mixture representation/class structure | Covariances, class priors and models | Classifier/corrected labels | Geometry does not uniquely name latent semantic classes |
| Do trajectories identify transitions? | [FedEFC, WACV2026](https://openaccess.thecvf.com/content/WACV2026/papers/Yu_FedEFC_Federated_Learning_Using_Enhanced_Forward_Correction_Against_Noisy_Labels_WACV_2026_paper.pdf) [method read] | CCN, useful pre-memorization global model | Accuracy reports, confidence/counts, updates | Transition estimates for forward correction | Correct-T consistency does not prove T's estimator uniquely identifies it |
| Loss grouping/soft-label correction? | [FedELC2024](https://arxiv.org/html/2408.04301v1) [method read] | Useful warm-up and loss separation | Per-class losses, models, local label variables | Relative groups/corrected labels | Own-label dependence persists; no general identifiability result found |
| Correction without clean-client identification? | [FedClean, ICML2025](https://proceedings.mlr.press/v267/jiang25m.html) [method read] | Useful local/global learners/correction | Selected sample counts and models | Corrected samples and sample-count aggregation | No wholly clean client required is not assumption-free semantic recovery; distinguished from Byzantine FedCLEAN |
| Signal-corruption mismatch? | [Scott, Luo & Ho2026](https://arxiv.org/html/2609.31454v1) [protocol read] | Particular empirical corruption designs | Prediction-label loss and uncertainty probes | Detection statistics | Non-IID detection confounding already recognized; not clean global update utility |
| Current noisy-FL landscape? | [Dong et al., Neural Networks2026](https://openaccess.city.ac.uk/id/eprint/37433/1/1-s2.0-S0893608026003503-main.pdf) [relevant survey read] | Survey | Method taxonomy | No new identifiable parameter | Reliability measures and hard/noisy separation are established concerns |
| Verified scores under malicious reporting? | [Zeno, ICML2019](https://proceedings.mlr.press/v97/xie19b/xie19b.pdf); [FLTrust, NDSS2021](https://www.ndss-symposium.org/wp-content/uploads/ndss2021_6C-2_24434_paper.pdf) [method read] | Trusted objective/root source | Submitted updates and server evaluation | Descent/trust proxies | Distinct Byzantine setting; trusted target coverage still matters |

### H3. Real-noise and deployment interpretation

[CIFAR-N](https://arxiv.org/html/2110.12088v2), [Clothing1M](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Xiao_Learning_From_Massive_2015_CVPR_paper.pdf), [institutional sepsis coding comparison](https://pubmed.ncbi.nlm.nih.gov/30431493/), [PheNorm](https://groups.csail.mit.edu/medg/ftp/psz-papers/J%20Am%20Med%20Inform%20Assoc%202017%20Yu.pdf), [Snorkel](https://arxiv.org/abs/1711.10160), and an [actual six-hospital RACOON FL study](https://academic.oup.com/jamia/article/32/1/193/7841978) document plausible human/weak/site-dependent annotation mechanisms. The [2026 medical real-noise FL benchmark](https://arxiv.org/html/2606.16868v1) was also inspected. Exact access and limitations are in `phase3_phase2_bridge.md` §6. Dataset annotator positions are not automatically fixed clients; fused/clinical labels are reference definitions, not infallible truth. None verifies the exact counterexample in production.

### H4. Novelty conclusion

Outcome D for the **tested formulation**: the broad impossibility is a centralized statistical/data-processing consequence; decision-specific recovery, independent-view reliability, semantic orientation and trusted correction have close prior art; FedDS directly occupies FL reliability identification. The clean-global-utility target distinction is valuable but no new FL-specific recovery theorem follows merely by renaming the target. The FedDS quantifier qualification is classical label-switching logic, not sufficient evidence of a new research program. FedSNC full access and exhaustive priority remain limitations. The Phase III decision is not to authorize this new direction; A′/B and large-scale work remain closed.

---

## Phase V addendum — explanation alignment / UCPA neighborhood (2026-10-02)

| Work | What it does | Relation to UCPA | Current collision judgment |
|---|---|---|---|
| Wasif et al., **xFedAlign**, ICML 2026 | Robustly aggregates sparse per-class attribution artifacts into one Global Explanation Prior and softly aligns local explanations | Same direct problem family; UCPA instead forms a personalized continuous peer prior using whole-explanation similarity and coordinate sampling uncertainty | **Closest direct neighbor; no exact collision found; medium novelty risk** |
| Zhang & Yu, **UncertainXFL**, 2025 | Generates concept/logical-rule explanations with uncertainty, ranks/selects reliable rules, and uses explanation reliability in model aggregation | Shares uncertainty + federated explanations, but uncertainty is rule/data confidence rather than sampling variance of attribution summaries; no per-client coordinate peer prior | Important conceptual neighbor, not direct mechanism collision |
| Hoefler et al., **FedXDS**, ICCV 2025 | Uses attribution to select task-relevant data elements for privacy-preserving sharing under heterogeneity | Explanations guide data sharing rather than explanation coordination | Different problem/mechanism |
| Gholizade et al., **FedXAI review**, 2026 | Taxonomy/review; highlights non-IID explanation behavior, stability, privacy and communication as open challenges | Supports motivation and benchmark gaps | No method collision |
| Ma et al., **PFedAtt**, ACML 2021 and later personalized-FL similarity/community methods | Similarity/attention/community-based collaboration for personalized models | Establishes that peer-selective collaboration itself is not novel | Narrows novelty claim to explanation-specific two-scale uncertainty-compatible coordination |

**Novelty wording to preserve:** do not claim similarity-aware collaboration, kernel smoothing, uncertainty weighting, personalization, JSD, or standardized variance tests as individually new. The candidate contribution is a federated explanation alignment rule that combines whole-explanation compatibility with coordinate-level sampling uncertainty so a client can borrow broadly in shared dimensions/regions and narrowly under genuine explanation heterogeneity.
