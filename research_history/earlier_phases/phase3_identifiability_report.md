# Phase III: what can a federated server know about clean update usefulness?

2026-10-02. Continuation of the existing project. Phase I/II implementations, reports and raw results were inspected; their experiments were not restarted. The prior large-scale gate remains closed. A′ and B remain rejected as method contributions.

**Decision: outcome D for the direction tested here.** The ambiguity is real, and a clean two-client counterexample survives even with low class-conditional corruption and an unidentified perfectly clean client. However, its mathematical content is substantially covered by noisy-label identification, statistical decision theory and crowdsourcing. FedDS already explicitly addresses federated reliability identifiability. No novel FL-specific recovery theorem or practical mechanism was established. This is a useful limitations synthesis and correction of the research record, not evidence for a new method.

Evidence labels: **[proof]** derived and checked here, without a novelty claim; **[known]** supported by linked primary literature; **[obs]** computed from recorded experiments; **[inference]** interpretation. Access limitations and search queries are recorded in [central literature audit](phase3_central_literature.md), [federated literature audit](phase3_federated_literature.md), and [Phase II bridge](phase3_phase2_bridge.md).

## 1. Question, protocol and threat models

There are n clients. Client i has clean law P_i(X,Y), observed label channel T_i(x)[y,z]=P(tilde Y=z | X=x,Y=y), and only observes samples (X,tilde Y). The server has current model theta, a specified evaluation target Q(X,Y), and an application scale a. Neither Q nor a is inferred implicitly from who happens to participate. A common choice is a declared mixture Q=sum_i lambda_i P_i; another is an external target population.

**Threat A, primary:** clients execute the specified training, evaluation and reporting honestly; clean labels are latent. Symmetric, class-conditional, instance-dependent and deterministic channels are admissible unless a proposition restricts them. Training and evaluation noise may differ, so an evaluation channel must be stated separately when used for correction.

**Threat B, separate extension:** a client may deliberately change reports or updates. No result about honest corrupted labels below is asserted to be Byzantine robust. An honestly computed statistic can still lack semantic information; authenticating its computation is not the same as verifying its clean meaning.

Let Z include the **joint** initially available data: client noisy datasets, public inputs, initial models, accessible pretrained models, metadata and any prior semantic information. Honest protocol randomness is independent of the latent world given Z, and the protocol is the same in both worlds. Let H be the full interactive transcript. Results comparing worlds require equal laws of all this available information, not just equal marginal client losses. A pretrained model with an assumed clean-competence guarantee is additional information/structure and must be included in the assumptions.

### Observation regimes

O1–O4 can be made cumulative by retaining earlier messages. They need not be strictly richer: if the server already has an executable model and the same reference inputs, predictions can be computed from O2. O5 and O6 are **alternative extensions**, not an automatic O5-subset-of-O6 hierarchy.

| Regime | Permitted observations | Added cost and privacy implications | Information boundary |
|---|---|---|---|
| O1 | One specified honest scalar s_i from own noisy labels: loss level or before/after difference | O(1) scalars/client; repeated queries still reveal information about private data | Scalar depends on both clean distribution and label channel; honesty alone gives no clean interpretation |
| O2 | O1 plus individual update Delta_i and protocol metadata | O(d) parameters/client, usual model-update costs; individual updates are unavailable under pure sum-only secure aggregation | Can reveal much more about observed training data, but cannot create unobserved clean semantics |
| O3 | O2 plus responses/prediction vectors from several specified server model/probe states | Typically O(kd) model downloads and O(kbC) prediction uploads, or O(k) loss summaries; costs depend on where probes run | A prediction vector requires explicit inputs and any local adaptation. Own-label risk vectors and prediction vectors are different packets; neither implies independent annotation |
| O4 | O3 plus jointly aligned predictions/agreement on a common unlabeled reference source R_X | b shared inputs or seed/access; O(nbC) predictions, or O(n²) pairwise summaries; exposes per-model behavior | Provides cross-model dependence/overlap, but no semantic orientation by itself; pairwise agreement is less than the full joint law |
| O5 | Earlier regime plus a trusted clean evaluation source, possibly resident at another client | b clean annotations; candidate model delivery/inference; m paired score sums and sample counts can replace raw-label uploads | Identifies target risk contrasts only with appropriate target sampling/transport and adequate support |
| O6 | Earlier regime plus multiple genuine noisy annotations/independent views of shared latent outcomes | Annotation/linkage costs and O(rb) labels, possibly aggregated sufficient moments; joint alignment can expose relationships | Recovery needs an identifiable multi-view model, orientation and target coverage; more correlated predictions are not equivalent |

Here d is parameter count, C classes, k probes, b reference items, m candidates and r annotators. These are order-of-magnitude communication counts, not measured network benchmarks. None of these protocols is automatically differentially private. Secure aggregation/DP may restrict the attainable observation regime; the no-information result remains valid under further processing.

## 2. Define the decision before estimating quality

For a fixed loss ell and target Q, let R_Q(theta)=E_Q ell(f_theta(X),Y). The primary target is

\[
U_i(Q,theta,a)=R_Q(theta)-R_Q(theta+a\Delta_i).
\]

Positive U_i means this particular candidate update improves the declared clean target. The smallest decision target for accept/reject is sign(U_i); to choose between two candidates it is sign(U_i-U_j). Exact risk, an entire noise matrix, or a client's clean/noisy status is unnecessary if the relevant contrast is identified.

| Candidate quantity | Definition | Does it answer the aggregation/routing decision? |
|---|---|---|
| Own clean risk | R_{P_i}(theta) | Measures local performance/need, not marginal global benefit |
| Own clean gain | R_{P_i}(theta)-R_{P_i}(theta+a Delta_i) | Appropriate for own-target routing only; does not automatically transfer to Q |
| Global clean gain | U_i above | Direct for a fixed candidate and applied scale; main target here |
| Gradient alignment | A_i=-grad R_Q(theta)^T Delta_i | First-order proxy; for L-smooth risk, U_i >= a A_i - L a² norm(Delta_i)²/2 |
| Label corruption rate | eta_i=P_i(tilde Y!=Y) | Neither model competence nor clean update usefulness; low noise can coexist with useless/redundant updates |
| Transition matrix | T_i(x) | Can support loss correction, but is more information than some decisions require |
| Improvement probability | P_training[U_i>0] | Requires a specified training/sample/randomness distribution; one realized update does not estimate it universally |
| Marginal contribution | R_Q(theta+a sum_{j in S}w_j Delta_j)-R_Q(theta+a sum_{j in S}w_j Delta_j+a w_i Delta_i) | Depends on S, weights and interactions; individual U_i values alone do not optimize a joint aggregate |

The CPU experiments use **binary scalar Brier loss (p-y)²**, not the two-coordinate sum convention, and fixed candidate updates. Their non-Brier scores are proxies for that declared decision. Phase II used 0–1 targets and is reported separately.

## 3. Fresh novelty audit

The following table condenses the two source audits. Those files record exact queries, versions, inspected sections and unavailable material. Absence from a search is not a proof of novelty.

| Question | Closest prior result | Assumptions | Observables | Identifiable quantity | Limitation relevant to FL |
|---|---|---|---|---|---|
| Can arbitrary noisy labels identify their channel? | [Liu, Cheng & Zhang, ICML 2023](https://proceedings.mlr.press/v202/liu23g/liu23g.pdf) | Informative independent views for recovery; single-view counterexamples | Joint noisy-label law at an instance | Channel/posterior up to permutation | Probes of one learner are not fresh annotations; permutation-free utility needs semantics |
| What rules out alternative asymmetric decompositions? | [Scott, Blanchard & Handy, COLT 2013](https://proceedings.mlr.press/v30/Scott13.pdf) | Mutual irreducibility and noise restrictions | Contaminated class-conditional distributions | Distinguished clean decomposition | Neither non-IID clients nor low forward flip rates imply irreducibility |
| Can a decision be identified without a full channel? | [Menon et al., ICML 2015](https://proceedings.mlr.press/v37/menon15.pdf) | Binary contamination model and positive ordering factor | Corrupted evaluation distributions | Balanced-error/AUC ordering | Not arbitrary target risk or cross-client unknown-scale comparisons |
| Are correction and two-world lower bounds new? | [van Rooyen & Williamson, JMLR 2018](https://jmlr.org/papers/volume18/16-315/16-315.pdf) | Reconstructible corruption for correction; decision-experiment framework for bounds | Corrupted statistical experiments | Expected losses and decision limits | Adaptive noisy-data-derived FL messages fit a standard data-processing argument |
| Can known noisy labels estimate a clean contrast? | [Natarajan et al., NeurIPS 2013](https://proceedings.neurips.cc/paper_files/paper/2013/file/3871bd64012152bfb53fdf04b401193f-Paper.pdf); [Cid-Sueiro, NeurIPS 2012](https://papers.neurips.cc/paper/4789-proper-losses-for-learning-from-partial-labels.pdf) | Suitable known channel/loss operator | Noisy/weak labels and predictions | Clean expected loss | Before/after subtraction is immediate; target coverage remains necessary |
| Can overlapping workers identify reliability? | [Dawid & Skene, 1979](https://doi.org/10.2307/2346806); [Zhang et al., NeurIPS 2014](https://proceedings.neurips.cc/paper_files/paper/2014/file/700c82b01b20a5e3e160dab2e511c48a-Paper.pdf) | Latent-class model; recovery adds rank, informative groups and orientation | Responses on shared items | Confusion matrices and latent labels | Disjoint private examples do not constitute shared-item responses |
| What do three views establish? | [Allman, Matias & Rhodes, 2009](https://www.cs.uaf.edu/~jrhodes/papers/Latent.pdf); [Fu et al., ICML 2020](https://proceedings.mlr.press/v119/fu20a/fu20a.pdf) | Conditional independence and nondegeneracy; semantic sign resolution | Joint views/agreement moments | Latent factors or accuracy magnitudes | Independence is substantive; orientation is separate |
| Can experts be scored without truth? | [Liu & Helmbold, AISTATS 2020](https://proceedings.mlr.press/v108/liu20d/liu20d.pdf); [Liu, Wang & Chen, EC 2020](https://arxiv.org/pdf/1802.09158) | Informative peer reference and calibration/orientation conditions | Repeated peer reports | Hidden-outcome scores/regret | The latter explicitly discusses counterfactual relabelling and an orientation bit; this is not a new minimal signal |
| Are systematically wrong labels necessarily uninformative? | [Ishida et al., ICML 2019](https://proceedings.mlr.press/v97/ishida19a/ishida19a.pdf) | Known complementary-label sampling design | Complementary labels | Ordinary risk for arbitrary losses | Channel knowledge matters more than raw error rate |
| Has FL reliability identifiability already been studied? | [FedDS, 2026](https://discovery.ucl.ac.uk/id/eprint/10223584/1/1-s2.0-S0020025526003567-main.pdf) | DS conditional independence plus identification/orientation assumptions | Uploaded models' predictions on shared unlabeled inputs | Prediction confusion matrices, macro-accuracy weights | Direct FL prior; not clean marginal update utility; see §7 |
| Can a clean calibration client estimate label noise? | [FNRE, 2026](https://file.techscience.com/files/onlinefirst/2026/4.8/TSP_CMC_75102/TSP_CMC_75102.pdf) | Calibration transfer and agreement/noise assumptions | Clean correct/total counts plus local prediction-label agreement | Local noise-rate estimates | A pooled clean accuracy is not automatically each rare client's clean accuracy |
| Are geometric/proxy signals new? | [FedSIR](https://arxiv.org/html/2604.20825v1), [FedSNC](https://doi.org/10.1016/j.neucom.2026.133647), [FedCova](https://arxiv.org/html/2603.04062v1) | Useful class geometry/proxy structure | Subspaces, proxy features, covariances | Detection/correction/classifiers | No general clean-utility identification established; FedSNC inspected only at abstract/introduction level |
| Are temporal/correction signals new? | [FedEFC, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/papers/Yu_FedEFC_Federated_Learning_Using_Enhanced_Forward_Correction_Against_Noisy_Labels_WACV_2026_paper.pdf), [FedELC](https://arxiv.org/html/2408.04301v1), [FedClean](https://proceedings.mlr.press/v267/jiang25m.html) | Useful early predictions, loss separation or correction assumptions | Model trajectories, losses, inferred labels | Transition/corrected labels or selected sample sets | Repeated use of corrupted supervision is not fresh semantic evidence |
| Has signal/corruption mismatch already been diagnosed? | [Scott, Luo & Ho, 2026](https://arxiv.org/html/2609.31454v1); [Dong et al. survey, 2026](https://openaccess.city.ac.uk/id/eprint/37433/1/1-s2.0-S0893608026003503-main.pdf) | Empirical corruption designs / review | Loss, uncertainty and client diagnostics | Detection comparisons / taxonomy | Recognizes confounding; does not make generic identification assumptions disappear |

Recent class-conditional/instance-dependent work was also inspected: [causal IDN identification, AAAI 2025](https://arxiv.org/html/2412.13516v2), [multinomial-mixture identification, 2025 version](https://arxiv.org/html/2301.01405v3), [feature-independence mixture estimation, 2026](https://arxiv.org/html/2604.07191v1), and [MIND, 2026](https://arxiv.org/pdf/2605.16081). They introduce structure such as latent factors, repeated labels, feature independence or anchors. None establishes unrestricted recovery from the original single noisy observation. Detailed qualifications are in the central audit; author claims were not silently elevated into verified general theorems.

## 4. Impossibility under honest corruption

### Proposition 1: identical inputs imply identical adaptive transcripts [known principle; proof here]

Let W and W′ give the same joint law of Z. For any fixed honest protocol using only Z and world-independent randomness,

\[
\mathcal L(H\mid W)=\mathcal L(H\mid W').
\]

**Proof.** The protocol induces the same Markov kernel K(dh|z) in either world. Integrating it against equal input laws gives equal transcript laws. Equivalently, condition on equal histories: adaptive server queries have the same law, and honest replies use the same conditional computation. Induct over rounds. This covers unlimited finite rounds and the induced infinite transcript when well-defined. Model updates, scalar losses, generated probes, reference predictions and trajectories are all included if they introduce no external semantic observation. This is ordinary statistical data processing, not a novel FL theorem.

If a binary decision differs between W and W′, any possibly randomized rule based on H has average success at most 1/2 under the equal prior on those two worlds. Therefore its worst-world error is at least 1/2. If U(W)=u and U(W′)=v, its worst-world expected absolute error is at least |u-v|/2, by the triangle inequality under the common output distribution. This is a worst-case limitation over the admissible worlds, not a prediction of chance accuracy on every benchmark.

### Proposition 2: low noise and an unknown clean client do not suffice [proof]

Use binary classification, uninformative X, independent client samples, and the same clean target Y~Bernoulli(p) for both clients. For forward channel parameters alpha=P(tilde Y=1|Y=0) and beta=P(tilde Y=0|Y=1), q=alpha+(1-alpha-beta)p.

| World | Clean p | Client 1 (alpha,beta) | Client 2 (alpha,beta) | Observed (q1,q2) |
|---|---:|---|---|---|
| W− | .4 | (1/3,0) | (0,0) | (.6,.4) |
| W+ | .6 | (0,0) | (0,1/3) | (.6,.4) |

Every channel is strictly diagonally dominant, every conditional flip rate is at most 1/3, and each world contains a perfectly clean client. The identity of that client is unknown. The noisy dataset laws match for **every sample size**, jointly across clients. The clean posteriors differ without needing channels worse than random.

Let baseline probability a0=.5 and honest population local Brier minimizers a1=.6, a2=.4. Since R_p(a)=p(1-a)²+(1-p)a²,

\[
(U_1,U_2)(W_-)=(-.03,.01),\qquad
(U_1,U_2)(W_+)=(.01,-.03).
\]

Both utility signs and their ordering reverse. Even observing all noisy samples, model parameters and arbitrary honest probes does not determine the correct choice. The worst-world absolute estimation error for either utility is at least .02. Empirical local minimizers have random values instead; transcript equality remains exact, while these fixed utility numbers apply to the population candidates. An actual empirical-mean update example is checked in §9.

The corresponding deterministic 0–1 candidates h1=1, h2=0 also reverse clean ordering. Relative to h0=0, client 1 has utility 2p-1=−.2 or +.2; client 2 has zero utility. Thus the obstruction is not peculiar to Brier loss.

**Scope.** This is an unrestricted impossibility witnessed by a simple admissible submodel. It does not assert that all useful data have uninformative features, or that known clean anchors cannot work. It is an elementary noisy-mixture counterexample; Scott/Liu and the broader corruption decision framework are the closest mathematical prior art.

## 5. What is minimally needed for a decision?

### Identification of a contrast, not necessarily a whole world

For an observation law o and model class A, define the identified set

\[
\mathcal I_U(o)=\{U(W):W\in A,\ \mathcal L(O\mid W)=o\}.
\]

Exact U is identified iff this set is a singleton. Accept/reject is identified if its sign is constant; pairwise ranking uses the same definition for U_i-U_j. This is the precise necessary-and-sufficient criterion, not a new algorithm. A signal is sufficient when it removes every decision-changing ambiguity; there is no universal smallest signal independent of the candidate set, loss, target and admissible channels.

### Proposition 3: known-channel functional recovery [elementary; prior loss-correction theory]

At a fixed x, write clean posterior p, observed posterior q=T^T p, and clean loss contrast d_y=ell(f_before(x),y)-ell(f_after(x),y). For known row-stochastic T, the contrast d^T p is identifiable **uniformly over the simplex** iff

\[
d\in\operatorname{col}(T).
\]

**Sufficiency.** Solve Tg=d. Then d^T p=g^T q, so the observed label score g_{tilde Y} has the desired conditional expectation.

**Necessity.** If d is outside col(T), there exists v in ker(T^T) with d^T v≠0. Since T1=1, 1^T v=0. Perturb an interior posterior to p±epsilon v, remaining in the simplex. They have equal q and different contrasts. Boundary-specific restrictions can change necessity; the statement is uniform over unrestricted posteriors.

With known Q_X and adequate support, integrate the recovered contrast over Q_X **provided the recovered posterior p(x) is Q(Y|x), or a justified conditional-label transport relation is available**. A known client channel, support and target feature weights alone do not transport arbitrary P_i(Y|x) to Q(Y|x). For unrestricted instance-dependent posteriors, require the column-space condition Q_X-almost everywhere. Full matrix invertibility is sufficient but stronger than needed. Example: T has rows (1,0,0),(1,0,0),(0,1,0). It cannot distinguish classes 1 and 2, but identifies any d=(a,a,b). This is operator/loss-correction algebra, not an established new FL contribution. Unknown T is a different inverse problem; the proposition does not estimate it.

### Trusted target evaluation and sample complexity

For m fixed candidates independent of a fresh clean audit set of b IID examples from Q, let D_i=ell_before-ell_after in [-1,1]. Paired sample averages estimate U_i directly. Hoeffding and a union bound give

\[
P(\max_i|\widehat U_i-U_i|>\epsilon)\le 2m\exp(-b\epsilon^2/2).
\]

Thus b≥2 epsilon^{-2} log(2m/delta) suffices for simultaneous error at most epsilon. Signs with |U_i|>epsilon and rankings with gaps >2epsilon are certified. This is standard validation, not a new estimator. Reusing the same audit set to adaptively choose candidates requires additional control or fresh data. CE needs clipping or tail assumptions because it is not bounded.

If the clean source is S rather than Q, transport needs appropriate support and a justified density ratio/conditional-label invariance. A clean source with no rare-region support cannot determine candidate utility there. A fixed small validation realization is not exact knowledge of an arbitrary population; identification refers to the observation-law model, while reliable estimation additionally needs samples and a margin.

### Assumption ladder

| Available information / assumption | What becomes identifiable? | What remains impossible? | Necessity, realism, checkability and costs |
|---|---|---|---|
| Honest O1–O4; unrestricted unknown channels | Observed behavior and noisy objective quantities | Clean utility/sign/rank in general | No additional communication can repair missing semantics if it only processes the same data |
| More samples/probes/rounds from that same law | Noisy quantities more precisely | Proposition 2 ambiguity | Temporal stability is checkable for observed data, not proof of correct labels |
| Binary symmetric rate eta<.5, same evaluation population | Sign/order of 0–1 differences for that source | Exact scale; cross-client magnitudes with unequal unknown rates; global transfer | Sufficient, not necessary; threshold alone is not enough for arbitrary asymmetric channels; cannot verify clean noise rate from one view |
| Known common positive affine risk factor | Cross-candidate ordering and gain sign | Absolute gain if scale unknown | Homogeneous corruption/target is substantive; no extra messages if justified externally |
| Strict diagonal dominance plus some unknown clean client | Not sufficient in general | Proposition 2 remains | Low corruption and a clean majority must not be silently assumed equivalent to an identified reference |
| Known evaluation channel and estimable loss contrast, with target posterior or justified conditional-label transport | Exact clean contrast on covered target | Other functionals outside col(T); unsupported target regions or different label semantics | Requires calibration/structural identification; correction variance can explode near singularity |
| Known clean representative source | Clean target contrasts, asymptotically; finite-sample bounds above | Uncovered target/novel candidates after uncontrolled adaptive reuse | Clean status is an external assumption; requires annotation, model evaluation and protected score transmission |
| Known clean source on disjoint support | Clean utility only there | Rare-region/global utility without transport | Support can be audited partially from features; semantic transport cannot be established from unlabeled agreement alone |
| Three informative CI views, positive class weights, sufficient rank | Latent-class parameters up to permutation in the applicable model | Semantic orientation; arbitrary prediction dependence | Sufficient route, not universally necessary. Shared items cost linkage/annotation; observed model fit is not proof of latent CI |
| Above plus a valid known orientation constraint and target coverage | Semantic reliability and corrected target contrasts under that model | Other tasks/populations or violated dependence model | A named dominant source, appropriate majority constraint, or anchors can orient; “some unknown good client” may not |
| Anchors/feature structure/known annotation design | Quantities allowed by that particular model | General IDN and unsupported transport | Convenient sufficient restrictions in many papers, not a universal minimal condition |

Two qualifications matter. First, a **known** binary symmetric eta>.5 is invertible and recoverable; eta=.5 loses label information. Second, the multiclass pair-noise margin changing sign at .5 is a risk-ordering phenomenon, not a universal rank-singularity/identifiability threshold. A known deterministic permutation is perfectly invertible. “High noise” and “unknown semantics” are different obstacles.

## 6. Connection to Phase II, including corrections

The [independent bridge](phase3_phase2_bridge.md) and [audit script](phase3_phase2_audit.py) recompute the delivered per-probe records. Stored score comparisons match exactly; this was not a retraining run, and cached per-example logits were unavailable. Original logs remain unchanged.

| Phase II finding | Recomputed evidence | Correct Phase III interpretation |
|---|---|---|
| Loss-level failures | All eight level AUROCs are 0 for RR versus selected symmetric-noise clients at round 150 | Perfect reverse discrimination in that comparison, **not proof of zero information** |
| CE/Brier reductions misrank | RR-versus-corrupted AUROC .2361/.4340; MAE/clipped CE 1.0000 | Estimator/target choice matters in that mixture; another observable already performs better, so this is not an all-estimator impossibility |
| Strong own/noise scores fail to predict global gain | MAE/clipped-CE Spearman with U_fed .0685/.0833; useful/harmful AUROC .5173/.5322 | Noise status, own benefit and global benefit are different quantities |
| Own/global disagreement | 130/336 round-150 probes have opposite U_own/U_fed signs; 119 corrupted probes have positive U_fed | Target and application scale both change; small empirical 0–1 gains limit interpretation |
| Pair noise above .5 | Wrong-sign fractions 7/36 at .55, 14/36 at .6, 36/36 at .8 and 1.0; MAE pair-high AUROC .7465 | Negative cell margins remove a universal ordering guarantee; not every update/statistic fails immediately above .5 |
| Deterministic relabelling | All 54 CF probes have observed positive gain and negative full-step own utility; gain=Delta D to 1.1e−16 | Fits semantic mismatch, but Phase II alone did not construct indistinguishable worlds |
| Claimed one-for-one CF clean harm | max absolute(observed gain + clean holdout gain)=.26 | Correction: gain=Delta D does not imply gain=−clean gain for arbitrary multiclass changes; third wrong classes matter |
| Heterogeneous symmetric noise | Known per-client positive/negative affine factors explain attenuation/reversal | Unknown positive scales can reverse cross-client rankings; offset cancellation is insufficient |
| Successful MAE/clipped CE | Reproduced, including finite-holdout variance advantages | Boundedness alone is insufficient: Brier is bounded too. MAE has symmetric-loss structure; clipped CE success is empirical |
| Chance-level guard | 512 recorded CF flags, no non-CF flag keys; blind to pair .75 | Useful under the learned model's competence/orientation assumptions; not universal semantics or guaranteed sensitivity over every possible CF exposure |

The Phase-II U_own uses a full local step on the own distribution; U_fed uses a one-tenth step on a 50:50 common/rotated target. At round 150 the median absolute U_fed is .000875. Differences cannot be attributed entirely to distribution shift or entirely to update scale, and finite test resolution matters. The old statement describing Brier as an “unbounded proper score” is incorrect; the update to `research_state.md` explicitly supersedes it.

**Answer to estimator versus information failure:** symmetric-noise CE failures often demonstrate a poor proxy, because other statistics succeed on the same observations. Unknown-channel ambiguity is demonstrated only by the new matched worlds. Structured-noise empirical failures are consistent with that mechanism but do not establish it for every Phase-II population. Both types of failure occur, and they must remain distinct.

## 7. FedDS and Dawid–Skene comparison

FedDS is the closest FL identification precedent. It treats uploaded local models as annotators of a shared unlabeled public set and uses DS estimation; the quantity weighted is average diagonal prediction reliability. This is reference-distribution prediction competence, not a private label-noise fraction or the marginal utility of a particular update. Its full procedure and source locations are documented in the [federated audit](phase3_federated_literature.md).

There are two separate theoretical obligations: (i) identify the latent factorization up to a common permutation using sufficient informative independent views; (ii) orient classes to their meanings. Model errors can be correlated given class because of shared initialization, common difficult examples or training feedback. Disjoint training records do not imply this independence. Rare reference classes have low precision; absent classes have unobserved confusion rows. A clean specialist may be weak on the reference population without having corrupted labels. Pair/deterministic corruption can be represented by a confusion matrix; it does **not automatically violate DS**. Orientation, dependence and rank are the specific issues.

**Counterexample to the literal unknown-existence orientation condition [proof; audit of FedDS §4.2].** Let Y be balanced and three views be conditionally independent binary symmetric predictors with accuracies (.9,.1,.8). Relabel latent Y′=1−Y, retaining every observed prediction. Accuracies become (.1,.9,.2). Their full joint observed distributions are equal and every channel is full rank. Both worlds contain at least one diagonally dominant client, but which one is dominant changes. Semantic reliability rankings reverse.

A **specified** client's known dominance can orient an already identified factorization. Existence of an unspecified dominant client cannot, as this example shows. Furthermore, one informative predictor plus any number of independent fair-coin predictors fails to identify its accuracy even if that predictor is named; enough informative views are also required. The FedDS proof's exclusion of permutations for a fixed j should not be read as a proof of general factorization uniqueness or of the weaker existential quantifier. This is a narrow logical audit, not a claim to have discovered label switching or to have invalidated FedDS's empirical performance.

FNRE illustrates a different signal: clean calibration accuracy combined with own-label agreement. Its utility depends on calibration error transferring to the evaluated client and on the assumed relationship between annotation/model errors. A rare client need not share the pooled calibration error. Thus it is relevant recovery prior art, but “one clean client exists” is not a substitute for its modeling assumptions. No peer-agreement algorithm was proposed before this comparison, and none is introduced here.

## 8. At most three candidate added signals

These are existing principled routes, not novel mechanism proposals. The general recovery gap is real; the audit found no unoccupied method claim.

| Candidate signal | New information and why it breaks ambiguity | Communication / computation / privacy | Closest prior and verdict |
|---|---|---|---|
| 1. Trusted paired loss contrast on the declared target | Evaluates before/after candidates against semantic labels; W− and W+ induce different clean observations | m model evaluations per audit item; m sums/counts per evaluator if models can be delivered; clean labels may stay local; repeated scores still leak and audit reuse needs control | Validation/contribution scoring, [Zeno](https://proceedings.mlr.press/v97/xie19b/xie19b.pdf), [FLTrust](https://www.ndss-symposium.org/wp-content/uploads/ndss2021_6C-2_24434_paper.pdf). Sufficient with coverage; not novel |
| 2. Calibration sufficient for the needed loss contrast | Supplies a known channel or just a justified operator relation Tg=d; excludes incompatible decompositions | Calibration annotation; up to C² channel parameters per stratum, or only required corrected-score summaries; linear solves and conditioning costs; class statistics can reveal sensitive patterns | Natarajan/Cid-Sueiro; FNRE is a more restrictive scalar-calibration neighbor. Sufficient when stated conditions hold; not a new full-matrix estimator |
| 3. Genuine independent annotations plus orientation | Adds joint views of the same latent outcome, identifying more than a marginal noisy label law; orientation selects semantic meaning | At least the applicable number of informative views, linked examples and inference/moment/EM computation; overlapping items may require privacy agreements; severe dependence defeats the model | DS/Allman/FlyingSquid/peer scoring; FedDS for model-view application. Conditional route, not new |

An unlabeled reference set alone, multiple model probes alone, or temporal consistency alone adds no guaranteed semantic information to Proposition 1. They can expose structure or improve estimation under additional assumptions, but are not fourth/fifth proposed mechanisms. A single orientation bit is minimal only in a model where exactly a two-way orientation ambiguity remains; Proposition 2 has more general unknown-channel ambiguity. No universal “one bit fixes FL” claim is made.

## 9. Cheap experimental validation

Specification: [experiment plan](phase3_experiment_plan.md). Main execution: [phase3_identifiability.py](phase3_identifiability.py). Raw outputs: `results/P3_matched_worlds.jsonl` (9,600 records), `results/P3_observation_grid.jsonl` (16,000 records), `results/P3_summary.json`, `results/P3_peer_dependence.json`. Every record is an estimator/world/repetition evaluation, **not an independent FL training run**. There are 200 repetitions at n=25,100,400,1600 with fixed seeds and candidates. No tuning, GPU work or new aggregation method was used.

### Same observations at every sample size

The two-client suite uses Proposition 2. Noisy samples are coupled, distinct latent labels are drawn from each world's exact conditional law, and the honest reporting/probe computation is executed separately on the two world datasets. See `P3_world_coupling.json`. Exact equality of population noisy laws is the proof; coupled execution is an implementation illustration. The initial duplicate-hash check was corrected after independent review and documented in the plan.

For every sample size, averaging the two worlds gives ranking accuracy **.5** and sign accuracy **.5** for each examined score. Increasing samples does not remove the ambiguity. The Brier score's variance falls from **3.1797e−4** at n=25 to **5.7715e−6** at n=1600, while the decision remains unresolved. One world's estimator can work perfectly if it fails in its counterpart; the theorem does not require identical single-world AUROCs.

Actual honest empirical local minimizers from two 10,000-example noisy datasets are (.5961,.4092). Their clean gains are (−.028455,.009915) versus (.009985,−.026405). Thus random empirical training can instantiate the reversal too; the fixed-candidate exact numbers were not improperly assigned to these updates.

### Observation grid and structured corruption

Two feature strata have clean posterior (.1,.9), with global target mass (.85,.15). Common and rare clients have masses (.9,.1) and (.1,.9). There are 22 candidate clients: both groups under clean, symmetric .2/.4, pair .65/.85, deterministic flip, asymmetric class-conditional (.1,.4), instance-dependent (.05,.75), and heterogeneous symmetric .1/.3/.45 channels. Candidate probabilities are one population Brier step from (.3,.55). Fixed population training isolates evaluation uncertainty.

Each of O1–O6 is evaluated on every channel. Binary pair noise equals binary symmetric noise; this is disclosed, with a separate three-class exact check separating their margins. O3 uses a rich local noisy-posterior probe estimate, an optimistic particular implementation. O2 update alignment and O4 consensus are illustrative scores of available packets, not optimal estimators over all observations. Full cumulative-regime impossibility comes from the proof, not a leaderboard of these scores.

The complemented latent-label world preserves every O1–O4 observation while changing the target and channel interpretation. O5 obtains fresh target labels in each world. O6 obtains three genuine independent annotations with symmetric errors (.1,.2,.3), a known positive orientation, and moment-based posterior estimates. This is a controlled special case of classical multi-view recovery, **not a FedDS reproduction**.

| n=1600 estimator | Original utility AUROC | Complemented utility AUROC | Original Spearman / Kendall | Original sign accuracy | Original utility MSE |
|---|---:|---:|---|---:|---:|
| O1 own Brier change | .3885 | .6115 | −.3691 / −.2513 | .4570 | .03920 |
| O2 mean-update alignment | .0000 | 1.0000 | −.9966 / −.9827 | .0000 | Not a utility estimator |
| O3 noisy-posterior probes | .3483 | .6517 | −.3902 / −.2671 | .4552 | .03457 |
| O4 consensus | 1.0000 | .0000 | .9989 / .9913 | 1.0000 | .0007861 |
| O5 trusted target labels | 1.0000 | 1.0000 | .9991 / .9935 | 1.0000 | 7.534e−6 |
| O6 independent oriented peers | 1.0000 | 1.0000 | .9973 / .9852 | 1.0000 | 4.059e−5 |

O4's perfect score in the original world is explicitly retained. Its failure in the indistinguishable counterpart is why this does not establish recovery. All 22 utility signs reverse in this grid, but utility magnitudes are not negatives and the full rankings are not exact reversals. Channel metadata describes the original world; in the complemented world the originally deterministic-flip clients are clean. These outcomes depend on the declared mixture and candidate set; they are not general comparisons between FL algorithms.

At n=25, original-world trusted/independent-peer sign accuracies are .9691/.9289 and MSEs 4.681e−4/7.978e−4. Both estimation errors shrink substantially with n. Own-Brier MSE stays .03972→.03920 while its variance shrinks .0009683→.00001449. This separates sampling precision from persistent bias. Clean-versus-corrupted AUROC is recorded separately in JSON; it is never substituted for utility AUROC. The summary's SDs are repetition standard deviations, not confidence intervals; undefined correlations for tied two-client scores are omitted, with no claim of matched-score superiority. [CPU tables](phase3_cpu_results.md) give both worlds and [channel tables](phase3_channel_results.md) give every corruption/common/rare case.

### Dependence, coverage and algebraic ablations

- **Dependent peers:** three copies of one annotation with error .2 at true p=.44 yield pair moments 1. The independent-view estimator converges to p=.464 instead of .44. The candidate .4 versus baseline .5 has true utility +.002, but limiting estimated utility −.0028. Observed sign accuracy deteriorates **.525→.090** from n=25 to1600: more data reinforces the wrong model. This violates the assumed independence and is reported as a misspecification test, not a refutation of DS.
- **Missing clean support:** a trusted source observes only the common stratum. Worlds with rare posterior .1 versus .9 have identical trusted observations but target gains −.033 versus +.015 for a rare-only change. A clean client somewhere is insufficient.
- **Known high-noise channel:** eta=.8 recovers p=.7 exactly by inversion. High error alone is not an information barrier.
- **Singular functional:** the rank-two three-class channel in §5 recovers the stated contrast while class-1 posterior remains .2 or .4.
- **Heterogeneous positive factors:** clean gains (.1,.06), attenuation factors (.2,1), observed gains (.02,.06) reverse the ranking while preserving signs.
- **DS unknown dominance:** the two three-view joint laws match to 6.94e−18; their semantic accuracy order reverses.
- **Three-class margins:** at eta=.6, pair correct-to-paired-label margin is −.2 whereas symmetric-noise margin is +.1. This checks the structured distinction absent in a binary-only comparison.

All eight exact numerical checks pass. Numerical assertions support the algebra; they are not proofs of generality. No real-noise model, neural architecture, partial-participation process or adaptive weighting loop was tested here.

![Sampling error versus missing information](figures/phase3_identifiability.png)

Figure: left, original-world utility estimation error; middle, paired-world ranking limitation; right, failure from pretending correlated labels are independent. [Standalone PDF](figures/phase3_identifiability.pdf). Figure values come directly from the saved repetitions.

## 10. Strategic reporting extension, kept separate

For weights w_i=g(s_i)/sum_j g(s_j), if a client can choose an unverifiable s_i with arbitrarily large g(s_i), while others remain fixed, its weight tends to one. If g or weights are bounded, the influence is bounded instead; the general claim of arbitrary weight is then false. This is elementary manipulation, already within the motivation of Byzantine-robust learning, not a novel result.

Server evaluation of a submitted update on hidden/fresh trusted data removes direct control over its scalar score. Cross-client verification requires trustworthy, noncolluding and informative evaluators; they may share corrupted semantics. Cryptographically verifying a calculation against committed labels verifies execution, not ground truth. Norm/weight caps limit influence but do not identify utility. Standard trimmed-mean/median/geometric defenses rely on their own honest-majority/dispersion assumptions; clean rare updates can be outliers. Existing [Zeno](https://proceedings.mlr.press/v97/xie19b/xie19b.pdf) and [FLTrust](https://www.ndss-symposium.org/wp-content/uploads/ndss2021_6C-2_24434_paper.pdf) already use external objective/trust information. No new Byzantine defense or empirical robustness claim is proposed.

## 11. Real-world relevance and limits

The ambiguity is a plausible model of uncertain annotation semantics, not a demonstrated property of every real federation. The [real-world audit](phase3_phase2_bridge.md) distinguishes documented annotation pipelines from hypothetical deployment claims.

- **CIFAR-10N/100N:** human annotations support nonuniform, instance-dependent error study. CIFAR-10N provides three annotation positions per image, not three fixed persistent annotators/FL clients. Different workers do not guarantee class-conditional error independence; shared image difficulty matters. The constructed “Worst” set uses clean labels in its definition. [Wei et al., ICLR 2022](https://arxiv.org/html/2110.12088v2).
- **Clothing1M:** shopping-page text produces natural weak labels with concentrated confusions, alongside manually refined data. It offers a real-noise audit source; an FL partition and target must still be specified, and its clean subsets must not leak into an O1-only estimator. [Xiao et al., CVPR 2015](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Xiao_Learning_From_Massive_2015_CVPR_paper.pdf).
- **Institution-specific clinical measurement:** a 193-hospital study documents substantial variation between administrative and clinical sepsis definitions and resulting outcome rankings. This motivates explicit target semantics; it is not an FL experiment or proof that a reference definition is perfect. [Rhee et al.](https://pubmed.ncbi.nlm.nih.gov/30431493/).
- **Actual FL annotation heterogeneity:** a six-hospital RACOON lung-CT study uses manual annotation at some sites and automatic preprocessing followed by manual correction at others. This is direct FL motivation, but does not establish our exact pair-noise rate or indistinguishable worlds. [Bujotzek et al., JAMIA](https://academic.oup.com/jamia/article/32/1/193/7841978).
- **Weak supervision:** ICD/NLP surrogates in [PheNorm](https://groups.csail.mit.edu/medg/ftp/psz-papers/J%20Am%20Med%20Inform%20Assoc%202017%20Yu.pdf) and labeling functions in [Snorkel](https://arxiv.org/abs/1711.10160) illustrate multiple imperfect signals whose dependencies must be modeled. Neither turns arbitrary FL models into independent annotators.
- A [2026 federated medical segmentation benchmark](https://arxiv.org/html/2606.16868v1) is a relevant future test source, but fused/expert reference labels still need interpretation. No training was performed on it.

No exact personal-device or regional-policy deployment of the counterexample was verified. Those examples are not presented as established findings. Legitimate site-specific label definitions may be different tasks rather than corruption of one universal truth; aggregation/routing must first declare which semantics it intends to serve.

## 12. Claim ledger, novelty risk and decision

| Claim | Status | Evidence | Closest prior work | Remaining risk |
|---|---|---|---|---|
| Honest clients can have identical observables and reversed clean update utility | Proved in a simple admissible setting; not novel by itself | Propositions 1–2; matched-world and empirical-update checks | Scott 2013; Liu 2023; van Rooyen & Williamson 2018 | Joint law, semantic side information and target must remain explicit |
| Low CCN rates and some unknown clean client suffice | Refuted | Both worlds have rates <=1/3 and a clean client | Unknown-channel decomposition theory | Stronger named-source/anchor assumptions change the result |
| Added probes/trajectories universally resolve ambiguity | Refuted under the same-input model | Transcript kernel proof | Data processing | External semantic content would be new information |
| Full noise-rate/matrix estimation is always required for a decision | Refuted | Contrast criterion and ordering examples | Menon 2015; weak-label/loss correction | Exact sufficient target information depends on the loss and model class |
| Known-channel contrast correction is a new FL recovery theorem | Not supported | Elementary column-space derivation | Natarajan; Cid-Sueiro; general corrupted experiments | More restrictive communication models might have new questions, not studied here |
| FedDS already addresses FL reliability identifiability | Verified | Full method/theory audit | FedDS 2026; classical DS | Exact semantic utility is a different target; not evidence of novelty alone |
| Some unknown dominant peer fixes orientation | Refuted as literally stated | Full-rank three-view counterexample | Classical latent label switching; FedDS §4.2 audit | A named dominant peer plus factorization conditions is stronger |
| Phase-II score failures alone prove non-identifiability | Rejected | Reversed level AUROC; other estimators succeed; no paired worlds in P2 | Noisy evaluation/robust loss theory | Empirical weakness does not quantify over all estimators |
| Trusted target data or oriented independent views recover utility in a controlled model | Supported, known mechanisms | O5/O6 error falls with samples, including structured-corruption candidate updates | Validation; DS; peer scoring; loss correction | Dependence, finite samples, coverage and adaptive reuse |
| A useful new algorithm follows | Not established; none proposed | Mechanism audit finds direct counterparts | FedDS/FNRE/FedSNC/FedEFC/Zeno and centralized priors | Exhaustive literature absence is impossible to certify |
| Real annotation structures motivate the question | Documented, bounded interpretation | Human labels, weak labels and actual heterogeneous FL annotation pipelines | Sources in §11 | Exact counterexample not verified in deployment |

**Why outcome D rather than A/B/C:** all established propositions are elementary consequences or specializations of existing identification/decision theory; the direct FL reliability setting is occupied by FedDS, and proposed recovery signals have close prior counterparts. The target distinction between prediction reliability and clean update utility is important, but stating it is not yet a novel recovery result. The FedDS quantifier counterexample is worth preserving as a precise qualification; its classical nature does not justify a new method program on its own.

This is a conclusion about the **tested formulation**, not a proof that every conceivable FL-specific identification question has already been solved. FedSNC full-method access remains incomplete, several recent works are preprints, and no exhaustive priority claim is made. These limitations weaken any novelty assertion; they do not justify forcing one. A future reopening would need a concrete theorem under a genuinely distinctive FL restriction—such as a specified limited-observation protocol—with a recovery or lower bound that is not an immediate centralized corollary. No such result was obtained here.

Recommendation: retain this report, the corrections and reproducible diagnostic suite; stop method development for this formulation and explore a different FL question. Do not revive A′/B or run large-scale validation on the basis of these results.

### Artifacts and reproducibility

From `fl_research`, run `python phase3_phase2_audit.py`, `python phase3_identifiability.py`, and `python phase3_summarize.py`. Only Phase III artifacts are written. The original-result manifest covers **37/37 prior result files**, all unchanged; the P2 audit separately verifies all 13 P2 files. The [independent review](phase3_review.md) checked the final simulation revision, proofs and reporting assumptions. The main report, research state and literature matrix are the required deliverables.

**DO NOT AUTHORIZE NEW DIRECTION**
