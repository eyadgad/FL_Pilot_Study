# Independent Phase III theory and CPU review

Review date: 2026-10-02. Scope: `phase3_identifiability.py`, its experiment specification, all matched-world and observation-grid rows, summaries, and the proposed theorem statements. This review does not rerun training, establish literature novelty, or change any existing script or result. The initial reviewed script has SHA256 `94360e125b6f4ea5d818ecccbf7b9645963a6e5d705ff0e8222fbd780b50cf48`.

**Assessment:** the elementary non-identifiability constructions and recovery calculations are correct within their stated models. The simulations support those calculations. They do not establish a novel FL theorem, optimality of an estimator, failure of every estimator from a single-world benchmark, or a practical recovery method for ordinary correlated FL predictions. The final report should use the mathematical indistinguishability argument for universal claims and the CPU results only as controlled illustrations.

## 1. Mild class-conditional corruption construction

Let `T[y,z] = P(noisy label=z | clean label=y)`. Write `a=P(0->1)` and `b=P(1->0)`, so an observed positive-label probability is `q=a+(1-a-b)p`.

| World | Clean positive probability p | Client 1 (a,b) | Client 2 (a,b) | Observed (q1,q2) |
|---|---:|---|---|---|
| Minus | .4 | (1/3,0) | (0,0) | (.6,.4) |
| Plus | .6 | (0,0) | (0,1/3) | (.6,.4) |

With uninformative features and independent client samples, these worlds have exactly the same **joint observed-data distribution**, not merely matching first moments. Every channel is strictly diagonally dominant and has both directional error probabilities below .5. A clean client exists in each world, but its identity differs. This correctly refutes recovery from merely assuming that an unidentified clean client exists. It does not refute recovery from a named, trusted client covering the relevant target distribution.

For binary Brier risk `R_p(t)=p(1-t)^2+(1-p)t^2`, baseline `.5`, and fixed candidate probabilities `(.6,.4)`, the clean utility vectors are `(-.03,.01)` and `(.01,-.03)`. Thus both the best-client ordering and the sign of each update reverse. The positive orientation of each channel does not rescue **Brier utility**. This is not a claim that the same mild-noise construction reverses the sign of 0-1 or MAE risk differences under their additional symmetric-noise assumptions; the targets must remain separate.

The fixed candidates are population local-risk minimizers. For empirical minimizers the same observed-law argument still holds, but realized candidates and utilities are random. The finite empirical check gives candidates `(.5961,.4092)`, with utilities `(-.02845521,.00991536)` versus `(.00998479,-.02640464)`. This is a valid illustrative realization, not a statement that every finite-sample realization has opposite signs. A sign reversal on an event of positive common observation probability suffices to contradict uniform decision recovery for that realized-output target.

## 2. Full adaptive transcript statement

Let `Z` include the joint client observed datasets, public reference features, initial states, all accessible pretrained models, and any other external side information available to the protocol. Let the honest protocol use a world-independent Markov kernel `K(H|Z)` to produce its full transcript, including server adaptation, local optimization, and randomized messages. If `P_minus(Z)=P_plus(Z)`, then

`P_minus(H in A) = integral K(A|z) dP_minus(z) = P_plus(H in A)`.

Equivalently, couple the entire observed datasets and protocol random seeds, then induct over rounds. The next query and honest response agree whenever the history agrees. The same statement holds if future observations are pre-sampled into `Z`, or if their conditional kernels are also identical in the two worlds.

The assumptions matter. Equality of each client's marginal distribution alone is insufficient when cross-client dependence differs and is observable. A fixed pretrained model can be included, but semantic side information that distinguishes the worlds invalidates the equality assumption. Probes or temporal observations that are functions of the existing data do not add identification; fresh trusted labels or genuinely different annotation views can. O5 and O6 therefore fall outside the identical-input premise unless their added observations are also coupled with identical laws.

For the two fixed opposite orderings, any possibly randomized server decision has complementary correctness probabilities in the worlds. With equal prior mass on the two worlds, its average success is .5; equivalently its worst-world error is at least .5. This does **not** require every estimator's AUROC to be .5 in either world separately.

This is a standard statistical data-processing consequence. Its FL interpretation is useful, but the transcript lift alone should not be claimed as a new information-theoretic contribution.

## 3. Recovery statements and their exact scope

### Known noise channel and a decision-specific functional

For known row-stochastic `T`, noisy posterior `q=T^T p`, and loss contrast vector `d`, the functional `d^T p` is identifiable for all clean posteriors in the simplex exactly when `d` lies in `col(T)`.

Sufficiency: if `Tg=d`, then `d^T p=g^T q`. Necessity over the unrestricted simplex: if `d` is outside `col(T)`, some `v` in `ker(T^T)` has `d^T v != 0`. Since `T 1 = 1`, `1^T v=0`. Perturb an interior posterior to `p+epsilon*v` and `p-epsilon*v`; both remain valid probabilities, have the same noisy posterior, and different functional values. Adding a constant to `d` causes no extra exception because the all-ones vector is itself in `col(T)`.

The criterion is for an unrestricted posterior family; restricted supports or known structural constraints can identify further functionals. For a feature-dependent loss contrast, the condition applies pointwise where the posterior is unrestricted, with target feature weights also known or estimable. Knowing a local posterior or channel does not by itself solve distribution shift to an unsupported target. Poor conditioning of `T` can make an identifiable functional difficult to estimate. The checked singular example has rank two and identifies `p1+p2=.5` while leaving `p1` undetermined.

Known binary symmetric corruption with error `.8` is invertible and correctly recovers `p=.7`; `.5` is a singularity for that family, not a universal boundary separating identifiable and unidentifiable noise.

### Trusted validation bound

For `m` fixed candidates (or candidates chosen independently of validation labels), define paired utility observations `D_ij=loss(before,Z_j)-loss(candidate_i,Z_j)` on `n` independent target examples. If each loss lies in `[0,1]`, then `D_ij` lies in `[-1,1]`. Hoeffding and a union bound yield

`P(max_i |Uhat_i-U_i| > epsilon) <= 2m exp(-n epsilon^2/2)`.

Thus `n >= (2/epsilon^2) log(2m/delta)` suffices for simultaneous error at most epsilon with probability at least `1-delta`. Shared validation samples across candidates are allowed; candidate independence is unnecessary. Utility signs are then correct when `|U_i|>epsilon`, and pair orderings when their utility gap exceeds `2epsilon`. Unrestricted CE is unbounded, and repeated adaptive reuse of the same validation labels is not covered. Binary Brier in this code has range `[0,1]`; using the unnormalized two-coordinate Brier score changes its range and the constant.

A trusted source with no support on a target stratum cannot determine utility there. The exact missing-support example correctly has common trusted observations but target utilities `-.033` and `.015`. Coverage is necessary for an unrestricted class; transportability assumptions can replace direct support only when stated and justified.

### Three peers and semantic orientation

For independent symmetric annotators with `Z_j` in `{-1,+1}` and positive orientation `a_j=1-2*eta_j`, conditional independence gives `E[Z_i Z_j]=a_i*a_j`. Three nondegenerate pair moments recover magnitudes, and a known orientation fixes their common sign. The code's conditional-stratum estimate is consistent in this special case. Its floors and clips produce finite-sample bias and instability; it is not a general DS maximum-likelihood implementation or a FedDS reproduction.

The existential-dominance counterexample is correct. With balanced latent classes and annotator accuracies `(.9,.1,.8)`, changing all accuracies to `(.1,.9,.2)` leaves the complete three-view observed law unchanged. All channels are full rank; each world has a dominant client, but the identity changes and semantic reliability reverses. A **specified** client's known orientation removes this permutation ambiguity only after enough information exists to identify the factorization. Existence of some unidentified dominant client does not do so. This is a scoped logical counterexample, not by itself a full audit of every assumption made in FedDS.

## 4. Numerical and raw-artifact checks

- All eight exact checks passed independently on review. All 37 earlier raw-result hashes match the preserved manifest. The reviewed script hash matches the integrity record.
- Recomputed every summary from the 9,600 matched-world and 16,000 grid rows: maximum discrepancy is zero. Independently computed pairwise AUROC and sign accuracy agree exactly for all 25,600 rows. An independently expanded Brier utility formula agrees with grid utilities within `1.6e-16`.
- O1-O4 scores agree exactly between paired worlds. Their paired useful-AUROCs sum to one in every raw repetition. This is a deliberately coupled experiment, not an empirical estimate of an accidental equality.
- All 22 grid utility signs reverse: ten clients are useful in the original world, twelve in the complemented world. Utility magnitudes are not negatives and full numerical rankings are not exact reversals. Minimum absolute utilities are `.0013082175` and `.0063223065`, respectively.
- Clean-status labels used by `clean_auc` are correct: original clean channels are positive in the original world; original deterministic-flip channels are positive in the complemented world. Row-swapping gives `a'=1-b`, `b'=1-a`, so the overall corruption probability becomes `1-eta`. The client metadata's channel names and noise values describe the **original** world.
- In the dependent-peer ablation, all annotators copy one corrupted annotation. The limiting posterior estimate is `.464` rather than `.44`; true utility is `.002` but estimated limiting utility is `-.0028`. Sign accuracy falls from `.525` at `n=25` to `.09` at `n=1600`, while estimator variance falls from `.0003708` to `.000005767`. The inconsistency is an exact population calculation; the trend is its finite-sample illustration.

## 5. Interpretation limits and implementation review

1. **Original packet fingerprint check is tautological.** The initially reviewed code computes two hashes from the same in-memory packet. This establishes equality of two copies, not an independent validation of two latent-world simulators. Exact equality of the observed Bernoulli laws and the protocol-kernel proof supply the real mathematical evidence. A revised coupled simulator can call the same honest protocol separately on world-specific records with coupled observed labels; it must avoid passing hidden labels into protocol computation. A final-revision review can be appended below.
2. **O1-O6 rows are scores, not proofs of regime limits.** O2 alignment, O3 noisy-posterior plug-in, and O4 hard consensus do not exhaust their available information. They also are not implemented as cumulative optimal estimators of nested observation regimes. The theorem remains valid even for a richer observed-data oracle; estimator failures alone do not prove it. O5 and O6 are alternative information augmentations, not automatically a strict information ordering.
3. **Losses and populations differ.** Global Brier utility is the declared target. Error, MAE, CE, and clipped CE are cross-loss proxy scores here. Even O1 Brier estimates a local noisy-population difference, while target utility is on global clean masses. Its bias includes both corruption and distribution shift. O3 uses known target masses and estimates the noisy posterior by stratum; its remaining ambiguity illustrates corruption after fixing that particular shift. Do not use this grid to claim that 0-1 utility, every bounded loss, or every possible local estimator necessarily fails in the same way.
4. **Training is fixed at population values.** Increasing `n` changes validation information only. O2 and O4 consequently have zero variance apart from floating-point roundoff. The experiment does not measure finite training randomness, neural-model learning, iterative aggregation, optimized consensus, or realistic cost/communication performance. Pair and symmetric noise coincide in binary classification; the multiclass check is an exact margin calculation rather than a multiclass FL experiment.
5. **Monte Carlo summaries are descriptive.** The `*_sd` fields are standard deviations over 200 repetitions, not standard errors or confidence intervals. Seeds are reused across sample sizes, so estimates across `n` should not be treated as independent. Correlations undefined because of tied two-client scores are omitted from their reported means; at `n=25`, some matched-world scores omit up to 19 of 200 repetitions, and update-magnitude correlations are always undefined. All grid correlations are defined. No selection-adjusted inferential claim is supplied.
6. **Floating ties can change small matched-world differences.** AUROC and rank correlations use exact floating comparisons, whereas sign accuracy uses a `1e-12` tie threshold. Algebraically equivalent rankings can therefore differ slightly when sample estimates tie mathematically. This does not affect the exact paired-world impossibility or the substantive grid conclusions. Small cross-loss differences in the two-client matched suite should not be interpreted as substantive superiority.
7. **Clean status and usefulness remain distinct.** `clean_auc` has two clean positives among 22 clients in each world, and uses a usefulness-oriented score to classify noise status. It is descriptive and cannot substitute for utility AUROC. Likewise, known corruption rate alone does not determine update utility on a chosen target.

No calculation reviewed here supplies novelty evidence or justifies large-scale experimentation. The honest-corruption constructions also make no Byzantine-resilience claim: arbitrary report falsification is a different observation model.

## 6. Final code revision and report review

The revised script has SHA256 `7667474f2ecde4d5d6d3c17ca101f867ed9753ebdc667371514d62a7079a7e6c`, matching its updated integrity record. The initial packet issue in section 5 item 1 is addressed: the revised function constructs both world-specific latent-label records and executes the honest computation separately on copies of the coupled observed labels. The computation reads the observed field and does not read the latent field.

The reverse conditional sampler uses `P(Y=1 | observed=z)=p*T[1,z]/q[z]`, with the correct world/client matrices. Together with the common observed marginal, this yields each intended joint clean/noisy law. The new `P3_world_coupling.json` records illustrative latent and noisy frequencies and empirical corruption. Sampling the latent labels has its own random generator, preserving the original noisy-data generator `31000+rep` and all score values.

Review reran `matched_worlds()` with the save function replaced by an in-memory capture, so no result files were overwritten. All 9,600 raw matched-world records and every coupling example reproduced exactly. Separately reconstructing the original noisy-label draws and scoring them gave maximum score difference zero. All 37 prior-result hashes remained unchanged. These checks validate this implementation; the equality-of-laws proof is still the source of the all-estimator statement.

The final report's propositions, Hoeffding constant, fixed-versus-empirical candidate distinction, DS orientation counterexample, and distinction between illustrative scores and available observation regimes were reviewed. One scope clarification was raised and corrected: integrating a recovered client conditional contrast over target feature mass requires the recovered posterior to equal `Q(Y|X)`, or a justified conditional-label transport relation. Known `Q_X` and support alone cannot transport arbitrary client semantics. The final report now states this explicitly in Proposition 3 and the assumption ladder. It also incorporates the distinction between reversed signs and numerical utility magnitudes/rankings, the world-specific meaning of client noise metadata, and the SD/undefined-correlation caveats. No remaining substantive mathematical or numerical error was identified in the reviewed scope. The review does not independently reproduce external literature or the Phase II audit, which have separate supporting artifacts.
