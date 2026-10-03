# Phase II report: does the discovery contain a real, general, explainable and novel contribution?

*Started 2026-10-02. Built on the session-1/2 package in this folder. Nothing in `results/E*.jsonl` was modified; Phase II results live in new files (`results/P2_*`).*

**Labels used throughout:**
- **[obs]** measured here;
- **[proof]** derived here, elementary unless said otherwise;
- **[known]** in prior work (cited);
- **[web]** verified on a fetched page or search result;
- **[interp]** interpretation;
- **[hyp]** untested.

**Scope warning.** All evidence is from a 784-128-64-10 NumPy MLP on MNIST / Fashion-MNIST, 50 clients, CPU only. Nothing here licenses claims about CNNs, CIFAR or real annotator noise.

## Summary

**Gate: DO NOT AUTHORIZE LARGE-SCALE VALIDATION. Framing: E (revise).** A′ and B are killed as method contributions. A narrowed measurement claim survives, but only in this testbed. Reasons follow.

1. **The discovery results are sound** (Stage 1). Every headline number recomputes from the raw logs, and 10/10 sampled runs re-execute bit-exactly. There is no leakage, overlap or stale data.
2. **The core phenomenon is real and replicates on fresh seeds** (Stage 7: 1,008 probe updates, pre-registered).
   - Cross-entropy and Brier *reductions* rank rare clean clients below noisy ones: AUROC 0.24 and 0.43, against 0.99–1.00 for bounded statistics.
   - Every loss *level* fails: AUROC 0.00 for all eight statistics.
   - The mechanism is confidence: noisy local updates raise predictive entropy, and CE rewards it.
3. **But "0–1 error" is not the answer — "bounded statistics" is.** MAE and clipped CE match or beat 0–1, with lower variance. A′-MAE beats A′ (0–1) on rare accuracy in all 9 FL setting × seed-set combinations. A per-client temperature fixes CE under symmetric noise, but not under class-conditional noise.
4. **The theory is correct and testable, but it is already published.**
   - The affine identity, its uniqueness and the margin form are known: Ghosh et al. 2017; Toner & Storkey 2024; Chen et al. 2021.
   - New here are the federated consequences: offset cancellation in before/after differences, exact sign reversal at negative margins (verified to 1e-16 for relabelled clients) and the chance-level guard.
5. **The mechanisms are not novel and are beaten by simpler alternatives.**
   - A′ is validation-improvement weighting (FedGA, FedFomo, RHFL) with a bounded statistic.
   - B is CLoVE with a noise-invariant statistic; a 0–1 argmin matches it on trained models.
   - Cluster-wise FedAvg beats both one-model A′ and B → A′.
6. **The boundaries are where the theory says.** Every own-label statistic fails for relabelled clients and for pair noise above ½. A chance-level test on the error *level* catches relabelled clients exactly (512/512 flags, 0 false), but is blind to pair noise.
7. **Pre-registered scorecard: 20 of 29 testable predictions passed and 9 failed.** One more was vacuous, one was untestable, and two carried no prediction. Every failure is reported where it occurs.

---

## Stage 1 — Verification of the discovery results from raw logs

**Method.** `phase2_verify.py` re-reads every raw record in `results/E*.jsonl` (20 experiment files), rebuilds the CSV, regenerates partitions, re-executes stored runs and recomputes every headline number. Output: `results/P2_verification.json`. "Reproduced" below means *recomputed from the raw logs*, except row V6, where runs were re-executed.

### Integrity checks

| # | Claim | Raw evidence | Reproduced? | Confidence | Notes |
|---|---|---|---|---|---|
| V1 | Raw logs are internally consistent | 0 duplicate (setting, method, seed) records in all 20 files; no record whose config deviates from its experiment's declared config; the "last-3" metrics recomputed from the stored histories: 6,528 checks, 0 mismatches; histories have the expected evaluation rounds | yes | high | — |
| V2 | Method comparisons are paired | within every (setting, seed), the client-selection shares are identical across methods (max difference 0.0) in all global-model experiments, so all methods saw the same client-sampling sequence | yes | high | Selection-based methods (PoC) are excluded by design |
| V3 | `experiments.csv` is not stale | rebuilt from raw JSONL by `summarize.build_csv`; identical (904 rows before Phase II, 0 differing lines). The Phase II rebuild (1,221 rows) leaves all 904 discovery rows cell-for-cell unchanged | yes | high | — |
| V4 | No train/test overlap or mismatched partitions | partitions regenerated twice are identical; 0 duplicate training images across clients; 0 client training images found in the test split (rotations undone before hashing); MNIST, FMNIST and CFL populations | yes | high | Clients' clean test sets come from the official test split |
| V5 | No oracle leakage in non-oracle methods | runtime invariance test: permuting the hidden fields (clean labels, client type, cluster id, noisy flag, η) leaves the final model **bit-identical** for all 28 non-oracle methods (17 global, 7 clustered, 4 routed); it changes the model for all 5 oracle methods, as it should | yes | high | A test of behaviour, not a code reading |
| V6 | Current code reproduces the logs | 10 stored runs (one per family: E1, E2, E2c, E3b, E3c, E4, E5b, E7, E8, E9) re-executed: every logged final metric bit-identical (8–37 metrics per run). After the Phase II code changes, 6 more runs (E1, E2, E3d, E4, E5b, E8) were re-executed, also bit-identical | yes (re-run) | high | Needs single-threaded BLAS; multi-threaded BLAS drifts by ~0.1 pt |

### Headline claims

| # | Claim | Raw evidence | Reproduced? | Confidence | Notes |
|---|---|---|---|---|---|
| C1 | Loss-proportional (FedEBA-style) weighting hands the budget to noisy clients | noisy weight share 0.654–0.900 vs FedAvg 0.302 (0.588 in the 60 %-noisy setting); 8 noisy settings × 3 seeds | yes | high (in scope) | Our FedEBA-style rule = FedEBA+'s aggregation weights only (no alignment step); not the official code |
| C2 | …which lowers covariate-shifted rare clients' accuracy | EBA − FedAvg rare accuracy −2.6 (pair η 0.4) to −14.7 pts (η 0.8); 7 settings, 3-seed means | yes | high (in scope) | **Label-rare: not supported** (+11.7, −12.3, −11.4, +16.3, +8.5 pts over 5 seeds); corrected in session 2 |
| C3 | The observed-label fairness metric rewards the failure | observed-label variance lower under EBA in 34/34 seed-settings, while clean worst-10 % accuracy is lower in 31/34 | yes | high | The 3 exceptions are label-rare seeds |
| C4 | CE reducibility (candidate A) does not fix it | η 0.6: noisy weight 0.404 vs FedAvg 0.302; rare accuracy 0.786 vs 0.833 | yes | medium-high | One gate setting, 3 development seeds |
| C5 | Noisy 0–1 reductions scale by a(η) | slope 0.325 vs predicted 0.333, r = 0.757 (135 noisy client-checkpoints); AUROC rare-vs-noisy of the 0–1 gap 0.935 / 0.880 / 0.731 vs CE gap 0.839 / 0.584 / 0.472 at rounds 25 / 75 / 150 | yes | medium | One setting, 3 development seeds → re-tested on fresh seeds in Stage 7 |
| C6 | A′ raises rare accuracy over FedAvg | 30/30 pairs (10 settings × seeds 0–2); per-setting mean +1.5 to +25.2 pts; median pair +4.75; 12/12 fresh-seed pairs (E8) | yes | high (in scope) | Seeds 0–2 are development seeds (τ was chosen on E1b). A′ trains on 80 % of local data, FedAvg on 100 % |
| C7 | …at a small common-accuracy cost | per-setting −0.4 to −2.6 pts (seeds 0–2); −0.97 mean on fresh seeds | yes | high | — |
| C8 | A′ ≈ the clean-label oracle | A′ − oracle rare: +0.10 pts (11/18 pairs, seeds 0–2); +0.63 (20/30, seeds 0–4) | yes | medium | The "oracle" is EBA on clean-label loss levels, not an optimal allocation. Supports "≈", not "better" |
| C9 | B recovers clusters under heterogeneous noise | ARI 1.00 in every seed of all 12 noisy settings at n = 500 (E2, E2b, E2c, E2d), and at n = 150 with η = 0.6; fresh seeds (E6) 11/12 (perm, seed 3: 0.932) | yes | high (in scope) | Breakdown at n = 150, η = 0.8: ARI 0.56 / 0.86 / 0.93 |
| C10 | CLoVE-style CE clustering collapses at η ≥ 0.5 | seed-mean ARI 0.12–0.49; 0.68 (pair η 0.4), 0.87 (η 0.4), 1.00 (η 0.3) | yes | high (in scope) | Our CLoVE re-implementation; the post-stability phase was omitted in E2–E6 |
| C11 | IFCA (CE argmin) is unreliable in this testbed | ARI 0.70 without noise; 0.09–0.41 with noise | yes | medium | **Gap found:** IFCA with a 0–1 argmin was never tested. It is noise-invariant by T1 and is a simpler alternative to B (Stage 11) |
| C12 | A′ fails with relabelled clients | common accuracy 0.712 / 0.950 / 0.056 (seeds 0–2); 0.802 / 0.771 / 0.757 (fresh seeds 3–5) | yes | high | FedEBA-style fails similarly (0.824 / 0.915 / 0.116) |
| C13 | Routing repairs the conflict failure | B4sc → FedAvg worst-10 % 0.91–0.92 in 6/6 seeds; B2c → A′ collapsed on development seed 2 (common 0.057) | yes | medium | The routing rule has no guarantee |
| C14 | Misreporting defeats A′ | attackers' weight share 0.408 (A′), 0.412 (EBA) vs a base share of 0.10 | yes | high | — |
| C15 | Personalisation beats one-model A′ when the rare group can be isolated | E9 rare accuracy 0.856 vs 0.725 (n_rr = 1); 0.899 vs 0.802 (n_rr = 2) | yes | medium | 3 seeds; 600-sample rare clients |

### Other checks requested in Stage 1

- **Post-hoc changes.** τ → 0 (E1b), the B-WG guard (E2e), the LCB ranking (E3f), K̂ selection (E2d) and the post-stability routing (E5b) were all chosen after seeing development results. They are listed as discovery evidence in Stage 2. None of the confirmatory runs was used to change a method.
- **Metric-definition changes.** The last-3 definition is identical across E1–E9 (V1). One metric was introduced after the fact: per-client clean worst-10 % in E5. This was disclosed in session 2.
- **Incorrect comparisons.** All method comparisons are paired on population and sampling (V2). One comparison is unfavourable to A′ and has been kept: A′ trains on 80 % of local data while FedAvg trains on 100 %. The shuffled-statistic control holds out the same data.
- **Baselines are re-implementations, none official:**
  - FedEBA-style: aggregation rule only;
  - FedPCA-style: our GMM on loss and feature dispersion, q = 1;
  - CLoVE-style: without the post-stability phase in E2–E6;
  - Power-of-Choice, q-FFL and IFCA: our code.
  
  The "unstable FedPCA-style" result on FMNIST (E4) is plausibly an artefact of our re-implementation.
- **Leakage and oracle leakage.** None found (V4, V5).

**Stage-1 verdict.** The raw logs support every headline number in the package as stated after the session-2 corrections. No integrity problem was found. The verification found one missing baseline (IFCA with a 0–1 argmin; C11), which is added in Stage 11.

---

## Stage 2 — Discovery vs confirmatory evidence

The ledger is in `research_state.md` §14.1 and was written before any Phase II run. In short:
- **Discovery:**
  - every session-1 experiment (seeds 0–2), which chose τ → 0, the 0–1 + z-normalisation for B, the B-WG guard, the LCB ranking and K̂ selection;
  - E5 and the E5b development seeds, which designed the post-stability routing.
- **Confirmatory (session 2):** E8 (seeds 3–4), E6 and the fresh E5b seeds (3–5). E7 and E9 were pre-registered but reuse seed numbers 0–2 on new populations or an attack, which is disclosed.
- **Phase II:** confirmatory seeds 10–12 (measurement, static B audit, dynamic CFL) and 20–22 (FL runs).
- **Reserved:** seeds 30–32 for any method changed after seeing Phase II results; seeds 100–104 for a future benchmark.

Every Phase II prediction (PM1–PM12, PF1–PF7, PR1–PR7, PB1–PB5) and every new guard and statistic was specified in `research_state.md` §14.2 before the first Phase II run (file time 14:13 UTC; first run 14:18 UTC).

---

## Stages 3–6 — Theory

Notation:
- K classes; client i has distribution P_i(x, y) and a label channel T_i(x)[y, j] = P(ỹ = j | y, x);
- R_i(h) = P(h(x) ≠ y) is the clean risk and R̃_i(h) = P(h(x) ≠ ỹ) the noisy risk;
- C(h)[y, j] = P(y, h(x) = j) is the joint confusion matrix; rows sum to π_y for every h.

Every algebraic statement below is checked numerically in `phase2_theory.py`: 16/16 checks pass (`results/P2_theory.json`), using the project's own noise generators. **Novelty status of each piece** (Stage 10 search, verified pages):
- the symmetric-noise identity and its uniqueness are **known**;
- the margin identity is **known** in its gap-to-optimum form;
- the rest are elementary corollaries.

The contribution, if any, is applying them to federated client statistics, not the algebra.

### Stage 3 — Clean vs noisy 0–1 risk

**Lemma 0 (master identity) [proof].** 1 − R̃(h) = E[T(x)[y, h(x)]], because P(h = ỹ) = E[P(ỹ = h(x) | x, y)].

**Lemma 1 (margin form) [proof; known in gap-to-optimum form: Chen et al., AAAI 2021, App. A Eq. 12].** For a class-conditional channel T and any two classifiers h, h′, with ΔC = C(h′) − C(h) and margins m_yj = T[y,y] − T[y,j]:

  ΔR̃ = Σ_y Σ_{j≠y} m_yj ΔC[y,j],   while   ΔR = Σ_y Σ_{j≠y} ΔC[y,j].

*Proof.* R̃ = 1 − Σ_{y,j} C_yj T_yj. Because Σ_j ΔC_yj = 0, we have Σ_j ΔC_yj T_yj = −Σ_{j≠y} ΔC_yj (T_yy − T_yj). ∎

For instance-dependent channels the same holds inside the expectation over x, with margins m_yj(x). In words: each unit of probability moved from "correct" into cell (y, j) costs 1 in clean risk and m_yj in noisy risk.

**Proposition 1 (symmetric channel) [known: Ghosh, Kumar & Sastry, AAAI 2017, Thm 1; Toner & Storkey, arXiv 2409.06830, Fact 1].** For T = (1−η)I + η/(K−1)(J−I), every margin equals a(η) = 1 − Kη/(K−1). Hence R̃ = η + a(η)·R and ΔR̃ = a(η)·ΔR.
- Ordering is preserved iff η < (K−1)/K (0.9 for K = 10).
- At η = 0.9 every classifier has noisy risk 0.9.
- Above 0.9 the ordering reverses.
- This matches the experiments' generator: `flsim.symmetric_noise` flips w.p. η to a uniformly random *other* class. The Monte-Carlo deviation from the formula is ≤ 0.001.

**Proposition 2 (exact proportionality only under symmetric noise) [proof; cf. Toner & Storkey Fact 2 for minimisers].** ΔR̃ = c·ΔR for every pair of classifiers holds iff all off-diagonal margins equal c. Rows of T sum to 1, so this forces a constant diagonal and constant off-diagonal: the symmetric channel. Random confusion pairs show sign disagreements between ΔR̃ and ΔR in:
- 0 % of cases for sym 0.6;
- 1.5 % for symmetric targets with class-dependent rates;
- 3.8 % for pair 0.4;
- 21 % for pair 0.75;
- 99 % for sym 0.95 (a < 0).

**Proposition 3 (sign preservation beyond symmetric noise) [proof].**
- (a) If T is diagonally dominant (all m_yj > 0) and an update is *cell-monotone* (it only removes errors: ΔC[y,j] ≤ 0 for all j ≠ y), then ΔR̃ and ΔR have the same sign, with m_min|ΔR| ≤ |ΔR̃| ≤ m_max|ΔR|.
- (b) For any update, |ΔR̃ − m̄ΔR| ≤ ½(m_max − m_min)·Σ_{off}|ΔC_yj|.
- (c) Mass moved into a cell with a **negative** margin lowers noisy risk while raising clean risk.
  - Pair noise y → σ(y): R̃ = η + (1−η)R − ηD, with D = P(h(x) = σ(y)). Moving correct predictions to σ(y) changes noisy risk by (1 − 2η) per unit: a reversal for η > ½. Monte Carlo with `population._noise('pair')`: +0.0098 at η 0.45, −0.0095 at η 0.55, −0.100 at η 1.0, per 0.1 moved.
  - Deterministic relabelling (CF): ΔR̃ = −ΔD exactly. An own-label statistic rewards learning the other labelling function, one for one. **No own-label statistic can detect this from reductions alone.**
  - Our IDN (ramp to 2η̄ on the hardest samples, target = the reference model's top wrong class): the target cell's margin is 1 − 2η(x). At η̄ = 0.4 it is negative on the hardest 37.5 % of samples.

**Proposition 4 (heterogeneous η across clients) [proof].**
- *Levels.* R̃_i = η_i + a_i·R_i, so noisy levels rank clients by noise rate whenever η varies more than a_i·R_i does. Example: a rare clean client with clean error 0.30 has noisy level 0.30, while a common client with η 0.6 and clean error 0.05 has noisy level 0.617.
- *Reductions.* ΔR̃_i = a_i·ΔR_i, so the offset cancels and the sign is preserved for every client with η_i < 0.9. Magnitudes are shrunk by a_i, though. For a_i < a_j and positive clean reductions, the pair is mis-ordered iff ΔR_j < ΔR_i < (a_j/a_i)·ΔR_j.

### Stage 4 — A′

A′'s statistic is r̂_i = ê_pre − ê_post on the client's noisy holdout. The holdout labels are corrupted independently of the training labels, so E[r̂_i | update] = ΔR̃_i.

**Factor α_i [proof].**
- Symmetric noise: α_i = a(η_i) = 1 − Kη_i/(K−1), giving E r̂_i = α_i·ρ_i, where ρ_i is the clean reduction on the client's distribution.
- Pair noise: there is no constant factor; E r̂_i = (1−η)ρ_i − η·ΔD_i, depending on how much of the update learns σ.
- CF: E r̂_i = −ΔD_i.

**Lemma A1 (scale invariance of the clamped softmax) [proof; check T2].** With τ → 0, FedEBA+'s clamp sets τ_t = range(s)/ln(1/(mε_p)). This gives

  w_i ∝ 10^{(s_i − s_max)/(s_max − s_min)} for m = 10, ε_p = 0.01,

where s_i = max(r̂_i, 0). The weights are invariant to s ↦ c·s for any c > 0 (the clipping at 0 commutes with positive scaling).

**Consequences [proof unless marked]:**
1. **Homogeneous η.** Symmetric noise shared by all sampled clients leaves A′'s *expected* weights exactly unchanged; only their variance grows.
2. **Heterogeneous η.** A noisy client's position is shrunk toward the minimum. Relative to a clean-label A′ its weight is multiplied by 10^{−(1−α_i)ρ_i/range} when the round's extremes are clean clients. A′ therefore approximates clean-label A′ when:
   - every α_i > 0 (no η ≥ 0.9, no pair noise above ½, no relabelling);
   - and the clients at the top of the clean ranking are clean, so their weights are unchanged.
3. **Where A′ departs from clean-label A′:**
   - a client that is both rare and noisy is demoted by α_i, even though its clean reducibility is high (tested with the RN probes in Stage 7);
   - beyond the identifiability boundary (CF, pair > ½, high-η IDN samples), a harmful update receives a positive statistic. This is the E3d failure, now explained. A CF client's r̂ equals the probability mass its update moves onto the relabelled classes (ΔD; verified to 1e-16 in Stage 9), and every unit of that mass is a unit of clean error.
4. **Normalising by α̂_i** recovers ρ_i in expectation but multiplies the variance by 1/α_i² (81× at η = 0.8). It would also hand back weight to noisy clients that A′'s goal is to discount. [interp] For allocation, the attenuation is a feature: a noisy client's expected update carries a fraction α_i of the clean signal under symmetric noise. **Not recommended.**
5. **"A′ ≈ oracle".** The session-1 oracle weights by clean-label loss *level* (need), while A′ weights by noisy *reducibility*. The two rank clients alike when the neediest clients are also the most reducible, which is the case for rare clean clients in the settings tested. The equality is therefore setting-dependent, not structural.

### Stage 5 — Finite samples

**Lemma F1 [proof; Monte-Carlo check within 2 %].** r̂ is a mean of d_k ∈ {−1, 0, 1}, where p10 = P(pre wrong, post right) and p01 = P(pre right, post wrong) on noisy labels. Then:
- E r̂ = p10 − p01 = α·ρ;
- Var r̂ = (p10 + p01 − (p10 − p01)²)/n_h ≤ q/n_h, where q = P(h_pre ≠ h_post); only samples whose prediction changes contribute.

**Mis-ranking.** For a rare clean client R and a noisy client N with independent holdouts:

  P(r̂_R < r̂_N) ≈ Φ(−(ρ_R − α_N ρ_N)/√((v_R + v_N)/n_h)).

Monte-Carlo deviation from the simulation is ≤ 0.004 for n_h = 25…400. The empirical check on real updates is PM11.

The holdout size needed for mis-ranking probability ≤ δ is n_h ≥ z²_{1−δ}·(v_R + v_N)/(ρ_R − α_N ρ_N)². The required n_h grows as the rare client's lead shrinks; late in training both ρ_R and the lead shrink.

**B's failure at low n and high η.** A client's z-normalised error vector is z(η + α·e) = z(e) for α > 0, so it is invariant in expectation. Its sampling noise per coordinate is ≈ √(e(1−e)/n)/(α·s_c), where s_c is the spread of clean errors across models.
- Session 2's recovery radius (T4), r_i = √(2 log(2NK/δ)/n_i)/(α_i·s_c), needs cluster separation Δ > 2r. Recovery is therefore governed by α√n.
- E2c's failure (η 0.8 → α 0.111; n 150 → α√n = 1.36; ARI 0.56 / 0.86 / 0.93) is consistent with this scaling. The constant in the bound is loose (session 2: the bound is vacuous, but its α√n scaling is calibrated).
- For α < 0 (η > 0.9) the vector flips: z(ẽ) = −z(e). The client then looks like the *mirror image* of its cluster, and misassignment is systematic, not random (PB3).

### Stage 6 — Why cross-entropy fails: structure, calibration, optimisation

1. **Structural [proof].** CE is not a symmetric loss:

     E_ỹ[−log p_ỹ] = −(1−η) log p_y − (η/(K−1)) Σ_{j≠y} log p_j.

   This depends on the probability on every wrong class and is unbounded. A noisy-calibrated predictor's expected noisy CE equals the noisy-label entropy H(ỹ|x) = h(η) + η·log(K−1): 0.94, 1.99 and 2.26 nats at η = 0.2, 0.6 and 0.8. Even the best possible model's CE *level* is a noise-rate meter.
2. **Calibration [proof; check T5].** Take a confidence-only update (argmax unchanged) where p_y goes from 0.99 to 0.40, at η = 0.6:

   | Statistic | Noisy-label reduction (positive = "helped") | Clean-label change |
   |---|---|---|
   | CE | +2.09 nats | 0.91 nats worse |
   | Brier | +0.39 | 0.40 worse |
   | MAE | −0.20 (the update is penalised) | — |
   | 0–1 | exactly 0 | 0 |

   Brier's confidence term ‖p‖² is not attenuated by noise: ΔB̃ = Δ‖p‖² − 2a·Δp_y. Flattening a confident correct prediction therefore lowers noisy Brier whenever a < ½, i.e. η > 0.45 (check passes at 0.44 / 0.46).
3. **Temperature scaling is not enough in principle [proof; check T5].** One temperature per model removes a *global* confidence change exactly (TS-CE reduction 0 for uniform flattening). It does not remove confidence *re-shaping*. An update that equalises heterogeneous confidences (0.99 and 0.60 → 0.80 and 0.80, argmax unchanged) earns a TS-CE "reduction" of 0.059 nats with zero clean 0–1 change. Whether this residual matters in practice is empirical; it is PM4, answered in Stage 7.
4. **Optimisation [hyp → tested in Stage 7].** Local SGD on noisy labels drives predictions toward the noisy posterior: it flattens them, and for pair or CF noise it learns the corrupted map. The local update therefore *produces* the confidence change that CE rewards. The prediction is that noisy probes' updates raise predictive entropy more than clean probes' updates do.

**Two strong results** (as requested; the rest is supporting algebra):
- **(R1) Federated cancellation-and-attenuation.** Under symmetric per-client noise, before/after 0–1 differences on a client's own noisy holdout equal a(η_i) × the clean difference. The η_i offset cancels and the sign is preserved up to η_i < (K−1)/K. Under class-conditional noise the difference is the margin-weighted confusion change (Lemma 1): sign-preserving for cell-monotone updates under diagonal dominance, and reversed exactly on negative-margin cells (pair > ½, relabelling). Together with Lemma A1 this gives A′'s exact behaviour: invariant to homogeneous noise, discounting heterogeneous noise, inverted past the boundary.
- **(R2) Confidence non-identifiability of CE and Brier under noise.** Noisy CE and Brier reductions contain a confidence term that the noise channel does not attenuate. It can dominate the correctness term and even reverse it (CE always; Brier once η > (K−1)/(2K)). A single temperature removes only its global component.

---

## Stage 7 — Decisive metric ablation (confirmatory seeds 10–12; `phase2_metrics.py`)

**Design** (pre-registered, §14.2).
- *Base federation and checkpoints:* a base federation (30 common, 5 rotated, 15 sym-0.6 clients) is trained by FedAvg and checkpointed at rounds 20, 60 and 150.
- *Probes:* 112 probe clients make one local update from each checkpoint. Their data are disjoint from the base federation (500 training + 400 holdout samples each). They cover:
  - 6 common datasets × 17 label channels;
  - 4 rotated clean datasets and the same 4 with sym-0.6 noise;
  - 2 label-rare (8/9) datasets.
- *Scale:* 1,008 probe updates.
- *Scoring:* every statistic is computed on the *same* holdout, model and update. It is scored against:
  - U_own: clean 0–1 reduction on a 10,000-image own-distribution test set;
  - U_hold: clean 0–1 reduction on the holdout images;
  - U_fed: balanced clean-error change after a 1/10 step.
- *Raw logs:* `results/P2_metrics_probes.jsonl` (one row per probe × checkpoint × seed) and `results/P2_metrics.jsonl` (scores).

**Main table.** Reductions at n_h = 400 unless stated. The AUROC is rare clean (RR) vs corrupted probes, i.e. P(an RR probe's statistic exceeds a corrupted probe's). Values are means over seeds; ± is the SD over seeds.

| Statistic | sym 0.2–0.8 (r60 / r150) | pair ≤ 0.45 (r150) | IDN (r150) | sym 0.2–0.95 (r150) | pair ≥ 0.55 (r150) | relabel (r150) | Spearman vs U_own (sym mix) | Pearson vs U_own | Pairwise acc. n_h 25 / 400 | AUROC at n_h 25 | Spearman under logit ×0.5 / ×2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **0–1 error** | 1.00 / 0.99 | 1.00 | 0.92 | 0.91 | 0.46 | 0.00 | 0.75 ± 0.04 | 0.71 | 0.58 / 0.78 | 0.87 | 1.00 / 1.00 |
| clipped CE, min(CE, log 10) | 1.00 / 1.00 | 1.00 | 1.00 | 1.00 | 0.53 | 0.00 | **0.80 ± 0.01** | 0.68 | **0.80 / 0.82** | **1.00** | 0.98 / 0.95 |
| MAE (1 − p_ỹ) | 1.00 / 1.00 | 1.00 | 1.00 | 1.00 | **0.75** | 0.00 | 0.67 ± 0.01 | 0.52 | 0.74 / 0.74 | 1.00 | 0.98 / 0.96 |
| GCE (q = 0.7) | 1.00 / 1.00 | 1.00 | 1.00 | 0.78 | 0.50 | 0.00 | 0.59 ± 0.03 | 0.42 | 0.71 / 0.71 | 0.98 | 0.97 / 0.94 |
| CE, temperature fitted per model on the client's noisy holdout (TS-local) | 1.00 / 1.00 | **0.31** | 0.84 | 1.00 | 0.00 | 0.00 | 0.48 ± 0.04 | 0.56 | 0.56 / 0.68 | 0.78 | 1.00 / 1.00 |
| CE, temperature fitted on clean server data (TS-global) | 0.44 / 0.26 | 0.05 | 0.33 | 0.18 | 0.00 | 0.00 | −0.65 ± 0.03 | −0.76 | 0.29 / 0.25 | 0.44 | (refit) |
| Brier | 0.54 / 0.43 | 0.52 | 0.75 | 0.29 | 0.00 | 0.00 | −0.36 ± 0.05 | −0.51 | 0.39 / 0.39 | 0.59 | 0.78 / 0.96 |
| CE | 0.33 / 0.24 | 0.21 | 0.63 | 0.16 | 0.00 | 0.00 | −0.54 ± 0.08 | −0.62 | 0.33 / 0.32 | 0.44 | 0.78 / 0.97 |

**Levels fail for every statistic.** At round 150 the AUROC of the noisy *level* for RR vs sym η ≥ 0.4 is **0.00 for all eight statistics** (PM2 ✓). Level Spearman with clean need is −0.11 to −0.24. The noise offset dominates whichever loss is used. Any allocation driven by loss levels (FedEBA+, q-FFL, Power-of-Choice) is confounded by heterogeneous label noise, and switching to 0–1 levels does not help (E1b's EBA on 0–1 levels, 0.775, agrees).

**Pre-registered predictions:**

| ID | Prediction | Result | Verdict |
|---|---|---|---|
| PM1 | Slope of noisy on clean 0–1 reduction (same holdout) within ±0.05 of a(η) | slopes 0.749 / 0.592 / 0.359 / 0.111 vs a = 0.778 / 0.556 / 0.333 / 0.111 (η 0.2 / 0.4 / 0.6 / 0.8); every 95 % CI contains a(η) | ✓ |
| PM2 | Levels fail for every statistic (AUROC ≤ 0.2) | 0.00 for all 8 | ✓ |
| PM3 | CE reduction AUROC ≥ 0.10 below 0–1 (r150) | 0.24 vs 0.99 | ✓ |
| PM4 | Calibrated CE: no prediction | TS-local equals 0–1 under symmetric noise (1.00) but fails under pair ≤ 0.45 (0.31 at r150) and IDN (0.84). TS-global fails everywhere | answered |
| PM5 | Brier ≥ 0.05 below 0–1 for η ≥ 0.6 | 0.00 vs 0.997 | ✓ |
| PM6 | MAE within 0.05 of 0–1 in AUROC; better pairwise accuracy at n_h ≤ 50 | AUROC 1.00 vs 0.99–1.00; pairwise 0.74 vs 0.58 (n_h 25) and 0.74 vs 0.65 (n_h 50) | ✓ |
| PM7 | Pair ≥ 0.55: reversal (r̃ > 0, U_own < 0) in ≥ 50 % of probe-checkpoints; ≤ 0.45: ≤ 20 % | 0.19 (0.55), 0.39 (0.6), 1.00 (0.8), 1.00 (1.0); for ≤ 0.45: 0.17 / 0.03 / 0.03, against a 0.14 base rate for clean common probes | **✗** — the reversal is graded, not a switch at 0.5 |
| PM8 | CF: reversal in ≥ 90 % | 100 % (36/36) | ✓ |
| PM9 | Sym 0.95: noisy reduction ≤ 0 where clean > 0 | condition never met (clean reduction always < 0: −0.35 / −0.46 / −0.54) | **vacuous as written** |
| PM10 | IDN harder than sym at equal rate (0–1, r150) | 0.86 vs 1.00 | ✓ |
| PM11 | Normal-approximation mis-ranking within 0.05 at every n_h | within 0.03 at n_h ≥ 100; off by up to 0.096 (n_h 25) and 0.085 (n_h 50) | **✗** |
| PM12 | Logit scaling moves CE's ranking (Spearman < 0.9 for ×0.5 and ×2); 0–1 and TS-local ≥ 0.99 | CE 0.78 (×0.5) but 0.97 (×2); 0–1 1.00, TS-local ≥ 0.999 | **✗** (CE reacts to flattening, not sharpening) |

Score: 7 pass (PM1, 2, 3, 5, 6, 8, 10), 3 fail (PM7, 11, 12), 1 vacuous (PM9), 1 without a prediction (PM4).
- **PM9** is vacuous, but the sign reversal it targeted was observed with the predicted size: noisy reductions of +0.020 / +0.024 / +0.028 against a(0.95)·clean = +0.019 / +0.026 / +0.030.
- **PM7** failed as written. The direction is right, though: the reversal fraction rises with η and saturates at 0.8, consistent with the (2η − 1) per-unit margin of Prop. 3 competing with error-fixing moves.

**What Stage 7 establishes:**
1. **Confirmed on fresh seeds.** Cross-entropy and Brier reductions misrank clients under heterogeneous label noise: they rank rare clean clients *below* noisy ones (AUROC 0.24–0.54; Spearman with clean utility −0.36 to −0.65). Calibrating on clean server data does not help (TS-global). The mechanism is confidence:
   - noisy probes' local updates *raise* predictive entropy, monotonically in η: sym 0.6 gives +0.60 / +1.13 / +1.28 nats at rounds 20 / 60 / 150; sym 0.95 gives +1.56;
   - clean probes' updates lower it (−0.24 to −1.15);
   - CE rewards this flattening (Stage 6, point 4, confirmed).
2. **0–1 error is *not* the uniquely right statistic** (the user's "do not assume 0–1 will win").
   - Every bounded, symmetric or truncated loss (MAE, GCE, CE clipped at log K) preserves the information as well as 0–1, and the continuous ones are *less noisy*.
   - At n_h = 25, 0–1's AUROC is 0.87; clipped CE and MAE reach ≥ 0.999.
   - Clipped CE has the best ranking fidelity overall (pairwise 0.82 vs 0.78).
   - 0–1's advantages are exact theory (PM1) and exact calibration invariance; its cost is variance.
3. **Calibrated CE fixes the symmetric case only.** A per-model temperature fitted on the client's own noisy holdout makes CE as good as 0–1 under symmetric noise. Under class-conditional noise below the reversal threshold it fails (AUROC 0.31): the local update learns the corruption (ΔD = +0.03 to +0.07 at pair 0.4–0.45), and a single temperature cannot undo a structured confidence gain. **The calibration story is therefore partly right:** the global component of confidence is the dominant failure under symmetric noise, while structured (class-conditional) confidence is a residual, structural failure.
4. **No own-label statistic survives relabelling** (AUROC 0.00 for all eight). Above pair 0.5 the best is MAE at 0.75, and the fraction of updates with the wrong sign grows with η. This boundary is fundamental (Prop. 3c), not a tuning issue.
5. **Federation-level utility is not tracked by any statistic** (Spearman with U_fed −0.07 to 0.46 in the symmetric mix). The evidence concerns a client's *own-distribution* clean utility, which is what A′ uses. It does not concern its value to the federation.
6. **Rare-and-noisy probes.** Their clean utility is positive early (+0.056 at round 20) and negative later (−0.16 at round 150). The 0–1 statistic follows this (AUROC vs common 0.90 → 0.01), so the predicted demotion of rare-noisy clients (Stage 4, point 3) had no harmful effect here.

---

## Stage 8 — Noise boundaries (measurement grids + FL runs on confirmatory seeds 20–22)

### 8a. Measurement-level boundaries (Stage 7 probes, plus a rarity grid added before running, §14.3)

| Axis | Finding | Theory |
|---|---|---|
| Sym rate η | 0–1 reductions scale as a(η) (PM1). Above η = 0.9 they reverse sign with the predicted size (sym 0.95: +0.020 / +0.024 / +0.028 vs a·clean = +0.019 / +0.026 / +0.030). Over sym 0.2–0.95, 0–1's AUROC falls to 0.91 at r150, while MAE, clipped CE and TS-local stay at 1.00 | Prop. 1; a < 0 beyond (K−1)/K |
| Noise type | Pair ≤ 0.45 and IDN: bounded statistics 0.92–1.00. Pair ≥ 0.55: graded failure (0–1 0.46, MAE 0.75). Relabelling: total failure for every statistic. TS-local CE fails already at pair ≤ 0.45 (0.31) | Prop. 3 (negative margins); structured confidence |
| Held-out n_h | 0–1 needs n_h ≳ 100 (AUROC 0.87 / 0.93 / 0.97 at n_h 25 / 50 / 100). MAE and clipped CE are ≥ 0.998 at n_h 25. The normal-approximation mis-ranking formula holds to 0.03 at n_h ≥ 100, not below (PM11 ✗) | Lemma F1; 0–1 is a ±1/0 mean, high variance |
| Rarity (1 / 5 / 10 rotated clients in the base federation) | As the rare group gets served, RR probes' clean utility at r150 falls from 0.31 to 0.030 to 0.003. 0–1 AUROC at r150 falls 1.00 → 0.99 → 0.94 (**pre-stated prediction ✓**). MAE and clipped CE stay at 1.00, TS-local 0.98, CE 0.54 → 0.24 → 0.19 | mis-ranking ∝ Φ(−(ρ_R − α ρ_N)/σ): the signal vanishes when ρ_R → 0 |

### 8b. FL boundary runs

Settings: 30 common + 5 rotated + 15 noisy clients; T = 200; seeds 20–22. Each cell is rare / common accuracy, mean of 3 seeds.

| Method | S1: sym 0.6 | HET: sym 0.2–0.95 | P45: pair 0.45 | P75: pair 0.75 | IDN: η̄ 0.4 |
|---|---|---|---|---|---|
| FedAvg | 0.842 / 0.951 | 0.840 / 0.951 | 0.829 / 0.954 | 0.832 / 0.927 | 0.817 / 0.936 |
| FedEBA-style | 0.756 / 0.901 | 0.757 / 0.886 | 0.794 / 0.890 | 0.748 / 0.648 | 0.800 / 0.860 |
| FedEBA-style, weights capped at 2/m | 0.802 / 0.931 | 0.816 / 0.933 | 0.813 / 0.911 | 0.786 / 0.835 | 0.815 / 0.890 |
| A (CE reduction) | 0.782 / 0.945 | 0.777 / 0.928 | 0.802 / 0.940 | 0.726 / 0.689 | 0.853 / 0.927 |
| **A′ (0–1 reduction)** | **0.877 / 0.948** | 0.858 / 0.922 | 0.845 / 0.938 | 0.713 / 0.759 | 0.854 / 0.920 |
| A′ shuffled (equal resources) | 0.831 / 0.947 | 0.825 / 0.948 | 0.818 / 0.949 | 0.815 / 0.862 | 0.805 / 0.933 |
| A′-MAE | **0.885 / 0.949** | **0.870 / 0.937** | **0.878 / 0.950** | 0.728 / 0.716 | **0.883 / 0.941** |
| A′-TS-local CE | 0.873 / 0.947 | 0.857 / 0.944 | 0.825 / 0.943 | 0.760 / 0.780 | 0.862 / 0.931 |
| A′-Brier | 0.806 / 0.943 | 0.794 / 0.937 | 0.802 / 0.930 | 0.693 / 0.687 | 0.855 / 0.921 |
| FedPCA-style (our re-implementation) | 0.841 / 0.952 | 0.854 / 0.953 | 0.469 / 0.955 (unstable: 0.87 / 0.34 / 0.19) | 0.779 / 0.954 | 0.829 / 0.949 |
| clean-label oracle (EBA on clean labels) | 0.875 / 0.947 | 0.874 / 0.948 | 0.867 / 0.949 | 0.855 / 0.865 | 0.855 / 0.923 |
| cluster-wise FedAvg (B4sc_n) | **0.922 / 0.955** | — | — | — | — |

| Prediction | Result | Verdict |
|---|---|---|
| PF1: S1, A′ > FedAvg in 3/3; EBA < FedAvg in 3/3 | +3.6 / +2.4 / +4.5; EBA −8.1 to −8.9 | ✓ |
| PF2: HET, A′ ≥ FedAvg + 2 (mean), > in ≥ 2/3; EBA < FedAvg | +2.2 / +0.6 / +2.5 (mean **+1.8**); EBA −8.4 | **✗** (by 0.2 pt); common cost 2.9 pts |
| PF3: P45, A′ ≥ FedAvg + 2; common cost ≤ 2 | +1.7 / +0.8 / +2.5 (mean **+1.7**); cost 1.5 | **✗** (rare-gain clause) |
| PF4: P75, A′ common ≤ FedAvg − 3 in ≥ 2/3 seeds | −25.0 / −24.7 / −0.6 | ✓ (predicted failure) |
| PF5: IDN, A′ ≥ FedAvg + 2; common cost ≤ 1.5 | +3.7 (✓); cost **1.6** | **✗** (by 0.1 pt) |
| PF6: variants (no direction) | A′-MAE beats A′ on rare accuracy in all 5 settings (+0.8 / +1.2 / +3.3 / +1.5 / +2.9) and on common in 4/5. TS-local CE: within ~1 pt in S1, HET, IDN; 2.1 below at P45. Brier and CE: 4–9 pts below A′ on rare accuracy in S1, HET and P45; level with A′ in IDN; everything fails in P75 | — |
| PF7: capped EBA ≥ 2 pts below A′ (S1, HET) | −7.5 / −4.2 | ✓ |
| PR7: T1′ guard silent on pair 0.75 | 0 flags; identical to A′ | ✓ (the guard's blind spot) |

**Reading** [obs; interp marked]:
1. **The symmetric-noise result replicates on new seeds** (PF1): A′ +3.5 pts over FedAvg, EBA −8.6.
2. **Outside symmetric noise A′'s margin is thin.** Mean rare gains are +1.7 to +3.7 pts, against common costs of 1.5–2.9 pts. Three of the four quantitative predictions for these settings missed their thresholds (by 0.1–0.3 pt).
3. **[interp] HET's 2.9-pt common cost** is consistent with the η = 0.95 clients (a < 0). In the measurement study their 0–1 reduction is *positive* (+0.02 to +0.03) while common clean clients' is ≤ 0, so after the max(·, 0) step they outrank clean clients.
4. **Past the pair-reversal boundary (P75) every own-label statistic fails.** A′ common drops to 0.64–0.69 in 2/3 seeds, and the clean-label oracle loses 6 pts as well. The T1′ guard does not fire, because pair-0.75 clients' observed error (≈ 0.77) is below chance.
5. **0–1 error is not the best statistic in FL either.** The MAE variant (pre-registered as a competitor) is better than 0–1 in every setting. The calibrated-CE variant (TS-local) is competitive under symmetric noise and IDN, as the measurement study predicted, but not under pair noise.
6. **FedPCA-style is unstable in our hands** (P45: 0.87 / 0.34 / 0.19). No conclusion about FedPCA itself should be drawn from it.

### 8c. Post-hoc check on reserved seeds 30–32 (`results/P2X.jsonl`; registered in §14.3 before running)

The measurement study ranked clipped CE first, so A′ with a clipped-CE statistic was added. Because the addition was chosen after seeing Phase II data, it was evaluated on the reserved seeds, with FedAvg, A′ (0–1) and A′-MAE alongside. Each cell is rare / common accuracy, mean of 3 seeds.

| Setting | FedAvg | A′ (0–1) | A′-clipped CE | A′-MAE |
|---|---|---|---|---|
| S1: sym 0.6 | 0.839 / 0.949 | 0.884 / 0.944 | 0.884 / 0.947 | **0.885 / 0.947** |
| HET: sym 0.2–0.95 | 0.845 / 0.951 | 0.860 / 0.919 | **0.880 / 0.943** | 0.872 / 0.942 |
| P45: pair 0.45 | 0.825 / 0.953 | 0.837 / 0.936 | 0.820 / 0.927 | **0.875 / 0.946** |
| IDN: η̄ 0.4 | 0.814 / 0.934 | 0.858 / 0.914 | 0.864 / 0.921 | **0.885 / 0.935** |

- **Prediction (A′-clipped CE within 1 pt of A′ in all four settings): ✗.** It holds in S1 and IDN; clipped CE is 2.0 pts better in HET and 1.7 pts worse in P45. The best measurement statistic did not carry over uniformly to FL.
- **A′-MAE replicates on reserved seeds.**
  - Against A′ (0–1): at least as good on rare and common accuracy in all four settings (rare +0.1 / +1.2 / +3.8 / +2.7; common +0.3 / +2.3 / +1.0 / +2.1).
  - Against FedAvg: rare +4.6 / +2.7 / +5.0 / +7.1 at common −0.2 / −0.9 / −0.7 / +0.1.
  - Over the eight setting × seed-set combinations outside P75 (S1, HET, P45 and IDN, each on seeds 20–22 and 30–32), A′-MAE has the **highest rare accuracy of all non-oracle one-model rules in 7 of 8**. The exception is clipped CE in HET on seeds 30–32. Its common accuracy is within 1.5 pts of FedAvg in all 8. A switch from 0–1 to MAE would be a method change made after Phase II data. By the ledger rule it now has one discovery evaluation (seeds 20–22, where it was a pre-registered competitor) and one confirmation (seeds 30–32).
- **A′ (0–1) replicates its pattern.** The rare gain is large under symmetric noise (+4.5), and outside it the common costs are 1.7–3.2 pts.

---

## Stage 9 — The relabelled-client failure as a boundary case (confirmatory seeds 20–22)

**Diagnosis (measurement level, Stage 7 probes).**
- Relabelled (CF) probes produce the **largest** 0–1 reductions of all 17 channels: r̃ = +0.41 / +0.47 / +0.46 at rounds 20 / 60 / 150, against +0.22 / +0.08 / +0.04 for rare clean probes.
- Meanwhile their clean utility is −0.58 / −0.64 / −0.64.
- The local update learns the swapped map: ΔD = +0.41 to +0.47. Prop. 3c says r̃ = ΔD exactly; the observed max |r̃ − ΔD| over all 54 CF probe-checkpoints is 1e-16.
- The reversal is total (PM8: 36/36 probe-checkpoints) and holds for **every** own-label statistic (AUROC 0.00 for all eight).
- This is not an A′ bug. It is the identifiability boundary of any statistic computed against a client's own labels.

**FL test** (population of E3d: 30 common, 5 rare, 10 sym-0.6, 5 CF clients). Guards were pre-specified in §14.2. Accuracy columns are seed values for seeds 20 / 21 / 22.

| Method | Common accuracy | Rare accuracy | Worst-10 % | CF weight share | Guard triggers |
|---|---|---|---|---|---|
| FedAvg | 0.949 / 0.914 / 0.952 | 0.840 / 0.853 / 0.826 | 0.81–0.82 | 0.09–0.10 | — |
| A′ (unguarded) | **0.554 / 0.417 / 0.780** | 0.820 / 0.735 / 0.808 | 0.34–0.68 | 0.25–0.27 | — |
| A′ + **T1′ exclusion** (observed error > 0.95 after round 20) | **0.948 / 0.947 / 0.948** | **0.888 / 0.888 / 0.877** | 0.86–0.87 | 0.03 (warm-up only) | 162–180, all on CF clients; 0 on others |
| A′ + weight cap 2/m | 0.816 / 0.935 / 0.945 | 0.834 / 0.827 / 0.818 | 0.71–0.81 | 0.17–0.19 | — |
| A′ + fallback to FedAvg in flagged rounds | 0.943 / 0.918 / 0.946 | 0.863 / 0.856 / 0.849 | 0.83–0.84 | 0.11–0.13 | 160–168, all on CF |
| B (K̂ ≤ 4, silhouette, post-stability) → FedAvg per cluster | 0.956 / 0.955 / 0.959 | **0.920 / 0.924 / 0.902** | **0.88–0.91** | own model (isolation 1.00) | — |
| B → A′ per cluster | 0.953 / 0.954 / 0.951 | 0.918 / 0.866 / 0.903 | 0.86–0.90 | own model (1.00) | — |
| oracle routing → A′ | 0.947 / 0.942 / 0.950 | 0.892 / 0.874 / 0.879 | 0.86–0.87 | own model | — |

| Prediction | Result | Verdict |
|---|---|---|
| PR1: A′ common < 0.90 in ≥ 1/3 seeds; FedAvg ≥ 0.92 in 3/3 | A′ < 0.90 in 3/3 (0.42–0.78); FedAvg 0.914 in seed 21 | **✗** (on the FedAvg clause: relabelled clients also dent FedAvg) |
| PR2: T1′ exclusion: common ≥ 0.94 in 3/3 and rare ≥ A′ on P0 − 2 pts | 0.947–0.948; rare 0.884 vs 0.877 | ✓ |
| PR3: weight cap does not remove the failure (worst seed < 0.92) | worst seed 0.816 | ✓ |
| PR4: fallback: common ≥ 0.94 in 3/3 | 0.918 in seed 21 | **✗** (falling back to FedAvg re-admits CF updates at FedAvg weight) |
| PR5: B4sc_n worst-10 % ≥ 0.85 in 3/3 | 0.883–0.909 | ✓ |
| PR6: on P0 (no CF), the T1′ guards cost ≤ 1 pt rare | never triggered; identical runs (Δ = 0.000) | ✓ |
| PR7: on pair-0.75 clients the T1′ test never triggers | see Stage 8 | (Stage 8) |

**Conclusions** [obs + proof]:
1. **The failure is a boundary of own-label statistics, and theory supplies the guard.** T1′ says that under symmetric noise a better-than-chance model never shows an observed error above (K−1)/K on any client. A Hoeffding-margin threshold on the global model's observed error (a *level*, not a reduction) isolates relabelled clients:
   - 512 / 512 flags on CF clients, none on others;
   - it restores A′ fully: common 0.948, rare 0.884;
   - it costs nothing when no CF client exists.
   
   The guard is specific to *deterministic or near-deterministic relabelling* (observed error near 1). Its blind spot is pair noise between 0.5 and the threshold (PR7, Stage 8).
2. **Clipping and fallback are not substitutes.** Capping shrinks but does not remove the damage, and falling back to FedAvg inherits FedAvg's own vulnerability.
3. **Cluster-wise FedAvg beats every guarded one-model rule** on rare accuracy (0.915 vs 0.884) and worst-10 % (0.90 vs 0.87), at no common cost. This replicates E5b and E9 on new seeds. Whenever a model per cluster is acceptable, personalisation dominates fairness re-weighting in this testbed.

---

## Stage 10 — Fresh novelty search

**Method.**
- Three literature sub-agents ran WebSearch / WebFetch on:
  - noisy-label FL and client quality;
  - validation-improvement weighting, selection and valuation;
  - noisy-label risk estimation outside FL.
- Every closest prior was then **re-fetched by hand** in this session (tag [web-verified] in `literature_matrix.md` §G).
- Items only a sub-agent read are tagged [agent-read]. Nothing below is cited beyond what was read.

### (1) The statistical principle

| Proposed contribution | Closest prior | Shared idea | Genuine difference | Why the difference matters | Novelty risk |
|---|---|---|---|---|---|
| Noisy 0–1 risk is affine in clean risk; ranking preserved iff η < (K−1)/K | Ghosh, Kumar & Sastry, AAAI 2017, Thm 1; **Toner & Storkey, arXiv 2409.06830, Fact 1** [web-verified] | identical formula; noisy validation accuracy selects the clean-best model | none in the identity | — | **very high (known)** |
| Exact proportionality only under symmetric noise | Toner & Storkey, Fact 2 [web-verified]: symmetric noise is the only model under which noisy and clean minimisers always coincide | uniqueness of symmetric noise | ours is stated for all pairwise differences, theirs for minimisers | small | **high** |
| Margin decomposition; sign kept for cell-monotone updates; reversal on negative margins | **Chen et al., AAAI 2021, App. A Eq. 12** [web-verified] (gap to the optimum under diagonal dominance); Natarajan 2013 and Menon 2015 (binary / balanced) [agent-read] | margin-weighted confusion | pairwise rather than gap-to-optimum; pair > ½ and relabelling named as reversal cells for client statistics | explains A′'s relabelled-client failure and its pair boundary | **high** (elementary corollary) |
| CE / Brier *reductions* of client updates conflate confidence with correctness; TS-local fixes only symmetric noise; every *level* fails | Ghosh 2017 and Ma 2020 (CE not robust as a training loss); Olmin & Lindsten 2022 (noisy-trained models are miscalibrated); TransTS 2026 (TS under noisy calibration labels) [agent-read] | CE is not noise-robust | used as a *valuation statistic of federated client updates*, not a training loss. Measured on 1,008 probe updates with pre-registered tests, including the TS-local / TS-global dissociation and the class-conditional residual | decides which statistics loss-driven FL may use | **medium** |
| "Bounded statistics suffice; 0–1 is not special" | Ghosh 2017 (MAE symmetric); Zhang & Sabuncu 2018 (GCE) [agent-read] | robust losses | as client-update statistics | MAE / clipped CE are the lower-variance choice | **medium-high** (follows from known robustness) |
| Federated framing: per-client offsets cancel in before/after differences, scaled by a(η_i) | Scott, Luo & Ho, arXiv 2609.31454 [web]; FedPCA [web]; RHFL, FedGA [web-verified] | client loss signals under noise | the cancellation-and-attenuation statement for client-level reductions | justifies own-holdout reductions and rules out levels | **medium** (one-line consequence of the known identity) |

### (2) A′

| Proposed contribution | Closest prior | Shared idea | Genuine difference | Why the difference matters | Novelty risk |
|---|---|---|---|---|---|
| Aggregation weight from a client's before/after improvement | **FedGA** (Zhang et al., CVPR 2023) [web-verified]: weights moved toward clients with a larger gap between global- and local-model loss on their training data | weight ∝ reducible loss of the client | held-out split; 0–1 instead of CE; exponential weights with clamp; label noise | a training-data CE drop rewards noisy clients. Our own `traindrop` baseline (E1) scored rare 0.780 < FedAvg 0.833 | **high** |
| Held-out improvement as weight | **FedFomo** (Zhang et al., ICLR 2021) [web-verified]: (L_i(old) − L_i(θ_n))/‖Δθ‖ on client i's own validation split, clipped at 0 | held-out improvement → weights | global model rather than personalised; noise | — | **high** |
| Loss-drop weights against noisy clients | **RHFL** (Fang & Ye, CVPR 2022) [web-verified]: (1/SL loss) × round-to-round SL drop, softmax | loss drop → softmax weights under label noise | held-out; 0–1; no 1/loss factor (which would penalise hard clean clients) | — | **medium-high** |
| Exponential weights on held-out learning progress | Graves et al., ICML 2017; DoReMi, NeurIPS 2023; RHO-LOSS, ICML 2022 [agent-read] | reducible vs irreducible loss | federated, client-level, no reference model | — | medium |
| Rare-vs-mislabelled allocation problem | **FedPCA** (arXiv 2503.10567) [web] | same three-group problem | signal | — | **high** (problem level) |

### (3) B

| Proposed contribution | Closest prior | Shared idea | Genuine difference | Why the difference matters | Novelty risk |
|---|---|---|---|---|---|
| Loss-vector clustering with cluster→model matching | **CLoVE** (arXiv 2506.22427; ICML 2026 seed paper) [web-verified] | identical pipeline | per-client z-normalised 0–1 vectors | invariance to per-client symmetric noise (CLoVE ARI 0.12–0.49 vs B 1.00 at η ≥ 0.5) | **high** ("CLoVE with a different statistic") |
| Within-client differencing removes client offsets | **LCFL** (Gu et al., arXiv 2407.09360) [web-verified] | loss differences within a client | 0–1; scale normalisation; noise model | — | medium |
| Assignment by best-fitting model | IFCA (NeurIPS 2020) [agent-read] | same signal family | k-means on z-vectors vs argmin | **Stage 11:** with trained models, the 0–1 argmin matches B to within 0.017 ARI. In dynamics, B avoids IFCA's merge failure, but that is CLoVE's k-means design | **high** |
| Noise-robust clustered PFL | RCC-PFL (arXiv 2503.19886); FB-NLL (arXiv 2604.19729) [agent-read] | same problem | uses labels | [interp] label-free clustering cannot separate clusters that differ only in labelling function | medium |
| Per-client noise-rate estimate | FedCorr, FedDiv, FedGR, FedA3I (loss-GMM fractions) [agent-read] | per-client η̂ | 0–1 offset against the cluster median | — | medium |
| Collaborator selection | FedSC (Comput. Commun. 2025); FedCollab (ICML 2023) [web-verified, abstracts] | who should share a model | they use distribution similarity; no label noise | — | low |

---

## Stage 11 — B audit: does the vector representation carry information that simpler alternatives lack?

### 11a. Static audit (seeds 10–12; `phase2_baudit.py`; `results/P2_baudit.jsonl`)

**Design.**
- K = 4 cluster models from oracle-routed CFL, on rotation and label-permutation structures, frozen at round 80 ("trained") and round 10 ("early").
- Probe pools of 40 clients (10 per cluster, 3 noisy), drawn from data disjoint from training: 8 noise levels × 5 sample sizes × 3 pools × 2 structures × 3 seeds.
- Each representation is clustered by k-means with K known, or assigned by argmin.

**ARI with trained models** (n = 500 / n = 100):

| Representation | η 0 | 0.2 | 0.4 | 0.6 | 0.8 | 0.9 | 0.95 | pair 0.4 |
|---|---|---|---|---|---|---|---|---|
| CE vector (CLoVE) | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.98 | 0.59 / 0.51 | 0.41 / 0.41 | 0.40 / 0.40 | 0.40 / 0.40 | 0.97 / 0.97 |
| z-CE | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.99 | 0.87 / 0.85 | 0.48 / 0.48 | 0.38 / 0.38 | 0.38 / 0.37 | 1.00 / 1.00 |
| TS-local CE vector | 1.00 / 0.95 | 1.00 / 0.97 | **0.71 / 0.56** | 0.41 / 0.40 | 0.41 / 0.40 | 0.41 / 0.40 | 0.41 / 0.40 | 0.51 / 0.47 |
| raw 0–1 vector | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.89 / 0.82 | 0.40 / 0.40 | 0.40 / 0.40 | 0.40 / 0.40 | 1.00 / 1.00 |
| **z-0–1 (B)** | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | **1.00 / 1.00** | **1.00 / 0.91** | 0.46 / 0.49 | 0.35 / 0.34 | 1.00 / 1.00 |
| z-Brier | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.94 | 0.66 / 0.57 | 0.40 / 0.40 | 0.37 / 0.37 | 1.00 / 1.00 |
| z-MAE | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | **1.00 / 0.94** | 0.50 / 0.47 | 0.35 / 0.33 | 1.00 / 1.00 |
| **argmin 0–1** (IFCA with 0–1, CE tie-break) | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | **1.00 / 1.00** | **1.00 / 0.91** | 0.46 / 0.47 | 0.33 / 0.33 | 1.00 / 1.00 |
| argmin CE (IFCA) | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.99 | 0.84 / 0.83 | 0.47 / 0.46 | 0.37 / 0.38 | 0.36 / 0.35 | 1.00 / 1.00 |
| scalar mean error | 0.38 / 0.23 | 0.14 / 0.07 | 0.18 / 0.06 | 0.17 / 0.08 | 0.15 / 0.08 | 0.16 / 0.09 | 0.18 / 0.09 | 0.16 / 0.09 |

Early models (round 10) give the same picture: z-0–1 = argmin 0–1 to within 0.01.

| Prediction | Result | Verdict |
|---|---|---|
| PB1: z-0–1 ≥ 0.95 for η ≤ 0.6 at n ≥ 300; raw CE < 0.6 at η ≥ 0.6 | 1.00 everywhere; raw CE 0.59 / 0.41 | ✓ |
| PB2: argmin 0–1 within 0.05 of z-0–1 in every cell with η ≤ 0.8 (trained) | max difference 0.017 | ✓ — **once models are trained, B's vector adds nothing beyond the 0–1 argmin** |
| PB3: η = 0.95 probes misassigned ≥ 50 % | 98.5 % (z-0–1), 98.8 % (argmin). At η = 0.9 (a = 0): 75 % = chance for K = 4 | ✓ — random at a = 0, mirror-image at a < 0, as Stage 5 predicts |
| PB4: failing cells at a√n within ×1.5 | only one failing cell (η 0.8, n 50: a√n = 0.79, ARI 0.84). ARI 0.91 at a√n = 1.11 and 0.985 at 1.36 | **not informative** — the comparison needs ≥ 2 failing cells, and only η = 0.8 approaches the threshold. The static threshold sits near a√n ≈ 1, lower than the dynamic breakdown in E2c (ARI 0.78 at a√n = 1.36) |

**Further static findings:**
- **Calibration hurts clustering vectors.** TS-local CE vectors fail from η = 0.4. Fitting a temperature per model on noisy data flattens every model toward the noise-entropy floor, which erases the between-model differences that clustering needs.
- **z-MAE matches or slightly beats z-0–1** (η 0.8, n 100: 0.94 vs 0.91). Normalised CE degrades from η = 0.6 (0.87 / 0.85) and normalised Brier from η = 0.8 (0.66 / 0.57).

### 11b. Dynamic audit (fresh CFL seeds 10–12; `results/P2B.jsonl`; E2 configuration T = 80, m = 20)

ARI per seed for seeds 10 / 11 / 12:

| Setting | B (z-0–1, k-means) | IFCA with 0–1 argmin (new) | IFCA (CE) | CLoVE-style (CE) |
|---|---|---|---|---|
| rot, no noise | 1.00 / 1.00 / 1.00 | 1.00 / **0.70** / 1.00 | 0.70 × 3 | 1.00 × 3 |
| rot, 30 % sym 0.6 | 1.00 × 3 | **0.70 / 0.70** / 1.00 | 0.35 × 3 | 0.41 / 0.35 / 0.35 |
| rot, 50 % sym 0.6 | 1.00 × 3 | **0.69 / 0.70** / 1.00 | 0.13 × 3 | 0.13 / 0.15 / 0.13 |
| rot, 30 % sym 0.8 | 1.00 / **0.62** / 1.00 | 0.69 / 0.66 / 1.00 | 0.35 × 3 | 0.35 × 3 |
| perm, 30 % sym 0.6 | 1.00 × 3 | 1.00 × 3 | 0.41 × 3 | 0.52 / 0.35 / 0.41 |

| Prediction | Result | Verdict |
|---|---|---|
| PB5: IFCA with 0–1 argmin below B in ≥ 3 of 5 settings | 4 of 5 (equal under permutation) | ✓ |

**Findings:**
- **New failure.** B failed on a fresh seed at η = 0.8, n = 500 (seed 11: ARI 0.62). Its record in that setting is now 8/9 seeds.

**B-audit verdict** [obs + interp]:
1. **Against the simplest alternative, the 0–1 argmin:**
   - Statically, B's vector carries **no** extra information (PB2).
   - Dynamically, k-means on the vectors avoids IFCA's cluster-merging failure (0.70 = two clusters merged) in rotation settings.
   - That advantage is **CLoVE's** design (k-means on loss vectors rather than an argmin), which CLoVE itself claims over IFCA. It is not B's.
2. **Against CLoVE**, B's entire advantage is the statistic. Replacing CLoVE's CE vector by the per-client z-normalised 0–1 vector (or z-MAE, statically) raises ARI from 0.13–0.52 to 1.00 in 11 of 12 noisy runs. The same switch of statistic also repairs IFCA (CE argmin 0.13–0.41 → 0–1 argmin 0.66–1.00).
3. B is therefore "CLoVE with a noise-invariant statistic". It survives only as an instantiation of the measurement principle, not as a standalone method. Its failure modes:
   - η ≥ 0.9 (mirror assignment);
   - a√n ≲ 1;
   - occasional dynamic failure at η = 0.8.

---

## Stage 12 — A′ audit: is A′ more than validation-performance aggregation?

**On mechanism: no.** A′ belongs to the published family that weights clients by validation improvement (FedGA, FedFomo, RHFL; Zeno/Zeno++ on the server side). Its distinctive choices are:
1. the improvement is measured on a held-out split of the client's *own, possibly noisy* data;
2. it uses a *bounded* statistic;
3. it uses FedEBA+'s clamped exponential weights with a fairness goal.

**Which of these matter.** Evidence, rare-accuracy differences in points:

| Variant | Statistic | Data | Evidence |
|---|---|---|---|
| traindrop (FedGA / RHFL-like) | CE drop | training data | E1 (seeds 0–2): 0.780 vs FedAvg 0.833 → **fails** |
| A | CE drop | held-out | E1: 0.786; Phase II S1: 0.782 vs FedAvg 0.842 → **fails** |
| A′-Brier | Brier drop | held-out | S1: 0.806 → **fails** |
| A′-TS-local | calibrated-CE drop | held-out | S1: 0.873 → works |
| **A′** | 0–1 drop | held-out | S1: 0.877 → works |
| A′-MAE | MAE drop | held-out | S1: 0.885 → works (best) |
| A′ shuffled | 0–1, permuted | held-out | S1: 0.831 → the gain needs the client–statistic link |
| capped EBA | CE level | training data | S1: 0.802 → bounding the weights is not enough |

So the ingredient that matters is the **bounded or calibrated statistic of the update**. The held-out split alone does not make CE work, and the clamp alone does not make level weighting work. Whether a bounded statistic also *needs* the held-out split was not tested. A′ is better described as *validation-improvement aggregation with a noise-robust statistic*. Within the robust class:
- under symmetric noise (S1) the choice barely matters: MAE ≈ 0–1 ≈ TS-local CE, within ~1 pt;
- outside symmetric noise MAE is consistently better: +1.2 to +3.8 pts over 0–1 (Stage 8).

**Verdict.** As an algorithm, A′ is incremental over FedGA / FedFomo / RHFL. Its defensible content is the statistic choice, which belongs to the measurement principle, not to A′ itself. In addition:
- cluster-wise FedAvg beats it whenever the rare group can be isolated (S1: 0.922 vs 0.877; P_CF: 0.915 vs 0.884 guarded);
- it fails without a guard when relabelled clients exist.

---

## Anti-confirmation checklist (each item asked for in the brief)

| Question | Answer | Evidence |
|---|---|---|
| Does calibrated CE solve the issue? | **Partly.** A per-model temperature fitted on the client's own noisy holdout (TS-local) solves *symmetric* noise; it fails under class-conditional noise. A temperature fitted on clean server data (TS-global) does not help at all. For clustering vectors, calibration hurts | Measurement AUROC: TS-local 1.00 (sym), 0.31 (pair ≤ 0.45); TS-global 0.26. FL: A′-TS-local within ~1 pt of A′ in S1 / HET / IDN, −2.1 at P45. Static B audit: TS-CE vectors fail from η = 0.4 |
| Does the Brier score perform equally well? | **No** | AUROC 0.43 (0.00 at η ≥ 0.6); FL 4–9 pts below A′. Theory: the confidence term ‖p‖² is not attenuated, and Brier reverses for flattening once η > 0.45 |
| Does clipping match A′? | **Weight clipping: no. Loss clipping: mixed.** | Capped EBA is 7.5 / 4.2 pts below A′ (S1 / HET). Clipped CE is the best *measurement* statistic, but in FL it is within 1 pt of A′ in only 2 of 4 settings (+2.0 in HET, −1.7 in P45; reserved seeds) |
| Does cluster-wise FedAvg match B + A′? | **It beats it** | P_CF rare 0.915 vs 0.896 (B → A′) vs 0.884 (guarded A′); S1 0.922 vs A′ 0.877; same in E5b and E9 |
| Does the phenomenon disappear on CNNs / is it an MLP-MNIST artefact? | **Unknown — not tested.** No GPU; PyTorch wheels are blocked. The mechanism (noisy updates raise predictive entropy; CE rewards it) is architecture-agnostic in theory, but magnitudes depend on calibration, which differs for CNNs | — |
| Do heterogeneous η or class-conditional noise break the theory? | **The theory holds and predicts where the *methods* break.** | Heterogeneous η: offset cancels and slope = a(η) (PM1), but η > 0.9 reverses (PM9 magnitude; HET common cost −2.9). Class-conditional: the margin identity is exact (r̃ = ΔD to 1e-16 for CF), with reversals graded in η (PM7 failed as a threshold but holds in direction) |
| Does prior work already contain the theorem? | **Yes** | Ghosh et al. 2017 Thm 1; Toner & Storkey 2024 Facts 1–2; Chen et al. 2021 Eq. 12 (all verified on fetched pages) |
| Is A′ an existing valuation method with a new metric? | **Essentially yes** | FedGA (CVPR 2023), FedFomo (ICLR 2021) and RHFL (CVPR 2022) weight clients by (held-out) loss improvement. A′'s substantive difference is the bounded statistic, and MAE does that job better |
| Is B just CLoVE with a different statistic? | **Yes** | Static: argmin 0–1 ≡ B (≤ 0.017 ARI). Dynamic: B's edge over argmin is CLoVE's k-means design. The CLoVE → B gain (ARI 0.13–0.52 → 1.00) comes entirely from the statistic |

## Paper-thesis test

The thesis: *"Federated systems frequently use client losses as proxies for usefulness, similarity or fairness … derive federated aggregation and clustering mechanisms from this principle."*

| Clause | Supported? | Evidence |
|---|---|---|
| FL systems use client losses as proxies for usefulness (PoC, Oort), similarity (CLoVE, IFCA) and fairness (FedEBA+, q-FFL) | yes [literature] | seed papers; §G of the literature matrix |
| Under heterogeneous label quality these proxies misrepresent clients | **yes, in this testbed** | every loss *level* fails (AUROC 0.00); CE / Brier *reductions* anti-correlate with clean utility; CE vectors and CE argmin fail at η ≥ 0.6 |
| 0–1-based statistics preserve the information under identifiable noise | **yes, but not specific to 0–1** | MAE and clipped CE do as well or better; only up to the identifiability boundary (pair > ½, relabelling, η ≥ (K−1)/K) |
| Aggregation and clustering mechanisms *derived* from the principle are a contribution | **no** | the derived mechanisms are published mechanisms with the statistic swapped (FedGA / FedFomo / RHFL; CLoVE / IFCA), and simpler alternatives beat them (cluster-wise FedAvg; MAE) |

**Verdict.** The thesis does not hold in every clause and should not be used as stated.

---

## Stage 13 — Decision gate

**Chosen framing: E — revise.** A′ and B are abandoned as method contributions. The only surviving content is a narrowed version of A (a measurement claim). At its current evidence level that is *not* a main-track contribution.

**Why not each alternative:**
- **B (A′ paper):** A′'s mechanism is published (FedGA, FedFomo, RHFL). Its own MAE variant dominates it. Outside symmetric noise its gains are +1.7 to +3.7 pts against 1.5–2.9 pt common costs; 3 of 5 quantitative FL predictions missed. Cluster-wise FedAvg beats it by 4.5 pts whenever a rare group can be isolated.
- **C (B paper):** B is CLoVE with a noise-invariant statistic. The static audit shows its vector adds nothing over a 0–1 argmin, and its dynamic edge is CLoVE's own design.
- **D (unified):** the "derive mechanisms" clause fails (paper-thesis test), and neither mechanism survives on its own.
- **A as originally framed ("0–1 error statistics"):** the "0–1" specificity is refuted (MAE and clipped CE are equal or better), and the risk identity is known (Ghosh 2017; Toner & Storkey 2024).

**What a narrowed A would claim**, with evidence that is solid *in this testbed*:
1. In federated learning with heterogeneous label quality, **no loss level** (CE, calibrated CE, 0–1, Brier, MAE, …) separates under-served clean clients from noisy ones.
2. **CE and Brier reductions** of client updates are confidence-confounded: noisy local updates flatten predictions, and these statistics reward that.
3. Reductions of **bounded statistics** (0–1, MAE, clipped CE) track own-distribution clean utility, up to an identifiability boundary that the margin identity locates exactly.
4. Per-client **temperature calibration** removes the symmetric-noise part of the CE failure but not the class-conditional part.
5. **Relabelled clients** defeat every own-label statistic but are caught by a chance-level test on the error *level*.

Items 1–5 are a characterisation of known robust-loss facts in a new setting. Their value depends on whether they hold beyond MNIST-scale synthetic noise, which is untested.

## Decision table

| Component | Decision | Evidence | Main risk | Next action |
|---|---|---|---|---|
| **Central hypothesis, part 1:** CE-based client statistics misrepresent clean client utility under heterogeneous label quality | **KEEP (confirmed, narrowed)** | Pre-registered, fresh seeds: CE reduction AUROC 0.24, Brier 0.43, every level 0.00 (PM2, PM3, PM5). Spearman with clean utility −0.54. Noisy updates raise predictive entropy by +0.6 to +1.6 nats. FL: A (CE) is 9.5 pts below A′ (S1) | MNIST / MLP only; own-distribution utility only (federation-level Spearman ≤ 0.46) | Only via a real-noise / CNN check, which is not authorised now (Stage 14) |
| **Central hypothesis, part 2:** suitably constructed 0–1 statistics preserve clean utility under identifiable noise | **REVISE** → "reductions of bounded / noise-affine statistics (0–1, MAE, clipped CE)". 0–1 is not special | PM1 (slopes = a(η)); AUROC ≈ 1.00 for 0–1, MAE, clipped CE; MAE better than 0–1 at small n_h (PM6) and on rare accuracy in all 9 FL setting × seed-set combinations | Theory already published | Cite the identity as known; claim only the federated characterisation |
| **A′** | **KILL** as a standalone method | Mechanism published; A′-MAE dominates; PF2 / PF3 / PF5 missed; collapses with relabelled clients unless guarded (common 0.42–0.78); fails at pair 0.75 | — | Keep only as an illustration of the statistic choice |
| **B** | **KILL** as a standalone method | Static ≡ 0–1 argmin; dynamic edge = CLoVE's k-means; the CLoVE → B gain is the statistic; 8/9 at η 0.8; mirror-assignment at η > 0.9 | "CLoVE with a different statistic" | None |
| **B + A′** | **KILL** | Cluster-wise FedAvg ≥ B → A′ (0.915 vs 0.896 rare, P_CF), replicating E5b and E9 | — | None |
| **Calibrated-CE alternative (TS-local)** | **KEEP as an explanatory control** | Fixes symmetric noise (AUROC 1.00; FL ≈ A′); fails pair ≤ 0.45 (0.31; FL −2.1); TS-global fails; hurts clustering vectors | — | Report as the calibration decomposition of the CE failure |
| **Brier alternative** | **KILL** | AUROC 0.43 (0.00 at η ≥ 0.6); FL 4–9 pts below A′; theory: unattenuated confidence term, reversal at η > 0.45 | — | None |
| MAE / clipped-CE statistics | **KEEP** as the recommended statistic if work continues | Best measurement fidelity (clipped CE pairwise 0.82; both ≥ 0.998 AUROC at n_h = 25). A′-MAE: highest rare accuracy among non-oracle one-model rules in 7/8 combinations | Clipped CE mixed in FL; MAE chosen after Phase II data (one confirmation, seeds 30–32) | Fresh-seed confirmation if anything continues |
| T1′ chance-level guard | **KEEP** (component) | 512/512 flags on relabelled clients, 0 false flags; restores A′ (common 0.948); silent with no relabelling | Blind to pair noise between 0.5 and the chance threshold (PR7) | None |
| Loss-level allocation (FedEBA-style, any loss) | **Negative result (keep)** | All levels AUROC 0.00; FedEBA-style −8.6 pts rare vs FedAvg on fresh seeds (S1); capping the weights does not rescue it | — | — |

---

## Stage 14 — Large-scale authorization gate

| Criterion | Met? | Reason |
|---|---|---|
| 1. The phenomenon survives falsification | **yes** (narrowed) | the CE / Brier confound and the universal level failure replicate on fresh seeds with pre-registered thresholds; the mechanism is confirmed |
| 2. A contribution is novel | **no** | the risk identity, its uniqueness and the margin form are published; A′ ≈ FedGA / FedFomo / RHFL; B ≈ CLoVE + statistic. What remains is an empirical characterisation of known robust-loss facts in FL |
| 3. Simple alternatives do not explain the results | **no** | known robust losses (MAE, clipped CE) match or beat 0–1; cluster-wise FedAvg beats A′ and B → A′; a one-line chance threshold fixes relabelling |
| 4. The theory makes testable predictions | **yes** | PM1, PM8, PM10, PB3, PF4 and PR7 were confirmed, and the PM9 magnitude matched; PM7, PM11 and PM12 failed. Score 20 / 29 |
| 5. The method survives at least one corruption model beyond symmetric noise | **partly** | A′ (0–1): positive but below its pre-registered margins under pair 0.45 and IDN. A′-MAE: clearly yes on two seed sets, but it is a post-Phase-II choice. Every variant fails at pair 0.75, and at relabelling without the guard |
| 6. The cost is justified | **no** | with 2 and 3 unmet, a GPU-scale study would mostly re-measure known facts |

### Final gate: **DO NOT AUTHORIZE LARGE-SCALE VALIDATION**

This is a resource decision. Criteria 2 and 3 fail on verified evidence:
- the theory is published;
- both mechanisms are published mechanisms with a swapped statistic;
- simpler or known alternatives match or beat them.

What survives is a characterisation that is narrower than any proposed method and is supported only on MNIST-scale synthetic noise.

**What would reopen the gate.** Each item is cheap and is listed in `research_state.md` §15.
1. A literature check showing that no FL paper already reports the "every loss level fails / CE reductions are confidence-confounded" characterisation. The current searches found none, but they were not exhaustive.
2. A small real-noise check of the measurement study (Stage 7 protocol, unchanged): CIFAR-10N annotator label sets as per-client channels, on a CPU-feasible model. It should show (a) the level failure and (b) the CE / Brier reversal with *human* noise.
3. A demonstration that a widely used loss-driven FL method (official FedEBA+, q-FFL or Oort code) actually misallocates under realistic label-quality heterogeneity. That would give the characterisation practical stakes.

---

## Stage 15 — Benchmark plan (conditional; **not authorised**)

Recorded only so that a future authorisation starts from a fixed protocol:
- **Settings:**
  - FedPCA's own setting (CIFAR-10 with Gaussian-noise rare clients, RSNA ICH, ISIC 2019);
  - CIFAR-10, CIFAR-100;
  - CIFAR-10N with per-client annotator label sets;
  - a ResNet-class CNN.
- **Methods:** FedAvg, FedEBA+ (with its LSR variant), q-FFL, FedPCA, CLoVE, IFCA, Ditto, cluster-wise FedAvg, and validation-improvement weighting with MAE, 0–1 and CE statistics.
- **Fairness of comparison:**
  - matched rounds, participation, local epochs and tuning budget (the same number of configurations per method, tuned on a validation split);
  - seeds 100–104;
  - primary metrics, pre-registered: rare-group clean accuracy, worst-10 % clean accuracy, common accuracy.
- **Official code.** A baseline is called "official" only after verifying the authors' released repository. **None was verified in Phase II.**

---

## Assumptions behind the results

1. **Label channels.**
   - The theory assumes class-conditional noise, T independent of x given y. The exception is the IDN statement, which works inside the expectation over x.
   - The experiments use synthetic channels: uniform symmetric, pair y → y+1, a deterministic swap, and our model-confusion IDN. **No human annotation noise was tested.**
2. **Independence of holdout and training noise.** The holdout labels are corrupted independently of the training labels. This makes r̂ unbiased for a·ρ, and it holds by construction in the simulator. Real annotators with persistent biases would correlate the two.
3. **Utility definition.** "Clean utility" is the clean 0–1 improvement on the client's *own* distribution. No statistic tracked the federation-level one-step utility (Spearman ≤ 0.46). Conclusions about "usefulness" are therefore about own-distribution usefulness only.
4. **Honest reporting.** Every statistic is self-reported. E7 showed that 10 % misreporting clients capture 41 % of the weight. No defence was tested.
5. **Model and scale.** NumPy MLP 784-128-64-10, MNIST / Fashion-MNIST, 50 clients, m = 10, one local epoch.
6. **Measurement design.** The Stage 7 probes make one local update from checkpoints of a base federation they did not train. External validity for participating clients is supported indirectly, by the FL runs agreeing with the probe predictions (P75 failure, η > 0.9 cost, MAE ≥ 0–1, TS-local fails at pair noise).

## Failure regimes (consolidated)

| Regime | What fails | Why | Guard / alternative |
|---|---|---|---|
| Any loss *level* under heterogeneous noise | FedEBA-style, q-FFL- and PoC-style allocation, with any loss | The level is η_i + a_i·R_i, dominated by the noise offset | Use reductions, not levels |
| CE / Brier *reductions* | A (CE), A′-Brier | Unattenuated confidence term; noisy updates flatten predictions | Bounded statistics; TS-local for symmetric noise only |
| Symmetric η ≥ (K−1)/K = 0.9 | Every own-label statistic; B mirror-assigns (99 % at η 0.95, 75 % = chance at η 0.9) | a(η) ≤ 0 | A chance-level test catches η > 0.9 in principle (observed error > 0.9), but the margin was set for relabelling |
| Pair noise above ½ | Every own-label statistic (graded: 19 % / 39 % / 100 % reversed at 0.55 / 0.6 / 0.8); A′ common accuracy 0.64–0.69 at pair 0.75 | Negative margin on the σ(y) cell | **None found.** The T1′ test is blind (observed error ≈ 0.77 < 0.95); FedPCA-style kept common 0.954 there |
| Deterministic relabelling | Every own-label statistic (100 % reversed); unguarded A′ collapses | r̃ = ΔD exactly | T1′ exclusion (fully effective); routing (B4sc) |
| Small holdouts | 0–1 statistic (AUROC 0.87 at n_h 25); normal approximation off by up to 0.1 at n_h ≤ 50 | Variance of a {−1, 0, 1} mean | MAE / clipped CE (≥ 0.998 at n_h 25) |
| Rare group already well served | 0–1 AUROC 0.94 when 10 rotated clients train the base model | Rare reducibility → 0 | — (the allocation question becomes moot) |
| a(η)·√n ≲ 1 | B / argmin clustering (ARI 0.84 at a√n 0.79); one dynamic failure at η 0.8, n 500 | Sampling noise of z-vectors | Larger n, or z-MAE (slightly better) |
| Isolable rare group | Every one-model fairness rule loses to cluster-wise FedAvg (by 4–5 pts) | Personalisation beats re-weighting | Cluster-wise FedAvg |
| Misreporting | A′ and FedEBA-style alike | Self-reported statistics; the clamp bounds a ratio, not a share | Untested |

## Unresolved rejection risks (for any write-up of the surviving claim)

1. **Novelty.** The risk identity, its uniqueness and the margin form are published. The FL mechanisms are published mechanisms with a different statistic. A reviewer can fairly say: "robust losses are known to be robust; using them as client statistics is the obvious corollary."
2. **Scale and realism.**
   - MNIST / FMNIST MLP, synthetic noise, 50 clients.
   - No CNN, no CIFAR, no human annotation noise, no real federated dataset.
   - The CNN question in the anti-confirmation list is open.
3. **Baselines are re-implementations.** FedEBA-style has no alignment step; FedPCA-style is unstable (P45: 0.87 / 0.34 / 0.19); CLoVE-style lacks the post-stability phase in the dynamic runs. None is the authors' code.
4. **Simple alternatives win.** Cluster-wise FedAvg beats every one-model rule. MAE beats 0–1. A one-line chance threshold handles relabelling.
5. **Utility scope.** The statistics track own-distribution utility, not federation-level value.
6. **Unaddressed attack surface.** Self-reported statistics can be gamed (E7).
7. **Missed predictions.** 9 of 29 pre-registered predictions failed. Several were near-misses on margins (PF2 by 0.2 pt, PF5 by 0.1 pt), but they count as failures.
8. **No guard for pair noise above ½.** A full method would need either labels from another source or a corruption-specific test.

## Files added in Phase II

- **Code:**
  - `phase2_verify.py` (Stage 1);
  - `phase2_theory.py` (Stages 3–6 checks);
  - `phase2_metrics.py` (Stages 7–8 measurement);
  - `phase2_pop.py` (heterogeneous-η and IDN populations);
  - `phase2_baudit.py` (Stage 11 static);
  - `phase2_analyze.py` (all verdicts and tables).
- **Code changes, all additive:**
  - `algos.py`: A′ statistics `r_brier`, `r_tsce`, `r_clip`; guards `t1g`, `cap`, `fb`; `eba_cap`; optional per-method counters;
  - `cfl.py`: `ifca_err`;
  - `run_exp.py`: Phase II population dispatch;
  - `summarize.py`: Phase II CSV rows.
  
  Six stored runs (E1, E2, E3d, E4, E5b, E8) re-executed bit-exactly with the final code (`results/P2_verification.json` → `reruns_after_phase2_code_changes`).
- **Raw logs (new files only):** `results/P2F.jsonl`, `P2R.jsonl`, `P2Rc.jsonl`, `P2B.jsonl`, `P2X.jsonl` (FL and CFL runs, same schema as E*).
- **Diagnostics:** `results/P2_metrics_probes.jsonl`, `P2_metrics.jsonl`, `P2_metrics_pm11.json`, `P2_metrics_rarity.jsonl`, `P2_baudit.jsonl`, `P2_theory.json`, `P2_verification.json`, `P2_summary.json` (every verdict).
- `experiments.csv` was rebuilt from the raw logs and now includes the Phase II runs and diagnostic rows. Discovery rows are unchanged.
- The probe logit cache (`results/cache_P2/`) is regenerable with `phase2_metrics.py generate` and is not part of the delivered package.
