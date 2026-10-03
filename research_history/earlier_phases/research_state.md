# Research state (compact living record)

*Last updated: 2026-10-02, end of Phase III. Current decision and corrections are in §16 and `phase3_identifiability_report.md`. §§1–15 remain the historical discovery/Phase-II record; their overbroad interpretations are superseded explicitly below.*

**Current status:** Phase III complete, outcome D for the tested identifiability formulation. **DO NOT AUTHORIZE NEW DIRECTION.** The elementary ambiguity/recovery results substantially overlap existing work; no novel FL-specific claim or algorithm was established. The earlier large-scale gate remains closed. A′ and B remain rejected as method contributions.

**Labels:**
- **[fact]** verified in an inspected source;
- **[obs]** from an experiment executed here;
- **[interp]** interpretation;
- **[hyp]** untested;
- **[spec]** speculation.

**Full narrative and tables:** `final_research_package.md`. **Per-run rows:** `experiments.csv`. **Raw histories:** `results/*.jsonl`.

## 0. Environment constraints [fact]
- 2 CPU cores, 7 GB RAM, no GPU. PyTorch wheels are blocked by the egress policy (download.pytorch.org returned 403), so all experiments use a NumPy MLP (`flsim.py`).
- **MNIST:** taken from the user's CSVs (standard 60k / 10k split). **Fashion-MNIST:** official files, md5 verified. `get_data.py` rebuilds both.
- **Literature:** searched with WebSearch / WebFetch. Novelty conclusions are limited to what those searches surfaced.

## 1. Landscape (seed set: 10 ICML 2026 papers, all read) — details in `literature_matrix.md`

**Shared assumptions [fact]:**
1. Loss is a faithful, honestly reported signal (FedEBA+, CLoVE; FedQueue for delays).
2. Participation is exogenous and homogeneous.
3. The population is stationary and closed, with per-client server state (DeMoA, ParFreFL, CLoVE, FedVeer).
4. Evaluation uses test sets drawn like training data, possibly with noisy labels.

**Conflict:** stale information is down-weighted by FedQueue but relied on by DeMoA and ParFreFL. This is already studied (FedStale, FedSteer, FedAU), so it was not pursued.

**Gap used:** no seed paper models heterogeneous label quality. In FedEBA+ [fact]:
- the App. E.1 toy example argues that up-weighting a pure-noise client reduces disparity;
- the noisy-label experiment corrupts all clients alike;
- App. E offers a CE + self-distillation robust variant (LSR).

## 2. Problems

- **P1 (selected).** Irreducible loss crowds out reducible need in loss-driven allocation and selection.
  - **Closest prior: FedPCA** (arXiv 2503.10567) [fact, web; re-checked 2026-10-02]. It poses the *same* three-group problem: common, covariate-shifted rare (Gaussian-noise images) and mislabeled clients.
  - FedPCA's mechanism: a GMM on (CE loss, feature dispersion), per-dataset q, temporal smoothing, SAM.
  - Our angle: a held-out, decision-level, self-referential statistic, plus the cross-family calibration diagnosis.
- **P2 (selected).** Heterogeneous label noise breaks loss-vector clustering (CLoVE).
  - Closest prior: RCC-PFL (label-agnostic feature clustering) [web].
- **P3 (not selected; became Candidate C).** Estimate a client's value from *other* clients' outcomes under randomised aggregation.

**Discarded directions** (each with its collision): stale-update value; DeMoA × heterogeneous participation; retention feedback; CLoVE drift; queue × data correlation; performative lag; misreporting as a standalone topic; compression-induced unfairness. Details are in `literature_matrix.md` §C.

## 3. Competing hypotheses for P1 (Stage 4) and outcomes

| ID | Hypothesis | Outcome |
|---|---|---|
| H1 crowding-out | loss-driven weights hand the budget to noisy clients | **supported** [obs] in 8/8 noisy settings (EBA noisy weight 0.65–0.90; rare accuracy 2.6–14.7 pts below FedAvg); replicated for Power-of-Choice |
| H2 weight variance | tempering suffices | **rejected** [obs]: τ = 1 gives 0.782 vs FedAvg 0.833 |
| H3 gradient corruption | noisy updates must be excluded | **rejected as the main cause** [obs]: oracle exclusion 0.854 < A′ 0.882 |
| H4 evaluation artefact | the observed-label fairness metric rewards the failure | **supported** [obs], 8/8 settings |
| H5 null | effect negligible | **rejected** [obs] |

## 4. Candidates (full specs: package §4; falsifiers fixed before testing)

| Candidate | Falsifier (pre-stated) | Result | Decision |
|---|---|---|---|
| **A**: held-out CE reducibility | does not beat EBA on rare accuracy with noisy clients, or a shuffled / tempered control matches it | noisy weight 0.40 > FedAvg 0.30; rare 0.786 < FedAvg 0.833 [obs] | **KILL** |
| **A′**: held-out 0-1 reducibility | same falsifiers, plus no gain with a noisy majority | not met. Rare accuracy above FedAvg in 30/30 seed pairs; above shuffled control in 18/18; ≈ clean-label oracle [obs] | **CONTINUE** |
| A′-select: PoC on stale / fresh / LCB reducibility | below FedAvg | met: all three variants below FedAvg in rotated-rare settings [obs] | **KILL** |
| **B**: z-normalised 0-1 error clustering | CLoVE does not degrade, raw error vectors suffice, or B fails under class-conditional noise | not met. CLoVE collapses at η ≥ 0.5; raw 0-1 and z-CE fail; B = 1.00 under pair noise [obs] | **CONTINUE** |
| B-W: within-cluster noise weights | — | harmful at η = 0.8 unguarded; guarded (B-WG) at oracle level (post-hoc) [obs] | **REVISE** |
| **C**: randomised aggregation evaluation | harmful-client AUROC ≈ 0.5, or a simple filter matches it | met: AUROC 0.42–0.60 [obs] | **KILL** |

**Why A failed** [obs, `diag_mechanism.py`, 3 seeds]:
- Noisy clients' CE gap grows +0.24 → +0.89 → +1.02 (rounds 25 / 75 / 150), while their clean-label CE gap is −0.59 → −0.93.
- The CE gap's AUROC for rare clients falls 0.84 → 0.58 → 0.47.
- The 0-1 gap scales with slope 0.325 vs the predicted a(0.6) = 0.333 (r = 0.76).

[interp] A was revised to A′ because a mechanism was identified (calibration), not by tuning.

## 5. Novelty-collision table (Stage 6; updated 2026-10-02)

| Ours | Closest prior | Shared | Difference | Risk |
|---|---|---|---|---|
| A′ problem framing | **FedPCA** (2025) | same three-group problem; protect rare clients from noise handling | none at problem level | **high** — cite as the problem's origin |
| A′ estimator | FedPCA; MRO / MERO; RHO-LOSS; DQA | separate noise from need; excess-risk idea; reducible holdout loss; learning progress | held-out + decision-level + from the client's own update; no features, no GMM, no per-dataset q; federated | **medium** — needs a head-to-head with official FedPCA code |
| Cross-family calibration diagnosis | Ghosh et al. 2017 (identity); FedEBA+ LSR (robust-loss variant) | symmetric-loss noise tolerance | shows CE level, CE reducibility, CE vectors and CE outcomes all fail in FL; metric artefact | medium-low |
| B | CLoVE; RCC-PFL | loss-vector clustering; noise-robust clustering | affine-normalised 0-1 errors, η̂, recovery in a(η)√n; separates labelling-function clusters | **medium** |
| C | FedLAW / FedHAW, BaFFLe, Shapley-style | data-driven client valuation | other clients' outcomes under SPSA | killed |
| Cross-family diagnosis (s2 search) | Scott, Luo & Ho, arXiv 2609.31454 (Sep 2026) | which signal exposes which corruption (loss ↔ label flips; uncertainty ↔ image noise) | per-sample detection vs client-level statistics driving allocation / selection / clustering; no calibration confound of reducibility | medium-low; must be cited |

## 6. Adversarial pre-review (Stage 7) — main objections and status

- **A′**
  - *Objective / convergence characterisation:* open (theory plan T2 / parity).
  - *Holdout cost:* covered by the equal-resource shuffled control.
  - *Simple alternatives (tempered EBA, loss-GMM):* tested; both fail somewhere.
  - *Novelty vs FedPCA:* **high**.
  - *Experiment most likely to break A′:* conflicting clients — and it did.
- **B**
  - *Invariance holds only in expectation and for symmetric noise:* pair noise tested, works.
  - *Normalisation alone or error alone?* Both are needed.
  - *Missing RCC-PFL baseline:* open.
- **C**
  - *SPSA power too low:* confirmed → KILL.

## 7. Experiment index [obs] (3 seeds each; MNIST unless stated; settings in `experiments.csv`)

| ID | Question | Key result |
|---|---|---|
| E1 | crowding-out gate (S0 = no noise, S1 = 30 % noisy, η = 0.6) | EBA rare 0.738 vs FedAvg 0.833 (noisy weight 0.73); oracle EBA 0.876; A (CE reducibility) 0.786; train-drop 0.780; τ = 1 0.782; metric artefact: observed-label variance 544 vs 645 |
| E1b | A′ on E1 settings | A′ 0.882 (S1), 0.881 (S0); shuffled 0.826; τ = 0.1 0.856; EBA on 0-1 level 0.775. **τ → 0 chosen here** |
| diag_mechanism | slope and AUROC of each statistic | slope 0.325 (pred. 0.333); AUROC of 0-1 gap 0.94 / 0.88 / 0.73 vs CE gap 0.84 / 0.58 / 0.47 |
| E3a | FedPCA-style, loss-GMM, PoC, stale PoC, C on S0 / S1 | loss-GMM deletes rare clients (0.189 at S0); PoC rare 0.730, noisy selection 0.87; C noisy weight 0.35 |
| E3b | stress grid: η 0.3 / 0.8, 60 % noisy, label-rare | A′ best non-oracle except 60 % noisy (loss-GMM 0.894 vs A′ 0.880); PoC common 0.597 at η = 0.8 |
| E3c | 5 local epochs | A′ 0.856 vs FedAvg 0.796; train-drop 0.710; EBA common 0.725 |
| E3d | relabelled (CF) clients | **A′ fails**: common 0.712 / 0.950 / 0.056 across seeds (CF weight 0.26); FedEBA fails similarly; FedAvg / FedPCA-style / loss-GMM do not |
| diag_structure | can a "structured disagreement" score gate CF clients? | CF 0.86–0.96, RR 0.43–0.54, NZ ≈ 0.31, CC 0.60–0.85 → separates CF from RR, not from CC; gate not adopted (would be test-set tuning) |
| diag_rae | does C detect harmful clients? | AUROC 0.42–0.53 (CE), 0.51–0.60 (0-1) → KILL C |
| E3e / E3f | A′-select with fresh / LCB ranking | below FedAvg in all 3 settings; noisy selection 0.41–0.74 → KILL (E3f post-hoc) |
| E4 | FMNIST; pair noise | A′ +7.3 / +7.7 (FMNIST no-noise / noisy) and +4.2 (pair) over FedAvg; FedPCA-style unstable (likely a re-implementation artefact) |
| E2 | B grid (rot / perm, η 0.3–0.8, 50 % noisy) | B ARI 1.00 everywhere; CLoVE-style 0.12–0.46 at η ≥ 0.6; raw 0-1 0.14–0.50; z-CE 0.22–0.71; B-W harmful at η = 0.8 |
| E2c | breakdown and threshold (η 0.4 / 0.5; n = 150) | CLoVE-style 0.87 at η = 0.4, 0.39 at η = 0.5; B fails only at a(η)√n ≈ 1.4 (ARI 0.78) |
| E2d | unknown K (6 models, silhouette) | B selects K = 4, ARI 1.00; CLoVE-style K ≈ 5.0–5.6, ARI 0.47–0.49 |
| E2b | B on FMNIST; pair noise | B ARI 1.00; η̂ MAE 0.034–0.044 |
| E2e | guarded B-W (post-hoc) | oracle-weight level in 4/4 settings |
| E5 (s2) | routing (oracle / B) composed with A′ when relabelled clients are present; P0 control | oracle routing repairs A′; B isolates CF in 6/6 runs; forced splits misroute clean clients (worst-10 % down to 0.01) |
| E5b (s2) | B + CLoVE post-stability routing; dev and fresh seeds | fixes forced splits on 3/3 fresh seeds; collapsed once on a development seed (froze at round 4) |
| E6 (s2) | fresh seeds for B-WG | safe vs B, not oracle-level; B ARI 1.00 in 11/12 |
| E7 (s2) | misreporting (5 noisy clients report the maximum) | attackers take 41 % of the weight; A′'s gain disappears |
| E8 (s2) | fresh seeds 3–4 for A′'s core claims | 12/12 pairs; label-rare crowding-out does not replicate |
| E9 (s2) | rare-group size 1 / 2 vs clustering | a dedicated model beats one-model A′ even for a single rare client |
| diag_t4 (s2) | T4 calibration on trained B models | deviation × a(η)√n = 0.70–1.26 where clustering succeeds; locates the breakdown |

**Paired statistics** (6 MNIST one-epoch settings, 18 seed pairs; Wilcoxon, descriptive). A′ minus FedAvg:
- rare accuracy +7.3 pts mean / +4.1 median, 18/18 pairs;
- common accuracy −0.7;
- mean over all clients +0.1 (n.s.).

A′ minus FedPCA-style: +4.7 rare, 15/18. A′ minus oracle: +0.1 (n.s.).

## 8. Decision gate (Stage 11, after session 1; updated in §13)

| Direction | Decision | Reason |
|---|---|---|
| A (CE reducibility) | KILL | falsified (E1); mechanism identified |
| A′ (0-1 reducibility, allocation) | **CONTINUE** | survives all falsifiers in 10 settings; known failure mode (conflicting clients) |
| A′-select | KILL | three variants fail |
| B (affine-invariant error clustering) | **CONTINUE** | ARI 1.00 in 12/12 noisy settings at n = 500; predicted breakdown observed |
| B-W | REVISE → B-WG | guard works but is post-hoc |
| C (RAE) | KILL | no signal (AUROC ≈ chance) |

## 9. Selected direction (Stage 12, after session 1; updated in §13)

**Working title:** "Decision-level, noise-affine client statistics for loss-driven FL", i.e. A′ + B under one thesis. CE-based client statistics are confounded by calibration under heterogeneous label noise, and held-out decision-level statistics fix allocation and clustering.

**Why this direction:**
- a mechanism is identified and confirmed;
- the effect replicates on two datasets and both noise types;
- predicted failures are observed.

**Main risks:**
- novelty vs FedPCA (high at problem level);
- MNIST-scale evidence only;
- the concept-conflict failure;
- re-implemented baselines (FedPCA-, FedEBA- and CLoVE-style).

## 10. Corrections made during the session (kept for integrity)

1. **Affine identity.** First recorded as "derived here". It is Ghosh, Kumar & Sastry (AAAI 2017) Thm 1, and is now cited as known.
2. **FedPCA's rare clients.** The matrix said FedPCA tested only label-imbalance rarity. Wrong: its rare clients are covariate-shifted (Gaussian-noise images). The problem-level novelty risk was raised to high.
3. **E2c table.** a(η)√n for η = 0.4, n = 500 was listed as 13.4; the correct value is 12.4.
4. **Summary ranges in the draft package**, recomputed from raw logs:
   - EBA vs FedAvg: −2.6 to −14.7 pts, not −4 to −15;
   - noisy weight 0.65–0.90;
   - PoC noisy selection 0.85–0.98;
   - B's ARI = 1.00 qualified to n = 500;
   - η̂ MAE 0.03–0.07;
   - the C kill rationale corrected (CE version did not lower rare accuracy at η = 0.6; 0-1 version did);
   - the fairness-metric claim narrowed (clean-label variance is higher in 5/8 settings; worst-10 % is lower in 8/8).
5. **An early single-seed CE-confidence diagnostic** (round 100, seed 0) was never saved to a file. It has been removed from the record and replaced by `diag_mechanism` (3 seeds, saved).
6. **Reviewing criteria** re-verified on the official NeurIPS / ICML / ICLR 2026 pages (2026-10-02).
7. **experiments.csv run IDs.** For the 366 global-model rows, the run_id did not match the raw records: the merge step had hashed the run without its config. The CSV is now rebuilt by `summarize.py csv` and the IDs match `results/*.jsonl`. All metric values are unchanged (checked cell by cell).
8. **Runners.** `run_exp.py` and `run_cfl.py` used to append CSV rows in an older schema, which would have corrupted `experiments.csv` on a re-run. They now write only the raw JSONL.
9. **Reproducibility check.** A logged run (E1b, A′, seed 0) was re-executed with the final code and matched bit for bit. `get_data.py` rebuilds both data files bit-exactly.

## 11. Next actions (after session 2; same as package §14)

1. **A routing rule with a guarantee.** Freeze only when the frozen centroids are pairwise distinct, and after the T1′ conflict signature has been seen. Pre-register; test on fresh seeds, with and without conflicting clients.
2. **Locate A′'s regime against personalisation.** Vary the rare group's data size (50–600 samples per client) and the task difficulty; compare with B routing and Ditto.
3. **A misreporting defence.** Weight caps or trimmed-rank weights, evaluated under the E7 attack.
4. **PyTorch port.** FedPCA's own setting and CIFAR-10N against official FedEBA+ (with LSR), FedPCA, CLoVE and Ditto code; 5 seeds.
5. **Theory.** The excess-risk-parity fixed point; convergence of the clamped weighted iteration.

## 12. Session 2 — pre-registered experiments (written 2026-10-02, BEFORE any of these runs)

Shared: MNIST, MLP, lr 0.05, batch 32, 1 local epoch, T = 200, m = 10 of 50 clients unless stated. Thresholds below are fixed now and will not be changed after seeing results.

**E5 — compose routing with A′ when concept-conflict clients are present.**
- Populations:
  - P_CF = E3d (30 CC + 5 RR + 10 NZ with η = 0.6 + 5 CF);
  - control P0 = 30 CC + 5 RR + 15 NZ (η = 0.6), no CF.
- Seeds: 0–2.
- Harness: `compose.py`. The K = 1 variants must reproduce E3d / E1b bit-exactly (harness check).
- Methods:
  - single model: FedAvg, A′;
  - oracle concept routing (CF → own model): FedAvg or A′ within cluster;
  - B routing: K = 2 fixed, or K̂ = 4 with silhouette selection over 2..4; A′ within cluster (and FedAvg within cluster for K̂ = 4).
- P5.1 (oracle routing repairs A′): common ≥ 0.94 in every seed, and mean rare ≥ 0.852 (= E3d FedAvg 0.832 + 2 pts).
- P5.2 (B finds the minority concept): CF isolation ≥ 0.8 in every seed, for both B variants. CF isolation = share of CF clients whose final model is not the model serving most CC clients. **Prior: low–medium.** The minority is 10 % of clients, about 1 CF client is sampled per round, and all models first learn the majority concept.
- P5.3 (composition):
  - if P5.2 holds, B→A′ is within 2 pts of oracle routing + A′ on common and rare accuracy;
  - if P5.2 fails, the composition is recorded as **not** a remedy at this participation level. Follow-up with m = 20 is allowed but labelled as such.
- P5.4 (control): in P0, B→A′ costs ≤ 1.5 pts common and ≤ 2 pts rare accuracy vs single-model A′.

**E6 — fresh-seed validation of B-WG.**
- Settings: E2e's four (rot 30 % η = 0.8; rot 30 % η = 0.6; rot 50 % η = 0.6; perm 30 % η = 0.6).
- Seeds: 3, 4, 5 (never used before).
- Methods: CLoVE-style, B, B-W, B-WG, oracle-W.
- P6.1: B-WG worst-cluster accuracy is within 1.0 pt of oracle-W (mean over seeds) in every setting, and never more than 0.5 pt below B.
- P6.2: B ARI = 1.00 in all 12 runs.
- P6.3 (does the original failure replicate?): unguarded B-W at η = 0.8 has ARI < 1, or worst cluster below B, in ≥ 1 of 3 seeds.

**E7 — misreporting.**
- Population: E1 S1. Seeds: 0–2.
- Attack: 5 of the 15 NZ clients always report the maximal statistic (A′: r = 1; FedEBA-style: F = 50).
- Analytic bound (m = 10, ε_p = 0.01): the clamp caps one attacker's share of a round's weight at 10/19 ≈ 0.53.
- P7.1: under attack, A′'s rare accuracy falls below FedAvg's (0.833).
- P7.2: the attackers' total weight share is ≥ 0.30 (their base share is 0.10).
- This measures the vulnerability only; no defence is tested.

**E8 — fresh seeds 3–4 for the A′ core claims.**
- Settings: the 6 MNIST one-epoch settings.
- Methods: FedAvg, A′, A′ shuffled, FedEBA-style, oracle EBA, FedPCA-style q = 1.
- These seeds are out-of-sample for every design choice, including τ.
- P8.1: A′ rare accuracy is above FedAvg's in ≥ 11 of 12 new seed pairs.
- P8.2: the mean common-accuracy cost is ≤ 1.5 pts.
- P8.3: FedEBA-style rare accuracy is below FedAvg's in all 10 new noisy seed pairs.

**E5b — post-hoc follow-up to E5. Pre-registered 2026-10-02 after E5, before any E5b run.**
- *What E5 showed:* with B routing, CF clients were isolated (P5.2 passed), and group-level common and rare accuracy matched oracle routing (P5.3 passed). But **per-client** worst-10 % clean accuracy collapsed in 2 of 3 seeds: 0.35 and 0.01.
- *Why [obs]:* clean CC clients ended on the CF model despite errors of 0.05 vs 0.99.
  - When a round samples no CF client (probability 0.31 with m = 10), k-means still forms K = 2 clusters.
  - With K = 2, every z-vector equals (−1, 1) up to ~1e-16 rounding, so k-means splits the clients arbitrarily (6 / 4 in a direct test).
  - The one-to-one matching then forces one group onto the CF model.
  - My first explanation (a singleton relocated into an empty cluster) was wrong: exactly identical points stay in one cluster.
- *Deviation from the pre-registration:* this per-client metric was not pre-registered for E5; the failure was found in analysis.
- *Fix tested:* CLoVE's own post-stability phase, which our CLoVE-style loop had omitted. Once ≥ 90 % of returning sampled clients keep their model for 3 consecutive rounds, the z-space centroids are frozen and each client is routed to its nearest frozen centroid. Method suffix `c`.
- *Runs:*
  - P_CF, dev seeds 0–2: B2c→A′, B4sc→A′, B4sc→FedAvg.
  - P_CF, **fresh seeds 3–5**: those three plus oracle→A′, oracle→FedAvg, FedAvg, A′, and B2→A′ (unfixed).
  - P0 control, fresh seeds 3–5: A′, B2c→A′, B4sc→A′.
- *Predictions (fresh seeds only):*
  - P5b.1: B2c→A′ has per-client worst-10 % clean accuracy ≥ 0.80 in every fresh seed, and within 2 pts of oracle→A′ on the mean.
  - P5b.2: CF isolation ≥ 0.8 and group common accuracy ≥ 0.94 in every fresh seed.
  - P5b.3: the unfixed B2→A′ has worst-10 % < 0.80 in ≥ 1 fresh seed (the failure replicates).
  - P5b.4 (P0 control): B2c→A′ is within 1.5 pts of single-model A′ on common accuracy and within 2 pts on worst-10 % (fresh-seed means).

## 13. Session 2 — outcomes against the pre-registered predictions [obs]

**E5** (`results/E5.jsonl`; harness check passed — the K = 1 variants reproduce E3d bit-exactly with single-threaded BLAS).

| prediction | result | verdict |
|---|---|---|
| P5.1 oracle routing → A′: common ≥ 0.94 every seed; mean rare ≥ 0.852 | common 0.949 / 0.947 / 0.948; rare 0.880 ± 0.011 (oracle → FedAvg 0.839) | **pass** |
| P5.2 B isolates CF (≥ 0.8 every seed, both variants) | CF isolation 1.00 in all 6 runs; my prior (low–medium) was wrong | **pass** |
| P5.3 B → A′ within 2 pts of oracle → A′ (common, rare) | B2: common 0.949, rare 0.881 → pass. B4s: common 0.952, rare 0.842 ± 0.059 → fail (rare −3.8) | **pass (B2) / fail (B4s)** |
| P5.4 control without CF: common within 1.5 pts, rare within 2 pts of single-model A′ | B2: 0.947 / 0.908; B4s: 0.945 / 0.928 (single A′ 0.945 / 0.882). Rotated clients often get their own model, which raises rare accuracy | **pass** |
| *not pre-registered:* per-client worst-10 % clean | B2 → A′: 0.867 / 0.347 / 0.009. Clean CC clients were forced onto the CF model in rounds without a CF client (forced K-way split + one-to-one matching) | **failure found in analysis** → E5b |

- [interp] Why routing finds the conflict (formalised as T1′ in the package): a model that is better than chance on the clean labels can never exceed chance-level error (0.9) under symmetric noise. Relabelled clients see 0.98 error on the majority model (FedAvg own-label accuracy 0.020), so any other model looks better to them.

**E7** (`results/E7.jsonl`).

| prediction | result | verdict |
|---|---|---|
| P7.1 A′ rare under attack < FedAvg (0.833) | 0.816 ± 0.011 (0.802 / 0.821 / 0.827); common 0.884 vs 0.945 honest | **pass — vulnerability confirmed** |
| P7.2 attackers' weight share ≥ 0.30 (base 0.10) | A′ 0.41 ± 0.01; FedEBA-style 0.41 ± 0.01 (honest A′: 0.06) | **pass** |

- [interp] The τ-clamp bounds the weight *ratio*, not a client's *share*. Five misreporting clients (10 %) capture 41 % of the weight under both rules. A′'s rare-client gain disappears, and common accuracy drops by 6 pts. Self-reported allocation statistics need a robust aggregation layer (e.g. per-client weight caps or trimmed ranks) before deployment.

**E6** (`results/E6.jsonl`; fresh seeds 3–5; CFL loop as in E2; worst-cluster accuracy on clean labels).

| setting | B ARI | worst cluster: B / B-W / B-WG / oracle-W | η̂ MAE (B-WG) |
|---|---|---|---|
| rot 30 %, η = 0.8 | 1.00 / 1.00 / 1.00 | 0.904 / **0.812** / 0.914 / 0.924 (B-W ARI 1.00 / 0.63 / 0.63) | 0.017–0.049 |
| rot 30 %, η = 0.6 | 1.00 / 1.00 / 1.00 | 0.910 / 0.918 / 0.916 / 0.920 | 0.019–0.046 |
| rot 50 %, η = 0.6 | 1.00 / 1.00 / 1.00 | 0.826 / 0.847 / 0.843 / 0.859 | 0.135–0.162 |
| perm 30 %, η = 0.6 | **0.93** / 1.00 / 1.00 | 0.879 / 0.915 / 0.886 / 0.920 | 0.019–0.047 |

CLoVE-style ARI on the same runs: 0.09–0.41.

| prediction | verdict |
|---|---|
| P6.1 B-WG within 1.0 pt of oracle-W in every setting, never > 0.5 pt below B | **fail** on the first clause: B-WG is 1.1 / 0.4 / 1.6 / 3.4 pts below oracle-W (in table order). The second clause holds: B-WG beats B by +0.6 to +1.7 in all 4 settings |
| P6.2 B ARI = 1.00 in all 12 runs | **fail**: 11/12 (perm, seed 3: ARI 0.93, worst cluster 0.82) |
| P6.3 the unguarded B-W failure at η = 0.8 replicates | **pass**: ARI 0.63 in 2 of 3 fresh seeds (worst cluster 0.77 / 0.75) |

- **Correction to the session-1 claims.** B-WG's "oracle-weight level" (E2e, same seeds as development) does **not** replicate on fresh seeds. B-WG is a safe improvement over B (never worse; fixes the B-W failure), but is not oracle-level. B's "ARI = 1.00 everywhere" becomes "11 of 12 fresh-seed runs".

**E5b** (`results/E5b.jsonl`).

| method (P_CF) | worst-10 % clean, dev seeds 0–2 | worst-10 % clean, fresh seeds 3–5 | common (fresh) | rare (fresh) | CF isolation | routing frozen at round |
|---|---|---|---|---|---|---|
| A′, one model | 0.58 / 0.79 / 0.04 | 0.69 / 0.68 / 0.61 | 0.80 / 0.77 / 0.76 | 0.811 | 0 | — |
| oracle routing → A′ | 0.87 / 0.88 / 0.87 | 0.87 / 0.88 / 0.88 | 0.94–0.95 | 0.883 | 1.00 | — |
| B (K = 2) → A′, forced split | 0.87 / 0.35 / 0.01 | **0.58 / 0.32 / 0.77** | 0.95 | 0.887 | 1.00 | — |
| B2c → A′ (post-stability) | 0.89 / 0.88 / **0.03** | 0.87 / 0.90 / 0.89 | 0.94–0.95 | 0.898 | 1.00 (dev seed 2: 0.07) | 7 / 6 / **4**; 5 / 4 / 12 |
| B4sc → A′ | 0.91 / 0.91 / 0.92 | 0.92 / 0.91 / 0.80 | 0.94–0.95 | 0.886 | 1.00 | 18 / 16 / 4; 5 / 4 / 5 |
| B4sc → FedAvg | 0.92 / 0.91 / 0.92 | 0.91 / 0.92 / 0.92 | 0.96 | 0.925 | 1.00 | same as above |

| prediction (fresh seeds) | verdict |
|---|---|
| P5b.1 B2c → A′ worst-10 % ≥ 0.80 every seed and within 2 pts of oracle → A′ | **pass** (0.866 / 0.899 / 0.894; mean 0.886 vs 0.875) |
| P5b.2 CF isolation ≥ 0.8 and common ≥ 0.94 every seed | **pass** |
| P5b.3 unfixed B2 → A′ worst-10 % < 0.80 in ≥ 1 seed | **pass** (all 3: 0.58 / 0.32 / 0.77) |
| P5b.4 P0 control: B2c → A′ within 1.5 pts common and 2 pts worst-10 % of single A′ | **pass** (0.952 vs 0.947; 0.919 vs 0.861) |

- **Caveat (development seed 2).** B2c froze its routing at round 4, before the conflicting clients were separated, and collapsed exactly like single-model A′ (common 0.057). Freezing is only safe once the routing has found the structure. A stricter stability rule is still needed (minimum warm-up, or requiring distinct frozen centroids); it is untested.
- **Unregistered observation [obs].** B4sc routed the rotated (rare) clients off the main model in all 6 B4sc → FedAvg runs and 5 of 6 B4sc → A′ runs. When it does, plain FedAvg inside each model is the best rare-accuracy configuration: 0.925 (fresh) vs 0.883 for oracle routing → A′, with worst-10 % 0.91–0.92 in all 6 seeds.
- [interp] Here personalisation beats fairness re-weighting, because the rare group (5 clients × 600 samples) is big enough for its own model. A′'s remaining regime is a rare group too small to support its own model, or a deployment that requires one model. This is tested next (E9).

**E9 — does A′ still matter when clustering is available? Pre-registered before running.**
- Populations: 30 CC + 15 NZ (η = 0.6) + n_rr ∈ {1, 2} rotated rare clients, no CF.
- Methods: FedAvg, A′ (one model), B4sc → FedAvg, B4sc → A′. Seeds 0–2.
- P9.1: with n_rr = 1, A′ (one model) has higher mean rare accuracy than B4sc → FedAvg. A single rare client cannot train a good dedicated model.
- P9.2: with n_rr = 2, the gap between A′ and B4sc → FedAvg is smaller than with n_rr = 1 (direction not predicted).
- Reference: with n_rr = 5 (P_CF, fresh seeds), B4sc → FedAvg reached rare accuracy 0.925 vs 0.883 for oracle routing → A′. In the P0 control, single-model A′ reached 0.873.

**E8** (`results/E8.jsonl`; fresh seeds 3–4).

| prediction | result | verdict |
|---|---|---|
| P8.1 A′ rare > FedAvg in ≥ 11/12 fresh pairs | 12/12 (mean +7.3, median +4.3) | **pass** |
| P8.2 mean common cost ≤ 1.5 pts | −0.97 (worst pair −4.6, label-rare) | **pass** |
| P8.3 FedEBA-style rare < FedAvg in all 10 fresh noisy pairs | 8/10: label-rare +16.3 and +8.5 | **fail** |

- Pooled over 5 seeds (6 MNIST settings, 30 pairs), A′ minus each baseline on rare accuracy:
  - FedAvg: +7.3 (median +4.1), 30/30;
  - FedPCA-style: +3.8, 26/30;
  - clean-label oracle: +0.6, 20/30;
  - shuffled r: +8.7, 30/30.
- A′ minus FedAvg on the other metrics: common −0.8; all-client mean −0.0 (n.s.).
- **Correction to session 1:** the crowding-out claim holds for covariate-shifted rare clients (every seed of every such setting), **not** for label-rare clients. There, FedEBA-style weighting is above FedAvg in 3 of 5 seeds.

**E9** (`results/E9.jsonl`). Mean rare accuracy over 3 seeds:

| n_rr | FedAvg | A′ (one model) | B4sc → FedAvg | B4sc → A′ |
|---|---|---|---|---|
| 1 | 0.409 | 0.725 | **0.856** | 0.780 |
| 2 | 0.645 | 0.802 | **0.899** | **0.899** |

| prediction | verdict |
|---|---|
| P9.1 A′ beats clustering with a single rare client | **fail**: a single rotated client with 600 samples trains a usable dedicated model |
| P9.2 the gap shrinks from n_rr = 1 to 2 | **pass** (−13.1 → −9.7 pts), but its sign never changes |

**Session 2 summary.** 18 predictions: 13 passed, 4 failed (P6.1, P6.2, P8.3, P9.1), 1 mixed (P5.3).

**Updated decision gate (Stage 11, after session 2):**

| Direction | Decision | Reason |
|---|---|---|
| A′ | **CONTINUE, scoped** | Robust one-model fairness rule (5 seeds, 42/42 pairs). Personalisation beats it whenever the rare group can be isolated. Misreporting defeats it without a defence. |
| B | **CONTINUE** | 11/12 fresh-seed runs perfect. Finds conflicting clients (T1′). Recovery theory proved. |
| B → A′ composition | **CONTINUE, conditional** | Repairs the conflict failure with oracle or stable routing. The routing rule needs a guarantee (R15). |
| B-WG | **KEEP as a safe default** | Never worse than B, but not oracle-level. |
| A′-select, A, C | KILL | unchanged |

**Updated selected direction (Stage 12).** The thesis — decision-level client statistics fix loss-driven FL under heterogeneous label noise — now centres on B plus routing: noise-invariant clustering that isolates conflicting labelling functions, with recovery theory. A′ is the one-model fairness component. The most important open problems are a routing rule with a guarantee, a misreporting defence, and locating A′'s regime against personalisation.

## 14. Phase II — evidence ledger and pre-registration (written 2026-10-02 14:13 UTC — file timestamp — BEFORE any Phase II experiment; the first Phase II run started 14:18 UTC)

### 14.1 Evidence ledger (Stage 2): what each seed set may and may not be used for

| Class | Runs | Seeds | Used to choose or modify a method? | May be cited as |
|---|---|---|---|---|
| **Discovery** | all session-1 experiments: E1, E1b, E2, E2b, E2c, E2d, E2e, E3a–E3f, E4, diag_mechanism, diag_structure, diag_rae | 0–2 | **yes**: A → A′ (diag_mechanism); τ → 0 (E1b); 0–1 + z-normalisation for B (E2); B-WG guard (E2e); LCB ranking (E3f); K̂ by silhouette (E2d); CF gate rejected (diag_structure) | existence of the phenomena; **not** independent confirmation |
| Discovery (session 2) | E5 (all), E5b development seeds | 0–2 | **yes**: the post-stability routing (suffix `c`) was designed from the E5 failure | as above |
| **Confirmatory (session 2)** | E8 (A′ core claims); E6 (B-WG); E5b fresh seeds | 3–4; 3–5; 3–5 | no | confirmation of the pre-registered predictions in §12–13 |
| Pre-registered, same seed numbers | E7 (misreporting), E9 (rare-group size) | 0–2 | no (pre-registered before running; new populations or attack) | tests of pre-registered predictions; the seed reuse is disclosed |
| **Confirmatory (Phase II)** | measurement ablation and B static audit | 10–12 | no | tests of the §14.2 predictions |
| **Confirmatory (Phase II)** | FL boundary runs (Stage 8), relabelled-client guards (Stage 9), dynamic B audit | 20–22 (FL); 10–12 (dynamic CFL) | no — guards and new statistics are fully specified below **before** running | tests of the §14.2 predictions |
| Reserved | any method modified after seeing Phase II results | 30–32 | — | first evaluation of a modified method |
| Reserved | the large-scale benchmark (if authorised) | 100–104 | — | must not be touched in Phase II |

**Rule.** A Phase II result that leads to a method change makes the changed method a *discovery* result. It must be re-tested on seeds 30–32 before being cited as confirmed.

### 14.2 Phase II pre-registration

**Shared.** MNIST, MLP 784-128-64-10, lr 0.05, batch 32, 1 local epoch, m = 10. Single-threaded BLAS. Thresholds are fixed here and will not be changed after seeing results.

**New statistics** (all computed on the client's noisy holdout, before and after its local update):
- CE;
- TS-local CE: a per-model temperature fitted on the client's noisy holdout by 2-fold cross-fitting;
- TS-global CE: a per-model temperature fitted on a 2,000-image clean server set (requires clean server data);
- clipped CE: min(CE, log 10);
- 0–1;
- Brier;
- MAE (1 − p_ỹ);
- GCE (q = 0.7).

**Noise channels:**
- *sym η*: flip to a uniformly random other class;
- *pair η*: y → y+1;
- *CF*: deterministic swap 0↔1, 2↔3, …;
- *IDN η̄ (our construction)*:
  - a reference MLP is trained 1 epoch on 10,000 clean images;
  - each sample's flip probability is a linear ramp over the client's difficulty ranks (difficulty = 1 − p_ref(y|x)), from 0 to 2η̄;
  - a flipped sample takes the reference model's most probable wrong class;
  - at η̄ = 0.4 the hardest 37.5 % of samples have flip probability > 0.5.

**Stage 7–8: measurement study (seeds 10–12; `phase2_metrics.py`).**
- *Base federation:* 30 CC + 5 RR (rotated) + 15 sym-η 0.6 clients, 600 samples each. Trained by FedAvg; checkpoints at rounds 20, 60 and 150.
- *Probes:* clients that never train the base model, built on data disjoint from it. Each has 500 training and 400 holdout samples.
  - 6 common datasets, each corrupted by every channel: clean; sym 0.2 / 0.4 / 0.6 / 0.8 / 0.9 / 0.95; pair 0.2 / 0.4 / 0.45 / 0.55 / 0.6 / 0.8 / 1.0; CF; IDN 0.2 / 0.4.
  - 4 rotated clean datasets; the same 4 with sym 0.6 (rare-and-noisy).
  - 2 label-rare datasets (classes 8–9).
  - Each probe makes one local update from each checkpoint.
- *Oracles:*
  - U_own: clean 0–1 reduction on the probe's own-distribution clean test set (10,000 images);
  - U_hold: clean reduction on the probe's own holdout images;
  - U_fed: change in balanced clean error over the common and rotated test sets after the partial step θ + Δ/10.
- PM1. For sym probes with η ≤ 0.8, the through-origin slope of the noisy 0–1 reduction on U_hold is within ±0.05 of a(η) = 1 − 10η/9.
- PM2. At round 150, for every statistic, the AUROC of the noisy **level** for RR vs sym η ≥ 0.4 is ≤ 0.2.
- PM3. At round 150, the AUROC (RR vs sym 0.2–0.8) of the CE **reduction** is at least 0.10 below that of the 0–1 reduction.
- PM4. Calibrated CE: **no directional prediction**. The result decides whether calibration alone explains the CE failure.
- PM5. At round 150, the Brier reduction's AUROC (RR vs sym η ≥ 0.6) is ≥ 0.05 below 0–1's. Theory: the confidence term is not attenuated, and Brier reverses for confidence-only updates when a(η) < 1/2, i.e. η > 0.45.
- PM6. The MAE reduction's AUROC (RR vs sym) is within 0.05 of 0–1's at n_h = 400, and MAE has higher pairwise ranking accuracy than 0–1 at n_h ≤ 50.
- PM7. Pair noise, rounds 60 and 150:
  - for η ≥ 0.55, the fraction of probe-checkpoints with r̃ > 0 and U_own < 0 is ≥ 0.5;
  - for η ≤ 0.45 it is ≤ 0.2.
- PM8. CF: r̃ > 0 and U_own < 0 in ≥ 90 % of probe-checkpoints at rounds 60 and 150.
- PM9. Sym η = 0.95 (a < 0): the mean noisy 0–1 reduction is ≤ 0 where the mean U_hold > 0.
- PM10. IDN at round 150: the 0–1 reduction's AUROC (RR vs IDN 0.4) is below its AUROC (RR vs sym 0.4).
- PM11. The normal-approximation probability that an RR probe's r̂ ranks below a sym-noisy probe's r̂ (from per-sample variances) matches the empirical subsampling frequency within 0.05 at every n_h ∈ {25, 50, 100, 200, 400}.
- PM12. Logit scaling s ∈ {0.5, 2} changes the CE-reduction ranking (Spearman with s = 1 below 0.9) but leaves 0–1 and TS-local CE at Spearman ≥ 0.99.

**Stage 8: FL boundary runs (seeds 20–22; `results/P2F.jsonl`).**
- *Settings:* 30 CC + 5 RR + 15 noisy clients, 600 samples each, T = 200:
  - S1: sym 0.6 — replicates E1 S1 on new seeds;
  - HET: sym η ∈ {0.2, 0.4, 0.6, 0.8, 0.95}, 3 clients each;
  - P45: pair 0.45;
  - P75: pair 0.75;
  - IDN: IDN 0.4.
- *Methods:*
  - FedAvg; FedEBA-style; EBA with weights capped at 2/m;
  - A (CE reduction); A′ (0–1); A′ shuffled; A′-MAE; A′-TS-local-CE; A′-Brier;
  - FedPCA-style; the clean-label oracle.
  
  All A′ variants use the same τ-clamp (τ → 0).
- PF1. S1: A′ rare accuracy > FedAvg in 3/3 seeds; EBA < FedAvg in 3/3.
- PF2. HET: A′ rare ≥ FedAvg + 2 pts (mean) and > FedAvg in ≥ 2/3 seeds; EBA < FedAvg (mean).
- PF3. P45: A′ rare ≥ FedAvg + 2 pts (mean); common cost ≤ 2 pts.
- PF4. P75 (beyond the pair reversal threshold 0.5): A′ common accuracy ≤ FedAvg − 3 pts in ≥ 2/3 seeds.
- PF5. IDN: A′ rare ≥ FedAvg + 2 pts (mean); common cost ≤ 1.5 pts.
- PF6. A′ variants: no directional prediction. "Matches" means within 1 pt of A′ on mean rare accuracy and within 1 pt on common.
- PF7. Capped EBA does **not** match A′: its rare accuracy is ≥ 2 pts below A′ in S1 and HET.

**Stage 9: relabelled clients (seeds 20–22; `results/P2R.jsonl`).**
- *Populations:* P_CF = 30 CC + 5 RR + 10 sym-0.6 + 5 CF (= E3d); P0 = S1 (no CF).
- *Guards on A′, fully specified now:*
  - **T1′ exclusion.** From round 20 (warm-up), a sampled client is dropped from the round if the global model's observed error on its local data exceeds (K−1)/K + √(ln(1/0.05)/(2n)), i.e. 0.95 for n = 600. A′ weights are computed over the remaining clients.
  - **Weight cap.** A′ weights capped at 2/m by water-filling.
  - **Fallback.** A round in which any client triggers the T1′ test uses FedAvg weights.
  - **Cluster-wise FedAvg.** B4sc_n.
  - **B then A′.** B4sc_aprime.
  - **Oracle routing then A′** (reference).
- PR1. A′ (unguarded) common accuracy < 0.90 in ≥ 1/3 seeds on P_CF; FedAvg ≥ 0.92 in 3/3.
- PR2. T1′ exclusion: common ≥ 0.94 in 3/3 seeds on P_CF, and rare ≥ (A′ on P0) − 2 pts.
- PR3. Weight cap: the worst seed's common accuracy is < 0.92 on P_CF (caps limit but do not remove the failure). Low confidence.
- PR4. Fallback: common ≥ 0.94 in 3/3 seeds on P_CF.
- PR5. B4sc_n: worst-10 % ≥ 0.85 in 3/3 seeds on P_CF.
- PR6. On P0, the T1′ guards cost ≤ 1 pt rare accuracy vs A′.
- PR7. On P75, the T1′ guard does not trigger after warm-up (observed error ≈ 0.77 < 0.95), so A′ + T1′ ≈ A′. The T1′ signature detects deterministic relabelling, not pair noise above 0.5.

**Stage 11: B audit (static: seeds 10–12; dynamic: fresh CFL seeds 10–12).**
- *Static design:*
  - K = 4 cluster models from oracle-routed CFL (E2 rot and perm populations, sym 0.6, 30 % noisy), taken at round 80 (trained) and round 10 (early);
  - probe pools of 40 clients (4 clusters × 10, 30 % noisy), drawn from data disjoint from training;
  - η ∈ {0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95} (sym) and pair 0.4;
  - n ∈ {50, 100, 150, 300, 500}.
- *Representations:* CE, z-CE, TS-local CE, 0–1, z-0–1 (B), Brier, z-Brier, MAE, z-MAE, argmin-0–1 (CE tie-break), argmin-CE, and a scalar mean error. k-means with K known; scored by ARI, NMI and purity.
- PB1. z-0–1: ARI ≥ 0.95 for η ≤ 0.6 at n ≥ 300 (trained models); raw CE: ARI < 0.6 at η ≥ 0.6.
- PB2. With trained models, argmin-0–1 is within 0.05 ARI of z-0–1 in every cell with η ≤ 0.8. If so, B's vector representation adds nothing beyond the 0–1 statistic once models are trained.
- PB3. At η = 0.95, both z-0–1 and argmin-0–1 misassign ≥ 50 % of the η-0.95 probes.
- PB4. The cells where z-0–1's ARI falls below 0.9 lie at a(η)√n within a factor 1.5 of each other (the T4 scaling).
- *Dynamic test:* new variant `ifca_err` (argmin of 0–1 error, CE tie-break) vs `ailc` (B), `ifca` and `clove_ce`, on 5 E2 settings (rot / perm sym 0.6 30 %; rot 0.8 30 %; rot 0.6 50 %; rot no noise).
- PB5. ifca_err has lower ARI than ailc in ≥ 3 of the 5 settings, because argmin inherits IFCA's initialisation collapse. If ifca_err matches ailc, B's representation adds nothing in the dynamics either.

**Code changes made for Phase II** (recorded per the integrity rules). All are additive. Existing methods are untouched, and a stored run will be re-executed after the change to confirm bit-exactness.
- `algos.py`:
  - A′ keys `r_brier` and `r_tsce`;
  - guards `t1g`, `cap`, `fb` on `rla*`;
  - `eba_cap`.
- `cfl.py`: variant `ifca_err`.
- New scripts:
  - `phase2_verify.py`, `phase2_theory.py`, `phase2_metrics.py`, `phase2_pop.py`, `phase2_baudit.py`.

### 14.3 Post-hoc addition (written 2026-10-02 14:58 UTC, after the measurement study, before running it)

- *What prompted it:* in the measurement study (seeds 10–12), **clipped CE**, min(CE, log 10), had the highest ranking fidelity of all statistics. It was a pre-registered measurement statistic but not a pre-registered FL variant. Pairwise accuracy vs U_own was 0.82 vs 0.78 for 0–1, and its AUROC was ≥ 0.998 at every n_h. MAE also beat 0–1 at small n_h.
- *Addition:* A′ with the clipped-CE statistic (`rla_clip_tau0.000001`), evaluated **only on reserved seeds 30–32**, together with FedAvg, A′ (0–1) and A′-MAE on the same seeds. Settings: S1, HET, P45, IDN. Because the variant was chosen after seeing Phase II data, this is its first (discovery-to-confirmation) evaluation.
- *Prediction:* A′-clipCE is within 1 pt of A′ (0–1) on mean rare accuracy and within 1 pt on common accuracy in each of the four settings. Prior: medium.
- *Rarity grid (measurement level; added 2026-10-02 15:05 UTC, before running).*
  - The pre-registered measurement design fixed 5 rotated clients in the base federation. To cover "rarity", the same probes are re-measured on base federations with **1** and **10** rotated clients (CC = 34 / 25; still 15 sym-0.6 clients), seeds 10–12.
  - *Prediction (from the Stage-5 mis-ranking formula):* the more rotated clients train the base model, the smaller the RR probes' reducibility late in training, and so the lower the AUROC (RR vs sym-noisy) at round 150. Concretely, AUROC(n_rr = 10) < AUROC(n_rr = 1) at round 150 for the 0–1 statistic.


## 15. Phase II outcomes (2026-10-02; full detail in `phase2_report.md`)

### 15.1 Current thesis (revised)

In federated learning with heterogeneous label quality:
1. **no client loss *level*** separates under-served clean clients from noisy ones;
2. **reductions of unbounded proper scores** (CE, Brier) are confidence-confounded;
3. **reductions of bounded statistics** (0–1, MAE, clipped CE) track a client's own-distribution clean utility, up to an identifiability boundary (pair noise > ½, relabelling, η ≥ (K−1)/K).

This is a *measurement* claim that builds on known robust-loss theory. It is not a method claim. **0–1 error is not special.**

### 15.2 Hypotheses

| Hypothesis | Status | Evidence (Phase II, confirmatory seeds) |
|---|---|---|
| CE-based client statistics misrepresent clean utility under heterogeneous noise | **survives** | Probe AUROC 0.24 (CE) and 0.43 (Brier) vs ≈ 1.00 (bounded); FL: A 9.5 pts below A′ |
| The CE failure is (mostly) calibration | **survives for symmetric noise only** | TS-local CE AUROC 1.00 (symmetric) vs 0.31 (pair ≤ 0.45); TS-global fails |
| Noisy local updates flatten predictions (optimisation mechanism) | **survives** | Predictive entropy +0.28 to +1.56 nats for noisy probes vs −0.24 to −1.15 for clean probes |
| 0–1 is the right statistic | **killed** | MAE and clipped CE equal or better (measurement); A′-MAE ≥ A′ in 9/9 FL combinations |
| Level statistics can be fixed by a better loss | **killed** | AUROC 0.00 for all 8 statistics' levels |
| A′ is a novel, competitive method | **killed** | FedGA / FedFomo / RHFL mechanism; A′-MAE and cluster-wise FedAvg dominate; 3/5 quantitative FL predictions missed |
| B is a novel, competitive method | **killed** | ≡ 0–1 argmin with trained models; dynamic edge = CLoVE's design; 8/9 at η 0.8 on fresh seeds |
| B → A′ composition | **killed** | Cluster-wise FedAvg ≥ B → A′ |
| The T1′ chance-level test detects relabelling | **survives** | 512/512 flags, 0 false; blind to pair 0.75 (as predicted) |
| The theory locates the boundaries | **survives** | PM1 slopes = a(η); r̃ = ΔD to 1e-16 (CF); η 0.95 reversal magnitude matched; η 0.9 → chance-level clustering (75 % misassigned) |

### 15.3 Pre-registered scorecard (§14.2–14.3)

- **Passed (20):** PF1, PF4, PF7; PR2, PR3, PR5, PR6, PR7; PM1, PM2, PM3, PM5, PM6, PM8, PM10; PB1, PB2, PB3, PB5; rarity.
- **Failed (9):** PF2, PF3, PF5; PR1 (FedAvg clause), PR4; PM7, PM11, PM12; clipped-CE post-hoc.
- **No verdict:** PM9 (vacuous as written; the magnitude check matched); PB4 (one failing cell, so not informative); PM4 and PF6 (no prediction).

### 15.4 Decision (Stage 13–14)

- **Framing: E (revise).** A′, B and B → A′ are killed as method contributions. The surviving content is the narrowed measurement claim of §15.1.
- **Gate: DO NOT AUTHORIZE LARGE-SCALE VALIDATION.**
  - Criterion 2 fails: novelty — the theory is published and the mechanisms are published.
  - Criterion 3 fails: simple alternatives (MAE / clipped CE, cluster-wise FedAvg, the chance threshold) match or beat the proposals.

### 15.5 Unknowns

- Behaviour with human annotation noise (CIFAR-10N) and with CNNs. Neither was testable here: no GPU, and PyTorch wheels are blocked.
- Whether any FL paper already reports the "levels always fail; CE reductions are confidence-confounded" characterisation. None was found, but the search was not exhaustive.
- A guard for pair noise between ½ and the chance threshold.
- Federation-level utility: no statistic tracked it (Spearman ≤ 0.46).
- Robustness to misreporting (E7) with a bounded-statistic rule.

### 15.6 Next experiments (only if the user wants to reopen the gate; all CPU-feasible)

1. A targeted novelty search for the characterisation itself:
   - "client loss level label noise fairness allocation fails";
   - "client valuation robust loss MAE";
   - "loss-based client selection noisy labels calibration".
2. The Stage 7 protocol on CIFAR-10N human label sets: per-client channels drawn from the annotator sets, with a CPU-trainable model and a fixed pre-registered threshold set.
3. A second fresh-seed confirmation (seeds 40–42, untouched so far) of A′-MAE vs A′ (0–1) vs cluster-wise FedAvg in S1, HET, P45 and IDN. MAE was a pre-registered competitor on seeds 20–22 and was confirmed once on the reserved seeds 30–32 after being identified as the best statistic.
4. A pair-noise guard: test whether disagreement between a client's own-label reduction and its reduction measured on *other* clients' holdouts (a cross-client check) separates pair > ½ clients. Pre-register before running.

## 16. Phase III — identifiability audit, completed 2026-10-02

### 16.1 Scope and target

The objective was read from the supplied goal-objective.md before proceeding. Work continued from the existing report, state, literature, P2 raw logs, implementations and theory. Primary threat model: honest clients with corrupted labels. Strategic scalar falsification was treated separately. No A′/B rescue, new aggregation architecture, GPU run or large-scale authorization occurred.

The declared decision target is `U_i=R_Q(theta)-R_Q(theta+a Delta_i)` for a specified clean target Q, applied scale a, and candidate update. Sign/rank can be sufficient without recovering an entire channel. Private noise rate, own risk, reference prediction accuracy and global utility are distinct. Simulations use binary scalar Brier loss; Phase II's clean 0–1 targets remain separate.

### 16.2 Main conclusions [proof / known / obs]

1. Two honest binary worlds share observed label probabilities (.6,.4), while clean target probabilities are .4 and .6. All forward class-conditional flip rates are <=1/3; each world has a perfectly clean client whose identity is unknown. Population local Brier candidates (.6,.4), baseline .5, have utilities (−.03,.01) versus (.01,−.03). Full observed-data equality implies equality of any same-information adaptive transcript. This is a valid elementary non-identifiability example, **not a novel FL theorem**.
2. More probes, unlabeled agreement or trajectories alone cannot resolve that ambiguity. Additional semantic observations or restrictions are necessary. This is a data-processing statement with explicit shared initial/side information; it does not dismiss externally trained semantic oracles under a justified competence assumption.
3. Known-channel contrast recovery only needs `d in col(T)` for `q=T^T p`, not always full invertibility. Trusted target validation or genuinely independent, oriented shared-item annotations can recover utility under their assumptions. These mechanisms are established in loss correction, validation, crowdsourcing and peer scoring.
4. **FedDS already explicitly studies client reliability identifiability in FL.** It estimates prediction confusion matrices on shared unlabeled reference data, not marginal clean update utility. Its literal “some unknown dominant client” orientation condition admits a full-rank three-view counterexample; a named dominant client plus factorization conditions is stronger. This qualification is useful but uses classical label-switching reasoning.
5. The tested formulation is substantially covered by existing theory. Outcome D is scoped to the actual claims established here; it is not an exhaustive assertion that all possible FL identification questions are solved. No surviving novel recovery result justifies a practical mechanism.

### 16.3 Corrections superseding Phase-II interpretations

The historical files/raw results are preserved. These corrections supersede corresponding interpretations in §15 and `phase2_report.md`:

- **AUROC 0 is reversed discrimination**, not absence of information. All eight loss levels reverse the selected rare-clean/corrupted comparison; that does not prove every estimator from the scalar fails.
- **Brier is bounded.** Boundedness alone is not sufficient; MAE has a symmetric-loss identity under symmetric noise, whereas clipped-CE's success here is empirical. The old “bounded statistics are the answer” formulation is too broad.
- **Corruption discrimination is not global utility recovery.** Round150 MAE/clipped-CE RR/corrupted AUROC is 1, but useful/harmful U_fed AUROC is .5173/.5322 and Spearman .0685/.0833. Across 336 probes, 130 have opposite U_own/U_fed signs; 119 corrupted probes have positive U_fed and 15 clean probes have negative U_fed. U_own is a full local step on own data; U_fed is a one-tenth step on a 50:50 common/rotated target. Step size and target are confounded, and gains are small.
- **Pair noise >.5 removes a guarantee; it does not force every update to fail.** Wrong-sign fractions at .55/.6 are 7/36 and 14/36; MAE pair-high discrimination AUROC is .7465.
- **Deterministic-relabelling gain equals Delta D**, but is not universally the negative clean gain in multiclass models. Residual with clean holdout gain reaches .26; third wrong classes matter.
- **Chance guard success uses model competence/orientation.** The logs record 512 CF flags and no non-CF flag keys, not a universal clean/noisy classifier or independently enumerated sensitivity across every possible exposure.
- The .5 pair-margin threshold is not a universal information-theoretic identification boundary. Known invertible high-noise/permutation channels permit correction; unknown asymmetric channels can remain ambiguous below .5.

### 16.4 CPU diagnostics and validation

- `phase3_phase2_audit.py` re-reads delivered P2 probe/run logs. Recomputed stored score discrepancy: 0. Cached logits were absent, so this is not retraining. All 13 P2 files unchanged.
- `phase3_identifiability.py`: 200 repetitions at evaluation n=25,100,400,1600, fixed candidates and seeds. 9,600 matched-world records and 16,000 grid records; counts represent estimator/world/repetition rows, not independent FL runs.
- All eight exact checks pass. Two-world average sign/ranking accuracy remains .5, while Brier score variance drops from 3.1797e−4 to 5.7715e−6.
- All O1–O6 regimes have illustrative diagnostics on symmetric/pair/class-conditional/deterministic/instance-dependent/heterogeneous channels and common/rare clients. Binary pair=symmetric is disclosed, with a separate three-class margin check. O2–O4 scores are not optimal estimators of their cumulative packets.
- On the original target, trusted-validation MSE drops 4.681e−4→7.534e−6; independent-oriented-peer MSE drops 7.978e−4→4.059e−5. Own-label Brier MSE remains .03972→.03920. The complemented world preserves O1–O4 scores; O5/O6 add new semantic observations.
- Correlated-copy peer ablation: sign accuracy .525→.090 as n rises. Three copied annotations do not satisfy independent-view assumptions.
- Initial identical-packet hash check was strengthened after review to separate protocol executions with hidden labels drawn from each world's exact conditional law; observed-data seeds/scores did not change. This is documented as an implementation audit amendment, not a new preregistration.
- Original-result manifest verifies **37/37 prior result files unchanged**. [Independent review](phase3_review.md) confirms numerical metrics and scopes the claims.

### 16.5 Literature, real-world scope and unresolved limits

Fresh primary-source searches cover Liu/Scott/Menon/general corruption decision theory, DS/multiview/weak supervision/peer scoring/complementary labels, recent CCN/IDN, and all required FedDS/FNRE/FedSIR/FedSNC/FedCova/FedEFC/FedELC/FedClean/Scott–Luo–Ho/survey comparisons. FedSNC was available only at abstract/introduction level; no full-method claim is made. Sources/search logs are in the two literature memos and `literature_matrix.md` §H.

Human CIFAR-N annotations, Clothing1M weak labels, institutional coding differences, Snorkel/PheNorm and a documented six-hospital RACOON FL study provide bounded motivation. No claim that the exact binary ambiguity was observed in production; no natural-noise training was performed. Private semantic disagreement may represent different tasks rather than erroneous labels.

### 16.6 Decision and artifacts

**DO NOT AUTHORIZE NEW DIRECTION.** Outcome D for this formulation. Retain the limitations synthesis and corrections; stop method development here and investigate a different FL question. Reopening requires a specific FL restriction and a theorem that is not an immediate centralized/noisy-label/crowdsourcing corollary. The current evidence does not justify large-scale work.

- Main deliverable: [phase3_identifiability_report.md](phase3_identifiability_report.md).
- Literature: [central audit](phase3_central_literature.md), [federated audit](phase3_federated_literature.md), matrix §H.
- Evidence: [Phase-II bridge](phase3_phase2_bridge.md), [CPU tables](phase3_cpu_results.md), [per-channel tables](phase3_channel_results.md), [review](phase3_review.md).
- Specification/code: `phase3_experiment_plan.md`, `phase3_identifiability.py`, `phase3_phase2_audit.py`, `phase3_summarize.py`.
- New raw/summary outputs: `results/P3_*`; prior `E*`/`P2*` results untouched.
- Exportable figure: `figures/phase3_identifiability.pdf` and `.png`.

---

## 16. Phase IV–V: explanation-alignment pivot and approved smoke test (2026-10-02)

### 16.1 Why the project pivoted

After Phase III closed the noisy-label/identifiability direction, a new gap scan across the ten seed-paper problem families found that queue/availability, heterogeneous architectures, and several other directions were already crowded by strong recent work. Explainable FL remained promising because xFedAlign uses one global explanation prior and explicitly notes limitations when groups are ambiguous or feature semantics drift.

### 16.2 Failed/revised explanation candidates

1. **REPA / random-effects personalized attribution shrinkage** — **KILL/REVISE**. It improved heterogeneous explanations but was decisively beaten by a simple coordinate variance gate and did not match global averaging under shared explanations.
2. **HGEA / hard coordinate heterogeneity gate** — initially passed untouched patch seeds, but **KILL as general approach** after independent rotation stress. It handles sparse feature-specific heterogeneity but misses coherent cross-feature shifts.
3. **Coordinate-only peer smoothing** — useful but incomplete; rotation improves, yet it lacks whole-explanation/client structure.
4. **UCPA** — combines whole-explanation JSD compatibility with coordinate-level standardized attribution compatibility.

### 16.3 Frozen UCPA

For clients `i,k` and coordinate `j`:

- `g_ik = exp(-JSD(E_i,E_k)/h)`
- `c_ikj = exp(-0.5*(|E_ij-E_kj|/(z*sqrt(V_ij+V_kj+floor)))^2)`
- `w_ikj = g_ik*c_ikj`

The aligned explanation is the normalized coordinatewise peer-weighted mean.

Development-selected parameters frozen before final confirmation:

- `z=2.0`
- `h=0.048`
- `lambda=0`
- variance-floor fraction `0.05`

A homogeneous-coordinate fallback was explicitly rejected before final confirmation because it traded away rotation robustness.

### 16.4 Final pre-registered confirmation

Fresh untouched seeds: **501–505**. No earlier JSONL contains these seed IDs.

Two qualitatively different stress families:

- sparse predictive patch heterogeneity (strong and moderate; shared/grouped/personal);
- coherent image-rotation heterogeneity (shared/grouped/personal).

All pre-registered criteria A–F passed. See `phase5_preregistered_plan.md` and `results/P5_confirmation_summary.json`.

Key heterogeneous JSD results:

| Family | UCPA | Local | Best oracle-tuned global | Improvement vs global |
|---|---:|---:|---:|---:|
| Patch grouped | 0.002690 | 0.010442 | 0.006521 | 58.7% |
| Patch personal | 0.002488 | 0.011155 | 0.005046 | 50.7% |
| Rotation grouped | 0.004408 | 0.008642 | 0.006303 | 30.1% |
| Rotation personal | 0.004808 | 0.009134 | 0.006639 | 27.6% |

UCPA beat Local and the selected global baseline on **5/5 fresh seeds in all four heterogeneous families**.

Equal-family heterogeneous mean JSD:

- UCPA 0.003599
- CoordPeer 0.004328
- HGEA 0.005552
- QGate 0.005764
- oracle-tuned Cluster 0.005819

Therefore UCPA is **SMOKE-TEST APPROVED FOR CONTINUATION**.

### 16.5 Novelty state

Current targeted search found no direct method collision for the exact two-scale explanation rule. Closest work:

- xFedAlign: one robust global explanation prior;
- UncertainXFL: logical-rule uncertainty used to rank explanations/model aggregation;
- FedXDS: attribution-guided data sharing;
- similarity-aware PFL: personalized model/prototype aggregation rather than explanation coordination.

Novelty risk remains **medium**. Kernel smoothing, uncertainty weighting, client similarity, and personalization are individually known; the potential contribution is their explanation-specific two-scale synthesis and the empirical failure analysis motivating it.

### 16.6 Current gate

**APPROVED ONLY FOR NEXT-STAGE VALIDATION.**

Do not call the method conference-ready yet. Required next evidence:

1. official xFedAlign integration/comparison;
2. nonlinear CNN and CIFAR-10 experiments;
3. explanation fidelity metrics (deletion/insertion AUC), EDI and Top-k overlap;
4. sparse top-k communication/privacy accounting;
5. partial participation, unequal client sizes, drift and poisoning stress;
6. theory/bias-variance analysis for two-scale uncertainty-compatible peer smoothing;
7. final novelty audit against concurrent work.
