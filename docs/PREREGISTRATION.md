# RWSSA preregistration v1.0 — frozen before any pilot/full run

**Status:** protocol freeze; no pilot (2401–2403) or confirmatory (2501–2510) runs executed within this package. Previous smoke seeds 1861–1865, 1871–1874 and 1881–1882 are exclusively development/history.

**Primary RWSSA hypothesis:** shared feature-level risk weighting (rho=.30, fixed grid of lambda, independent selection/calibration) improves explanation alignment and task-model oracle fidelity without the support/function collapse seen in earlier work. It must beat not only xFedAlign-style beta=.2, but also global-mean and quality-weighted aggregates and ablations.

**Primary endpoints (held-out):** sample-to-task-IG JSD ↓, pairwise EDI ↓, deletion AUC ↓, insertion AUC ↑, oracle top-k overlap ↑. Task accuracy is a validity control; artifact bytes and additional client quality-channel bytes are audited. The same trained global CNN and local surrogate/IG data support all methods within each run.

**Pilot gate P:** 3 real-data families × 3 new seeds; at least 2/3 families with >=4/5 directional metric wins against fixed xFedAlign, valid task learning, no severe Local-XAI fidelity degradation, and communication overhead <=12× the xFedAlign-style artifact size. Strongest-baseline per-metric wins are reported. Failing means stop before Tier A. The code's `evaluate_rwssa_gates.py` is the executable gate definition.

**Confirmatory gate A:** 8 families × 10 new seeds; at least 4/6 heterogeneous families meet >=4/5 directional metric wins versus xFedAlign-style median in >=7/10 seeds, functional non-collapse relative to Local, acceptable communication, plus ablation superiority in >=50% of matched JSD comparisons. The actual executable gate reports detailed family-level results. A 10-seed inference is still exploratory as a conference acceptance predictor.

**Baseline disclosure:** global mean, iFLASH-inspired quality-weighted reproduction, frozen-beta xFedAlign-style median, beta=.4 stronger control, and ablations. The study is **not** a verified reproduction of official SHAP/iFLASH or official xFedAlign implementation; comparison to author code is a future research prerequisite.

**Risk and privacy disclosure:** JSD/top-k-only CRC surrogate is the primary frozen procedure; deletion/insertion are evaluated separately. Dense safety-score transmission is not private or DP-certified and must be charged to RWSSA. Sparse 8-bit gain reports are a novel **exploratory** variant, not the confirmed algorithm; no post-hoc substitution if dense RWSSA fails. The four-metric budget is independently evaluated in Tier B, not used to change Tier A.

**Analysis:** preserve all failed seeds, raw logs, config hashes, model checkpoints and per-client selection diagnostics. Compare seed-matched differences, bootstrap intervals, qualitative effects and negative results; do not change confirmatory parameters after inspecting outcomes. Tier C cannot be used to re-interpret failure as preregistered success.
