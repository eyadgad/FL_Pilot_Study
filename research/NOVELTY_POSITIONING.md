# Novelty and positioning — CFBA-CRC

## Narrow novelty claim
CFBA-CRC is **not** proposed as a new conformal framework. It applies finite-sample risk control to a specific unresolved FedXAI decision: how strongly each client should accept a shared explanation prior when alignment can improve consistency but damage fidelity to the deployed task model.

The candidate contribution is the combination of:
1. client-private calibration of federated explanation alignment strength;
2. a bounded multi-metric *excess explanation-fidelity* loss relative to Local-XAI;
3. a globally certified control showing when one beta is over-conservative under heterogeneous clients;
4. attack-time recalibration against the actual received prior;
5. no additional high-dimensional explanation payload beyond the matched global-prior baseline.

## Closest research families
- **xFedAlign (ICML 2026):** closest FedXAI method; builds a robust Global Explanation Prior and aligns local explanations. CFBA reuses that prior family deliberately and changes the decision rule for alignment strength.
- **Conformal Risk Control (ICLR 2024):** provides the finite-sample bounded-risk machinery. CRC itself is prior art.
- **Aligning Model Properties via Conformal Risk Control (NeurIPS 2024):** establishes that CRC can calibrate model/property post-processing. This increases novelty risk and prevents a broad claim that “CRC alignment” is new.
- **Conformal feature-attribution explanation work (2026):** provides confidence/sufficiency guarantees for feature attribution; CFBA must not claim that conformal XAI is new.
- **Site-conditional federated CRC (MICCAI 2026 workshop):** directly shows that pooled/global calibration can fail individual sites. Therefore per-client federated CRC by itself is not novel; the paper must focus on explanation alignment and its multi-metric fidelity budget.
- **2026 FedXAI surveys/reviews:** identify non-IID explanation stability, security, communication, and evaluation as unresolved challenges. These support significance, not originality.

## Current novelty risk
**Medium-high.** The full paper is only defensible if the empirical study shows that the explanation-specific risk formulation and client-specific alignment decision add value over both fixed xFedAlign and a globally certified CRC control.

## Verification links
- xFedAlign (ICML 2026): https://proceedings.mlr.press/v306/wasif26a.html
- Conformal Risk Control (ICLR 2024): https://openreview.net/forum?id=33XGfHLtZg
- Aligning Model Properties via Conformal Risk Control (NeurIPS 2024): https://papers.nips.cc/paper_files/paper/2024/hash/c79625091a4f8b5d3abe29f3b14fa43a-Abstract-Conference.html
- Turning Feature Attributions into Sufficient Explanations Using Conformal Prediction (2026): https://proceedings.mlr.press/v329/alkhatib26a.html
- When Average Calibration Fails: Site-Conditional Federated Conformal Risk Control (MICCAI 2026 workshop): https://papers.miccai.org/miccai-2026-sat/DeCaF_004.html
- Federated explainable artificial intelligence review (2026): https://www.sciencedirect.com/science/article/pii/S0957417426020920
