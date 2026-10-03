# Novelty and current-research positioning

**Search date:** 2026-10-02. This is a living audit, not a guarantee of novelty.

## Closest direct work

### xFedAlign — Wasif, Moore, Lu & Cho, ICML 2026

Published contribution: a model-agnostic federated XAI framework using local surrogates, sparse top-k attribution artifacts, and a **single Global Explanation Prior** robustly aggregated at the server; it reports IID/non-IID image, text and tabular results, deletion/insertion AUC, privacy, and attribution-poisoning experiments.

UCPA difference: replaces the single global explanation object with a continuous personalized peer prior. Peer influence depends jointly on whole-explanation similarity and coordinate-level sampling uncertainty.

Paper: https://proceedings.mlr.press/v306/wasif26a.html  
Official code: https://github.com/dawoodwasif/xFedAlign

### UncertainXFL — Zhang & Yu, 2025

UncertainXFL explicitly evaluates uncertainty of logical/rule explanations and uses explanation quality/uncertainty to prioritize clients during model aggregation. Its uncertainty object and objective differ from UCPA: UCPA estimates sampling uncertainty of attribution coordinates to decide *where explanation information should be shared*, while leaving task-model aggregation unchanged.

Paper: https://arxiv.org/abs/2503.05194

### FedXDS — Hoefler, Mueller & Samek, ICCV 2025

Uses feature attribution to guide selective data sharing for heterogeneous FL. It uses XAI as a mechanism for data exchange/model performance, not as a personalized explanation-alignment prior.

Paper record: https://mlanthology.org/iccv/2025/hoefler2025iccv-fedxds/

### FedUP — Qi et al., IJCAI 2026

Models class prototypes probabilistically and uses uncertainty-aware prototype aggregation/personalization. This is important adjacent prior art because it establishes uncertainty-weighted federated knowledge sharing, but the shared object is a class prototype distribution rather than an explanation, and it does not use UCPA's product of global explanation relevance and coordinate sampling compatibility.

Paper: https://www.ijcai.org/proceedings/2026/171

## Broader personalized-FL collision risk

Similarity-aware collaboration, soft clustering, mixture-of-experts, and multi-level personalization are established ideas in personalized FL. Therefore the paper must not claim that continuous peer weighting is itself new. UCPA must be justified by the structure of **federated explanation estimation**: attribution coordinates have heterogeneous sampling uncertainty and can be genuinely client-specific.

## Why the problem is current

The 2026 FedXAI review describes fragmented evaluation practices and lack of standardized benchmarking for explanation quality, consistency under non-IID data, privacy leakage, and communication costs. The full suite is intentionally built around those axes rather than a single explanation-drift number.

Review: https://arxiv.org/abs/2607.13045

## Conference alignment

ICML 2026 asks reviewers to assess soundness, presentation, significance and originality; it explicitly asks whether experimental claims are well designed/supported and whether the work is properly distinguished from current literature. The package therefore separates development from confirmatory seeds, uses disjoint client subsets, includes close baselines and ablations, and records failures.

ICML reviewer guidance: https://icml.cc/Conferences/2026/ReviewerInstructions

NeurIPS 2026 likewise treats significance and originality broadly but requires clearly supported claims and relevant comparison to prior work. Its guidelines note that simple, practical ideas can be significant and that originality can come from a well-motivated combination, but the reasoning and evidence must explain why that combination is effective.

NeurIPS reviewer guidance: https://neurips.cc/Conferences/2026/ReviewerGuidelines

## Current novelty assessment

**Medium risk.** No direct 2026 collision found for the exact two-scale explanation rule during this audit. The components are individually familiar, and new papers may appear before submission. A final search must be repeated after full results and immediately before paper submission.

### Metric-Guided Attribution Fusion — Schuler et al., 2026 preprint

This July 2026 work studies explanation quality in FL and formulates multi-objective fusion of multiple XAI methods based on explanation metrics. It is relevant because it also treats explanation aggregation/fusion as an optimization problem. Its fusion axis is **across XAI methods**, whereas UCPA's axis is **across federated clients and attribution coordinates under heterogeneous sampling uncertainty**. It should be cited in a final paper if the preprint remains current.

Record: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7129255
