# Phase V novelty audit: UCPA

Date: 2026-10-02

## Candidate being audited

**UCPA — Uncertainty-Compatible Peer Alignment** forms a personalized federated explanation prior by multiplying:

1. a client-pair kernel based on whole-explanation Jensen-Shannon divergence, and
2. a coordinate-level kernel based on the attribution difference standardized by both clients' estimated sampling variance for that coordinate.

The server therefore borrows explanation information continuously from statistically compatible peers instead of forcing all clients toward one global explanation prior or assigning every client to a discrete cluster.

The mathematical ingredients (kernel smoothing, JSD, standardized differences, heteroskedastic uncertainty) are not themselves claimed as new. The candidate novelty is their use as a **two-scale explanation-coordination rule in federated learning**, specifically to distinguish estimation noise from genuine explanation heterogeneity.

## Closest inspected work

### xFedAlign — ICML 2026

Wasif et al., *Explainable Federated Learning via Global–Local Attribution Alignment*, ICML 2026.

- Coordinates sparse per-class attribution artifacts in a group space.
- The server robustly aggregates client artifacts into a **Global Explanation Prior**.
- Local explanations are softly aligned to that global prior while task training remains separate.
- The supplied paper explicitly notes that its approach assumes a meaningful, relatively stable group space and says ambiguous groups or drifting feature semantics may require adaptive grouping or drift-aware updates.

**Collision assessment:** closest direct method. UCPA differs by constructing a **client-specific continuous peer prior**, using whole-explanation compatibility and coordinate sampling uncertainty instead of one shared global prior. This difference is substantive enough to test, but novelty risk remains medium until a full implementation against xFedAlign's official code is completed.

Source: https://proceedings.mlr.press/v306/wasif26a.html

### UncertainXFL — 2025 preprint

Zhang & Yu, *Uncertainty-Aware Explainable Federated Learning*, arXiv:2503.05194.

- Explicitly attaches uncertainty to federated explanations.
- Uses concept/logical-rule explanations.
- Ranks/selects rules based on rule uncertainty and validity/accuracy.
- Uses the resulting rule reliability to influence model aggregation.

**Collision assessment:** important conceptual neighbor because it combines FL, explanations, and uncertainty. It does not estimate sampling variance of attribution summaries and does not construct per-client, per-coordinate compatible-peer explanation priors.

Source: https://arxiv.org/abs/2503.05194

### FedXDS — ICCV 2025

Hoefler et al., *FedXDS: Leveraging Model Attribution Methods to Counteract Data Heterogeneity in Federated Learning*.

- Uses attribution methods to identify task-relevant data elements for selective data sharing under heterogeneity.
- The explanation signal serves data-sharing/model-training utility rather than cross-client explanation alignment.

**Collision assessment:** shares attribution-based FL but not the UCPA problem or mechanism.

Source: https://arxiv.org/abs/2606.31742 and ICCV 2025 proceedings.

### Federated XAI review — 2026

Gholizade et al., *Federated explainable artificial intelligence: roles, architectures, evaluation, and open challenges*.

- Reviews explainability across the FL lifecycle.
- Identifies non-IID explanation behavior, stability/consistency, privacy, communication, and benchmark standardization as open challenges.

**Collision assessment:** supports the problem motivation but is not a method collision.

Source: https://arxiv.org/abs/2607.13045

### Personalized/similarity-aware FL methods

PFedAtt and later community/prototype methods use client similarity, attention, clustering, or prototypes to personalize **models/representations**. These establish that peer-selective collaboration is a known FL principle.

**Collision assessment:** they reduce the breadth of any novelty claim. UCPA must not claim that similarity-aware peer collaboration itself is new. Its potential novelty is explanation-specific, two-scale, sampling-uncertainty-aware coordination.

## Targeted search outcome

Targeted searches included combinations of:

- federated explanation alignment + uncertainty;
- personalized federated explanations;
- attribution aggregation + client similarity;
- attribution variance + federated explanations;
- peer explanation smoothing;
- adaptive/global explanation priors;
- SHAP/Integrated Gradients + federated uncertainty aggregation.

No inspected source used the same two-scale rule: whole-explanation similarity multiplied by coordinate-level standardized attribution compatibility to create a personalized explanation prior.

This is **not proof of novelty**. Search coverage is finite, terminology can differ, and new/concurrent work may exist.

## Novelty decision

**Current novelty status: PLAUSIBLE / MEDIUM RISK.**

UCPA is sufficiently differentiated to justify continued research after its smoke-test pass, but not yet sufficiently audited to make a publication-level novelty claim.

The next novelty gate must include:

1. an official-code xFedAlign comparison and close reading of all its ablations;
2. a broader citation-network search around FedAttr-Agg, Fed-XAI, UncertainXFL, personalized FL, and explanation aggregation;
3. checking concurrent OpenReview/arXiv work before submission;
4. a precise claim that avoids treating kernel smoothing, uncertainty weighting, similarity-aware collaboration, or personalization as independently novel.
