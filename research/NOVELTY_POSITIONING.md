# NC-SACPA novelty positioning — audit date 2026-10-04

## Closest direct work: xFedAlign (ICML 2026)

xFedAlign constructs compact per-class top-k attribution artifacts, robustly aggregates them into **one Global Explanation Prior**, and softly aligns client explanations to that prior while decoupling explanation coordination from task optimization.

NC-SACPA changes the coordination object: there is no single global explanation prior. Each client receives a peer prior based on a self-tuned explanation-space neighborhood, and sparse features absent from the target are imported only when multiple nearby peers corroborate them.

Paper: https://proceedings.mlr.press/v306/wasif26a.html

## UncertainXFL (2025)

UncertainXFL is important because it explicitly treats uncertainty in federated explanations. It produces logical/rule explanations with uncertainty and uses explanation quality to influence model aggregation. NC-SACPA does not claim that "uncertainty-aware FedXAI" is new; in fact the final NC-SACPA rule removes UCPA v1's variance channel entirely.

Paper: https://arxiv.org/abs/2503.05194

## Personalized-FL collision risk

Adaptive collaboration is established beyond FedXAI.

- Fermanian et al. (ICML 2026) learn collaboration weights via kernel mean embeddings and multi-task averaging.
- Yin et al. (ICML 2026) analyze collaboration-graph degeneration toward global consensus or spontaneous clustering and propose controlled collaboration geometry.

These works mean NC-SACPA cannot claim novelty for self-tuned kernels, similarity-weighted peers, collaboration graphs, or continuous personalization in general. The contribution must be tied to the **federated explanation artifact problem** and to the specific sparse-support failure identified experimentally.

References:
- https://proceedings.mlr.press/v306/fermanian26a.html
- https://proceedings.mlr.press/v306/yin26j.html

## Why the problem remains current

The 2026 FedXAI survey identifies non-IID explanation behavior, explanation stability, security, communication cost, and lack of standardized evaluation as open challenges. The package therefore makes local fidelity primary and jointly measures consistency, perturbation fidelity, attacks, privacy exposure, and communication.

Survey: https://arxiv.org/abs/2607.13045

## Candidate novelty statement

> Existing global-prior explanation alignment can over-pool genuinely heterogeneous clients, while fixed-bandwidth personalized alignment can collapse when explanation-distance scale changes and global support voting can import sparse features from a semantically distant majority. NC-SACPA forms a client-specific sparse explanation prior using a self-tuned explanation-space peer kernel and requires support corroboration inside the target's nearest explanation neighborhood before importing absent features.

This statement is **provisional** until the full study and a final pre-submission literature search.

## Novelty risk

**Medium.** No direct collision was found in the 2026-10-04 audit for the exact combination of self-tuned explanation-space collaboration plus neighborhood-corroborated sparse support. Every major ingredient is known individually, so the paper must demonstrate a non-obvious FedXAI failure mode and evidence that both parts of the rule matter. If a simpler adaptive global/local mixture matches NC-SACPA in the full study, the novelty claim should be revised or killed.
