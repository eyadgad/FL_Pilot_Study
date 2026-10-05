# Baseline and predecessor provenance

## xFedAlign — closest direct explanation-coordination baseline

- Wasif et al., *Explainable Federated Learning via Global–Local Attribution Alignment*, ICML 2026.
- PMLR: https://proceedings.mlr.press/v306/wasif26a.html
- Official code: https://github.com/dawoodwasif/xFedAlign

xFedAlign communicates compact per-class top-k attribution artifacts, robustly aggregates them into one Global Explanation Prior, and softly aligns local explanations while leaving task-model training decoupled.

`xfedalign_median` in this package is a **self-contained mechanism reproduction**, not a byte-for-byte run of the official source. It uses the same trained task model and local explanation samples as the other methods so the coordination mechanism can be isolated.

## Local-XAI

`local` keeps each client's own explanation and provides the no-sharing fidelity reference.

## FedAttr-style mean

`fedattr_mean` is a transparent server-side global mean attribution-template comparator. It is an operational baseline, not an official external implementation.

## Cluster comparator

`cluster` is a deliberately strong generic baseline: k-means over client explanation vectors followed by cluster-wise pooling. It is not claimed to reproduce a named paper. Its role is to test whether continuous personalized collaboration adds value beyond a discrete grouping solution.

## Frozen negative predecessors

- `ucpa`: fixed-bandwidth, uncertainty-compatible UCPA v1. The returned full experiment killed this method; peer weights collapsed at realistic artifact scales.
- `sacpa_global_count`: Phase-VII scale-adaptive predecessor. It passed most smoke criteria but failed strong grouped sparse heterogeneity because a distant group could authorize foreign support through raw counts.

The full research history and gate reports are preserved under `research_history/`.

## Adjacent literature that constrains novelty

- UncertainXFL (2025) explicitly integrates explanation uncertainty into federated learning, but its uncertainty is attached to logical/rule explanations and is used to prioritize model/rule aggregation rather than to create NC-SACPA's sparse personalized attribution prior: https://arxiv.org/abs/2503.05194
- Adaptive PFL via kernel mean embeddings (ICML 2026) learns data-dependent collaboration weights for personalized model learning; it is important prior art showing that adaptive peer weighting is not itself novel: https://proceedings.mlr.press/v306/fermanian26a.html
- Controlled Collaboration Geometry (ICML 2026) studies collapse/degeneracy of collaboration graphs in personalized FL and is relevant prior art for the claim that collaboration structure must be controlled: https://proceedings.mlr.press/v306/yin26j.html
