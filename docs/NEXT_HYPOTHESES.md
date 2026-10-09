# Next iteration: what would actually be novel?

## Hypothesis A — out-of-client-class support transfer

A global field should help primarily when a client has few direct-IG teacher examples of a **prediction class**. Federated class-conditional fields must be compared against a full-budget private class-conditioned field with shrinkage and a class-free shared field; otherwise adding class conditioning is not novel. Test support strata `{0, 1–3, 4–7, >=8}`, with predeclared sparse-class JSD and shared-versus-private interaction effect. No class may receive a manufactured teacher map where data are absent. A privacy/privacy-leakage audit is mandatory because class support counts can themselves leak sensitive information.

## Hypothesis B — sparse negative-transfer rejection

Rather than selecting global `lambda` on 12 validation images, test a **client-validated spatial reliability mask** inferred solely from teacher statistics. The mask should reject peer statistics at coordinates with unstable local/peer likelihood ratios. It must transmit no raw images or IG maps and must be compared to a private spatial shrinkage model under the same teacher and byte budgets. This needs genuinely new seeds; current pilot checkpoints are now *development data*.

## Hypothesis C — communication Pareto and placebo peers

Evaluate private-field, fixed-peer, adaptive-peer, shuffled-peer (negative control), and a public shared constant field on the same CNN/IG seeds. Sweep teacher budgets `{8, 16, 32, 48}` and wire-bit budgets at `{256, 512, 1024, 2366, 5510}` **including protocol and candidate/validation selection traffic**. A federated claim requires a consistent nontrivial gain over the strong zero-communication private option, not merely lower JSD relative to linear surrogate. At alpha=.05, demonstrate that adaptive transfer can avoid regressions *without deciding its gate by seeing test-set JSD*.

## Literature & claim boundaries

- Wasif et al., 2026, **xFedAlign**, ICML, https://proceedings.mlr.press/v306/wasif26a.html — already uses compact global-local explanation alignment.
- Schuler et al., 2026, **Metric-Guided Attribution Fusion**, https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7129255 — optimization and metric-aware attribution fusion may limit novelty of simple validation-selected weights.
- Personalized federated learning and negative-transfer control are established ideas. A new paper must demonstrate an *attribution-specific mechanism* and robust evidence, not only rename a selector.
