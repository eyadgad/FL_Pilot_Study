# Phase IV-C pre-registration: independent rotation stress for HGEA

Registered 2026-10-02 (America/Toronto), after the Phase-IV-B patch confirmation passed and before any rotation-stress seed was run.

Purpose: test whether the fixed HGEA rule (`c=4`) generalizes to a qualitatively different explanation-heterogeneity mechanism. No HGEA parameter may change.

## Setting
Binary MNIST 3-vs-8, same FedAvg logistic task model and exact linear integrated-gradient summaries as Phase IV-B.

Fresh seeds: 401, 402, 403, 404, 405.

Client transforms:
- shared: all clients 0 degrees,
- grouped: four rotation groups {-20, -7, +7, +20} degrees,
- personal: 12 client-specific angles linearly spaced from -24 to +24 degrees.

Explanation sample sizes: 8, 16, 32, 64 positive examples/client.

## Comparators
- Local
- HGEA fixed c=4
- QGate q=.01 + FDR + mean prior
- Global-mean soft alignment with beta in {0.1, .2, .4, .6, .8, 1.0}
- Global-median soft alignment with same beta grid
- Clustered mean/median-style alignment using k in {2,3,4,6}, beta grid

For this secondary stress only, global/cluster competitors may be selected *post hoc by oracle JSD over the complete rotation stress*. This intentionally gives them an unrealistically favorable tuning advantage. HGEA remains fixed and untuned.

## Stress-pass criterion
This secondary stress supports HGEA only if:
1. shared: HGEA is within 5% JSD of the oracle-tuned best global comparator;
2. grouped and personal: HGEA improves JSD by >=10% over Local and >=5% over the oracle-tuned best single-global comparator;
3. personal: HGEA is no worse than the oracle-tuned best cluster comparator by more than 5%;
4. grouped/personal: HGEA beats Local on at least 4/5 seeds averaged over sample sizes.

Failure does not erase the pre-registered Phase-IV-B result, but it downgrades HGEA from "approved smoke-tested approach" to "patch-specific preliminary result" and requires revision.
