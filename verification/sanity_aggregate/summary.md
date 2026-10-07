# Aggregate results

Runs found: 8

Primary paired tables compare RWSSA against xFedAlign-style, iFLASH-inspired, global-mean, and ablation controls. Negative differences are favorable for lower-is-better metrics; positive for insertion/overlap.

## sanity / clean / fedattr_mean (n=1)
- artifact fidelity JSD: 0.402908 [0.402908, 0.402908]
- pairwise EDI: 0.000000
- deletion / insertion AUC: 0.259343 / 0.254806

## sanity / clean / iflash_proxy (n=1)
- artifact fidelity JSD: 0.402438 [0.402438, 0.402438]
- pairwise EDI: 0.000000
- deletion / insertion AUC: 0.259343 / 0.254806

## sanity / clean / local (n=1)
- artifact fidelity JSD: 0.542274 [0.542274, 0.542274]
- pairwise EDI: 0.627412
- deletion / insertion AUC: 0.256978 / 0.257015

## sanity / clean / rwssa (n=1)
- artifact fidelity JSD: 0.531334 [0.531334, 0.531334]
- pairwise EDI: 0.358356
- deletion / insertion AUC: 0.256707 / 0.257425

## sanity / clean / rwssa_binary (n=1)
- artifact fidelity JSD: 0.529273 [0.529273, 0.529273]
- pairwise EDI: 0.398271
- deletion / insertion AUC: 0.256696 / 0.257516

## sanity / clean / rwssa_no_calibration (n=1)
- artifact fidelity JSD: 0.531334 [0.531334, 0.531334]
- pairwise EDI: 0.358356
- deletion / insertion AUC: 0.256707 / 0.257425

## sanity / clean / rwssa_sparse (n=1)
- artifact fidelity JSD: 0.531211 [0.531211, 0.531211]
- pairwise EDI: 0.341986
- deletion / insertion AUC: 0.256425 / 0.257584

## sanity / clean / xfedalign_median (n=1)
- artifact fidelity JSD: 0.532837 [0.532837, 0.532837]
- pairwise EDI: 0.463374
- deletion / insertion AUC: 0.257001 / 0.257007
