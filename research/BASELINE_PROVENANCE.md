# Baseline provenance

## Official xFedAlign reference

- Paper: Wasif et al., *Explainable Federated Learning via Global–Local Attribution Alignment*, ICML 2026.
- PMLR: https://proceedings.mlr.press/v306/wasif26a.html
- Official code: https://github.com/dawoodwasif/xFedAlign
- Official repository is MIT licensed at the time of the 2026-10-02 audit.

The paper/repository describe 8 clients, 15 communication rounds, one local epoch, batch 64, SGD momentum .9, learning rate .01, Dirichlet alpha .1 for non-IID, top-k sparse attribution artifacts, clipping, 8-bit quantization, light Gaussian noise, a coordinatewise-median global prior, and alignment beta .2 after warm-up. It reports Local-XAI, FedAttr-Agg and Fed-XAI comparators and explanation drift/deletion/insertion metrics.

## What this package implements

`xfedalign_median` is a **self-contained xFedAlign-style mechanism reproduction**, not a byte-for-byte execution of the authors' code. It shares the same trained task model with all explanation methods, fits a local sparse linear surrogate, transmits sparse clipped/quantized/noised class summaries, forms a coordinatewise-median global prior, and mixes local explanation with that prior at beta=.2.

This design intentionally isolates the explanation-coordination mechanism. It should be described as a reproduction in any paper until results have also been checked against the official code.

`fedattr_mean` is a server-side global mean attribution-template comparator implemented inside this package. It should likewise be called a reproduction/operational baseline, not an official third-party implementation.

`cluster` is an intentionally strong generic comparator: k-means on full client explanation vectors followed by cluster-wise explanation pooling. It is not claimed to be a named published method.

## Optional official-code check

Run:

```bash
bash scripts/fetch_official_xfedalign.sh
```

when internet access is available. Record the commit in `external/xfedalign_commit.txt`, then execute the authors' MNIST/CIFAR scripts separately. Do not merge their result files with this package's output unless the experimental settings are matched and provenance is retained.
