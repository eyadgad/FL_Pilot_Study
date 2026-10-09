# APGF v6: Selective Federated Attribution Calibration — REAL MNIST ONLY

**Status:** EXPLORATORY / **DO NOT ADVANCE** to publication-scale claims.

This is a standalone **implementation-only** supplement to FAGC v5. It includes no raw MNIST data, model checkpoints or simulated digit samples. It includes *metrics calculated from genuine MNIST* in `outputs/`.

## Why this iteration exists

The independent full-study audit of FAGC found a large margin versus xFedAlign-style but very small federated-only benefit versus a strong private field. The full class-aware mechanism often lost to class-free FAGC, and extreme non-IID (Dirichlet alpha=0.05) had consistent negative transfer. APGF removes the weak class exponent and lets each client choose whether to borrow quantized peer sufficient statistics based on *separate real IG teacher validation samples*. This is not a formal statistical certificate, privacy protection, nor original paper-level novelty by itself.

## What was actually tested

Four **already-trained** and independently seeded MNIST CNN checkpoints from the previous v5 research package: rotation 6101/6103 and patch 6102/6104. All task accuracies were independently checked on the complete 10,000-image test split: 97.35%, 97.42%, 95.16%, 95.80%, respectively. No new CNN training occurred during the APGF smoke. Their training used genuine CSV-derived MNIST, 12k training-image development subset, 3 clients, 8 rounds, 2 local SGD epochs per round. We reuse the existing disjoint client index partitions.

For each client we calculate 48 direct-task-CNN Integrated Gradients (IG) teacher records, of which 36 fit a quantized gain field and 12 are withheld to select the peer weight; a disjoint 48 genuine-image final evaluation set is never used to choose the peer weight. The stronger no-gate control uses **all 48** teacher records for private calibration. Five client-relative candidate weights: 0, 0.1, 0.2, 0.4, 0.8; accept a peer weight only if private-validation mean JSD gain >=0.001 and >=60% of validation records show improvement. Validation selection is heuristic, not a guaranteed upper bound. The wire transport consists of two 790-byte quantized statistic uplinks and five 786-byte candidate gain downlinks per client (5,510 bytes per simulated exchange). All methods use the same checkpoint within a seed.

**A crucial negative result:** Compared with the *properly strengthened* 48-teacher fixed-sharing baseline, APGF is on average worse by 0.000310 JSD. APGF beats the 48-teacher private field by just 0.000652 mean JSD (3/4 seeds), with a seed-level 95% interval that includes zero. It fails the prior >=0.005 absolute improvement threshold. No research-readiness score >=75 is awarded.

See `docs/APGF_V6_INDEPENDENT_SMOKE_REPORT.md` for the full numeric and failure analysis. Final-evaluation class-conditioned map disagreement is a *diagnostic* and must not be represented as the prior artifact-based EDI definition. We do not report deletion/insertion scores in this targeted smoke.

## Setup: provide **real MNIST** separately

```bash
python -m pip install -e '.[dev]'
python scripts/import_mnist_csv.py --source-zip /path/to/project.zip --dataset-root data
python scripts/verify_real_mnist.py --dataset-root data
python -m pytest -q tests/
```

The MNIST source ZIP must include 6 training CSV shards and 1 test CSV in the original `data/mnist/` layout. Do not substitute synthetic data, TorchVision downloads, or a different dataset. The CSV importer verifies 60,000/10,000 and records cryptographic provenance. The source package does **not** contain the original pre-trained v5 checkpoints. To reproduce the four development smokes, keep the complete original v5 archive extracted and use:

```bash
python scripts/smoke_real_mnist.py --v5-root /path/to/verified_v5_package_root \
  --kind rotation --seed 6101 --out outputs/recomputed
```

Repeat for `patch 6102`, `rotation 6103`, `patch 6104`. The `--v5-root` must contain the original `data`, `fagc_independent_checkpoints`, `configs`, and `ucpa_fl` subfolders. Saved metrics are in `outputs/` and require **no** experimental data in this package.

## Prospective full pipeline — not executed or validated as scientific evidence

Six future scenarios × five **new** seeds = 30 prospective runs, with full original training image split, five clients, 25 rounds, two local epochs, and 10,000-image model gates. Frozen, SHA-256-pinned YAMLs are in `configs/apgf_full_*.yaml`. Important alpha=0.05 negative transfer checks are mandatory. Run one representative training seed first:

```bash
python scripts/run_apgf_prospective.py --config configs/apgf_full_noniid_005.yaml \
  --seed 8601 --device cuda --threads 4 --output-root apgf_prospective_outputs
```

`--device cpu` is supported if CUDA is unavailable. The new full runner trains from scratch and saves checkpoint and exact split indices; it fails closed if real-MNIST integrity or task accuracy is insufficient, and does not silently re-seed. Before CUDA runs set `CUBLAS_WORKSPACE_CONFIG=:4096:8` (also set automatically before torch import in the driver); verify deterministic behavior on your hardware. Do not rewrite or retune frozen YAMLs based on observed test output. After all 30 seeds:

```bash
python scripts/aggregate_apgf_prospective.py --root apgf_prospective_outputs \
  --out apgf_prospective_aggregate.json
```

Numeric promotion gates include positive gains versus full-48 private and fixed-sharing controls, a predeclared ≥0.005 absolute improvement versus private, no alpha=.05 negative-transfer seeds, and nonregression on top-k. Passing numeric gates would **not** independently establish novelty, privacy, communication efficiency, or superiority over *official* xFedAlign code. The current package is an **engineering/prospective protocol**, not a prospective experimental result.

## Important limitations

- Literature overlap with existing personalized federated learning, adaptive fusion, and global-local attribution alignment must be checked before proposing novelty.
- The true private-only method needs no peer communication: its smaller wire footprint is a separate efficiency advantage. The matched quantized gain codecs do **not** erase this cost.
- The exploratory smoke has only four previously used checkpoint seeds, not independent future confirmation, and excludes extreme non-IID direct IG measurements.
- Prior `Results.zip` contained CIFAR outcomes but not the matching CIFAR implementation; this v6 code does not claim CIFAR support.
- Only validation teaches the selector; the full 48-teacher private baseline is a stricter control than the 36-teacher private calibration used to build the selector.
