# QF-SCAD v8 expanded real-data study — SOURCE + verified engineering smoke

**Scientific status: experimental protocol and software verified, real-MNIST 5-round/5-local-epoch smoke complete; the full 138-run study is NOT executed. CUDA implementation is present but not physically tested here. Official xFedAlign is NOT integrated, so a publication-level SOTA comparison is still incomplete.**

## Design

- **Datasets**: genuine MNIST (60,000 original train/10,000 test) and genuine CIFAR-10 (50,000 train/10,000 test), supplied separately. No synthetic dataset and no automatic downloads. Original MNIST IDX SHA-256 is pinned.
- **Primary matrix**: 2 datasets × 7 IID/non-IID/shift/participation conditions × **5 new independent seeds** = **70 runs**; 10 clients, 50 rounds, 5 local epochs. 5-class, 10-class, and 20-client behavior is not inferred from this core experiment.
- **Sensitivity matrix**: change exactly one main training dimension at a time, holding others fixed and pairing **the three seed values** used in the primary IID condition, plus matched FedProx comparisons. 3, 5, 10, 20 clients; 5/10/25/50 rounds; 1/2/5/10 local epochs; split controls IID, Dirichlet alpha 1.0, 0.35, 0.05, rotation, patch and partial participation. Total **138 planned run assignments**, including 2 separately tagged engineering smoke configs.
- **Optimizer**: SGD momentum 0.9, FedAvg; additional separate task-model training optimizer **FedProx (mu=0.01)** on paired seeds. FedProx results are not a substitute for an FL-XAI explanation baseline.
- **Frozen CNN**: every explanation method consumes the same CNN checkpoint for a seed; 95% complete-original-MNIST test accuracy floor and 55% complete-original-CIFAR-10 floor. Undertrained seeds fail closed, never tuned against test set.
- **Evaluation**: 48 private genuine IG teachers/client, 64 disjoint genuine held-out records/client, spatial IG with eight midpoint steps for full study, top-k JSD/top-k overlap, class-summary disagreement, and 20-step deletion/insertion; exact explanation payload bytes and partition indices logged. Client-level pixels are not transmitted.
- **Published explanation controls**: gradient saliency (Simonyan et al.) and Grad-CAM (Selvaraju et al.), independently implemented against the exact frozen CNN. IG (Sundararajan et al.) is the reference attribution.
- **Strong matched controls**: same-architecture private convolution with 48 teachers, private convolution with shared prior, fixed and global gradient fields, class-template **PROXY (NOT official FedAttr-Agg)**, APGF v6, QF-SCAD v7 ablations.
- **Outstanding baseline**: official xFedAlign (ICML 2026) needs a paired-source port. See `docs/PUBLISHED_BASELINE_AUDIT.md`. Until then **no SOTA / publication readiness claim**.

## Prepare original datasets

To convert the user-provided CSV MNIST ZIP without downloading:

```bash
python scripts/import_mnist_csv.py --source-zip /path/to/mnist.zip --dataset-root data
```

CIFAR-10 requires official local `data/cifar-10-batches-py/{data_batch_1,..,data_batch_5,test_batch,batches.meta}`. No CIFAR images are bundled.

## Install and verify

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests/
python scripts/preflight_expanded.py --dataset mnist --data-root data --out mnist_preflight.json
python scripts/preflight_expanded.py --dataset cifar10 --data-root data --out cifar_preflight.json
python scripts/launch_expanded.py --dataset both --stage primary --dry-run
```

## Re-execute engineering smoke — **5 rounds × 5 local epochs** on genuine MNIST

```bash
python scripts/smoke_5r5e_real_mnist.py --data-root data --out smoke_5r5e --device auto --train-per-client 500 --test-gate 0.70
```

This uses an explicitly **reduced two-convolution CNN** and genuine disjoint subsets of original MNIST, and evaluates on ALL 10,000 test images. The low 70% **smoke-only** gate must never replace the 95% scientific full-run gate. The checkpoint is freshly trained in the smoke, not loaded from archival data. This smoke completed with 93.27% test accuracy in this CPU environment; see `evidence/real_mnist_5round_5epoch_smoke.json`.

## Full study (GPU, parallel / stages)

```bash
# Main confirmatory runs; 70 seeds, one process per GPU by default
python scripts/launch_expanded.py --dataset both --stage primary --data-root data --output-root expanded_results --device auto --workers 1 --threads 4

# On a two-GPU host: distinct CUDA subprocesses, no memory oversubscription
python scripts/launch_expanded.py --dataset both --stage clients --data-root data --output-root expanded_results --device cuda --workers 2 --threads 3

# On a single 16-GB GPU: optional explicit same-device parallelism, monitor VRAM
python scripts/launch_expanded.py --dataset mnist --stage epochs --data-root data --output-root expanded_results --device cuda --workers 2 --allow-gpu-sharing --threads 2

# Run each remaining stage; never overwrite an incomplete run
python scripts/launch_expanded.py --dataset both --stage rounds --data-root data --output-root expanded_results --device auto
python scripts/launch_expanded.py --dataset both --stage epochs --data-root data --output-root expanded_results --device auto
python scripts/launch_expanded.py --dataset both --stage optimizer --data-root data --output-root expanded_results --device auto

# All run assignment completeness and per-example arithmetic
python scripts/aggregate_expanded.py --root expanded_results --out expanded_analysis
```

Every launcher job is a separate subprocess, CPU threading is limited per process, outputs are isolated, and completed outputs can be resumed after verifying their frozen config SHA. The `--workers` switch controls independent **run-level parallelism**, not client training concurrently inside a single run. CUDA process support is source-verified but untested on actual NVIDIA hardware in this environment (`torch 2.10.0+cpu`). For single-GPU hosts, default one job per GPU is safest; `--allow-gpu-sharing` must be used explicitly and risks OOM.

## Verification and research limitations

- Real MNIST source validates against pinned original image/label checksums. All **69 MNIST prospective+smoke seed/split feasibility checks** passed on real images. CIFAR-10 data are absent here: no CIFAR smoke/preflight pass is claimed.
- **18/18 code tests** passed (plus fresh extraction test after packaging). Five real-MNIST rounds × five local epochs ran to completion with 93.27% full-test accuracy using the smaller smoke CNN.
- The smoke results (48 held-out records) are not independently seeded, not trained on the full data, and cannot justify publication-level claims.
- The main study has a 95% MNIST / 55% CIFAR accuracy gate and never silently bypasses either. No full study runs were executed here.
- The **official xFedAlign implementation is not integrated**. A source-verified, equal-budget, paired official xFedAlign comparison is mandatory before manuscript claims about popular FL-XAI SOTA. The internal class-template method must NEVER be renamed or reported as actual FedAttr-Agg.
- Task optimizer comparisons (FedAvg versus FedProx) use separately trained CNN checkpoints; explanation comparisons within a training run reuse exactly one frozen CNN. Interpret as two distinct experimental questions.

The run manager freezes `configs/expanded/*.yaml` via `configs/EXPANDED_CONFIG_HASHES.json`; the experiments and their precommitted seed assignments are defined in `configs/EXPANDED_STUDY_PLAN.json`. All analysis recomputes raw mean JSD, top-k and deletion/insertion AUC from saved per-example measurements and refuses a publication SOTA promotion while official xFedAlign is absent.
