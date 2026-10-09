# APGF v6: independent exploratory MNIST smoke report
**Date:** 2026-10-08 | **Decision:** NO ADVANCE / insufficient evidence for 75/100 paper-readiness

## 1. Prior full-study finding motivating this iteration

The supplied `Results.zip` contains five independently seeded tasks per condition (30 MNIST / 30 CIFAR-10), but neither the full FAGC class factor nor raw global sharing establishes a material advantage over strong private gradient-field calibration. On MNIST, mean FAGC oracle-IG JSD was 0.081709 versus 0.082476 private gradient field and 0.081626 class-free FAGC. On CIFAR the values were 0.091011, 0.091220 and 0.090662, respectively. The extreme MNIST Dirichlet alpha 0.05 condition lost to private gradient calibration on all five seeds, mean 0.001049 JSD worse. Saved metrics and run records are internally consistent, but the executed CIFAR source/config snapshot and checkpoints were not included. Reference: `FAGC_Independent_Study_Audit_2026-10-08/INDEPENDENT_FULL_STUDY_ASSESSMENT.md` delivered with the audit.

## 2. Method: APGF, client-validated selective peer transfer

Let client i transmit two quantized sufficient statistics from 36 genuine-MNIST IG teacher records:

`A_i = sum_r normalised_IG(x_r)` and `B_i = sum_r x_r^1.2`.

For a peer weight w, the server computes a candidate 784-pixel field from **decoded wire packets**:

`g_i(w) = smooth((A_i + w * sum_(k!=i) A_k)/(B_i + w * sum_(k!=i) B_k + 0.01*(n_i + w * sum_(k!=i)n_k)))`.

Quantize the field as a 786-byte gain packet before client evaluation. The client privately evaluates weights `[0, 0.1, 0.2, 0.4, 0.8]` on **12 separate calibration IG records**; a peer weight is accepted only if its estimated paired JSD gain over private-only is >=0.001 and its per-record win fraction >=0.6. Otherwise the client abstains and uses its original quantized private field. This gate uses no final evaluation labels/oracles. It is a **heuristic**, not a formal confidence bound or privacy guarantee. The original private-only control uses no peer information and logically needs no communication.

The corrected prospective wire ledger charges each client **two 790-byte quantized statistic uplinks and five 786-byte quantized candidate-field downlinks**, totalling **5,510 bytes/client** (1,580 upstream, 3,930 downstream), before any transport/security framing. All five candidates must reach the client for *private* validation-based selection. The first four real-MNIST smoke metrics were computed by a single-process simulator that logged only one 786-byte downlink; see `outputs/LEDGER_ERRATA.json` for the exact unmodified source hashes and corrected accounting. Therefore the 5,510-byte estimate has **not been verified on a network**, and APGF is much more expensive than the zero-additional-communication private model. No privacy guarantee is claimed.

## 3. Real-MNIST experiment and provenance

The source is the originally supplied genuine handwritten MNIST CSV shards, already losslessly converted and SHA-256 checked in the prior v5 project: 60,000 training images and 10,000 test images. **No synthetic dataset, synthetic smoke, or network download was used.** Four already-trained, independent-task-seeded FedAvg CNNs were loaded from prior v5 (each originally trained on 12,000 genuine MNIST training examples, three clients, eight rounds, two local epochs). We independently rechecked every CNN on all 10,000 test images and refused to calculate IG unless its accuracy reached 95%. Client splits match the hashed original partitions. Teachers, the reserved calibration holdout, and final eval each originate from strictly disjoint index lists or disjoint slices of the calibration partition.

Per seed: 3 clients × (36 fitting + 12 weight-selection + 48 final held-out) examples; 144 final evaluation explanations per seed, 576 in total. All direct task-model IG uses four midpoint steps and a zero baseline. All compared maps use the identical frozen CNN for that seed. The four seeded runs were previously known and are **not** an independent post-development preregistered confirmation set.

| Real-MNIST task condition | Seed | Full 10k test CNN accuracy | Private48 JSD | FixedPeer48 JSD | APGF JSD | APGF selection weights (clients 0,1,2) |
|---|---:|---:|---:|---:|---:|---|
| Rotation | 6101 | 97.35% | 0.082900 | **0.082209** | 0.083105 | 0.8, 0.4, 0.8 |
| Rotation | 6103 | 97.42% | 0.080202 | **0.079023** | 0.079622 | 0.0, 0.8, 0.0 |
| Patch | 6102 | 95.16% | 0.077154 | 0.075856 | **0.075196** | 0.8, 0.0, 0.8 |
| Patch | 6104 | 95.80% | 0.073330 | **0.072649** | 0.073055 | 0.8, 0.0, 0.8 |

Metrics above rounded only for display; `outputs/*_records.csv`, `outputs/*_summary.json` and `outputs/APGF_V6B_SMOKE_SUMMARY.json` retain machine-readable precision.

### Mean results (four checkpoint seeds, each with 144 held-out eval records)

| Explanation method | Oracle IG JSD ↓ | Top-48 overlap ↑ | Held-out predicted-class summary disagreement ↓ |
|---|---:|---:|---:|
| Input-only pixel saliency | 0.083258 | 0.547924 | 0.149911 |
| Private gain 36 fit records | 0.079460 | 0.573278 | 0.146680 |
| **Private gain 48 fit records** | **0.078396** | (see summary JSON) | (see summary JSON) |
| Fixed peer weight 0.2, 36 fit | 0.077947 | 0.576570 | 0.144460 |
| **Fixed peer weight 0.2, 48 fit** | **0.077434** | (see summary JSON) | (see summary JSON) |
| **APGF, 36 fit + 12 gated validation** | **0.077744** | **0.578451** | **0.145123** |

The held-out class-summary metric here is **not equivalent to the artifact-derived EDI** used in the full study. We do not claim deletion/insertion AUC changes from this focused experiment. Teacher/validation records were not reused in final evaluation, but only the per-run metadata and hashed partitions are included in the implementation-only ZIP.

### Pairwise differences evaluated at seed level

- APGF minus *strong private48*, expressed as private48 JSD − APGF JSD: **+0.000652 mean**, 3 of 4 seeds positive; approximate t-based 95% interval **[-0.000826, +0.002130]**.
- APGF minus *strong fixed48*: **−0.000310 mean** (APGF **worse**), only 1 of 4 seeds positive; approximate t-based 95% interval **[-0.001388, +0.000768]**.
- APGF exceeds a weaker 36-teacher private control by 0.001716, but this is **not** the decision comparator after the strengthened audit.
- Current evidence is **below** the previously required **+0.005** absolute JSD advantage over an equally teacher-budgeted private control. The small and uncertain improvements do not justify a ≥75 readiness score.

These confidence intervals use four checkpoint seeds as units, not hundreds of image-level samples. This is severely underpowered, not a hypothesis-test certificate. The first development iteration `research_history/v6a_initial_unfair_comparison` is retained rather than silently discarded.

## 4. Failure analysis and implications

The adaptive selector's 12 validation images sometimes favor a larger peer weight (0.8). Yet fixed sharing trained on the full 48 teachers can outperform the selector trained on 36: this cost of withholding validation examples can exceed the benefit of adaptive selection. Where the selector abstains, it prevents reliance on peer statistics by construction, but does not guarantee a lower test JSD than the fixed-peer model. Under rotation seed 6103 the mean JSD of adaptive is worse than fixed36 and fixed48.

For publishable novelty, APGF as presently formulated is too close to common adaptive personalization, shrinkage, and local validation methods. Review and distinguish from published xFedAlign global-local attribution alignment (ICML 2026) and other attribution-fusion approaches. The empirical mechanism (positive transfer under client scarcity and rigorous rejection under true class-skew) must be demonstrated on genuinely new seeds with enough validation teacher data and bit-budget equality.

## 5. Forward plan and frozen numeric gates

The six prospective configs provide five *new* seeds each (30 total); full MNIST 60k/10k, 5 clients, 25 FedAvg rounds, 2 local epochs, source IDX provenance, accuracy gate 95%, and strong 48-teacher private/fixed controls. The α=.05 non-IID setting is compulsory. Each output must include exact split index arrays, model checkpoint, hashes, raw metrics, per-client gate decisions, and byte ledger.

Before promoting a subsequent method, require: positive lower paired seed CI for JSD gain over strong private48 **and** fixed48, mean absolute advantage of **at least 0.005 over strong private48**, no negative-transfer α=.05 seed relative to private48, top-k non-inferiority, no task-accuracy gate failures included, no metric cherry-picking, and balanced communication-rate comparisons. Then separately test official xFedAlign implementation, CIFAR-10, privacy/attacks, deletion/insertion and conceptual novelty. Failing any condition means **NO ADVANCE** regardless of software-test success.

## 6. Reproduction and software verification

- `python -m pytest -q tests/` checks codec integrity and abstention plus future-run completeness/frozen configurations (these are *unit tests*, not synthetic-data scientific smoke tests).
- `python scripts/smoke_real_mnist.py --v5-root /path/to/verified_v5_full_package --kind patch --seed 6102 --out outputs/repeat` reexecutes on genuine MNIST and verifies the original checkpoint and partition hashes.
- `python scripts/run_apgf_prospective.py --config configs/apgf_full_noniid_005.yaml --seed 8601 --device cpu` is for a *future* complete train-to-evaluate run. It has passed CLI preflight, not an end-to-end prospective smoke in this continuation.

**Final verdict:** The work produced a testable algorithm and falsified its claimed superiority against stronger controls. This is useful negative scientific evidence. It does not establish a new ≥75/100 publication-ready direction.
