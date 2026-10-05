# NC-SACPA: Full Reproducible Federated Explanation Research Pipeline

**Current status:** NC-SACPA passed a fresh five-seed Phase-VIII smoke-test gate; the full MNIST/CIFAR-10 confirmatory study has **not** been run. Fresh full-study seeds **9101–9105** are intentionally untouched.

This package is the next-stage validation of **Neighborhood-Corroborated Scale-Adaptive Peer Alignment (NC-SACPA)**, developed after two predecessor failures were preserved rather than hidden:

- UCPA v1 was killed by full validation because its fixed JSD bandwidth collapsed peer sharing at realistic sparse-artifact scales.
- SACPA v1 fixed scale collapse but failed a preregistered strong grouped-patch case because global support counts allowed a distant group to authorize foreign features.

NC-SACPA uses a self-tuned explanation-space kernel and requires local-neighborhood corroboration before importing a feature missing from the target client's sparse artifact.

## Smoke-test evidence that authorized this package

On untouched Phase-VIII seeds 801–805, NC-SACPA passed all 7 frozen smoke gates. Equal-family heterogeneous oracle JSD was:

- **NC-SACPA:** 0.07603
- oracle-tuned GlobalMedian: 0.09040
- oracle-tuned Cluster: 0.09387
- Local: 0.13231

NC-SACPA beat Local, GlobalMedian, and Cluster on 5/5 seeds; it beat GlobalMedian in every tested family and patch strength. This is **smoke-test approval only**, not publication evidence.

The complete smoke preregistration, raw confirmation JSONL, evaluator, and gate report are under `research_history/phase8_ncsacpa_smoke_approved/`.

## Frozen method

For each client/class:

1. compute pairwise JSD between sparse normalized explanation summaries;
2. set each client's local distance scale to its median peer distance;
3. use a symmetric self-tuned kernel `exp(-(d/sqrt(sigma_i sigma_k))^2)`;
4. form a peer-only prior;
5. for coordinates missing from the target, import only when at least **2 of the target's 3 nearest explanation peers** report the feature;
6. output `normalize(0.2 * local + 0.8 * peer_prior)`.

See `research/METHOD.md` for the exact formulation.

## Why the full study is stricter

- MNIST **and CIFAR-10**.
- Four disjoint per-client partitions: task training / surrogate fitting / artifact estimation / final explanation evaluation.
- Fresh confirmatory seeds **9101–9105**.
- Same FedAvg model and local explanation samples for every coordination method.
- Local-XAI, FedAttr mean, xFedAlign-style global median, clustering, killed UCPA, rejected SACPA predecessor, and NC-SACPA in the same pipeline.
- Direct task-model Integrated Gradients as the disjoint fidelity oracle.
- Primary local fidelity, explanation drift, deletion/insertion AUC, top-k overlap, communication, task validity, two poisoning attacks, and empirical privacy audit.
- Raw JSONL logs, environment/config manifests, SHA-256 per-run integrity files, duplicate-run provenance, paired seed analysis, and frozen decision gates.
- Exploratory sensitivity is separated from confirmatory evidence.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
python scripts/verify_package.py
```

## Run the study

```bash
# Core confirmation
python scripts/run_suite.py --tier A
python aggregate_results.py --outputs outputs --out aggregate_tierA

# Stress + robustness required for the final gate
python scripts/run_suite.py --tier B
python aggregate_results.py --outputs outputs --out aggregate_all
python evaluate_ncsacpa_gates.py --runs aggregate_all/runs.csv --out aggregate_all/ncsacpa_gate_report.json

# Exploratory sensitivity/communication
python scripts/run_suite.py --tier C
python aggregate_results.py --outputs outputs --out aggregate_all

# Verify raw run integrity
python scripts/verify_runs.py --outputs outputs
```

The authoritative experiment list is `configs/suite_manifest.yaml` (suite 2.1; 28 configurations).

## Full-study decision

The executable frozen gate is `evaluate_ncsacpa_gates.py`:

- G1 local fidelity vs xFedAlign-style global median;
- G2 coordination benefit vs Local-XAI;
- G3 functional non-inferiority;
- G4 neighborhood-corroboration revision test vs SACPA predecessor;
- G5 CIFAR task validity;
- G6 communication non-inferiority;
- G7 poisoning robustness.

Only a complete pass returns `CONTINUE`. Otherwise the outcome is `REVISE_OR_KILL`.

## Baseline honesty

`xfedalign_median` is a self-contained reproduction of the published xFedAlign coordination mechanism, **not the authors' exact code**. Official paper/code:

- https://proceedings.mlr.press/v306/wasif26a.html
- https://github.com/dawoodwasif/xFedAlign

Use `scripts/fetch_official_xfedalign.sh` for a separate official-code cross-check when internet access is available.

## Important files

- `ucpa_fl/` — implementation and logging pipeline.
- `configs/` — frozen suite.
- `research/PREREGISTRATION.md` — full-study hypotheses and gates.
- `research/METHOD.md` — exact NC-SACPA formulation.
- `research/NOVELTY_POSITIONING.md` — current prior-art audit.
- `research/BASELINE_PROVENANCE.md` — official vs reproduced baselines.
- `research/KNOWN_LIMITATIONS.md` — explicit boundaries.
- `research_history/` — negative/revised earlier phases and raw smoke confirmation.
- `verification/` — engineering-only sanity evidence.

Do not interpret the smoke result as evidence that NC-SACPA is conference-ready. The purpose of this package is to try to falsify it at realistic scale before theory/paper development.
