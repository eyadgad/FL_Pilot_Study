# UCPA-FL: Full Reproducible Experiment Package

This package takes the smoke-tested **Uncertainty-Compatible Peer Alignment (UCPA)** idea into a preregistered, auditable full experiment designed to test whether the approach deserves continued development toward a top-tier ML submission.

It does **not** contain full-scale results yet. The full-study seeds are frozen and intentionally left for you to run. The included result is only an end-to-end synthetic sanity verification created during packaging.

## Research question

A global federated explanation prior can improve consistency, but it can also erase real client-specific feature semantics. Pure local explanations preserve those semantics but can be noisy. UCPA builds a **personalized explanation prior** by borrowing from peers only when:

1. their whole explanation is similar, and
2. the coordinate-level difference is small relative to the two clients' sampling uncertainty.

See `research/METHOD.md` for the equations and `research/PREREGISTRATION.md` for the frozen decision gates.

## Why this package is stricter than the smoke test

- MNIST **and CIFAR-10**.
- Four disjoint client subsets: task training / surrogate fitting / artifact estimation / final explanation evaluation.
- Fresh five-seed confirmatory block: **6101–6105**.
- Same FedAvg task model and same local explanation samples for every coordination method.
- Close baselines: Local-XAI, FedAttr-Agg reproduction, xFedAlign-style median global prior, clustering, and UCPA ablations.
- Direct task-model integrated gradients as an independent local-fidelity oracle.
- Local fidelity, pairwise drift, xFedAlign-style reference EDI, deletion/insertion AUC, top-k overlap, communication cost, task performance, poisoning robustness, and direct audit of privacy exposure from UCPA's variance channel.
- Raw append-only JSONL logs, complete environment/config manifests, SHA-256 integrity files, and deterministic seed handling.
- Pre-registered automatic gates in `evaluate_preregistered_gates.py`.

## Quick start

Recommended: Python 3.10–3.12 with a CUDA-enabled PyTorch environment.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e '.[dev]'

python scripts/verify_package.py
python run_experiment.py --config configs/sanity.yaml
python scripts/verify_runs.py --outputs outputs
```

### Run the full frozen study

```bash
# Core confirmatory evidence first
python scripts/run_suite.py --tier A
python aggregate_results.py --outputs outputs --out aggregate_tierA

# Stress + robustness
python scripts/run_suite.py --tier B

# Communication/privacy/parameter sensitivity
python scripts/run_suite.py --tier C

# Final aggregation and preregistered gate
python aggregate_results.py --outputs outputs --out aggregate_all
python evaluate_preregistered_gates.py --runs aggregate_all/runs.csv --out aggregate_all/preregistered_gate_report.json
python scripts/verify_runs.py --outputs outputs
```

Or run all three tiers with:

```bash
python scripts/run_suite.py --tier all
```

See `research/RUNBOOK.md` before running the full suite.

## Study tiers

`configs/suite_manifest.yaml` is the authoritative experiment list.

- **Tier A — required core evidence:** MNIST/CIFAR-10 label-skew controls plus rotation, sparse-patch, and photometric client heterogeneity.
- **Tier B — stress/robustness:** unequal client sizes, partial task participation, time-varying shift, direct-IG explanation source, and two attribution-poisoning attacks.
- **Tier C — sensitivity:** sparse communication, artifact noise, and UCPA parameter sensitivity.

The core UCPA values `z=2`, `h=0.048`, `beta=.2` are frozen from the previous smoke-test phase. Tier-C runs are exploratory and cannot be used to retroactively tune Tier A.

## Main files

- `ucpa_fl/` — experiment implementation.
- `configs/` — frozen YAML configs and suite manifest.
- `research/PREREGISTRATION.md` — hypotheses, outcomes and pass/revise/kill rules.
- `research/NOVELTY_POSITIONING.md` — current literature positioning as of 2026-10-02.
- `research/BASELINE_PROVENANCE.md` — what is official vs reproduced.
- `research/KNOWN_LIMITATIONS.md` — claims the package cannot support by itself.
- `research_history/` — selected earlier negative phases, smoke-test approval, and the ten user-provided seed papers.
- `outputs/` — immutable per-seed run logs (created by runs).
- `aggregate_results.py` — regenerates CSV/Markdown summaries.
- `evaluate_preregistered_gates.py` — applies the frozen research gate.

## Baseline honesty

The `xfedalign_median` implementation is a **self-contained reproduction of the published mechanism**, not the authors' exact source code. The official ICML 2026 code is at:

https://github.com/dawoodwasif/xFedAlign

Use `bash scripts/fetch_official_xfedalign.sh` when internet access is available and keep official-code outputs separate. A publication should include an official-code cross-check before claiming superiority to xFedAlign.

## Result handling

Do not edit raw run folders. Every completed run ends with `integrity.sha256`. `aggregate_results.py` selects the latest completed duplicate if a seed/config was accidentally rerun and writes `duplicate_runs.json` so this is auditable rather than silent.

When the full run is finished, return the `outputs/` and `aggregate_all/` folders. The next analysis should start from the preregistered gates and paired UCPA-vs-xFedAlign results, not from post-hoc selection of favorable metrics.
