# RWSSA Federated Explainability Research Pipeline — v1.0

**What is included:** a runnable, source-available research pipeline for the smoke-tested RWSSA hypothesis, with a **real-data qualification pilot**, **10-seed main confirmation**, robustness tests, exploratory experiments, reproducible raw logs, checksums, and strict evidence gates. The CNN/FedAvg task model is trained once for each configuration/seed and all explanation methods share the same task model, client partitions and held-out data.

**Status:** engineering sanity verified on synthetic data. The real-data pilot/full GPU results are **not run** and **not approved**. Smoke evidence for RWSSA was generated on controlled synthetic data, not this implementation's MNIST/CIFAR pipeline.

## Quick start

From the extracted folder:

```bash
python -m pip install -e '.[dev]'
python scripts/verify_package.py
# Optional offline end-to-end synthetic sanity:
python scripts/run_suite.py --tier S
python scripts/verify_runs.py --outputs verification/sanity_outputs
```

The real datasets download on first use from torchvision (Internet needed once); you can prepopulate `./data`. For a single GPU run tiers **sequentially** to avoid interference and mixed runtime estimates. Only one instance of a given config/seed should run at a time.

### 1. Pilot FIRST — early kill gate

```bash
python scripts/run_suite.py --tier P
python aggregate_results.py --outputs outputs --out aggregate_pilot
python evaluate_rwssa_gates.py --tier P --runs aggregate_pilot/runs.csv --out aggregate_pilot/pilot_gate.json
python scripts/verify_runs.py --outputs outputs
```

Review `aggregate_pilot/pilot_gate.json`. Only if its decision is `AUTHORIZE_FULL_STUDY` consider the expensive study. The gate may say `STOP_AND_REVIEW` or `PENDING`; do not ignore that to preserve a desired hypothesis. **This is not an acceptance probability.**

### 2. Main confirmation — 10 independent seeds per setting

```bash
python scripts/run_suite.py --tier A
python aggregate_results.py --outputs outputs --out aggregate_main
python evaluate_rwssa_gates.py --tier A --runs aggregate_main/runs.csv --out aggregate_main/confirmation_gate.json
```

### 3. Stress + exploratory tiers

```bash
python scripts/run_suite.py --tier B
python scripts/run_suite.py --tier C
python aggregate_results.py --outputs outputs --out aggregate_all
python scripts/verify_runs.py --outputs outputs
```

To upload results for independent analysis, include `outputs/`, `aggregate_pilot/`, `aggregate_main/`, and `aggregate_all/`. **Keep original checkpoints** (they can be large); if compressing to upload, keep at least all manifests, `final_results.json`, `integrity.sha256`, CSVs, and policy/safety logs. Omitting checkpoints prevents model re-inference verification.

Each suite config runs individual seeds with resume support. Restarting after a failure re-runs only missing completed seeds. The suite ledger is append-only and raw run directories are immutable.

## Experiment matrix

| Tier | Configurations | Seeds per config | Purpose |
|---|---:|---:|---|
| P | 3 | 3 | MNIST/Fashion-MNIST/CIFAR-10 early qualification |
| A | 8 | 10 | rotation, sparse patch, color, no-shift controls |
| B | 7 | 5 | participation, data volume, drift, direct IG, artifact attacks, four-risk calibration |
| C | 4 | 5 | feature-channel/DP/noise/safety sensitivity; **exploratory only** |

All hyperparameters, dataset partitions, seeds, training rounds, learning rates, local batch size, local epochs and metrics are in `configs/*.yaml` and logged inside each run's `manifest.json`.

## Methods evaluated on the same held-out records

- `local`: unaligned client explainer.
- `fedattr_mean`: mean summary aggregation.
- `xfedalign_median`: xFedAlign-**style** coordinate-median prior and a common beta=0.2 (local reproduction; **NOT official xFedAlign code**).
- `xfedalign_beta_0p4`: stronger common-beta baseline.
- `iflash_proxy`: **iFLASH-inspired** fidelity-weighted client explanation summaries; uses direct-IG accuracy/JSD instead of author SHAP faithfulness, **NOT official iFLASH**.
- `rwssa`: **smoke-rule exact** shared positive feature-gain map and residual floor rho=0.30, local scale chosen by two-metric JSD/top-k CRC-style calibration on independent samples.
- `rwssa_sparse`: **new exploratory sparse/quantized safety channel**, using the existing artifact top-k indices; **not smoke-approved**.
- `rwssa_binary`: binary safety-mask ablation.
- `rwssa_no_safety`: uniform coordinate weights ablation.
- `rwssa_no_calibration`: frozen lambda=1 ablation.
- `rwssa_class_scalar`: classwise scalar rather than coordinate-specific gain.

## Scientific guardrails and known limitations

1. **Five independent client partitions:** task training (50%), local surrogate (15%), artifact estimation (10%), policy selection + calibration (15%, split internally without overlap), and held-out evaluation (10%). Client data never goes to a real remote server; this is an FL simulation.
2. The main RWSSA policy preserves the **two-metric** (JSD/top-k) calibration rule tested in the cheap smoke. The selector does **not** certify deletion/insertion outcomes or offer a simultaneous high-probability guarantee. A separate Tier B config tests a stronger four-component loss.
3. Using direct task-model IG as a private *teacher* is permitted for policy selection but not for reporting held-out results. The final evaluation is disjoint and uses independent records. Results on `task_ig` as the local explainer are separately labeled.
4. **Privacy caution:** raw per-feature gain vectors reveal aspects of client explainer-vs-task-model discrepancies. Main RWSSA logs dense float32 gain transmission (`C × D × 4` bytes per client), and a novel sparse/8-bit candidate is separately measured. There is **no differential-privacy guarantee** for this signal. Artifact Gaussian noise is a robustness mechanism, not a proven DP protocol. The direct private gain vector makes this a **research-only simulator**, not deployable privacy-preserving FL.
5. The iFLASH proxy operates on IG/surrogate artifacts, not full per-example SHAP. Any performance claim against *official* iFLASH must wait for an appropriate author-code reproduction. Likewise the xFedAlign proxy is not official author code. Sources and official code links are in `docs/RELATED_WORK.md`.
6. The server can infer more from gain vectors than from xFedAlign artifacts; byte overhead and threat model are explicit. **An apparent accuracy gain cannot by itself justify this cost.**
7. The training seeds are fixed but bitwise equality on every GPU/version is not guaranteed (CUDA nondeterministic operations can remain). The pipeline logs CUDA/PyTorch version and outputs/hashes.
8. The preregistered gate compares against stronger alternative baselines and refuses to pass incomplete, non-finite or invalid-task data. Exploratory Tier C cannot change the frozen A gate.
9. This is not a guarantee of a top-tier publication. Novelty remains medium-risk and the real-data results may fail. Reproducing the **published official xFedAlign** and substantive iFLASH-equivalent comparator is still necessary for a strong paper.

## Source provenance

Prior synthetic 5-seed smoke report: `reference_smoke/SMOKE_STUDY_REPORT.md`. Exact input code comes from the verified smoke archive, with the paper-oriented integration in `ucpa_fl/rwssa.py`. The original CFBA pipeline supplies the FedAvg + surrogate + artifact framework. No historical results are reused as confirmatory evidence.
