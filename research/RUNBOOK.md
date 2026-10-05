# Full NC-SACPA experiment runbook

## 1. Environment

Recommended: Python 3.10–3.12 and a recent CUDA-enabled PyTorch environment.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
```

If PyTorch/CUDA requires a platform-specific wheel, install PyTorch first using its official installer and then run `pip install -e '.[dev]'`.

## 2. Verify the frozen package

```bash
python scripts/verify_package.py
python run_experiment.py --config configs/sanity.yaml
python scripts/verify_runs.py --outputs verification/ncsacpa_sanity/output
```

The package already contains an engineering sanity run under `verification/`; rerunning `configs/sanity.yaml` is optional but recommended after installing on a new system.

## 3. Run the study without retuning

### Tier A — core confirmatory evidence

```bash
python scripts/run_suite.py --tier A
python aggregate_results.py --outputs outputs --out aggregate_tierA
```

Do **not** use Tier-A results to change NC-SACPA's frozen `p=2, beta=.8, q=3, min_support=2`.

### Tier B — stress and robustness

```bash
python scripts/run_suite.py --tier B
python aggregate_results.py --outputs outputs --out aggregate_AB
```

After A+B, the seven-gate decision can be evaluated:

```bash
python aggregate_results.py --outputs outputs --out aggregate_all
python evaluate_ncsacpa_gates.py --runs aggregate_all/runs.csv --out aggregate_all/ncsacpa_gate_report.json
```

### Tier C — exploratory sensitivity and communication

```bash
python scripts/run_suite.py --tier C
python aggregate_results.py --outputs outputs --out aggregate_all
```

Tier C cannot retroactively tune or redefine the confirmatory result.

## 4. Logging and integrity

Each seed creates an immutable run directory with:

- complete `manifest.json`, including config hash, environment, package versions, seed, and lifecycle status;
- `events.jsonl` and `metrics.jsonl` append-only logs;
- `client_partition_summary.json`;
- saved clean artifacts when enabled;
- empirical artifact-membership audit;
- `final_results.json`;
- `integrity.sha256` for finalized run files.

`aggregate_results.py` never silently double-counts reruns: it selects the latest completed duplicate and writes `duplicate_runs.json`.

Verify all completed runs with:

```bash
python scripts/verify_runs.py --outputs outputs
```

## 5. Official xFedAlign cross-check

When internet access is available:

```bash
bash scripts/fetch_official_xfedalign.sh
```

Run the official repository in a separate environment and retain its commit hash and outputs separately. Do not label the built-in `xfedalign_median` results as official-code results.

## 6. What to return for final analysis

Return the project with:

- `outputs/`
- `aggregate_all/`
- `aggregate_all/ncsacpa_gate_report.json`

The final analysis must begin from the frozen gates and paired seed results, including failures, rather than selecting favorable Tier-C settings after the fact.
