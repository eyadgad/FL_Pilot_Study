# Full experiment runbook

## 1. Environment

Recommended: Python 3.10–3.12, recent PyTorch with CUDA, one NVIDIA GPU for practical runtime.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
```

PyTorch/CUDA wheels vary by GPU driver; if needed install PyTorch first using the command from https://pytorch.org/get-started/locally/ and then `pip install -e '.[dev]'`.

## 2. Verify package before experiments

```bash
python scripts/verify_package.py
python run_experiment.py --config configs/sanity.yaml
python scripts/verify_runs.py --outputs outputs
```

The first real MNIST/CIFAR run will download torchvision data to `./data` unless already present.

## 3. Run the frozen study

Start with Tier A. Do not inspect Tier-A results and change UCPA parameters before finishing all Tier-A configs.

```bash
python scripts/run_suite.py --tier A
python aggregate_results.py --outputs outputs --out aggregate_tierA
```

Then run stress/robustness:

```bash
python scripts/run_suite.py --tier B
python aggregate_results.py --outputs outputs --out aggregate_AB
```

Then sensitivity/communication:

```bash
python scripts/run_suite.py --tier C
python aggregate_results.py --outputs outputs --out aggregate_all
```

Or run everything:

```bash
python scripts/run_suite.py --tier all
```

## 4. Logging

Every seed creates an immutable run directory containing:

- `manifest.json`: complete config, environment and pip freeze, config hash, status.
- `events.jsonl`: lifecycle/selected-client/attack events.
- `metrics.jsonl`: append-only scalar metrics.
- `client_partition_summary.json`: per-client sample counts and class histograms for all four splits.
- `artifacts/clean_artifacts.npz`: transmitted clean explanation artifacts when enabled.
- `artifact_membership_audit.json`: per-client privacy audit of mean-only vs mean+variance artifacts.
- `final_results.json`: task/explanation/attack results.
- `integrity.sha256`: hashes of all finalized run files.

Raw JSONL is the source of truth. Aggregated CSVs are regenerable.

## 5. After runs

```bash
python scripts/verify_runs.py --outputs outputs
python aggregate_results.py --outputs outputs --out aggregate_all
```

Then send back the entire `outputs/` and `aggregate_all/` directories, or zip this project with those folders. Analysis should begin from the paired UCPA-vs-xFedAlign CSV and the preregistered gates, not from whichever metric looks best.

## 6. Official xFedAlign reference run

If internet is available:

```bash
bash scripts/fetch_official_xfedalign.sh
```

Follow the official repository's own environment/run instructions in a separate environment. Keep its outputs separate and record the fetched commit. Use it as an external reproduction check, not as a hidden dependency of this pipeline.
