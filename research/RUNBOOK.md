# CFBA-CRC runbook

## 1. Install and verify
```bash
pip install -e '.[dev]'
python scripts/verify_package.py
```

## 2. Run untouched confirmation
```bash
python scripts/run_suite.py --tier A
```
Do not inspect Tier C and alter the frozen method. Seeds 1701–1705 are confirmatory.

## 3. Run required stress/robustness
```bash
python scripts/run_suite.py --tier B
```

## 4. Aggregate and execute the frozen gate
```bash
python aggregate_results.py --outputs outputs --out aggregate_all
python evaluate_cfba_gates.py \
  --runs aggregate_all/runs.csv \
  --out aggregate_all/cfba_gate_report.json
```

## 5. Exploratory sensitivity only
```bash
python scripts/run_suite.py --tier C
python aggregate_results.py --outputs outputs --out aggregate_all
```
Tier C cannot retroactively redefine Tier A/B success.

## 6. Integrity
```bash
python scripts/verify_runs.py --outputs outputs
```

## Parallel execution
Different tiers/configs write separate run directories, but one GPU may become compute-bound even when VRAM is abundant. For the cleanest confirmatory record, run Tier A without competing GPU workloads; Tier B/C may be parallelized if desired. Do not compare wall-clock method times across different contention conditions.
