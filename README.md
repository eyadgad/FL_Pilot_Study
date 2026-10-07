# CFBA-CRC — Full Reproducible Federated Explanation Research Pipeline

**Current status:** CFBA-CRC passed an adversarial pre-full-study smoke program and scores **79/100 on the project's paper-direction readiness rubric**. This is permission to run the frozen full validation, **not a 79% acceptance probability**. Full confirmatory seeds **1701–1705 have not been run**.

CFBA-CRC (Conformal Fidelity-Budgeted Alignment) reuses an xFedAlign-style robust global attribution prior, but each client privately chooses the strongest alignment it can certify under a bounded multi-metric excess-fidelity budget. The critical control `global_crc` uses the same information and the single strongest beta safe for every client.

## Why this direction survived
Earlier methods were killed when harder testing exposed scale collapse, oversmoothing, functional-fidelity loss, or over-conservative heuristics. CFBA directly controls the failure that remained: explanation alignment may reduce cross-client drift while degrading fidelity to the deployed task model.

The final smoke used high-accuracy non-IID synthetic tasks, disjoint calibration/test examples, direct task-model IG, and a valid finite-sample CRC correction. Across five rotation seeds, CFBA reduced EDI by ~19% versus the globally safe control while keeping mean held-out composite excess risk ~.008. Patch/erasing tests and a risk–consistency frontier are documented in `research/SMOKE_TEST_REPORT.md`.

## Install
```bash
pip install -e '.[dev]'
python scripts/verify_package.py
```

## Frozen study
```bash
# Confirmatory core
python scripts/run_suite.py --tier A

# Required robustness/stress
python scripts/run_suite.py --tier B

python aggregate_results.py --outputs outputs --out aggregate_all
python evaluate_cfba_gates.py \
  --runs aggregate_all/runs.csv \
  --out aggregate_all/cfba_gate_report.json

# Exploratory sensitivities only
python scripts/run_suite.py --tier C
python aggregate_results.py --outputs outputs --out aggregate_all

python scripts/verify_runs.py --outputs outputs
```

## What is frozen
- confirmatory seeds: 1701–1705;
- alpha=.05;
- beta grid `[0,.1,.2,.4,.6,.8]`;
- four-component bounded excess-fidelity loss;
- five disjoint client data partitions;
- seven executable decision gates in `evaluate_cfba_gates.py`.

## Key files
- `research/METHOD.md` — exact algorithm and guarantee scope.
- `research/PREREGISTRATION.md` — frozen study and gates.
- `research/SMOKE_TEST_REPORT.md` — adversarial smoke evidence and discarded failures.
- `research/READINESS_SCORE.md` — 79/100 direction score and interpretation.
- `research/NOVELTY_POSITIONING.md` — closest prior art and narrow novelty claim.
- `research/RUNBOOK.md` — execution instructions.
- `research_history/phase14_cfba_crc_smoke/` — raw valid smoke evidence.

## Important interpretation
A `CONTINUE` gate means the direction survived this validation stage. It does not mean the paper is accepted or finished. Official xFedAlign code comparison, more independent seeds, stronger architectures/datasets, and a paper-level theorem/assumption treatment remain subsequent work.
