# CFBA-CRC adversarial smoke-test report

## Purpose
This smoke stage was intentionally used to reject weak directions before another multi-day MNIST/CIFAR suite. Several earlier candidates were killed before CFBA: UCPA, SACPA/NC-SACPA, simple FBA, support-preserving residual alignment, and RPGA.

## Engineering failures discarded
- An early vectorized deletion/insertion implementation integrated over unit-spaced perturbation steps, allowing AUC > 1. It was fixed to integrate over an x-axis normalized to `[0,1]`; all earlier runs were discarded.
- A compact smoke model reached only ~41% task accuracy. That batch was discarded as scientifically invalid.
- A server coordinate-descent policy under client caps was tested and rejected because artifact-level EDI did not reliably predict held-out per-sample drift.

## Final smoke protocol
- non-IID synthetic image clients with high task accuracy;
- direct task-model IG as the private calibration/evaluation reference;
- disjoint calibration and final evaluation examples;
- alpha=.05;
- beta grid `[0,.1,.2,.4,.6,.8]`;
- four-component bounded loss: JSD, deletion, insertion, and top-k excess harm;
- methods: Local, globally certified beta, client-specific CFBA, fixed xFedAlign beta=.2.

Raw valid runs are preserved under `research_history/phase14_cfba_crc_smoke/`.

## Five-seed rotation (1601–1605)
Mean metrics:

| Method | JSD ↓ | EDI ↓ | Excess risk ↓ | Deletion ↓ | Insertion ↑ | Top-k ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Local | .141263 | .138127 | .000000 | .345133 | .681665 | .386816 |
| Global CRC | .141263 | .138127 | .000000 | .345133 | .681665 | .386816 |
| **CFBA** | .143057 | **.111232** | .008005 | **.298540** | **.711156** | **.397949** |
| xFedAlign .2 | **.140729** | .096074 | .014192 | .286281 | .714852 | .397900 |

CFBA beats the globally certified control on EDI in **5/5** seeds. In four seeds, one restrictive client drives the single globally safe beta to zero while other clients still have nonzero certified caps.

## Five-seed patch (1601–1605)

| Method | JSD ↓ | EDI ↓ | Excess risk ↓ | Deletion ↓ | Insertion ↑ | Top-k ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Local | .127907 | .128408 | .000000 | .382077 | .837488 | .432422 |
| Global CRC | .133433 | .065876 | .008732 | .241333 | .884965 | .493408 |
| **CFBA** | .150027 | **.059571** | .011522 | **.237741** | **.886094** | .491064 |
| xFedAlign .2 | **.121976** | .087822 | .003446 | .251219 | .878649 | .477197 |

CFBA beats global CRC on EDI in **3/5** seeds and fixed xFedAlign in **5/5**. The larger raw JSD penalty is why JSD excess harm was added to the frozen composite risk before this final smoke series.

## Erasing stress
On seed 1601 all clients certified beta=.4, so CFBA correctly collapses to the globally certified policy. EDI fell from about .1264 (Local) to .0600 while held-out excess risk stayed about .0093.

## Risk–consistency frontier
On rotation seed 1606, increasing fixed beta progressively lowers EDI while consuming more fidelity risk. CFBA uses heterogeneous certified caps `[.2,.4,.4,.4]`, producing EDI about .0726 at held-out risk about .0119; fixed beta=.6 lowers EDI further (~.0410) but consumes ~.0429 risk. This demonstrates the intended trade-off rather than universal metric dominance.

## Smoke conclusion
CFBA clears the pre-full-study direction bar because it has:
- a finite-sample risk-control mechanism tied to the observed failure mode;
- a strong globally certified control;
- repeated evidence that client heterogeneity can make a single safe beta unnecessarily conservative;
- held-out risk well below alpha in the final smoke;
- a clear failure/abstention mode (beta=0).

It does **not** dominate fixed xFedAlign on every raw metric. The intended paper claim is certified, client-specific explanation alignment under a fidelity budget.
