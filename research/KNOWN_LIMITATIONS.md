# Known limitations

- The CRC theorem requires the calibration and future examples to satisfy the relevant exchangeability conditions. Strong temporal or adversarial distribution shift can invalidate the interpretation of the certificate.
- Direct task-model IG is used as a private calibration reference. IG is not ground-truth explanation and adds client-side computation.
- The guarantee controls the defined bounded composite loss, not every conceivable explanation-quality property.
- The four components are combined with a max rather than learned weights. This is conservative and transparent, but not necessarily optimal.
- The finite-sample correction makes small calibration sets conservative. With alpha=.05, very small `n` cannot certify even the zero-excess fallback in the formal CRC inequality; the pipeline logs uncertified cases and the full gate requires complete certification.
- xFedAlign in this package is a self-contained reproduction of the published global-prior mechanism, not the authors' official code. A paper-stage study should include the official implementation when possible.
- Five confirmatory seeds are a research gate, not sufficient by themselves for strong final inferential claims.
- CIFAR-10 uses the package's compact CNN and is not intended to establish state-of-the-art task accuracy.
- Client-private calibration changes compute cost even though it does not add a high-dimensional communication payload. Compute/time should be reported in the final paper.
- Generic federated conformal risk control and conformal explanation methods already exist; the novelty claim is narrower: risk-controlled *federated explanation alignment* using a multi-metric local explanation-fidelity budget.
