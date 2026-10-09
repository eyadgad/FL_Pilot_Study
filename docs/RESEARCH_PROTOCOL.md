# Research protocol, hypotheses, and audit caveats

## Primary question

Does input-dependent quantized federated attribution correction significantly outperform a matched **full-precision local convolutional corrector** and locally calibrated + fixed-peer gradient fields, under original-image heterogeneous FL and communication constraints?

### Falsifiable hypotheses

- H1: Within the same training seed/shift, QF-SCAD lowers oracle-IG JSD relative to the full-precision `private_conv_48` by at least 0.005 absolute (not just relative to fixed fields).
- H2: Fidelity gain does not cause a meaningful increase in held-out predicted-class explanation-summary disagreement versus fixed-field control.
- H3: Benefits survive α=0.05 class skew without making valid independent-seed task models fail more often than control conditions; task checkpoints are shared across explanation methods.
- H4: Deletion AUC (lower) and insertion AUC (higher) are not worse on paired, same-example perturbation tests.
- H5: The communication-fidelity Pareto position is justified after counting transmitted teacher-stat and int8 weight packets in **both directions**.

### Study details

All 12 SHA-pinned YAMLs are under `configs`. For each seed train one FedAvg CNN from scratch on the genuine training images and test once on all original 10,000 test images; fail closed if it misses the frozen accuracy gate. Client images are divided into disjoint task/surrogate/artifact/teacher/evaluation sets with indexes in `partition_indices.npz`. No synthetic images, generated replacements, or download fallback. No evaluation record, original test example or downstream metric is used to fit the explanation corrector or select its parameters.

All explanations use frozen-model predicted target and the same 8-step midpoint absolute IG with zero baseline. For CIFAR, sum RGB-channel absolute IG to 32×32 coordinates; use mean RGB brightness to drive the 1-channel corrector (77 parameters total). For functional tests, delete/insert full RGB pixels; for MNIST, single channel. Preserve the exact original target class, spatial coordinate ordering and baseline for every method.

No client images or teacher IG maps are intentionally communicated in QF-SCAD; the server only sees its serialized quantized sufficient-statistic packets and model-snapshot packets. This is an **engineering boundary**, not a certified privacy guarantee. The separate `fedattr_class_template_proxy` does communicate compact IG class summaries as an explicit high-traffic baseline and must not be conflated with QF-SCAD's privacy story.

### Interpretation

Check both the paired mean improvement and 95% *seed-level* interval. Sample-level confidence intervals pooled across clients are pseudoreplication. The primary 0.005 threshold must hold against a strong matched local conv control, not a much weaker static control. Qualifying gates must be evaluated by scenario and dataset, especially extreme Dirichlet α=0.05, not merely across all records.

The baseline called `fedattr_class_template_proxy` is **not official xFedAlign**. Official xFedAlign code is at https://github.com/dawoodwasif/xFedAlign (authors' protocol differs, including surrogate fitting, communication, datasets and attribution artifacts). The benchmark here is an internally paired method-development harness. For publication, reproduce official xFedAlign under matching checkpoints and comparable privacy/bandwidth before claiming superiority.

### Status as packaged

- MNIST local images present in the execution environment and 30/30 source-index/split preflight checks passed.
- One historical previously trained CNN / genuine-MNIST attribution-cache smoke passed, comparing all matched baseline architecture categories needed for that smoke. No new FedAvg training seed was completed in this release.
- CIFAR original images were not available in the execution environment; the CIFAR branch has code and frozen configs but **no real-CIFAR data smoke or correctness claim**.
- 60-run prospective study is not executed. Return the entire `full_study_outputs` and `study_analysis` folders for an independent full assessment.
