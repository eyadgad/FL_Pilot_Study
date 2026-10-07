# Theory sketch — CFBA-CRC

## Setup
For client `i`, let `beta` index an ordered finite family of explanation-alignment actions. Let `L_i(X,beta) in [0,1]` be the four-component excess-fidelity loss defined in `METHOD.md` on an exchangeable client-local example `X`.

Because the raw explanation loss need not be monotone in beta, define the nested envelope

`Ltilde_i(X,beta_g) = max_{h<=g} L_i(X,beta_h)`.

This loss is bounded and nondecreasing in alignment strength. Equivalently, under `lambda=1-beta`, it is nonincreasing in the CRC control parameter.

## CRC selector
Given `n_i` calibration examples, define

`U_i(beta) = n_i/(n_i+1) * mean_j Ltilde_i(X_j,beta) + 1/(n_i+1)`.

CFBA selects the largest beta with `U_i(beta)<=alpha`.

Under the standard CRC assumptions for bounded monotone risk, this gives the corresponding finite-sample control of the selected action's expected `Ltilde` risk. Since `L <= Ltilde` and each individual excess-harm component is <= `L`, the same budget upper-bounds each constituent expected excess harm.

## Why client-specific calibration can matter
Let `beta_i^max` denote client i's maximal certified action. A globally certified policy restricted to one shared beta must use at most

`beta_global = min_i beta_i^max`.

If clients have heterogeneous risk curves, there can be clients with `beta_i^max > beta_global`. CFBA can align those clients more strongly without changing the certification rule for the restrictive client. The empirical contribution must establish that this extra admissible alignment translates into lower cross-client explanation drift; it does not follow from the risk theorem alone.

## What remains to prove/write carefully
- State the exact CRC theorem with the package's beta/lambda reparameterization.
- Make explicit that the monotone envelope changes the controlled loss to a conservative nested loss.
- Separate the theorem (expected risk under exchangeability) from empirical held-out tests.
- Do not claim simultaneous high-probability per-client coverage unless a separate theorem is added.
- Analyze calibration computation and how the risk budget scales with finite `n`.
