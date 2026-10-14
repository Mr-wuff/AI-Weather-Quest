# CIDM-v2 B2-R9/P0 official frozen timing result; B2-R10/P0 gradient-credit qualification (2026-10-14)

## Original user-supplied result

- `CIDM_v2_B2_R9_P0_ACTIVATION_ENERGY_REPORTS.zip` SHA256 `3b90fded1769d199da2a8ca93e1505acc451d074f4e45e8c720be3a15861389f`, matches sidecar; CRC PASS.
- `status=FROZEN_ACTIVATION_TIMING_COMPLETED_NO_TRAINING` with official PDEBench MD5 3/3 PASS, seed17 original parent replay 3/3 PASS, B2-R8 per-window replay max error 0, pre-onset parity zero, equal-perturbation exposure max difference 1.4e-9; 315 curves and 5,040 paired window MSE records. No new training.
- Runtime sync JSON records successful HF model ZIP/checksum/source uploads and successful GitHub Release, source and status updates.

## Actual fixed 0.3% latent-RMS energy-matched activation effects (pooled lead9+12 MSE gains relative to original v2)

| PDE | Onset1 | Onset4 matched energy | Onset7 matched energy |
|---|---:|---:|---:|
| Burgers | 2.414% | 2.013% | 1.609% |
| Reaction–Diffusion | 7.236% | 4.842% | 2.508% |
| Shallow-Water | 1.883% | 1.785% | 1.551% |

Matched energy means equal *sum of squared relative latent-state perturbation* (0.000108 for budget 0.3%), not equal physical energy or forecast effect.

- Early lead1/3 window >2% relative-MSE regressions (16 windows/task): Burgers Onset1 7/16 -> Onset4 matched 0/16; RD 8/16 -> 0/16; SWE 0/16 in both. Burgers lead9-or-12 regression-window count 6/16 ->4/16 (Onset7 matched 1/16). Corresponding gains decrease.
- These reused 16 VAL windows are **not untouched external test data**, and the trained controllers use seed17 only; `timing_gate.json`'s `candidate_diagnostic_pass` means a frozen diagnostic passed, **not training approval**.

## Retrospective action-switch headroom (future truth Oracle, NOT issue-time usable)

For a hypothetical selector choosing only `Onset1` vs `Onset4 matched energy` per validation window with future lead9+12 truth, extra improvement over the *best fixed schedule* is **0.786 percentage points Burgers**, **zero Reaction–Diffusion**, **0.010 percentage points SWE**. Selection frequencies Onset1 vs Onset4 were 10/6, 16/0, 13/3. Hence a broad, trainable issue-time onset gate has tiny retrospective headroom on this dataset and should **NOT** be the next expensive model training.

## B2-R10/P0 experiment delivered (no official experiment executed yet)

Standalone Colab Notebook: `CIDM_v2_B2_R10_P0_GradientCreditHorizon_Colab_STANDALONE.ipynb`, new exact Python driver `b2_r10_gradient_credit_audit.py`, detailed Chinese result audit, paired-window metrics, true user-provided R6/R9 ZIP evidence, full delivery ZIP and SHA256. Exact eight upstream sources, original six R6 trained heads and real R7/R8/R9 reports are embedded/verified. Original 18-run B0-R1 parent SHA256 `69552b29e169ec22ee85d9c1b1a678fdb188f6f25dc18f33b927b88f4dddffd4`; HF Model/GH Release discovery first; strict original official data MD5/splits and seed17 parent replay. No Google Drive.

Q0 **zero training** on two official TRAIN windows per PDE and the frozen learned Natural and Rotated controller. Identical 12-step causal rollout and objective `0.5*(MSE9+MSE12)+0.125*(MSE1+MSE3)`; compute control-head parameter gradients with autograd detach horizon K=1, K=3 and full K=12. All original F/L/S and decoder core **must be differentiated through** despite frozen core weights; avoid prior `torch.no_grad` that blocked multi-step credit. Compare cosine and relative L2 error of truncated vs reference 12-step gradient, forward parity <=2e-6, finite gradients and GPU memory / seconds. Future truth is used as offline TRAIN supervision only; not a model prediction input.

Q0 cannot grant architecture training permission by itself. If K3 materially improves training-gradient fidelity over K1 and resource cost permits, consider **separate pre-registered Q1 small paired BPTT(1) vs BPTT(3)**, risk monitoring and independent fresh data/multiple training seeds before full CIDM-v2 training.

## Program validation

Full 11-code-cell notebook CPU smoke executed PASS; exact code/archived report SHA verified; original R6 heads exercised on 1D and 2D; K1/K3/K12 gradients for Natural and Rotated heads 12 synthetic reports, original R9 forward output parity PASS; ZIP SHA/CRC PASS. **Real official B2-R10 not run.** No full architecture training, no AI Weather Quest RPSS qualification.
