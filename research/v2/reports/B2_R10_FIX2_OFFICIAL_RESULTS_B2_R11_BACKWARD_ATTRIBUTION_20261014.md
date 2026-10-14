# CIDM-v2 B2-R10 FIX2 official gradient results and B2-R11/P0 backward attribution contract — 2026-10-14

## Actual uploaded B2-R10 FIX2 result, verified
- ZIP SHA256 `bfc54b95d54714eee6ddd1ef3f1d158418f26f8e00a5fc42faf0b8cc5eeba6bc`; uploaded sidecar matches, CRC passes, internal FILE_SHA256_MANIFEST independently verifies.
- `report.json.status=FROZEN_GRADIENT_CREDIT_AUDIT_COMPLETED_NO_TRAINING` and 18/18 gradient-horizon rows from all Burgers, Reaction–Diffusion, SWE (original seed17, Natural and Rotated heads, K=1/3/12).
- Gate: `DIAGNOSTIC_ONLY_NO_TRAINING`; numeric status `SEVERE_GRADIENT_AMPLIFICATION_NO_TRAINING`; candidate_for_future_3step_pilot=false.
- All 3 official PDEBench HDF5 MD5 checks, parent checkpoint RMSE exact replay PASS, forward parity zero and no optimizer updates. HF/GitHub sync user log reports success.

## Exact FP64 K12 controller-gradient norms / K12 divided by K3
| task | head | K1 cosine vs K12 | K3 cosine vs K12 | K12 norm | K12/K3 |
|---|---|---:|---:|---:|---:|
| Burgers | Natural | .660690 | .703934 | .00424055 | 4.7596 |
| Burgers | Rotated | .650414 | .652121 | .00024463 | 2.4625 |
| Reaction–Diffusion | Natural | .024792 | .191177 | 2.39433e-5 | 6.8612 |
| Reaction–Diffusion | Rotated | .913302 | .939019 | 7.33112e-5 | 10.2239 |
| SWE | Natural | -.092224 | -.013738 | **9.04094e19** | **2.26226e24** |
| SWE | Rotated | -.096751 | -.078625 | **1.09105e14** | **3.98090e18** |

FP32 gradient norm computation overflowed for SWE Natural but FP64 shows its true very large magnitude (all gradient elements finite). SWE Natural largest absolute parameter gradient ~4.599e19. Full-gradient cosine fidelity differs by PDE and control geometry; K3 is NOT justified as a training change. K12 is a mathematical autograd reference, not a stable optimizer direction or forecast skill. Training data: two TRAIN windows per task only.

## B2-R11/P0 preregistered zero-training experiment (created/delivered in ChatGPT conversation)
Standalone Colab Notebook `CIDM_v2_B2_R11_P0_BackwardPathAttribution_Colab_STANDALONE.ipynb` and exact source `b2_r11_backward_attribution.py` (SHA256 `5e413a85e0cf8c3ddd6584a220a965c4228978cf2d50300670b5ea86dabefebe`). Complete delivery ZIP SHA256 `445032eeea288fa3abfb69b4e33b2e94cd6545e498c50b33b3197ed428fbc634`.

- Pin **exact B2-R10 FIX2 results ZIP bytes and SHA**, real B2-R6 six trained heads, R7/R8/R9 real evidence and 8 original SOURCE hashes. Original 18-run B0 parent SHA256 `69552b29e169ec22ee85d9c1b1a678fdb188f6f25dc18f33b927b88f4dddffd4`; HF **Model** / GH Release parent discovery first; official PDEBench data fixed MD5, original RMSE test replay. No Google Drive.
- Real 12-step FP32 forward unchanged from R9/R10 including correct zero-safe stable exact RMS. Freeze model and heads; 2 original TRAIN windows per task, original seed17 Natural vs Rotated heads, same physical loss `0.5*(MSE9+MSE12)+0.125*(MSE1+MSE3)`. Full gradient backprop ONLY; zero parameter updates.
- **Temporal parameter contribution**: extract `dL/dgates_t` for each of 12 head calls, re-evaluate that head with detached current-state inputs, form per-timestep `(dgates_t/dtheta)^T(dL/dgates_t)`; enforce exact sum reconstructs complete K12 controller gradient within 1e-4 relative error.
- **Local operator VJP**: each step measure identity, Flux, Local, Skew contributions with correct effects of the local normalized basis; separately compare summed VJPs with an independently recomputed *local* autograd derivative (<=1e-3). Do **not** directly compare with original `z_old.grad`: DAG includes earlier decoder-side branches and they are different cotangents.
- Expected official groups: 3 PDE × 2 heads = 6; 72 temporal head-adjoint rows + 72 operator local VJP rows. Outputs corresponding CSVs, case summary, gates, report ZIP and checksum. This is an attribution diagnostic, **not** causal operator ablation, physical conservation audit, Jacobian spectral norm or deployment.
- All original upstream hashes and actual FIX2 result SHA are enforced before official run; any input/forward/gradient mismatch blocks. Work remains strict original train data and zero new weights.

## Actual notebook test
Whole B2-R11 Notebook executed all code cells in CPU smoke mode with exact user archived sources and real six trained R6 heads, both synthetic 1D and 2D tasks; 4 test pairs, 48 time rows, 48 local VJP rows. Exact R9 forward parity, parameter-contribution reconstruction, and local-VJP reconstruction passed. ZIP/CRC/SHA256 packaging passed.
**Official B2-R11 three-task run has not occurred**, so no claim about which time step or operator causes SWE amplification is made. Subsequent training gates stay CLOSED, as do v1-superiority, unseen external holdout, and AI Weather Quest RPSS qualification.
