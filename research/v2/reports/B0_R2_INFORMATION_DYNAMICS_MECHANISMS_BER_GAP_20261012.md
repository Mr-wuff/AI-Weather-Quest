# CIDM-v2 B0-R2: component necessity, BER identifiability and v1 comparison (2026-10-12)

## Verified frozen-inference evidence
User-uploaded `CIDM_v2_B0_R2_FROZEN_DIAGNOSTIC_REPORTS.zip`: `FROZEN_DIAGNOSTIC_NO_TRAINING_NO_ARCHITECTURE_QUALIFICATION`; 18 model replays, 9 paired v1/v2 seed/task entries, official-data MD5 validation claimed in original runtime. Testing windows 32 per task.
Frozen removal of v2 components, percent change in 12-step **MSE** (positive = model worsens when removed), means over 3 seeds:
| Intervention | Burgers | Reaction-Diffusion | Shallow-Water |
|---|---:|---:|---:|
| noFlux | +104.56 | +332.67 | +44.67 |
| noLocal | +46.31 | +253.60 | -0.37 |
| noSkew | +33.64 | +33.35 | -1.36 |
| frozenLatent | +165.30 | +696.00 | +26.44 |

v2 prediction increment power / true: Burgers .381, RD .987, SWE .230. v1 corresponding .557, 1.903, 1.018. SWE v2 under-response is striking; neither flux activation nor skew algebra guarantees correct decoded dynamics.

## Actual code contracts
- B0 `V1Archived` uses the original archived `CIDM_BER` **with SOC and BER enabled**, but `port=None`, i.e. zero external input. BER then remains an ordinary state-dependent residual: this **does not validate identifiable external-state exchange**.
- PDE-ported B0 v2 `PortedRestoredCIDM` has only Encoder, antisymmetric Flux, Local, Skew and direct incremental Decoder; **NO BER, no SOC, no v1-style internal hidden reservoir**. Its Sync parameters exist but inference/training uses `mode='off'`.
- Flux periodic telescoping sum is a latent spatial-budget identity, not physical mass or entropy conservation. Skew makes `z^T A z=0` instantaneously, but explicit Euler does not conserve norm.
- SWE Lowpass residual RMS v1 .006465 vs v2 .020691; highpass .004449 vs .015510; interior .008939 vs .032668. Boundary diagnostic is not the main observed dominant gap.
- Only same-checkpoint inference masking was done; Local/Skew deletion decisions require **trained** capacity-matched ablations and component interaction controls.

## Accuracy v2 vs v1 paired 3-seed B0-R1 training
- Burgers RMSE 6-step v1 .056367 vs v2 .091537 (v2 -62.39% relative); 12-step .094384 vs .130207 (-37.95%).
- RD2D 6-step .007144 vs .003144 (+55.99%); 12-step .010051 vs .005508 (+45.20%).
- SWE2D 6-step .004362 vs .021037 (-382.31%); 12-step .008519 vs .030632 (-259.59%).
- v2 wins 1 of 3 PDE families, 2/6 task-horizon cells; no evidence that long 24/48-step drift, Sync, open-system dynamics, eight PDE generalization or AI Weather Quest RPSS has been solved.

## Next economical gate: B1/P0 BER
A standalone Colab Notebook attached in ChatGPT runs 3 independent seeds x3 independently trained models: no BER, zero-port BER (same capacity as port BER), causally accessible future-port BER. Matched-input histories with alternate independently issued future external ports; train on two generic dynamics, hold out a third whole family; evaluate future-state MSE and future-port CF sensitivity. Both BER architectures share parameters; no PDE coefficients/family IDs used by models. BER requires proven port dependence and improved heldout conditional forecasting.
Notebook executable from one .ipynb without external code/data; entire CPU smoke run verified, scientific status `SMOKE_DIAGNOSTIC_ONLY`. **The full GPU pilot is NOT run.** Program must not be used to claim external information prediction when future port is unknown.

Decision: Keep Flux and latent dynamics as core research candidate; freeze SOC removal; Local/Skew retrained ablations first before removing; Sync not established; introduce BER only after causal-port information gate. RTX PRO 6000 global training remains blocked until eight-PDE and official AIWQ gates.