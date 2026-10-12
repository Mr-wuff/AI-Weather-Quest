# CIDM-v2 A3-R3 result: retire current SOC from v2 mainline

**Date:** 2026-10-12. **Decision:** Existing SOC0/SOC1 cross-scale exchange is no longer a mandatory CIDM-v2 core component. The original abstract information-evolution hypothesis remains separate research, not an assumed proven mechanism. CIDM is a domain-general PREDICTION architecture, not a weather-only product.

## Grounded uploaded A3-R3 results
- 53,740 parameter single CIDM with SOC; 10 epochs x 40 updates; best epoch 9; gate **6/12** passed; status `A3_R3_INCONCLUSIVE_UNSEEN_SIGNAL_AND_GATE_FAILED`.
- Novel multi-lead causal and latent auxiliary supervision improved selected long-lead field forecasts compared with historical A3-R1, but **this is not an attribution to SOC**.
- Same-trained-weight SOC disabled field MSE vs enabled: Transport at 24 steps 0.015978 vs 0.016154 (disabled better ~1.09%); Wave at 24 0.008956 vs 0.009032 (disabled better ~0.85%); Reaction at 24 0.014537 vs 0.014444 (SOC better ~0.64%); unseen Coupled at 24 0.010481 vs 0.010372 (SOC better ~1.05%).
- 12-step seen-family true cross-response signal evaluable in six of six: SOC contribution to CF skill ranges from about -0.0083 to +0.0097; NONE reaches preregistered +0.02.
- Unseen Coupled coarse-to-fine 12-step CF skill ~-13.37, SOC OFF ~-13.31; unseen fine-to-coarse true response/input signal ratio ~0.0033, below preregistered 0.005 floor. This is an inconclusive evaluation for the latter direction, not evidence of guaranteed model failure.
- 16-step strong-naive baseline gate failed. The user-uploaded synthetic evidence does NOT support stable, general and beneficial SOC.

## CIDM-v2 architectural decision
- Remove all current SOC exchange parameters and SOC forward calls from the **mainline candidate**; do not merely ablate SOC at inference.
- Keep historical encoder, antisymmetric local Flux, trainable Local+Skew evolution, prediction-only self-reencoding Sync, and supervised long-range causal response objectives, without PDE-specific upwind equations.
- Treat BER and abstract open-system information exchange as future independent mechanism research, not an excuse to add another free residual module.
- Next experiment A3-R4 **one newly trained no-SOC model** with 36,104 parameters (vs A3-R3 53,740; ~32.8% less), same seen-family training generator, data budget, seed, loss weighting, and selection policy. Report new independent heldout field 8/16/24, causal pair leads 4/8/12/24; evaluate unseen Coupled family after freezing checkpoint only.
- Coupled blind cross-direction signal audit uses lead 24 because prior lead 12 one direction failed signal sufficiency. Signal thresholds are not reduced and failing independent skill remains a failure.
- This is NOT an equal-parameter causal ablation or official AI Weather Quest benchmark. PRO 6000 global spending remains BLOCKED regardless of synthetic success.

## Deliverables (in ChatGPT conversation)
- `CIDM_v2_A3_R4_NoSOC_Core_Qualification_Colab_STANDALONE.ipynb` self-contained; source SHA256 verified; full mini-run passed.
- `CIDM_v2_A3_R4_NoSOC_Core_Colab_FULL.zip` source, research report, experiment contract, historical score chart and hashes.
- Small-budget smoke achieved no scientific Gate; no full 10-epoch experiment or performance claim is yet available.

## Security
Repository synchronization from hosted training must use newly rotated environment secrets. Do not reuse any previously disclosed personal access tokens.