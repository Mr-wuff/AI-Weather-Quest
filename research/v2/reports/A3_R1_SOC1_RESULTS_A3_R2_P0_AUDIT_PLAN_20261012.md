# CIDM-v2 A3-R1 SOC1 actual results and A3-R2/P0 research gate (2026-10-12)

## Genuine uploaded result: not qualified
- Source: user-provided completed Colab notebook and CIDM_v2_A3_R1_SOC1_QUALIFICATION_REPORTS.zip; all seven SHA256 entries in the package manifest verified.
- Single shared CIDM (52,658 parameters); 9 epochs ×48 updates; best validation checkpoint epoch 9 (1-based).
- Gate 5/10 PASS: causal pair, trained-family short vs persistence, unseen-family short vs persistence, existence of cross signal, long-vs-persistence. Fails: OOD 16-step vs strong baseline, positive cross-scale response, SOC positive impact, direction-specific SOC value, unseen-family cross-response.
- 24-step Transport MSE 0.020807 vs persistence 0.019237; unseen Coupled 0.015719 vs persistence 0.016564.

## SOC frozen-weight mechanism increments (counterfactual skill FULL minus OFF)
| Family | coarse -> fine | fine -> coarse |
|---|---:|---:|
| transport | -0.002688 | -0.011940 |
| reaction | +0.045775 | -0.038255 |
| wave | -0.002201 | +0.010729 |
| coupled (unseen) | -0.463821 | -1.295813 |

Coupled true opposite-band power is very small (1.034e-5 for low perturbation, 6.016e-6 for high perturbation). Full-model predicted response power/true response power is approximately 22.45/31.80. Negative normalized skill from such small denominators needs additional signal identifiability checking; it is not standalone proof of an intrinsically non-generalizable SOC operator.

## Critical code-level limitations
- SOC's Haar latent bands do not by definition align with the input-space low/high bands measured by 4x4 average pooling; the learned encoder includes convolution and GroupNorm.
- Readout bypasses SOC with direct current-field and field-difference inputs. SOC gate is bounded and comparatively weak.
- Counterfactual training loss stayed nearly flat despite lower validation field MSE; trained horizon 8 vs evaluation 12/24 steps.
- Inference-only frozen-weight SOC disable is not an equal-parameter retraining control.

## A3-R2/P0: strict frozen scale-information sufficiency audit (ready)
Do not train a larger SOC yet. A standalone Colab Notebook embeds the actual A3-R1 Epoch 9 model-only checkpoint and entire original source, without requiring a ZIP upload or any new access token.
- Audit four dynamical families, both directions, 24 independent pairs each, leads 1/4/8/12/16/24.
- Separate actual cross-band signal energy, relative injected energy, encoder latent scale alignment, prediction delta energy, weighted normalized error, and paired SOC MSE benefit.
- Mark an evaluation group **INCONCLUSIVE** when true cross-band power <1e-6 or relative to input perturbation power <0.005. These are diagnostic thresholds, not universal physical constants.
- Use ratio-of-means, not mean of 1-error_i/signal_i over nearly zero targets.
- No training, no HF/GitHub writes, no token use. Synthetic evidence cannot authorize RTX PRO 6000 global AI Weather Quest training.

Release available as user-facing ChatGPT artifacts:
- CIDM_v2_A3_R2_P0_Frozen_Scale_Information_Audit_Colab_STANDALONE.ipynb
- CIDM_v2_A3_R2_P0_Frozen_Scale_Information_Audit_FULL.zip
Frozen checkpoint is NOT uploaded to public GitHub.