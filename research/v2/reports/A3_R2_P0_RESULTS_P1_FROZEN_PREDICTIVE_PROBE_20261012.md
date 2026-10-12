# CIDM-v2 A3-R2/P0 frozen SOC audit and P1 linear predictive-readout analysis

2026-10-12. **CIDM is a general prediction architecture**, not a domain-specific weather module.

## Verified P0 user-run evidence
- Frozen A3-R1 Epoch 9 model SHA256 `8616737266efd776c1c95e63d440cc14dd8df02abe7bc901a240f22ff3381603`.
- 192 paired cases × six forecast leads = 1,152 records; 24 paired seeds per family/direction; eight direction/family groups.
- P0 stage: `SIGNAL_CONTRACT_INSUFFICIENT_FOR_BLIND_SOC_CLAIM`. Seven of eight directional 12-step signals passed the user-run audit's relative/absolute power threshold; unseen Coupled fine-to-coarse failed.
- 12-step SOC positive effects: Reaction coarse-to-fine +0.0434 and Wave fine-to-coarse +0.0107. Other six groups negative; most confident gains are weak. The early positive effect does not generalize through 24-step rolling.
- Unseen Coupled 12-step true response/input signal ratios ~0.00581 and ~0.00337, while full model predicted response power ~22.4 and ~32.3 times truth; first passes the preset signal floor but still has a real amplitude-overreaction concern.
- Input perturbation opposite-band leakage ~1e-14 is guaranteed by construction, NOT proof of a predictive latent representation. Expected latent scale power fraction ~0.62–0.64 (coarse), ~0.81–0.83 (fine), but energy correspondence is not predictive information sufficiency.

## P1 exploratory probe (assistant CPU-reproduced, no new CIDM training)
- A single Colab notebook embeds the EXACT frozen Epoch-9 model checkpoint and original synthetic simulator; only small fixed-regularization local 3x3 ridge readouts are fit to Transport/Reaction/Wave training seeds. Entire Coupled family is blind.
- Probe against raw paired input, frozen latent, expected latent scale, opposite latent scale and immediate post-SOC latent. No access to future simulator family labels, coefficients, or targets in features.
- The actual local full P1 run completed with distinct training, validation, test and blind seeds; e.g. Wave fine-to-coarse at lead 12: raw probe +0.338, latent +0.765, post-SOC +0.766; Reaction coarse-to-fine: raw -0.166, latent -0.436, post-SOC -0.437; Coupled fine-to-coarse: raw -5.162, latent -14.667, post-SOC -14.703.
- Interpretation: latent has some linearly decodable future cross-scale information for Wave. SOC's INITIAL exchange hardly changes local linear decodability; evidence does not support blindly increasing SOC gain or declaring all information was destroyed by encoder.
- LIMITATION: linear readout is a restricted lower bound, different representation feature counts, paired inference assumes two last-frame scenarios; no information-theoretic sufficiency proof or capacity-matched SOC ablation.

## Investment / next research decision
- A3-R1/R2 have NOT established SOC's abstract cross-dynamical predictive transfer. Do not scale SOC or pay for RTX PRO 6000 global weather training.
- Next candidate A3-R3 should test multi-lead prediction-specific latent supervision and *incremental* cross-scale rollout response, keeping task/PDE coefficient labels absent and unseen Coupled excluded from selection.
- Self-contained P1 notebook, report, and locally executed exploratory probe ZIP supplied through ChatGPT as `CIDM_v2_A3_R2_P1_Frozen_Predictive_Scale_Audit_FULL.zip`.