# CIDM-v2 A3-R4 NoSOC — Actual Results and A3-R5/P0 Frozen Core Audit

Date: 2026-10-12. User-provided result ZIP + executed Colab notebook reviewed.

**Actual experimental gate: 8/11 PASS, status A3_R4_NoSOC_INCONCLUSIVE_BLIND_SIGNAL_AND_OTHER_FAILS. Formal global RTX PRO 6000 training remains BLOCKED.**

## Real model and training
- CIDM Lean NoSOC 36,104 params; SOC parameter count exactly 0; 10 epochs x40 optimizer updates, best validation Epoch 9; full preflight passes.
- Independent field MSE at 24 steps: Transport 0.00953024, Reaction 0.01437853, Wave 0.00759363, Coupled 0.01344473.
- All four 24-step systems beat persistence; Transport and Reaction **fail** stronger Linear Trend at 24 steps (0.00811139 and 0.00750192 respectively). At 16 steps, only Wave among seen systems beats the stronger naive baseline.

## Frozen inference dependence / structural bypass
- Turning OFF learned Flux changes MSE in the original 12 family x horizon aggregates between -0.0182% and +0.0276%: negligible output dependence even though antisymmetric edge-budget numerical preflight passes.
- Turning OFF state synchronization improves 11/12 original field aggregates; Reaction-24 improves from 0.01437853 to 0.01416427, while Wave-24 gets slightly worse.
- Grad sum in preflight: encoder 6.6533, Flux 4.83e-5, Local 0.04669; Flux has two small multiplicative update strengths and may be suppressed or bypassed.
- These are **same-trained-model masked inference interventions**, not parameter-matched retrained ablations. They cannot prove conceptual Flux/Sync invalid.

## Causal cross-scale and blind limits
- Seen families 12-step opposite-scale response: 5/6 evaluable and 4/6 with normalized skill >0.10, without any SOC exchange module.
- Coupled 24-step one direction true source-normalized response ~0.006415, evaluable but predicted response power ~57.7x truth and normalized CF skill ~-64.81; other direction ~0.002585, **below** required 0.005 and inconclusive.
- Coupled has already been repeatedly examined in *this research history*, so newly sampled Coupled cases are NOT a fresh, untouched family-level blind test.

## Immediate next experiment: A3-R5/P0 (inference only)
- A fully standalone Colab Notebook with original Epoch 9 model checkpoint embedded and SHA256 verified; **no new training**.
- Exact replay of original network forward checked before frozen interventions. Nine modes: original, no_flux, no_sync, no_flux_no_sync, frozen_latent, no_latent_readout, no_explicit_trend, no_skew, no_local.
- Pairwise bootstrap CIs on newly seeded field forecasts at 8/16/24/36; stronger naive baseline (min of persistence and linear trend); causal coarse-to-fine and fine-to-coarse at 4/8/12/24; explicit weak-signal INCONCLUSIVE tagging.
- Complete 1-seed and 2-seed CPU end-to-end smoke tests passed; default 8+8 sample full audit has **not yet been fully executed**. No official AIWQ skill or world-training approval is inferred.
- Deliverables in ChatGPT file links: CIDM_v2_A3_R5_P0_Frozen_Core_Bypass_Audit_Colab_STANDALONE.ipynb and CIDM_v2_A3_R5_P0_Frozen_Core_Bypass_Audit_FULL.zip.

Research principle: CIDM is a generic prediction architecture, not a weather/ocean product. Retire unqualified SOC implementation without rejecting the abstract information hypothesis; next focus on latent dynamics being genuinely used rather than bypassed.