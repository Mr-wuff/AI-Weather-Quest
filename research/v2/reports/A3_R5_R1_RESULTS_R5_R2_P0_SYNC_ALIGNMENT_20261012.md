# CIDM-v2 R5-R1 official user run and R5-R2/P0 frozen sync audit

Date 2026-10-12. **CIDM is a generic prediction architecture, not a weather/ocean-only module.**

## User-run R5-R1 results
- Trained one 34,664-parameter NoSOC CIDM for 9 epochs x40 steps, best epoch 9; Gate 5/8, `NOT_QUALIFIED`.
- Flux isolated coefficient 0.15250 after training. At 24 steps, masking Flux increases MSE by 20.26% Transport, 15.35% Wave, 35.84% Coupled; Reaction **improves** 10.62% when Flux is masked. Frozen ablation is not capacity-matched retraining.
- Freezing latent dynamics raises 24-step MSE by 15.66%, 22.83%, 40.8%, 89.0% across Transport/Reaction/Wave/Coupled. This indicates true latent-state contribution.
- 9/9 validation epochs choose Sync=`off`. Final validation 12-step MSE: off 0.00100634, soft 0.00106788, hard 0.00127579.
- 24-step seen-family strong baseline passes two of three; 36-step Transport still trails persistence. Coupled 24-step counterfactual both directions fail joint signal/skill Gate.
- Previous Foundation A3 Flux and A4 interval hard-reencoding findings are different architectures, tasks, optimization and evaluation contracts. Current results do not refute them.

## Assistant-executed NEW R5-R2/P0 audit (no CIDM retraining)
- Standalone code embeds user's *exact* Epoch9 model-only state_dict; torch weights_only=True safe load and exact Off/Soft/Hard original forward replay verified.
- Predeclared 21 policies: off or 5 reencode intervals x 4 strengths. Select only on NEW seen-family validation seeds; Coupled not used for policy selection. Non-off must improve validation at least 1% to be selected.
- New validation 24/36 combined MSE: off 0.0257450; best nonoff = every12 beta0.275 MSE 0.0255679, only 0.688% better. **Selected off**.
- New independent test (n=5 cases/family) at 24 steps: best nonoff relative MSE vs Off +1.46% Transport, +3.81% Reaction, -0.49% Wave, +3.01% Coupled. Not a stable general gain.
- Uncorrected evolving z vs re-encoded model-generated rolling history cosine at step 6->36: Transport .976->.419, Reaction .971->.367, Wave .979->.480, Coupled .977->.403.
- Evaluation-only true-history encoding tends to be closer to model-generated reencoding than to independently evolved z, but replacing z with reencoding worsens forecast in many cases; possible dynamics/decoder gauge mismatch. This is a hypothesis, not demonstrated causality.
- New artifact in ChatGPT: CIDM_v2_A3_R5_R2_P0_Frozen_Sync_Alignment_Colab_STANDALONE.ipynb and CIDM_v2_A3_R5_R2_P0_Sync_Alignment_Colab_FULL.zip; full DEFAULT CPU audit executed and archived, small notebook end-to-end test completed.

## Decision
- Preserve Flux as a core candidate. Do not restore abandoned SOC just to boost synthetic metrics.
- Don't discard State Synchronization theory; **freeze the current hard/soft injection implementation** pending self-consistency and decoder-compatibility training.
- Next possible trainable R5-R2 candidate should include a low-budget mechanism gate on prediction-only reencoding alignment with matched controls, before formal training.
- No real ECMWF AI Weather Quest RPSS. No RTX PRO6000 authorization. Coupled family has been repeatedly examined across the program and is NOT research-level untouched blind.

Credentials: Never upload tokens. Only use rotated secrets for runtime sync if needed.