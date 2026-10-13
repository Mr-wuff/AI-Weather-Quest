# CIDM-v2 B1-R3/R1 real results and B1-R4/P0 frozen external-event consequence audit

Date: 2026-10-13. Continue CIDM-v2, NOT v3. User-uploaded complete BER normalized causal-gradient report ZIP and executed Notebook reviewed.

## Original user-run result: Q0 passed; heldout mechanism Gate failed
- B1-R3/R1 has 12 trained checkpoints: 3 seeds x4 same-parameter 19,038 models x240 updates. Q0 passed 6/6, then formal 1/4; status `B1R3R1_HELDOUT_NOT_QUALIFIED`.
- Four models: normalized-loss State-Port, original raw-loss State-Port, normalized Readout-Port, normalized Zero-Port.
- Q0 normalized State continuous MSE benefit +71.67% vs Zero-Port and Readout +59.78%; normalized state early pulse response CF skill ~0.745; normalized vs raw pulse CF +1.1199; pulse power ratio .816. These are seen-family validation only.
- Across independent test seeds, seen early-pulse 12-step post-event CF skill: normalized State +0.7261 vs raw State -0.5928 vs Readout +0.5467.
- Unseen internal Coupled family early-pulse 12-step: normalized State post-event CF +0.0494, Readout +0.1109. At 24 steps: State -1.0016, Readout -0.6088, raw-loss State -3.8898. State 24-step response power ratio ~0.927, **directional cosine -0.0388**, so wrong direction/alignment rather than simply wrong amplitude.
- State-vs-Readout unseen 24-step field MSE relative gains by seed: +8.76%, -9.26%, -1.86%; State response CF gains negative all three seeds. Power stabilized at 0.923–0.932 but skill failed. Normalization repaired seen short-range response, not out-of-family long-range causal consequence propagation.
- Evidence limitations: Coupled family external source injection law is shared with seen systems; Coupled has been reused throughout this program, so not a pristine novel holdout. Access to future external port is simulator-issued, not inferred future forcing.

## B1-R4/P0 follow-up delivered (no training)
- New standalone Colab Notebook embeds and SHA256-verifies all 12 exact completed model checkpoints; source code self-contained. Replays fresh paired seeds for seen early pulse12/24 and reused-family early/mid pulse12/24.
- Compares actual causal free rollout to **non-deployable oracle one-step predictions using true future observation history**, and critically to **oracle persistence using exactly the same true history**. Never mistake oracle high CF score for an operational forecast or drift cure.
- Tracks lead-resolved true response power, predicted power, directional cosine, CF skill, latent paired divergence, field MSE, with grouped reporting; no architecture Gate claim.
- Whole Notebook end-to-end CPU smoke passed. Independently ran FULL frozen audit with 3 model seeds ×3 variants ×5 scenarios, 12 new pairs per family. New full local audit confirms heldout early24 state free CF -1.037, response cosine -0.060, Readout CF -0.619; oracle true-history state CF +0.970 but true-history persistence +0.985 (**oracle model fails to beat persistence**). This result CANNOT be interpreted as a viable Sync strategy.
- Artifacts included in ChatGPT: `CIDM_v2_B1_R4_P0_Frozen_BER_Causal_Geometry_Audit_Colab_STANDALONE.ipynb`, `CIDM_v2_B1_R4_P0_Frozen_BER_Causal_Geometry_FULL.zip`, and locally executed report.

## Next training decision
Do NOT add a new BER module or simply increase normalized causal weight. The newly identified gap is correct *direction* of post-event response under autonomous rollout and different internal dynamics. A future trainable candidate must prequalify multi-lead direction and free-rollout state compatibility on genuinely new dynamical family, using the same-known-port Readout baseline; no future observation at inference. No global PRO6000/AI Weather Quest claims. Original eight-PDE v1-superiority and long-horizon gates remain unmet.