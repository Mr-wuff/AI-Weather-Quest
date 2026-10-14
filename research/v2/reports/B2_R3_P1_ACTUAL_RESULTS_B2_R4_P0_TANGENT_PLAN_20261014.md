# CIDM-v2 B2-R3/P1 actual pilot results and B2-R4/P0 diagnostic preregistration (2026-10-14)

The user-provided report archive SHA256 is `21845870ce91a2492b1021053e0e061d90eb9f582259f318910015c728877024` and ZIP CRC PASS.

## Verified actual pilot
- `status=SINGLE_SEED_PILOT_COMPLETED_NOT_FULL_TRAINING`; `pilot_gate=PILOT_NO_UNIFIED_LATENT_SPECIFIC_GAIN`.
- Seed 17 only, 100 updates per same-parameter OBS/LATENT head per family, frozen original core, baseline archived RMSE replay on test PASS. 16 validation windows for step selection and 16 for separate probe audit; every best step = 100 (convergence unproven).
- Late (leads 9+12) MSE: Burgers baseline .039372319, OBS .039172299, LATENT .039697820. OBS improves 0.508%, LATENT regresses 0.827%; latent-v-obs -1.34%.
- RD baseline .00006508129, OBS .00005032729, LATENT .00004489073. OBS improves 22.67%, LATENT 31.02%, and LATENT beats OBS by 10.80%.
- SWE baseline .001573960, OBS .001258365, LATENT .001293502. OBS improves 20.05%, LATENT 17.82% (LATENT worse than OBS 2.79%). Both OBS/LATENT regress first-lead MSE by ~19.9%.
- No general latent-specific advantage and no permission for full retraining, v1 superiority, or AI Weather Quest RPSS.
- Outer val is not unseen relative to the original B0 model checkpoint selection.

## Next: B2-R4/P0 frozen Decoder Tangent and Operator Span Controllability audit
Deliverables created in ChatGPT: `CIDM_v2_B2_R4_P0_Frozen_Tangent_Controllability_Colab_STANDALONE.ipynb`, self-contained sources and ZIP, Chinese science review and SHA256.

Preregistered design:
- Pin original 18-run B0 parent SHA256 `69552b29e169ec22ee85d9c1b1a678fdb188f6f25dc18f33b927b88f4dddffd4`; original four FIX2 source hashes checked. HF Model / GitHub Releases first, one-time upload only if unavailable; no Google Drive.
- Nine exact original v2 checkpoints and 12-step historical test RMSE replay required (strict 0.1% or 1e-5 tolerance), plus exact forward parity. Test must not be used for oracle diagnostics.
- Q1 no optimizer, no training. Three PDE × 3 seeds × five leads (1,3,6,9,12) × first eight validation windows = 45 groups, 360 sample-lead rows.
- From frozen latent z compare three **validation truth-informed diagnostic oracles**: (a) steepest descent latent perturbation in the full space, (b) steepest descent projected into 3-vector per-sample span of Flux/Local/Skew effective updates, and (c) equal-size random perturbation. Budgets 1% and 5% of latent RMS (minimum 1). Record corrected diagnostic loss ratios, saturation, gradient RMS, cosine and gradient projection fraction.
- **Future truth is used only to calculate retrospective oracle gradients**; it is NOT available at operational issue-time, NEVER fed into baseline or subsequent rollout. Hence diagnostic oracle scores are not forecasts nor evidence of deployable skill.
- Full-gradient effective / operator-span ineffective would motivate a separate learnable causal dynamics direction Q0 trainable pilot; both ineffective might point to decoder reachability/budget/saturation, requiring further audit. No automatic trainable Gate.
- Entire notebook all executable cells passed locally in CPU `CIDM_B2R4_MODE=smoke` with both 1D/2D data and 20 synthetic diagnostic rows, ZIP+SHA256 exported. **Official B2-R4/P0 has not run**.

Maintain strict scientific status: B2-R1 flux cap Q0 FAILED, B2-R3 latent specific Q0 FAILED, CIDM-v2 8 PDE and AI Weather Quest competition gates NOT passed.
