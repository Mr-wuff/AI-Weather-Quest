# CIDM-v2 B2-R4/P0 actual result and B2-R5/P0 spatial-operator-span preregistration

Date: 2026-10-14. **Do not infer deployable skill from future-truth Oracle diagnostics.**

## User-provided official B2-R4 artifacts
- Report ZIP `CIDM_v2_B2_R4_P0_TANGENT_CONTROLLABILITY_REPORTS.zip` SHA256 `32fd05222848c8ffeb62bb1c558525598f60625f4139a61e514c3056b5c739e3` equals uploaded sidecar; ZIP CRC PASS.
- `status=FROZEN_TANGENT_DIAGNOSTIC_COMPLETED_NO_TRAINING`, 9/9 unchanged official CIDM-v2 parent test replays, 45/45 task×seed×lead groups, 360 label-conditioned val sample diagnostics, zero trained parameters/optimizer updates.
- Original parent binary SHA256 `69552b29e169ec22ee85d9c1b1a678fdb188f6f25dc18f33b927b88f4dddffd4`. HF/GitHub sync status JSON records success for model ZIP, sidecar, source and release.

## Scientifically important B2-R4 metrics (24 samples per PDE×lead)
The archived report *averages samplewise MSE ratios*, not `sum(corrected MSE)/sum(baseline MSE)`; both must be shown.
- Burgers lead1 Full-gradient 5%: mean sample ratio **7.466**, pooled ratio **0.859**, p90 sample ratio **25.714**, only **14/24** samples improved. This is a high-risk distribution even when total loss is lower.
- Burgers lead12 Full-gradient 1% mean sample ratio **0.849**; 5% mean ratio **1.081**, pooled ratio **0.883**. Global operator projection of gradient at lead12 **2.18%**.
- RD lead12 Full-gradient 5% mean ratio **0.736**, pooled **0.724**; global 3-direction gradient capture **0.62%**.
- SWE lead12 Full-gradient 5% mean ratio **0.924**, pooled **0.965**; global 3-direction gradient capture **0.060%**.
- **No training qualification:** a low gradient-projection fraction in 3 global vectors is strongly confounded by extremely low subspace dimension. The observed full-gradient gains used validation future truth.

## New B2-R5/P0 experiment delivered to user as complete self-contained Colab Notebook
Name `CIDM_v2_B2_R5_P0_SpatialOperatorSpan_Qualification_Colab_STANDALONE.ipynb`, Python `b2_r5_local_geometry.py`. Source: seven exact SHA256 verified upstream B2-R4 modules plus new script, embedded with checks before imports. Model weights are not reconstructed; original full 18-run B0 parent is retrieved from HF Model `wuff-mann/AI_Weather_Quest` or GitHub release; one-time upload only if necessary. No Google Drive.

- Q0 frozen analysis ONLY: Burgers/RD/SWE, seeds 17/41/73, fixed lead 1/3/6/9/12, eight val windows **8–15**, not B2-R4 windows 0–7. Prior B0 checkpoint selection means neither validation group is fresh *outer* held out data.
- Five diagnostic moves: Full-gradient, **global** 3-operator span, **per-gridpoint** 3-operator span, per-gridpoint **effective-rank-matched** random span, random ambient perturbation. Diagnostic gradients and best-of-budgets line search use future validation labels **only after frozen baseline**; these are not forecast policies.
- Predetermined latent RMS budgets 0.1%, 0.3%, 1%, 3%, 5%. Distinguish pooled error ratio (primary), macro mean sample ratio, win fraction and 90th-percentile risk.
- Matched random per-gridpoint control uses the operator's *local Gram-Schmidt active-mask* so the random basis has the same effective rank per gridpoint. Output local operator mean rank and projected gradient fractions for interpretation.
- Strict official MD5/splits, nine archived checkpoint test RMSE replays, original-output forward parity, all 45 groups / 360 samples. Failure blocks; no new models trained.
- On Colab, optionally publish full results ZIP, SHA256, new source, and status to HF **Model** and GitHub Release/contents using explicit write tokens; always export browser downloads and sync manifest.

## Program validation and next Gate
Complete notebook smoke executed on CPU with both 1D and 2D data, 10 groups/20 records; all eight exact modules hashed, ZIP/CRC/SHA256 passed. **Official B2-R5 has not been run.** Only after comparing natural local span vs rank-matched random local span on the official nine checkpoints should a tiny trainable *issue-time causal* local-coefficient probe be considered. B2-R1 Flux-Cap and B2-R3 universal latent-readout pilots previously failed; across-PDE v1 superiority and AI Weather Quest RPSS gates remain closed.
