# CIDM-v2 B2-R2/P0: observable alignment frozen diagnostic (2026-10-13)

## Prerequisite verified on GitHub
- B0-R1 formal PDEBench (three families, seeds 17/41/73) archived v1/v2 horizon-12 mean RMSE: Burgers 0.094384/0.130207, Reaction-Diffusion 0.010051/0.005508, SWE 0.008519/0.030632. v2 **not** qualified across PDEs.
- B2/P0 lead-12 Flux RMS: Burgers 43.2001, RD 2.4985, SWE 24.0586; corresponding Local RMS 0.0506/0.0375/0.0225. Large latent sensitivity is not evidence of equivalent observable error or physical conservation.
- User-run B2-R1 FIX2: nine original checkpoint RMSE replays PASS, original/wrapped outputs exactly equal, no Q1 training. Frozen Q0 failed: best unified cap4 geometric validation MSE ratio 0.981674 (1.83% improvement <3% gate), Burgers family ratio 1.013398 and 1/3 paired seed wins. Do not keep tuning Flux cap values.

## Proposed B2-R2/P0 — frozen, read-only, NO training
Continue the exact FIX2 canonical evaluator and 18-model B0-R1 parent, never reconstruct a different model from generic CIDM reference.
1. Pin original parent B0 ZIP SHA256: `69552b29e169ec22ee85d9c1b1a678fdb188f6f25dc18f33b927b88f4dddffd4`. Verify ZIP CRC and checkpoint count before execution.
2. Strictly reuse original PDEBench official data/MD5 and repaired `b2_r1_flux_control.py` scorer (avoid batch=2 Tensor-unpacking pitfall).
3. Observe frozen Flux/Local/Skew/decoder through read-only hooks at steps 1,3,6,9,12. Reconstruct effective latent update from actual operators. If additive decomposition residual >10% or ill-conditioned, mark component attribution invalid.
4. For identifiable decomposition, compute each component's **decoder-space** effect from controlled latent counterfactuals. These are NOT automatically true observable field increments.
5. When original encoder API permits, re-encode only rolling **model-predicted** history and measure functional/decoder discrepancy; never feed future truth to the model.
6. Fail closed for missing/incompatible source, absent verified checkpoints, invalid official replay, incomplete 3 family × 3 seed coverage, or attempted optimizer step. B2-R2 never unlocks formal training by itself.

## Repository persistence (no Google Drive)
- HF target is **model** repo `wuff-mann/AI_Weather_Quest` (not a Dataset repository). Auto-discover parent/source there and from *actual* GitHub Releases.
- GitHub HEAD currently contains compact B2-R1 result JSON but no binary checkpoint in that JSON; GitHub Releases query returned empty as checked at this handoff.
- If remote full assets are absent, a **single verified upload** of B0 ZIP and FIX2 source is required before HF Model and GitHub Release archival. Need write-scoped runtime `HF_TOKEN` and `GH_TOKEN`; ChatGPT's linked HF scope is read-only.
- Results: local ZIP+SHA256 via Colab download, model artifacts/results to HF, compact source/report GitHub Contents and binary GitHub Releases. Only confirmed server success counts as synced.

## Delivery and validation status
- Created self-contained notebook `CIDM_v2_B2_R2_P0_Observable_Dynamics_Colab_STANDALONE.ipynb`, observer and repository-sync utilities, plus a detailed Chinese research report/ZIP delivered in the current conversation.
- Executed complete notebook CPU `PROGRAM_ONLY_SMOKE`; observer unit test additive reconstruction residual ~1e-6, with automatic smoke ZIP. This is **not official PDEBench evidence**.
- B2-R2 full official replay/diagnostics remain unexecuted pending original verified parent weight ZIP and FIX2 driver source. A source interface compatibility gate may still require adaptation to exact runtime code; do not substitute synthetic data.
- No new architecture qualified and no AI Weather Quest RPSS investment release.
