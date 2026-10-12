# CIDM-v2 B0 — exact CIDM-Complete vs channel-ported v2 PDEBench qualification

Date: 2026-10-12. The project remains **CIDM-v2**. No CIDM-v3 is proposed.

## User-defined v2 graduation criteria
- Resolve the original v1 limitations in long rollout, latent state drift, robust synchronization, generalization and external information handling, with positive mechanistic controls.
- Beat the original CIDM-Complete on **every** selected public PDE benchmark in independently re-trained, same-data, same-budget evaluation, across seeds. Then compare to pinned official FNO/UNet/Transolver models and require top-tier rank.
- Only after PDE qualification: issue-time causal AI Weather Quest data and official hindcast RPSS versus proper operational baselines, before any PRO6000 scale spend.

## B0 benchmark supplied as standalone ChatGPT Colab Notebook
- Exact v1 archived executable CIDM_BER source SHA256 `f9b1cf4ff7630f5540bb3adde20e4c4b81fff3218d9325ea3105bf28b1d97665` (not reconstructed or reinterpreted).
- v2 from R5/R1 flux-restoration backbone, transparently **ported** from 2-channel synthetic grid to 1/3/4-channel and 1D lifted grid, without SOC, no extra simulator forcing. Unused synthetic-only auxiliary scale head removed. All v2 PDE checkpoints train afresh.
- Parameters matched to 500,000 +/- 5%, actual architecture widths and parameter counts stored in data_and_architecture_contract.json.
- Original Foundation A2 DaRUS PDEBench eight task URLs and MD5 hashes; identical train/val/test trajectory IDs, windows, train-only moments, fixed LR, optimizer updates, per-step train sample indices for both models, and 6/12-step RMSE.
- Profiles: pilot 3 public PDE / 1 seed / 120 updates; formal8 8 PDE / 3 seeds / 600 updates; canonical8 8 PDE /5 seeds /1000 updates; smoke synthetic and always diagnostic-only.
- Resumable optimizer-complete checkpoints, per-30-step reports, best model exports, paired-seed summary, GPU latency/VRAM, v2 frozen Flux-dependence check, zip, SHA256. Prior to formal benchmark data download, local complete Notebook CPU smoke passed; **no actual PDE performance result is claimed**.
- Caveats: pilot/formal8 short training is not proof of full convergence. Historical v1 published RMSE uses different windows, so only the new paired run supports numerical improvement. B0 does not rerun external official baselines, does not establish long 24/48-step property, and is not AIWQ qualification.

Artifacts are attached in the ChatGPT conversation as `CIDM_v2_B0_OriginalComplete_vs_V2_Fair_PDEBench_Colab_STANDALONE.ipynb` and `CIDM_v2_B0_Fair_PDEBench_Qualification_FULL.zip`.