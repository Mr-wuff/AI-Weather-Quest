# CIDM-v2 B2-R11/P0 official result and FIX1 strict gradient-partition numerical audit — 2026-10-14

## Actual user-run R11 evidence
- Uploaded ZIP `CIDM_v2_B2_R11_P0_BACKWARD_PATH_ATTRIBUTION_REPORTS.zip` SHA256 `713d3c327fdaad77ae97ffe18f7c9bae9e9436d8143f88a5ea25a20d94a67e46`, sidecar match, CRC PASS.
- Status `BLOCKED_OFFICIAL_BACKWARD_ATTRIBUTION_NO_TRAINING`. Three PDEBench source MD5 PASS. Burgers-1D and Reaction–Diffusion-2D exact seed17 checkpoint test-RMSE replay PASS.
- 4/6 head groups complete, 48/72 temporal and 48/72 operator local VJP rows; no `case_summary.csv`, final `report.json`, or qualification Gate because `swe_2d operator` failed `HEAD_PARTITION_GRADIENT_FAILED` with relative reconstruction error **0.10675963707489294** vs prespecified **1e-4**.
- Four successful groups' local VJP reconstruction error <= **5.03e-8**. Burgers Natural lead1 identity VJP L2 `0.084617`, Flux `31.80013`, Local `0.00381`, Skew `0.00462`. This is frozen local sensitivity geometry, NOT SWE root-cause identification, Jacobian spectral norm, or physical causal ablation.
- Existing B2-R10 FIX2 SWE Natural 12-step controller gradient FP64 norm `9.04094e19`: extremely ill-conditioned. The new ~10.68% difference may be numerical accumulation/cancellation or graph-replay inconsistency; evidence does NOT yet distinguish them. Prior official R11 did not log the SWE step-level partition or its cancellation condition.
- Runtime sync manifest shows HF ZIP/checksum/source and GitHub source/release/status SUCCESS; this only archives a blocked report.

## New FIX1 self-contained rerun (delivered with Notebook/ZIP in user chat)
`CIDM_v2_B2_R11_P0_FIX1_GradientPartitionNumerics_Colab_STANDALONE.ipynb`.
- Keeps exact original FP32 forward trajectory, 12-step loss, backward, original model/core, B2-R6 six trained heads, B0 full parent SHA256 `69552b29e169ec22ee85d9c1b1a678fdb188f6f25dc18f33b927b88f4dddffd4`, official data MD5, strict 1e-4 numerical Gate, zero optimizer steps, no Google Drive.
- Separately computes sum of 12 FP32 local head gradients reduced in FP64 / FP32, and each local controller head VJP re-evaluated in FP64 with original frozen FP32 cotangent. Records head FP64 replay difference, FP32/FP64 gradient discrepancies, numerical residual, and gradient cancellation ratio `sum(||g_t||)/||sum(g_t)||`.
- **A strict reconstruction failure remains FAIL**, but nonblocking diagnostics can continue to capture 6 head and 72+72 time/operator rows for root-cause isolation (subject to the separate local VJP strict Gate). No long training permission.
- True three-PDE official FIX1 experiment **not run**. User must execute Colab for actual Shallow-Water result.
- Full Notebook CPU smoke ran ALL code cells PASS with **1D Natural** and **2D Rotated** real archived B2-R6 trained heads on synthetic data. New reconstruction errors ~8.98e-8, 4.98e-8; independent double local-head replay errors ~1.13e-7, 7.73e-8. ZIP+CRC+SHA verified. Smoke is not external official data.
- HF **Model**/GH Release original parent automatic discovery, one-time upload if absent; runtime archive code and browser downloads preserved.

## Gate status
No CIDM-v2 full training, no approved BPTT(3), no unseen outer holdout/multiseed-v1 superiority, no AI Weather Quest official RPSS qualification. Next experiment after real FIX1 may require FP64 staged graph recomputation or controlled Jacobian/normalization VJP stability diagnosis.
