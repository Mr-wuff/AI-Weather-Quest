# CIDM-v2 B2-R10/P0 FIX1 official results and FIX2 FP64 gradient-condition audit — 2026-10-14

## Actually uploaded FIX1 report

- ZIP SHA256 `fc9cd9c22af49948aaf34cc9ef39729959e6a4829c8e4cd87bc591d93baa18ad`, sidecar match, CRC PASS.
- Report final state `BLOCKED_OFFICIAL_GRADIENT_CREDIT_NO_TRAINING` because `write_json(gradient_qualification_gate.json)` rejected an `inf`: `ValueError: Out of range float values are not JSON compliant: inf`. This is **not** an officially completed scientific gate.
- Original parent sourced from HF Model, official PDEBench three HDF5 MD5 checks PASS. All three tasks ran and wrote **18 horizon records** and 6 paired records. Source/control head forward replay max diff 0 in all 18.
- The FIX1 source checked each gradient *element* for nonfinite values then computed `gradient_l2=float(torch.linalg.vector_norm(vec))` on an FP32 vector. FP32 norm reduction may overflow even when all elements are finite. For SWE Natural K12 the logged norm is `inf`, whereas SWE Rotated K12 is finite but very large (~9.188753e13). Do **NOT** infer that full gradient is numerically well-conditioned.
- Horizon gradient cosine K1/K3 relative to K12:

| Task / head | K1 | K3 |
|---|---:|---:|
| Burgers Natural | .660690 | .703934 |
| Burgers Rotated | .650415 | .652122 |
| Reaction–Diffusion Natural | .024792 | .191177 |
| Reaction–Diffusion Rotated | .913302 | .939019 |
| SWE Natural | .005669 | -.014393 |
| SWE Rotated | .344659 | .121550 |

K3 is **not uniformly better**; SWE increases divergence and exhibits enormous horizon sensitivity. These gradients are TRAIN-window sensitivity geometry, not improvements in forecast error. Only two training windows per task, seed17 controller.

## FIX2 released in this ChatGPT turn

New self-contained `CIDM_v2_B2_R10_P0_FIX2_FP64GradientConditioning_Colab_STANDALONE.ipynb` plus Python source `b2_r10_gradient_credit_audit.py` SHA256 `744dba22c10435e21f5040b5399dd5bddd7b8752b234a61f0829cf27da27b6b1`.

- Preserves all exact source modules, archived real R6 heads, R7/R8/R9 evidence, and embeds the actual FIX1 blocked ZIP (SHA256 pinned).
- Same 12-step autoregressive **FP32** forward/backward, K=1/3/12, original model and controller parameters frozen; no gradient clipping, optimization, trajectory or loss changes.
- **After** backward, FP64 computes L2, RMS, extrema, per-parameter gradient health and K12/K3 norm amplification. Checks whether old FP32 norm would overflow; maintains strict JSON (NaN/inf forbidden), flags extreme-conditioning diagnostics and blocks any later training automatically.
- Original 18-run B0 parent SHA256 `69552b29e169ec22ee85d9c1b1a678fdb188f6f25dc18f33b927b88f4dddffd4`, official PDEBench fixed MD5, checkpoint replay fail closed. HF **Model**/GitHub Releases remote-first; one-time upload only if original parent genuinely absent. No Google Drive.
- Full 13-code-cell Notebook executed **CPU program-only smoke PASS** using original R6 trained heads and actual archived results. Includes zero backward singularity, an FP32 norm-overflow vector regression test, 1D+2D K1/K3/K12 forward parity and stable FP64 output fields, ZIP SHA/CRC. Official FIX2 **not executed**; SWE K12 FP64 magnitude and layer amplifiers remain to be measured.
- Intended next step after official FIX2: inspect severe SWE long-horizon gradient condition and sign instability before authorizing any BPTT(3) pilot. CIDM-v2 fair v1 PDE superiority, new outer-holdout, multi-seed and AI Weather Quest RPSS gates are closed.

No models were trained by this work.