# CIDM-v2 B2-R10/P0 actual user-run result and FIX1 numerical gradient audit repair (2026-10-14)

## Original uploaded report (actual execution; not full result)
- Original ZIP SHA256 `a17f7694441bd5de14d8e2c51a6363994f578476d978c2de2ccf9a6c666d6eab`, matches user sidecar, ZIP CRC PASS.
- Three official PDEBench source MD5s were recorded as PASS. Parent full B0-R1 weights sourced from HF Model. 2/3 PDE tasks complete: Burgers and Reaction–Diffusion, each Natural and Rotated controller, horizons K=1/3/12; 12 valid rows in `gradient_credit_horizons.csv`.
- Original final state: **`BLOCKED_OFFICIAL_GRADIENT_CREDIT_NO_TRAINING`** because Shallow-Water triggered `NONFINITE_HEAD_GRADIENT` in `frozen_gradient`; no official `report.json` or overall gradient gate. B2-R10 therefore **NOT qualified**.
- Burgers Natural K1 cosine against K12 = 0.640789, K3 = 0.685239, K3 relative L2 error=0.868754.
- Reaction–Diffusion Natural K1=0.024791, K3=0.191175, K3 relative L2 error=0.982607. Reaction–Diffusion Rotated K1=0.913303, K3=0.939020. These are gradients for two TRAIN windows, not a predictive-skill result.
- Original HF/GitHub sync JSON reports SUCCESS for ZIP/source/release/status even though the run itself was blocked; availability of a ZIP does not imply a passed scientific gate.

## Reproducible numerical concern
The original gradient-only code computed `sqrt(mean(x**2))` followed by `clamp_min` for operator basis and normalization. On **exact zero** tensors in PyTorch this yields a finite forward but `nan` backward due to `sqrt(0)` derivative singularity. Demonstrated by direct PyTorch CPU reproduction. This is a plausible cause, **not conclusively proven root cause of actual SWE run**: other effects such as gradient explosion remain possible.

## Delivered B2-R10/P0 FIX1 self-contained Colab Notebook
- `CIDM_v2_B2_R10_P0_FIX1_StableGradient_Colab_STANDALONE.ipynb`, new source `b2_r10_gradient_credit_audit.py` SHA256 `d636c9e122f708ad1abbd1b48a0979be84db3e3bac3f7120b6975738015a327d`, all 8 original source modules and actual B2-R6/R7/R8/R9 evidence still embedded and hash locked; B0-R1 original ZIP SHA locked.
- In gradient **diagnostic only**, use `sqrt(clamp_min(mean(x²), 1e-30))` then restore exact zero forward, for basis norms and per-sample RMS. No core architecture modifications, no gradient clipping/substitution. The original R9 `rollout_timing` is still the zero/finite forward reference, strict parity <= 2e-6.
- Explicit zero-vector backward regression tests, per-parameter nonfinite localization and per-task numeric failure evidence; fail closed if SWE or other case remains nonfinite. Prior verified B2-R10 data cache is included in discovery to avoid ~28 GB unnecessary redownload.
- Distinct `B2_R10_P0_FIX1` HF Model/GitHub Results/Source/Release archive paths, automatic browser ZIP/checksum download with write tokens, **no Google Drive**.

## Verified execution
- **Every code cell in new FIX1 Notebook** executed in CPU smoke mode PASS; real original six R6 heads and all stored prior evidence SHA checked, both synthetic 1D and 2D paths, both heads K1/K3/K12, zero-singularity backward, exact forward value tests, original R9 parity, and packaged results ZIP CRC/SHA PASS.
- **Real three-PDE FIX1 official gradient audit has NOT been run yet**. Only the user's new execution can establish full B2-R10 qualification.

## Project stage
CIDM-v2 has several credible local-operator geometric and causal-perturbation **mechanism-level signals**, but has NOT demonstrated a validated universal performance edge over frozen CIDM-v1 across Burgers, RD and SWE. Original B0 v2 performed worse for Burgers/SWE and better for RD. Original B2-R1 Flux-cap gate, B2-R3 latent-readout cross-family gate and B2-R6 causal-head full-training gate did not pass. The main unresolved issue is scalable long-horizon credit assignment with robust geometry, trained/validated across independent data and seeds. No official full CIDM-v2 retraining, unseen outer-test or AI Weather Quest RPSS qualification authorized.
