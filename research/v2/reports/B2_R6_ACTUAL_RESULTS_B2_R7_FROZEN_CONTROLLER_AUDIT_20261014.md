# CIDM-v2 B2-R6/P1 actual causal-head result and B2-R7/P0 frozen controller audit (2026-10-14)

## Integrity
User supplied `CIDM_v2_B2_R6_P1_CAUSAL_LOCAL_COEFFICIENT_REPORTS.zip`, SHA256 `d923eb6af43c294f28ec4f236c707feb0c4bcf1e6e5fed519bf697599730ef89`, exact uploaded .sha256 match, zip CRC PASS and all 26 inner manifest entries hash-verified. Status `PILOT_SINGLE_SEED_FAIL_NO_FULL_TRAINING`. Official MD5 and seed17 parent checkpoint RMSE replay passed in original report.

## Actual seed17 causal pilot
Three PDE families, 60 optimizer steps per matched 4.5–4.8k-param coefficient head; *full CIDM-v2 backbone frozen*, no full training. Natural learned local coefficients based on issue-time/predicted-state features versus fixed-orthogonally-rotated same-local-Gram operator control. VAL 0–7 checkpoint selection, VAL 16–31 audit; original parent had been validated on this dataset, so these are not untouched outer holdouts.

| PDE | natural late MSE/base | rotated late MSE/base | natural late MSE/rotated | best step natural/rotated |
|---|---:|---:|---:|---|
| Burgers-1D | 0.999011231 | 0.999149788 | 0.999861326 | 20 / 60 |
| Reaction–Diffusion-2D | 0.984169268 | 0.999054758 | 0.985100426 | 60 / 60 |
| Shallow-Water-2D | 0.991214193 | 0.997876550 | 0.993323466 | 40 / 40 |

RD has a useful single-seed candidate signal but not a replicated cross-PDE success. Original preregistered Burger/SWE improvement >=3%, vs rotated >=1%, early regression <=2%, RD <=2% regression did **not** pass. Therefore no 3-seed long-train escalation and no official weather competition eligibility.

## Research interpretation
B2-R5 future-label Oracle showed operator-span geometric utility; B2-R6 shows converting that into a causal learned coefficient policy is **harder**, with tiny Burgers advantage and modest RD/SWE. The result does **not** establish a specific root cause. Possible axes are underused bounded 1% latent RMS intervention, erroneous spatial coefficient direction, and BPTT(1)-limited temporal credit assignment. Best training step at 20/60 for Burgers versus 60/60 for RD, so blindly extending all training is not a justified universal fix.

## Next delivered B2-R7/P0 (no training)
A complete standalone Colab Notebook and package were delivered in ChatGPT: `CIDM_v2_B2_R7_P0_FrozenControllerSensitivity_Colab_STANDALONE.ipynb` + `CIDM_v2_B2_R7_P0_Complete_Research_Delivery.zip`. The Notebook SHA-locks eight original source modules, exact B2-R6 controller source and the full original B2-R6 evidence zip including all six trained heads. SHA-pinned HF Model/GitHub Release discovery for original 18-run B0-R1 parent; official MD5 and parent checkpoint exact test-RMSE replay; no Google Drive.

Freeze parent and trained coefficient heads. Use validation audit windows 16–31 only; compare fixed signed gains `0/0.5/1/2/4` with positive/negative direction across natural and rotated heads, audit lead1/3/6/9/12 per-task physical MSE and actual relative latent update RMS, gate magnitudes, clipping frequency, sample tail. Exact 1× replay of B2-R6 archived validation MSE must match before any interpretation. Test set **only** replays original B0 model performance. No model updates, no new hyperparameter deployment selection. Full-gradient-oracle skill is never substituted for issue-time predictions.

Program tests: all nine executable notebook cells CPU smoke PASS; existing real B2-R6 trained heads are loaded and exercised on 1D/2D synthetic cases, forecast parity check, output CSV, zipped report and CRC/SHA256 PASS. Official frozen B2-R7 study remains unexecuted.

Remote runtime, with valid credentials, will attempt GitHub source/Release/status + HF **model** results/source. Lack of write scopes is logged rather than misreported. No AI Weather Quest RPSS qualification.