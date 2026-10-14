# CIDM-v2 B2-R8/P0 official results and B2-R9/P0 causal activation timing pre-registration (2026-10-14)

## Source integrity and execution
User-provided `CIDM_v2_B2_R8_P0_EQUAL_RMS_CAUSAL_CONTROL_REPORTS.zip`: SHA256 `ad8334d8ac0632ccd29965305bc613d6628fdcc89b34906ba30ca64604947159`, sidecar match and ZIP CRC PASS. Frozen audit `FROZEN_EQUAL_RMS_AUDIT_COMPLETED_NO_TRAINING`: three original seed17 archived checkpoint RMSE replays PASS, three official dataset MD5 PASS, core forward max diff 0, 345 curve rows, 5,040 validation window MSE rows, six genuine B2-R6 heads frozen. All runtime HF Model / GH release/source/status uploads report SUCCESS in user `repository_sync_status(5).json`. No new training.

## Main preregistered 0.3% equal-realized-RMS budget, pooled late-lead (9+12) MSE
| Task | Natural gain vs baseline | Natural gain vs equal-exposure Rotated | Worst pooled lead1/3 regression | Per-window early tail |
|---|---:|---:|---:|---|
| Burgers-1D | +2.4140% | +1.7380% | +0.2197% | 7/16 windows >2% regression at lead3 (archived risk contract) |
| Reaction-Diffusion-2D | +7.2365% | +4.8963% | +0.3557% | 8/16 windows >2% at lead3 |
| Shallow-Water-2D | +1.8828% | +1.4221% | +0.0449% | 0/16 >2% at lead3 |

Same sample, same step, Natural vs Rotated actual latent RMS matches at 0.003 to numerical tolerance. This is stronger direction evidence than unmatched multiplier in B2-R7, but cannot be generalized from one trained head seed and pre-used validation data.

Using paired **window** resampling (5,000 draws) on 16 potentially correlated windows, **DESCRIPTIVE ONLY** 95% bootstrap intervals for Natural late improvement over equal-RMS Rotated: Burgers [-1.78%, 3.97%], RD [3.49%,5.95%], SWE [1.17%,1.66%]. Not valid independent-sample inferential significance or external model skill. Burgers result is fragile.

## New B2-R9/P0 experiment (delivered in user chat; OFFICIAL NOT YET RUN)
- `CIDM_v2_B2_R9_P0_CausalActivation_EqualEnergy_Colab_STANDALONE.ipynb`, new source `b2_r9_activation_timing.py`, complete research ZIP, true B2-R8 review in Chinese, checksums. SHA-lock exact eight prior source modules, six actual trained B2-R6 heads, actual B2-R7 and B2-R8 full archives, plus R8 source.
- Original 18-run B0 parent exact SHA256 `69552b29e169ec22ee85d9c1b1a678fdb188f6f25dc18f33b927b88f4dddffd4` retrieved HF **Model** or GH Release first; official HDF5 MD5 and per-window original baseline/0.1%/0.3% replay required. Reuse B2-R8 / B2-R7 authenticated local official data if available; no Google Drive.
- **Zero-optimizer frozen counterfactual:** activate the existing causal Natural or Rotated controller at known rollout lead 1 vs 4 vs 7, primary fixed 0.3% and secondary 0.1% per-step latent RMS. Delayed policies tested with (a) unchanged per-active-step amplitude, versus (b) **equal total 12-step squared relative RMS energy** via per-step amplitude `b*sqrt(12/(13-onset))`. Source implements per-sample equal exposure across Natural/Rotated and asserts identical baseline output until the scheduled onset. Labels used ONLY to score post-forecast.
- Diagnostic Gate (not training permission): pooled late leads 9/12 improvement >=1.5% vs baseline and >=1% vs matched Rotated, max lead1/3 pooled ratio<=1.005, and max fraction of late windows regressing >2% <=20%. No validation-only selection of deployed onset or budget. Original B0 checkpoint reused same VAL, so **not fresh outer holdout**.
- Outputs curves, 16-window paired risk and per-sample MSE, stringent gates and partial progress, ZIP SHA256 and HF/GH attempts. Requires new outer validation / many training seeds before escalation, no official AI Weather Quest RPSS.

## Program tests
Whole 12-cell notebook executed (CPU smoke), passed all original/driver source hashes and true six head loads, 1D/2D causal 12-step rollouts, zero pre-onset forecast discrepancy and equal actual per-step RMS across Natural/Rotated. 12-step cumulative energy matches baseline-onset: fixed 0.3% total energy 12×0.003²=0.000108; onset 4/7 matched-energy versions also =0.000108, while fixed-step versions have lower energy. Local archive SHA/CRC pass. **Official B2-R9 not yet executed.** B2-R6 single seed original long training Gate FAIL and CIDM-v2 full v1-superiority gate remains closed.
