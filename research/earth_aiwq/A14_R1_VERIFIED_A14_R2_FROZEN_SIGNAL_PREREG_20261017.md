# CIDM-MHC A14-R1 official uploaded evidence → A14-R2 frozen-signal falsification
2026-10-17

## Verified A14-R1
- User result ZIP SHA256 `4c3b3fbce8a64b308bf271abf0238b06ad92e50cd2a94aaa38b383ed715d0986`; SHA sidecar and ZIP CRC PASS; actual Notebook all 7 code cells executed.
- 2008–11 training: 102 issues; 2012 validation: 24; 2013–14 repeatedly observed **development**: 50; 2015 historical inputs only: 26. WB2 1.5° ERA5 retrospective proxy; coarse 3×4 outputs; **NO** official probability RPSS, NO CIDM neural optimization and NO issue-time operational availability audit.
- Original shared-alpha selector returned `alpha=0` in every model. Headwise rescoring produced nonzero but fragile improvements.
- Headwise dev MSE improvement **versus climatology** in order W3 MSLP, W3 TAS, W4 MSLP, W4 TAS:
  - surface: `+0.1764%, -0.1624%, -0.5912%, 0%`
  - surface+Z500: `+0.6057%, +0.1628%, -0.2427%, 0%`
  - surface+SST: `+0.2554%, +0.3917%, -0.6237%, 0%`
  - surface+both: `+0.4553%, +0.4530%, -0.7687%, 0%`
  - wrong-time sham: `+0.3268%, -2.0012%, +0.0569%, 0%`.
- **Increment over paired surface**:
  - Z500: `+0.4301%, +0.3247%, +0.3465%, 0%`.
  - SST: `+0.0792%, +0.5532%, -0.0323%, 0%`.
  - Both: `+0.2794%, +0.6144%, -0.1764%, 0%`.
- W4 TAS always `alpha=0`. Pre-registered A14-R1 gate fails: weak gains do not jointly survive both windows and placebo. **No architecture or long training authorization.**

## A14-R2 pre-registration: fixed hyperparameters and falsification, no re-tuning
- Embed the exact original user A14-R1 ZIP in a standalone Colab notebook and refuse any SHA drift.
- Freeze all four A14-R1 headwise per-model `alpha`/`lambda` values; refit on train+val 2008–12 only as originally allowed, evaluate already-inspected dev 2013–14 (not fresh blind).
- No new physics features or neural models. Compare model predictions against the same paired surface case errors.
- For each of Z500, SST, BOTH: 128 year-and-season matched time-permutation placebo evaluations with **frozen weights**; compute one-sided permutation p-values against the observed aligned gain.
- Statistical feasibility fix: 64 permutations cannot attain p≤0.0125 (minimum p=1/65). Frozen contract uses 128 permutations (minimum p=1/129), validated through entire offline Notebook.
- 1000 year-stratified four-biweekly-issue moving-block paired bootstrap for 95% CI; report each year independently, each variable and each lead.
- Exploratory mechanism signal requires one target variable W3 and W4 both improve >=0.5% vs surface, permutation p<=0.0125 per head, year-wise gains positive, both block-bootstrap 95% lower bounds positive, and no other head regresses >2%. Passing is only a candidate, not permission for formal training.
- If not established: end repeated tiny Ridge/shrinkage exercises; shift to time-available S2S hindcast/reforecast input and a properly competitive probability baseline before investment.
- Original 2013–14 has already been used extensively. It cannot be promoted to an untouched holdout regardless of bootstrap outcome.
- Workbook includes synthetic positive-signal smoke and fail-closed missing data path; entire 10-code-cell Notebook synthetic mode passes offline. Actual A14-R2 WB2 online experiment **NOT** executed as of this preregistration.
- Research delivery ZIP SHA256 `d79981288f12f04371737dfd7c4c3e62bfbcf7caa1370c861661898612cf3074`.
- Notebook `CIDM_MHC_REAL_A14_R2_Stability_Falsification_Colab_STANDALONE.ipynb`, source `a14_r2_stability_falsification.py`, SHA-lock A14 parent source and A14-R1 evidence all within the ChatGPT delivered ZIP.
- GitHub/HF lightweight runtime sync opt-in; NEVER publish WB2 ERA5 caches or user secrets.

## Competition guardrail
The current official ECMWF AI Weather Quest evaluation expects 1.5° global 5-quintile probabilities for days 19–25 and 26–32 with official RPSS, and a realistic issue-time provenance of available input data. A14-R2 coarse deterministic MSE **does not** fulfill the competition protocol.
