# CIDM-MHC-Real A3-FIX1 official completed — A4/P0 historical risk audit preregistration
2026-10-16

## Verified original user report
User ZIP `CIDM_MHC_REAL_A3_FIX1_PHYSICAL_STATE_REPORTS.zip` SHA256 `98fd14a9c4c295b342cdb274525b399081988be959d9284a27f190a07cbdf4b1`; supplied sidecar exact match; ZIP CRC PASS; upload receipt SUCCESS HF commit 17e6e4ef74aaf87d8d1701c051d06f12455b2a86.
Official `reports/status.json`: `FROZEN_PHYSICAL_STATE_INFORMATION_COMPLETED_NO_TRAINING`. Raw annual ERA5 official file hashes and real seed17 frozen checkpoint hash logged by Colab; raw bytes not uploaded to chat. A2 historical scores replay PASS, 2017 sealed, zero model optimizer steps.
2016 48 previously used validation issues per lead, three latitude belts, MSLP/TAS coarse residual squared error, **not full-grid physical forecast or ECMWF AI Weather Quest RPSS**.
W3 MSE Instant .002407063033221274, Old MHC Memory .0023797314385967192, Physical History .002168703640537171. Physical gain vs Instant +9.902499%, vs Memory +8.867715%; wins only 27/48 vs Instant, 26/48 vs Memory.
W4 Instant .002331048521826151, Memory .002306800992316113, Physical .0021729604188103504. Physical gain +6.781845% vs Instant, +5.801999% vs Memory; wins only 25/48 and 26/48.
A3 4-week-block descriptive bootstrap improvement CI vs Instant: W3 [-2.3818%, 19.8130%], W4 [-13.2166%, 23.7933%]; both contain zero. Requirement >=32/48 wins and CI lower >0 not met: both physical_signal_gate=false, full_mhc_training_authorized=false.
Frozen A3 source 16 global + 16 quarter-controlled sham perturbations per history type and lead, 128 rows; for W4 at least one quarter-shuffled Physical sham beat observed Physical. Single fixed 2016 validation repeatedly examined; not an independent outer test.

## Newly computed regime risk decomposition from 2016 actual per-issue outputs
Physical vs Instant pooled quarter-weighted MSE gains W3: Q1 +28.91%, Q2 -17.18%, Q3 +7.43%, Q4 -0.57%; W4 Q1 +34.11%, Q2 -43.67%, Q3 +9.23%, Q4 -3.67%. Large spring failures and low win counts preclude global always-on use.

## Next delivered A4/P0 notebook (frozen retrospective ONLY)
Self-contained `CIDM_MHC_REAL_A4_P0_FrozenSeasonalRisk_Colab_STANDALONE.ipynb` embeds exact immutable user A3 ZIP and audit driver `mhc_real_a4_risk_regime_audit.py` SHA256 `dd8944212db1d3389f5b7900376681a2e0626a33bfd16ba60705df03bf84c7fb`. All five executable cells actually ran nbclient CPU, ZIP/CRC/SHA PASS, independent user A3 source score means replayed.
Outputs per-quarter score/risk, per-issue absolute loss differences and extremes, sham distributions, 4-week block confidence intervals, retrospective Q1-only / Q1+Q3 seasonal selection and truth-using **non-deployable** per-window Oracle upper bound. Calendar choices derive from observed 2016 and therefore are **post-hoc hypotheses, not new validation or operational policy**.
Complete user delivery ZIP SHA256 `99362ba14706ca823141023aea2f2d78d64b8089cac5face5bcc863339ed9a8a`. No new data, no weather training, no AIWQ RPSS, 2017 remains sealed.
Use no Google Drive. Result archive immutable; optional single atomic HF Model commit for ZIP+SHA if Colab HF_TOKEN is explicitly configured and `MHC_A4_SYNC_HF=1`; lightweight GitHub optional.

## Future conditional physical-state research decision
Freeze **one** causal physical/seasonal gate chosen on pre-2017 development data, refrain from training-time future ERA5 predictors; test its effect on genuinely fresh 2018+ years with no adaptive reuse and matched capacity and budgets. A strong current-state fallback is essential, but a date-only posthoc selector cannot be declared qualified from 2016 alone. Eventually score official quintile probabilistic AIWQ RPSS on genuine dates, not the coarse linear residual probe.