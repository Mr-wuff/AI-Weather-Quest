# CIDM-Earth A15-P4 verified real holdout FAIL → A15-P5 six-case frozen failure attribution (2026-10-18)

## A15-P4 uploaded results verified
- Input ZIP `CIDM_EARTH_A15_P4_DELIVERY_REPORTS.zip` SHA256 `10e471b02b94583c0b6a59857d1f9f6339c3ab2a498845e429a621bb34a25871`; user SHA sidecar matches, ZIP CRC PASS, per-case SHA matches.
- Real GEFSv12 research reforecast on 2019-01-02 and 2019-07-03: 2 cases × 5 members × 2 variables × 2 leads × 28 six-hour fields = 1120 real GRIB2 records, paired with WB2 ERA5 truth. Parent train-only threshold ZIP SHA `0a60d66483bdf81d9e7bdeacf72c40d7eb1e4cabd8f7d6cb6691cc61c903ebd9`.
- Pooled 2019 **research-proxy** RPSS for raw five-member bins: mslp_W3 -0.2632281, mslp_W4 -0.2085836, tas_W3 -0.1928415, tas_W4 -0.2431099.
- Frozen half-count probabilities: -0.1065795, -0.0567049, +0.0034700, -0.0288814 respectively. tas_W3 is +0.0709132 in January but -0.0623764 in July. None of four targets have positive RPSS on **both** 2019 cases. Preregistered P4 qualification FAIL. No CIDM weight training / official AI Weather Quest RPSS / operational input time verification.

## A15-P5 follow-up already run on genuine historical tensors; diagnostic only
- Objective: do not tune another smoothing coefficient on repeated evaluation cases. Identify whether forecast weakness is associated with ensemble overconfidence, insufficient spread, seasonal climatic category drift, geographic localization, and/or mean bias.
- Three exact SHA-locked real parent archives:
  - A15-P2 (four 2017/2018 cases) current uploaded byte SHA `dd380fb310546b080314ecb8ad78a2edbecad68413ac21f7696ea6a9feda486c`. NOTE this differs from its original sidecar; internal per-case SHA and CRC valid.
  - A15-P3 2008-16 train-only 20/40/60/80% ERA5 27-sample seasonal thresholds SHA `0a60d66483bdf81d9e7bdeacf72c40d7eb1e4cabd8f7d6cb6691cc61c903ebd9`.
  - A15-P4 (two 2019 cases) SHA `10e471b02b94583c0b6a59857d1f9f6339c3ab2a498845e429a621bb34a25871`.
- Local replay on six real forecast/truth tensors completed and generated report ZIP SHA `611dce6f6de444c62edbe0339c639d818edb209b2d41065793c74e85e12e4c51`, full JSON per case/region/bin.
- Six-case pooled proxy RPSS by mslp_W3/mslp_W4/tas_W3/tas_W4:
  - Raw -0.1854 / -0.1134 / -0.1574 / -0.2183.
  - Frozen half-count -0.0485 / +0.0008 / +0.0258 / -0.0154.
- Area with any predicted 5-bin class mass ≥0.8 (raw empirical probabilities): MSLP W3 13.55%, W4 14.02%; TAS W3 34.15%, W4 33.77%.
- Ensemble spread/RMSE (mean per case) MSLP W3 .967/W4 .968; TAS W3 .856/W4 .824. TAS suggests relative underdispersion but these are descriptive scores on just six cases, not a complete reliability analysis.
- Observed W3 TAS 5-bin category frequencies across 6 cases: 14.8%,16.8%,19.2%,21.5%,27.7% rather than 20% each: the train-only climate threshold bins do not have uniform distribution on this small evaluation slice. Possible historical/seasonal mismatch, not a statistically proven climate regime change.
- Conditional geographic diagnostics cover tropics/midlat/polar and Northern/Southern Hemisphere, plus four threshold cumulative Brier losses and reliability bins.
- No hyperparameter tuning, no scientific mechanism attribution to CIDM, no new unseen years, no neural training, and no official RPSS. GEFS is retrospective reforecast, not proven historically available at issuance.

## Deliverables provided in ChatGPT conversation
- Full independently generated REAL diagnostic report `CIDM_EARTH_A15_P5_Real_SixCase_Attribution_REPORTS.zip` SHA `611dce6f6de444c62edbe0339c639d818edb209b2d41065793c74e85e12e4c51`.
- Small reproducibility ZIP `CIDM_EARTH_A15_P5_LITE_Reproducibility_FULL.zip`, SHA `dd7929db33f70777a897457e8554b606a06d2cf0639a36c8ee01b490d82ba068`. Includes small ~14KB Colab Notebook, independent Python source, Chinese evaluation report and README. Large weather archives are **not embedded**.
- Notebook: prefer cached/Drive/HF prior results, fallback to one-time Colab upload, strict file SHA; auto-download ZIP+SHA in success/BLOCKED. Standalone Python source is distributed in the downloadable package; **this GitHub commit archives reproducible protocol/results, not the new Python engine**.
- Full notebook 5 code cells executed successfully with actual 6 cases in local QA; deliberately corrupted/mismatched parent triggers BLOCKED and no stale success reports. Genuine remote HF datasets were not verified to contain these ZIPs.

## Next research gate
1. Do **not** search more smoothing coefficients on these six already-seen cases.
2. Fix train-only probabilistic calibration validation with many more independent initialization dates, broader historical reforecasts and operational-availability audit, with geographic reliability and lead stratification.
3. Introduce CIDM cross-scale information dynamics as a **small, preregistered correction** against a strong calibrated probabilistic baseline, with independent held-out dates, before full training.
