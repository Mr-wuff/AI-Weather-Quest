# A14-R2 REAL outcome → A15-P0 ECMWF ECDS S2S port source qualification (2026-10-17)

## Verified A14-R2 result
- User-uploaded ZIP SHA256 `6c79e1930018395df1570c752e67864ceea4520f02fc430d9d921d83a6abd75a`; sidecar match, CRC PASS.
- Status `REAL_RETROSPECTIVE_STABILITY_AUDIT_COMPLETE`; train 2008–11 (102), inner-val 2012 (24), reused 2013–14 development (50); 128 within-year/season time-permutation negative controls and 1000 moving-block bootstrap replicates. No neural training and no official RPSS.
- Extra inputs vs surface history (relative MSE error reductions, %): Z500 W3 MSLP +0.4301, W3 TAS +0.3247, W4 MSLP +0.3465, W4 TAS zero; SST +0.0792, +0.5532, −0.0323, zero; both +0.2794, +0.6144, −0.1764, zero.
- SST/W3 TAS matched placebo p=0.007752, block bootstrap 95% CI [+0.0577%,+1.0485%], positive in 2013 and 2014; both/W3 TAS p=0.007752, CI [+0.2011%,+1.0800%]. BUT W4 fails, including W4 TAS alpha 0 for all. `stable_incremental_signal=false`, `formal_CIDM_training_authorized=false`.
- This closes repeated 3x4 coarse-grid ridge/alpha tuning, not all possible Z500/SST physics or CIDM-v2 architectures. Only two development years; no fresh blind holdout.

## A15-P0 preregistered source-port audit; no model training
- ECDS datasets `s2s-forecasts` and `s2s-reforecasts` exist; **the prototype uses archived real-time `s2s-forecasts` only** and avoids substituting later-version on-the-fly hindcasts as historical operations.
- Conservative tiny pilot pair: initialization Monday 2018-01-01 00 UTC, hypothetical Thursday issue 2018-01-04 00 UTC. ECMWF policy has 48h public latency; this pair is policy-compatible, but actual 2018 release timestamp remains **UNVERIFIED**.
- Control forecast, single level, 2m temperature + mean sea level pressure, target days 19–25 / 26–32 **start** represented by lead 504h and 672h from Monday init. These are only two time samples, NOT weekly-mean weather forecasts.
- Strict ingest checks source GRIB SHA, data variables, K/Pa units, initialization and lead, 121x240 regular latitude/longitude. Also synthetic-only full-global quintile shape/simplex/RPS mathematical unit checks; **never** call it official RPSS.
- Credential missing, terms unaccepted, absent source, incompatible time or fields => BLOCKED with useful traceback and report ZIP, never imply forecast skill FAIL.
- Notebook offline 10 code cells end-to-end PASS, includes intentional bad-lead fake GRIB rejection and temporal negatives; **ECDS real download not run**. 0 CIDM optimization steps, 0 official scored issues.
- Downloadable deliverable in current conversation: `CIDM_EARTH_A15_P0_S2S_Port_Qualification_FULL.zip`, SHA256 `52428c2b8d6ce1a6fd76d6bbd18cff05f9841c49c6ac8e79ab65cdc5bf198e67`. Includes standalone Notebook, independent script, synthetic QA, Chinese contract/README/manifest.
- Publication: no proprietary raw GRIB, ERA5 cache, API tokens or secrets. HF model repo `wuff-mann/AI_Weather_Quest` and GitHub `Mr-wuff/AI-Weather-Quest` are configured in Notebook for optional *explicitly authorized* lightweight run-time sync.

## Crucial official-rule distinction
ECMWF AI Weather Quest's **submitting-forecasts** page permits forecast inputs with timestamps **not later than Thursday 00 UTC**, even when they become available during the four-day submission window. Thus this P0 prototype's policy-public-before-Thursday requirement is a deliberately MORE CONSERVATIVE research gate, **not an official rule**. Future P1 may use a Thursday-timestamp source with 48h public latency if obtained before the Sunday deadline. Competition rules require significant original AI/ML transformation; unchanged third-party dynamical forecasts cannot be passed off as original AI submissions.

Official documentation:
- https://aiweatherquest.ecmwf.int/submitting-forecasts/
- https://aiweatherquest.ecmwf.int/terms-of-ai-weather-quest/
- https://aiweatherquest.ecmwf.int/evaluation-system/
- https://ecds.ecmwf.int/how-to-api
- https://ecds.ecmwf.int/datasets/s2s-forecasts?tab=download

Next A15-P1 requires complete time-dependent multi-member S2S, official quintile threshold calibration, full-field probabilistic baselines, fresh date blocks, and operational ingestion evidence BEFORE new CIDM-v2 formal training.
