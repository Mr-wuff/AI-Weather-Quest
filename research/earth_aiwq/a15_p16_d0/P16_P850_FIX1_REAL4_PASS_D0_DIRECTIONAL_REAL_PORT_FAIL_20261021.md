# CIDM-Earth P16-P850-FIX1 genuine 850 hPa Pilot4 PASS; P16-D0 real-port qualification FAIL

Date 2026-10-21. Two distinct conclusions: **physical wind-source gate PASS**, **simple wind-driven diagnostic forecast-skill mechanism gate FAIL**.

## Original uploaded user evidence
- FIX1 genuine report `P16_WIND_P850_E0_DELIVERY_REPORTS(1).zip` SHA256 `141573a935f5503e50bb3b686c6b68e591fcdc0a62638296bc43156e09332d43`, sidecar exact match, CRC PASS.
- User full 850hPa recovery ZIP 81,539,009 bytes SHA256 `602ae04e2bafc1e8e09fea0ac8136bde7d88b7abee06a6c959985c6f8b7a263e`, sidecar match, CRC PASS; all 56 signed manifest entries size/SHA verified.
- Result status `REAL_P850_SOURCE_GATE_PASS_RESEARCH_ONLY`: **4/4** real retrospective GEFSv12 issues 2009010700, 2012040400, 2013070300, 2015100700. Five members×U850/V850×16 6-hour leads = 640 true GRIB range decoded records. Source IDX strictly selects `850 mb` from 2,500 records/group (100 target 850 records), genuine label `456 hour fcst` etc, no other level substitutes. Full per-issue tensor `[5,2,2,8,121,240]` and signed patch `[5,2,2,8,11,12]`. Independent raw full->patch recomputation within 2.5e-5 m/s.
- W3 `f456..f498` and W4 `f624..f666` each 8 nodes covering 42 hours; 126h gap MUST NOT be used as a 6h rollover. ERA5 future is exclusively verification label.

## Actual NEW P16-D0 minimal real wind-port qualification
- Independently merged SHA-verified authentic 4-case MSLP/TAS forecast/ERA5 label 11x12 arrays, authentic 10m U/V patch arrays and four raw 850hPa full tensor projected patch arrays; no new NOAA downloads.
- Signed independent ~988KB panel SHA256 `d399e91c974a8a1cb08e2b36b5a3baddcfea81f8b86cfe977de1ab89aad4a0f1`. 2009/2012 development-only supervised fitting of 8 frozen per-variable/week/height coefficients; 2013/2015 **previously inspected** assessment (NOT new independent blind generalization). Predictors only GEFS same-issue forecast F(t), forecast F(t+6), same-issue forecast U10/V10 or U850/V850. ERA5 truth *only* for train fit and offline score.
- Fixed 6h conservative area-weighted upwind transport correction of forecast field anomaly, shared edges, periodic longitude, no north-south/pole wrap. Max source-score-weighted conservation relative residual `5.6195e-17` for MSLP diagnostic, NOT physical atmospheric conservation.
- Freeze ridge penalty equal .5× feature energy and coefficient clamp ±1. Null baseline F(t+6) versus correctly aligned wind, reversed wind, +4 patch-longitude shifted wind, time-shifted wind. No neural net training, no hyperparameter selection from audit dates.
- Genuine past-viewed audit RMSE improvement vs raw original GEFS:
  - MSLP: 10m **-0.0126%**, 850hPa **+0.0846%**.
  - TAS: 10m **+0.2326%**, 850hPa **+0.2278%**.
- Gain of correct over +4 longitude-shifted negative wind:
  - MSLP 10m **-0.898%**, 850 **-1.217%**;
  - TAS 10m **+0.411%**, 850 **+0.282%**.
- Fixed exploratory threshold each variable/port: ≥1% raw improvement, ≥0.5% shifted wind advantage, ≥3/4 favorable issue×week comparisons. **ALL 4 variable×height combinations FAIL**. Several MSLP fitted alpha clamp at +1; do not conclude wind has no information in general, only this highly restricted coarse-scale correction was not qualified.
- No true autoregressive CIDM state prediction, full 1.5° model, official RPSS, operational evidence of historical issue-time GEFS availability, or fair CNN comparison. 2014/2016 remain untouched. **Long training remains forbidden.** Need new independent training and unseen issue dates and comparable neural controls before evaluating directed CIDM.

## Reproducibility
- Source SHA-locked authentic panel and engine runnable entirely OFFLINE. Notebook `CIDM_EARTH_A15_P16_D0_REAL4_10m_vs_850hPa_Port_Qualification_Colab.ipynb` 1,530,826 bytes SHA `63d9a7a5f21b30b62afa432f3e70f5939c3d25592305773d8dac72a562784d0b`.
- Actual computed result ZIP `CIDM_EARTH_A15_P16_D0_DELIVERY_REPORTS.zip` SHA `92b83a5d3ec8400879c9fa0929b6f89da890aaf9e36475529ca6d3159dffb208` with per-case CSV and status JSON.
- Full user handoff ZIP `CIDM_EARTH_A15_P16_D0_REAL4_10m_850hPa_Qualification_FULL.zip` 3,038,421 bytes SHA `06f2de010ddc865f7e0852900dafa43f2ea0a819491c3be18c0d9269defe67e3`; member hash manifest and CRC all PASS; includes source, raw-to-panel provenance code, signed real panel, standalone notebook, Chinese report, unit tests, and actual results.
- Entire Notebook independent true-panel rerun PASS, panel checksum tamper blocked, offline QA PASS with real source, tampered embedded source bootstrap correctly exports BLOCKED and not fake scientific result. Public PR stores only text contract and source SHA fingerprints, NOT user's full weather arrays. Draft; do not merge.
