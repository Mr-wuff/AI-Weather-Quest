# A15-P2 user evidence audit → A15-P3 fixed train-only climatological quintile proxy preregistration (2026-10-18)

## Genuine uploaded results
- User supplied A15-P2 output ZIP plus separate SHA256 and executed Colab Notebook.
- Actual current uploaded ZIP SHA256: `dd380fb310546b080314ecb8ad78a2edbecad68413ac21f7696ea6a9feda486c`.
- Separate user-provided sidecar and Notebook post-pack output SHA: `91292d2cc1e6a47b5a0eb031cddb12b360967bb7fa221b28be8fe95602e5b75d`. **These differ. The current bytes are NOT a verified byte-for-byte copy of the original Colab archive.**
- Nevertheless current archive ZIP CRC PASS, 4 per-case paired NPZ SHA256 all match their internal report, each NPZ shape `forecast:[5,2,2,121,240]` and `truth:[2,2,121,240]`, all finite and physically reasonable, and all four area-weighted RMSE metrics independently recomputed to match reports.
- Actual 2017-01-04, 2017-07-05, 2018-01-03 and 2018-07-04 GEFSv12 reforecast–WB2 ERA5 same-time pairing completed. 2018-01-03 reused A15-P1 SHA-locked source; three other dates downloaded.
- Four-case arithmetic means: MSLP W3/W4 ensemble RMSE 5.342016 / 5.446953 hPa vs control 6.901595 / 6.601644; TAS W3/W4 ensemble RMSE 2.129737 / 2.225383 degC vs control 2.578051 / 2.678495. Relative gain against control is about 22.60%, 17.49%, 17.39%, 16.92%.
- Mean empirical five-member CRPS 2.636659 / 2.592335 hPa and 1.011514 / 1.058226 degC respectively. These reflect **GEFS**, not any trained CIDM model. Mean ensemble spread ~5.0275 / 5.2731 hPa and 1.8124 / 1.8639 degC.
- Only four already-inspected cases; 0 official RPSS, 0 CIDM training, no historical issue-time availability proof. GEFS 0.5° node subsampling versus WB2 ERA5 conservative 1.5° is not identical regridding.

## A15-P3 preregistered experiment
- Pure forecast verification: **no model fitting or CIDM training**.
- Embed current uploaded A15-P2 ZIP byte version inside standalone Notebook and SHA lock to the ACTUAL digest above, explicitly record original sidecar mismatch. It is not a repair of the original Colab artifact.
- Compute 0.2/0.4/0.6/0.8 climatological quantiles of ERA5 *weekly mean*, per season Jan/Jul, per variable MSLP/TAS, per lead W3/W4, for training years 2008–2016, using first Wednesday of Jan and July plus -1/+1 week: 9 × 3 = **27 independent historical initialization dates per season**, 54 total. Each weekly mean uses exactly 28 6h valid times matching Thursday issue +18..24 or +25..31 days.
- Classify each of five actual GEFS weekly means into 5 bins against **train-only** thresholds, empirical probabilities as counts/5; evaluate latitude-area-weighted RPS and `proxy_rpss = 1 - RPS_forecast/RPS_uniform_reference` on only four 2017/18 dates. Half-count=.5 smoothing is a separate *prespecified sensitivity* and must not be optimized on test outcomes. Persist per-case distributions and honest scores.
- This is only a ***research proxy RPSS***. 27 climate weeks is insufficiently robust for official climatology; official ECMWF AI Weather Quest thresholds/weighting/evaluation must be checked separately. Four years/months examples are not independent untouched benchmark.
- **No formal training authorization**, no automatic promotion to competition ranking even if proxy RPSS > 0.
- Cache each one of 54 historic weekly mean tensors locally for resume, default real data access and Google Colab browser auto-download ZIP+SHA256 on both pass and block.
- Synthetic offline smoke and full 8-code-cell Notebook execution PASS; modifying original input ZIP byte intentionally triggers `BLOCKED_QUALIFICATION_NOT_SKILL_RESULT`. Real historical 2008–2016 WB2 quantile construction has NOT been executed in assistant environment. Network request/actual proxy skill not known.
- Full delivery in ChatGPT: `CIDM_EARTH_A15_P3_TrainOnly_QuintileProxy_FULL.zip`, digest `ab01684e79867b89a199def3cae94554e003894f4d6e56ea40d853b692c0e8a0`; source, selfcontained notebook, docs and two contrasting original vs recalculated source SHA256 sidecars. No GEFS original GRIB, no ECDS token/private data is published.
- Public source: WeatherBench2 6h ERA5 global 1.5° conservative Zarr; NOAA GEFSv12 retrospective AWS.
- Follow-on after successful proxy baseline must include independent fresh dates/years, train-only climatological threshold breadth, official quintile definition/real operational input compliance, recalibration to strong baselines, and *mechanism prerequisite tests* before spending on CIDM-v2 training.
