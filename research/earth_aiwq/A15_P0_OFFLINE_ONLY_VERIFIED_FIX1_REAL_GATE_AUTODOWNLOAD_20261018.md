# CIDM-Earth A15-P0 result audit and A15-P0-FIX1 real source gate (2026-10-18)

## User result (verified)
- File `CIDM_EARTH_A15_P0_DELIVERY_REPORTS.zip` SHA256 `80d886cfca0f4e9b90cf202edbc116c1bedf5f02184273fd14db9290a82272f2` matches sidecar; ZIP CRC PASS.
- Executed Notebook has 10 numbered code cells; source SHA locked and offline synthetic math and fake GRIB checks passed.
- Notebook output: `Real fetch requested: False Local GRIB provided: False`; `REAL GATE NOT RUN`. Source manifest `real_status:null`, and `offline_smoke:OFFLINE_SMOKE_PASS_NO_REAL_FORECAST`.
- Hence this is **not a failed real data download** and **not a successful real S2S forecast ingestion**. No real ECDS access occurred, no CIDM training or official RPSS.

## Repair and evaluation contract
- New stand-alone FIX1 Notebook `CIDM_EARTH_A15_P0_FIX1_RealECDS_AutoDownload_Colab_STANDALONE.ipynb` is delivered in ChatGPT's local ZIP. It now defaults to attempting the real data gate. It does not silently pass offline mode. Credentials are read from authorized Colab Secrets; user's dataset license must be manually accepted and user explicitly sets `ACCEPTED_ECDS_TERMS=True`. No credentials/terms leads to a `BLOCKED_REAL_SOURCE_NOT_ASSESSED` report.
- ECDS request includes the verified `grid:1.5/1.5`, needed for 121×240 grid audit. `cdsapi` interface follows ECDS reference syntax.
- Revised physical GRIB audit aggregates two lead hours across *multiple* cfgrib groups (separate variables/steps), validates 504/672 h, initialization, valid times where provided, units K/Pa, and full regular grid.
- Notebook ZIP is packed with status, runtime log, traceback on failure, provenance, SHA and no secrets or raw GRIB. Default `AUTO_DOWNLOAD_RESULTS=True` invokes Google Colab `files.download` after either PASS or BLOCKED. Browser permissions are outside script control; a manual file-path fallback is printed.
- Offline entire FIX1 Notebook executed successfully in a no-secret environment and emitted BLOCKED and ZIP without exception. Independent 4-group simulated GRIB positive test PASS; missing W4 MSLP negative control rejected.
- No real ECDS forecast download has yet been run for FIX1; no model training, official RPSS, or operational availability validation. The 504/672 h outputs are snapshots at target-window STARTS, NOT weekly mean prediction labels.
- Required next step: user runs FIX1 in Colab after ECDS account dataset license acceptance and enabling Secret; uploads ZIP and SHA. If gate passes, separately qualify complete weekly S2S time series, ensemble members, strong probability baseline, historical available-at-issue provenance and fresh date block before CIDM-v2 neural mechanism training.

## Delivery files (in ChatGPT)
`CIDM_EARTH_A15_P0_FIX1_REAL_GATE_AUTODOWNLOAD_FULL.zip`
SHA256 `c1295aeb46d1eea244f25d6480fa59a361185a920e2e296d7fc64be571bc9472`.

## Repositories
- Research source/report PR archived here as a draft; actual trained checkpoints do not exist.
- HF repo `wuff-mann/AI_Weather_Quest` not modified via this review. User can explicitly opt in to lightweight report synchronization in Colab; publishing original S2S GRIB or API secrets is prohibited.

## Sources
- https://ecds.ecmwf.int/how-to-api
- https://confluence.ecmwf.int/spaces/DAC/pages/540541961/Examples%2Bof%2Bthe%2Bcurrent%2Band%2Bfuture%2Bretrieval%2Brequests
- https://ecds.ecmwf.int/datasets/s2s-forecasts?tab=download
