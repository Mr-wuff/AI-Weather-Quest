# CIDM-Earth A15-P0-FIX1 result audit → FIX2 ECDS server-auth gate (2026-10-18)

## Verified real user upload
- `CIDM_EARTH_A15_P0_FIX1_DELIVERY_REPORTS.zip`: SHA256 `58da532ecab12467208f0cde79c105c7296336cada0c0d7c09e1e61ddd939d43`, CRC PASS.
- User-executed notebook: 7 code cells completed, mathematical/synthetic smoke PASS, Colab auto-download requested.
- Real-stage program entered; report has `real_gate_status=BLOCKED_REAL_SOURCE_NOT_ASSESSED`, return=2, no real GRIB, no training, no official RPSS.
- Actual traceback `TERMS_NOT_ACKNOWLEDGED: accept on ECDS portal, set ECDS_TERMS_ACCEPTED=1`. Notebook shows `Secret present: False`, `Accepted ECDS terms: False`, `Local GRIB provided: False`.
- **No actual ECDS remote request**: prior `online_data_test_attempted=true` indicated entry to real-stage path, not network submission. Distinguish with new `remote_request_submitted` boolean.

## A15-P0-FIX2 reproducibility contract
- User must log into https://ecds.ecmwf.int/how-to-api and obtain Personal Access Token, separately **manually accept the S2S dataset licence** at https://ecds.ecmwf.int/datasets/s2s-forecasts?tab=download; token is passed privately through Colab Secret `ECDS_API_KEY`.
- Remove extra local `ECDS_TERMS_ACCEPTED` boolean, which was redundant and easy to leave False; the ECDS API SERVER remains sole authorization authority, so **no permission bypass**.
- Default actual fetch when credentials exist; no token => `BLOCKED_USER_SETUP_API_KEY_MISSING`; server-side terms => `BLOCKED_ECDS_DATASET_LICENCE_REQUIRED`; auth / network / GRIB QC are separated and captured with sanitized traceback.
- Forecast request frozen: actual archived S2S `s2s-forecasts`, ECMWF control, 2018-01-01 00UTC init, 2018-01-04 00UTC issue, 1.5° global, MSLP and TAS at 504/672h. These are *instantaneous forecast fields at W3/W4 starts*, NOT weekly means; historical release time not verified.
- Automatically package full diagnostics with code/manifest/source SHA and **trigger Colab local browser download** of ZIP+SHA256 whether PASS or BLOCKED (user browser may need allow multiple downloads). Never publish raw GRIB or secrets.
- Offline full notebook 6 code cells PASS with no token and correctly stops at user-setup status. Fake cfgrib 4-group complete/one missing lead tests PASS. Simulated server-side licence rejection PASS, verifies token redaction and `remote_request_submitted=true`.
- Real ECMWF S2S retrieval **not executed in assistant environment**, genuine forecast compatibility remains untested. No CIDM training or official RPSS; next A15-P1 only after real port PASS.
- Downloadable full source package in conversation `CIDM_EARTH_A15_P0_FIX2_ServerAuth_AutoDownload_FULL.zip`, SHA256 `5cc560475cc50eed9f4a08c615f84eee709eae943800053788d440ad4eda2655`.

## Reference
- https://ecds.ecmwf.int/how-to-api
- https://ecds.ecmwf.int/datasets/s2s-forecasts?tab=download
- https://confluence.ecmwf.int/spaces/DAC/pages/540541961/Examples%2Bof%2Bthe%2Bcurrent%2Band%2Bfuture%2Bretrieval%2Brequests
