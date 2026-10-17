# CIDM-Earth: A15-P0-FIX2 REAL ECDS outcome → A15-P0-FIX3 MARS ambiguity diagnostics (2026-10-18)

## Verified user result
- Uploaded `CIDM_EARTH_A15_P0_FIX2_DELIVERY_REPORTS.zip`, SHA256 **ffc439d7fd4db0e75bceb5801f5cb2d8675270aa24a205c254b1d92b077bad5b**; provided SHA sidecar matches; ZIP CRC PASS.
- Actual user Colab detected an ECDS key and submitted remote request ID **30a41402-bef0-4381-a33b-3c9ee29d510c**; ECDS stages **accepted → running → failed** (2026-10-17 16:49–16:50 UTC).
- The ECDS backend returned HTTP 400 with MARS:
  - `Ambiguous : s2 could be SWEDEN or SWITZERLAND`
  - `Ambiguous : 2m_temperature could be 2M TEMPERATURE ANOMALY OF AT MOST -1K or ... -2K`
- This is **MARS request-selector ambiguity** rather than a transient network outage, despite FIX2's inaccurate `BLOCKED_ECDS_TRANSIENT_OR_NETWORK` label. No GRIB was delivered, no official RPSS or CIDM optimization.
- The request used ECDS `s2s-forecasts`, archived control forecast, 2018-01-01 00Z, ECMWF origin, `2m_temperature` / `mean_sea_level_pressure`, steps 504/672h, 1.5° grid. These two steps are start-time **snapshots**, not 7-day forecast means.

## A15-P0-FIX3 preregistration
1. Start with the **exact ECDS official tutorial minimal sample** (`s2s-forecasts`, ECMWF control, `2m_temperature`, 2023-01-30 00Z, 024h). A failed minimal sample with same `class=s2` ambiguity provides evidence for a ECDS/MARS service-side mapping issue; do not blindly change MARS class tokens.
2. Only upon genuine successful minimal GRIB + decoding: request the 2018-01-01 W3/W4-start 504/672h MSLP and TAS separately, with original 1.5° grid. Strict GRIB time, variable, units, finite field, and 121x240 structure checks.
3. New mutually distinct failure categories: `BLOCKED_ECDS_MARS_CLASS_AMBIGUITY`, `BLOCKED_ECDS_MARS_PARAMETER_AMBIGUITY`, `BLOCKED_ECDS_MARS_REQUEST_REJECTED`, `BLOCKED_ECDS_AUTH`, `BLOCKED_ECDS_NETWORK_OR_BUSY`, `BLOCKED_USER_TOKEN_MISSING`. Any MARS ambiguity halts repeated requests and exports a sanitized, copy-ready ECMWF support ticket.
4. No raw GRIB, ECDS key, or private data in diagnostic ZIP; optional user-supplied legitimate local GRIB for independent QC. Default Colab automatic download of ZIP and SHA256 in PASS or BLOCKED.
5. Engine includes 6/6-code-cell offline Notebook execution, no-token fail-closed test, simulated MARS error and token redaction, and full three-stage GRIB validation with a **fake** cfgrib backend. No assistant-side real ECDS network run and no official score claim.
6. **Do not authorize CIDM-v2 long training** even on P0 PASS. Next phase must construct full weekly-mean S2S ensembles, issue-time availability, robust calibrated baseline, actual quintile scores on fresh held-out dates.

## Deliverables in current conversation
- `CIDM_EARTH_A15_P0_FIX3_MARS_OfficialProbe_AutoDownload_Colab.ipynb`
- `a15_p0_fix3_mars_triage.py`
- `test_a15_p0_fix3_offline.py`
- Chinese source review, contract and README.
- Research ZIP SHA256 **29ddd8399b95c3aa1de4b6bae579fde9fed2149fef64d135c399784cdbfec965**.
- Hugging Face `wuff-mann/AI_Weather_Quest` remains unchanged without actual P0 success or user-run credentials; raw GRIB should never be uploaded there.

## Sources
- https://ecds.ecmwf.int/how-to-api
- https://confluence.ecmwf.int/spaces/DAC/pages/540541961/Examples%2Bof%2Bthe%2Bcurrent%2Band%2Bfuture%2Bretrieval%2Brequests
- https://ecds.ecmwf.int/datasets/s2s-forecasts?tab=download
