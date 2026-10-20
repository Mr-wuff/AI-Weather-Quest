# A15-P16 P850 E0 actual source gate BLOCKED -> FIX1 strictly parse 850hPa IDX

Date: 2026-10-20. This draft PR archives the genuine observed failure and an offline-qualified minimal code repair; no NOAA 850hPa GRIB fetched during assistant QA.

## Actual user's uploaded scientific evidence (no fabricated PASS)
- Real ZIP `P16_WIND_P850_E0_DELIVERY_REPORTS.zip`: exactly 707 bytes, SHA256 `e84e70f1dce2b1db09bd100ccb9756bde6e2adf0bfc70a2db4f029e4753805f0`, matches user sidecar, CRC PASS. Only `reports/STATUS.json` in ZIP: `stage=pilot1`, `status=BLOCKED_P850_SOURCE_OR_GRIB`, `exception=Blocked('GRIB_INDEX_FORECAST_LEAD_AMBIGUOUS')`.
- User's actually executed 5-cell Colab Notebook reports Drive credential propagation failure with nonfatal volatile fallback, then aborts in `run->load_group->get_source->parse_index`. No complete 850hPa paired wind case, no CIDM trained, no physics scores. Drive mount failure is NOT the reason for the source gate failure.
- Source parser incorrectly demanded **exactly one** `fNNN` time expression in **every pressure-level index row**, BEFORE selecting requested UGRD/VGRD and 850 mb layer. Real source index has at least one row not matching; the particular actual IDX string is not included in uploaded results, so its true syntax remains unverified.

## FIX1 precise source parser repair (keeps original data and physics Gate)
1. Validate all row counters, UTC `d=YYYYMMDDHH`, strict increasing GRIB byte offsets, S3 HEAD length and ETag/IDX hash. Then filter ONLY exact requested `UGRD/VGRD` and `850 mb` or `850 hPa`, and ONLY THEN parse forecast lead for a **target** row.
2. Accept one whole time-descriptor field matching `f456`, `456 hour fcst`, `456 hr fcst`, or `456 hour forecast`; reject unknown/ambiguous or accumulated/average spans such as `456-462 hour ave`. Non-target levels can have unrelated textual descriptions without prematurely blocking the target stream.
3. Preserve exactly `LEADS=456..498` and `624..666` at 6h intervals, 16 distinct time nodes. Duplicate/absent 850hPa target steps fail closed.
4. Keep unchanged ecCodes verification of actual GRIB dataDate/dataTime, endStep, parameterCategory/Number, pressure level 850hPa, units m/s, grid orientation, numeric finiteness, and original 121x240 full fields + 11x12 patch outputs.
5. If source index fails again, emit real source provenance + sample target index rows to `P850_INDEX_DIAGNOSTIC.json` and `reports/IDX_FORMAT_DIAGNOSTIC.json` inside a BLOCKED result ZIP. Never claim a blocked read succeeded.
6. Pilot1 2009010700 five members×U850/V850×16 leads => at most 160 GRIB messages, then pilot4 four earlier actual 2009/2012/2013/2015 candidate dates if and only if real Pilot1 passes. No ERA5 future labels used as predictor inputs; 2014/2016 reserve untouched, W3/W4 126h gap still protected, GEFSv12 historic reforecast not proven operational issue-time source. No long CIDM training or official RPSS claims.

## Independent offline QA, scope and user artifacts
- Fully executed corrected five-cell Colab Notebook in real Jupyter kernel with `P16_P850_OFFLINE_QA=1`: PASS, status `OFFLINE_QA_ONLY_NOT_REAL_850_HPA`.
- Additional source tests: fNNN and `NNN hour fcst` and `NNN hr fcst` accepted, a 700 mb non-target index row with unknown time ignored, 850 averaged spans/wrong timestamp/wrong layer/wrong parameter rejected, failure diagnostics included in CRC PASS ZIP. ZERO new genuine 850hPa weather fields downloaded during QA; actual Colab pilot1 is PENDING.
- User standalone patched Notebook `CIDM_EARTH_A15_P16_P850_E0_FIX1_StrictIDX_MultiLevel_Colab.ipynb`: 29613 bytes SHA256 `4fa6e068f6a89a2768ae7c37350a43a43c9de97bcf9378649cc952fadfc8e845`.
- Readable full Python `p16_p850_fix1_source.py`: SHA256 `2c711db778859d9b78978aef1758554eb78290309235bc7c4f33a23264382ac3`; engineering test script, original actual BLOCKED status JSON, Chinese audit and README in the complete user ZIP.
- Complete audited delivery `CIDM_EARTH_A15_P16_P850_E0_FIX1_StrictIDX_MultiLevel_FULL.zip`: SHA256 `9b05226b2348b1ba9c2b395fd3b53ef997f6c36a4bee34f2f048e15b806e13d7`, ZIP CRC + signed per-member manifest verified. User code is delivered as direct ChatGPT download, not claimed to be in this public GitHub PR.
- Scientific gate: zero new actual 850hPa data; REAL 850hPa qualification remains BLOCKED until Colab retest. Do not misstate as passing the wind/flux/CIDM model performance gate.

Public PR contains this text audit+hashes only, not user's weather arrays/Colab notebook; remains DRAFT and should not be merged automatically.
