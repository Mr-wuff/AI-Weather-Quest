# CIDM-Earth A15-P0-FIX4 VERIFIED → A15-P1 full-week GEFS ensemble qualification

Date: 2026-10-18.

## Real result verified from user-uploaded report
- Source ZIP SHA256 `acbbf29007afcd5c48cc3d99cd16df27f642dc7fad16b76809f399624ceb9d6b`; user SHA sidecar matches and CRC PASS.
- Genuine anonymous S3 listing for GEFSv12 reforecast 2018-01-03 00 UTC control `c00`: 232 objects under the date/member prefix.
- Two genuine 35-day GRIB2 files: pres_msl 28,117,924 bytes, tmp_2m 23,385,000 bytes.
- Each index includes 100 messages; W3 f456...f618 (28 x 6h), W4 f624...f786 (28 x 6h), both variables 56/56 matching IDX entries.
- Two actual ranged GRIB2 records f456 decoded: msl paramId=151 Pa, 2t paramId=167 K, both regular latitude-longitude Ni=720,Nj=361 (0.5°). These were only **one decoded time per variable**. Remaining 55 leads were verified by index, not yet decoded as complete fields.
- Verified status `REAL_GEFS_REFORECAST_PORT_SAMPLE_PASS_RESEARCH_ONLY`, not official AIWQ eligibility. It is a **retrospective reforecast** (not the 2018 historical operational forecast), and there is no official RPSS or CIDM training.

## A15-P1 preregistered narrow next step
- Same fixed 2018-01-03 00 UTC source with reference Thursday 2018-01-04 00 UTC, five members `c00,p01,p02,p03,p04` x MSLP,T2m. Preflight HEAD and 100-row IDX existence/lead time for all 10 objects before expensive reading.
- Up to 5 x 2 x 56=560 individual GRIB2 ranged message reads, no full annual or object downloads. 8 MB per record absolute cap; bounded SDK retries.
- For each real message verify GRIB2 edition/length/end marker, exact initial date + 6h forecast step, variable/units/paramId, global regular_ll 361x720, finite values and physical bounds. Convert 0.5° grid to aligned 1.5° 121x240 nodes by exact *coordinate subsampling*, NOT area-conservative regridding.
- For each member, W3 and W4 weekly means of 28 six-hourly instantaneous MSLP/T2m fields; store a derived portable NPZ shape `[5,2,2,121,240]`, MSLP hPa, TAS °C, ensemble mean/sample spread. This is a physical data readiness test, not a calibrated probability forecast.
- Every four records checkpoint cumulative sums and counts atomically; resumes only after index SHA256/object ETag match. If interrupted, preserve failure report and resume caches locally; browser automatically downloads ZIP+SHA256 on normal completion or BLOCKED.
- No model fitting; `training_authorized=false`, `official_rpss=null`, `operational_issue_time_verified=false`. No download or use of future ERA5 truth, no official ECMWF score, no claim that reforecasts could have been used operationally during 2018.
- Offline tested whole self-contained Colab (8 executed code cells) and synthetic fake-S3 full-week, missing-index, interrupted/restart resumption; **real P1 cloud retrieval has NOT yet executed**.
- Complete ChatGPT deliverable ZIP SHA256 `d857330567a0f91b0eaef0dd69c788e8bebb30ce73b57dcffb0a21aedfcd563d`. Contains the runnable self-contained Notebook, independent Python driver, synthetic S3 test, Chinese formal review, execution README, and internal file hash manifest.

## Publication and next gate
- GitHub research-only preregistration on draft PR; source/notebook distributed as user downloadable package in chat.
- HF repo `wuff-mann/AI_Weather_Quest` has NOT been modified; automatic publication is disabled, user secrets and NOAA raw GRIB not uploaded.
- After genuine P1, next design must test multiple uninspected dates/year blocks, physical GEFS biases and ensemble reliability, carefully obtain independent climatological quintiles and legal issue-time operational predictors before CIDM-v2 + official RPSS benchmark. GEFS retrospective source data must not be labeled real-time competition compliant.

NOAA description: https://noaa-gefs-retrospective.s3.amazonaws.com/Description_of_reforecast_data.pdf
AIWQ official: https://aiweatherquest.ecmwf.int/submitting-forecasts/
