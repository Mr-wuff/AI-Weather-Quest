# CIDM-Earth A15-P11 real G1-R2 failure triage → FIX1 real IDX parser repair
Date 2026-10-19

## User uploaded actual evidence and reproducibility
- User report ZIP `CIDM_EARTH_A15_P11_DELIVERY_REPORTS.zip`, exact SHA256 `8d0dd475baa03ef6dde8a567da0b4d65a4ad749ecede0baf78370ed9a0dd915b`. User sidecar matches; ZIP CRC PASS. User Colab ran all four code cells.
- Actual status `G1_R2_PARTIAL_OR_BLOCKED_RESEARCH_ONLY`, 24 scheduled issues, 0/240 source index objects PASS, 0/48 physical samples checked. All 240 objects failed with the **same** `QualificationError: IDX_INIT_DATE_MISSING_OR_MISMATCH`.
- Root-cause in **P11 code**: `tags[0]=='d='+date[:8]` requires exactly 8 digits, while normal GRIB IDX metadata can carry a ten-digit initialization time `d=YYYYMMDDHH`. Earlier P10 used an 8-digit substring presence check and passed c00 48/48. P11 error occurs BEFORE W3/W4 index completeness and real GRIB2 physical metadata checks. The uploaded ZIP did not contain raw .idx lines, so the exact actual remote syntax remains to be audited in the FIX1 re-run; no source-availability conclusion can be drawn from 0/240.

## Code change
- New **readable complete FIX1 engine committed to this branch**: `research/earth_aiwq/a15_p11_fix1/a15_p11_fix1_full_ensemble_gate.py`, pinned Git commit `df938844f1facd3b8d2c211751fae457fd911ec1`, SHA256 `3afd20ea9cdab5ceb8025580cadbd148a3cc5e10fbad8b77c85bf495e2e217fb`. Server-side file fetched and contents independently checked by byte length/FNV32.
- Accept strict `d=YYYYMMDDHH` only if it **exactly equals** the required ten-digit initialization. Legacy eight-digit `d=YYYYMMDD` is accepted as a lower-assurance tag with `strict_utc_hour_verified_by_idx=false` and still requires sampled real GRIB2 ecCodes time verification. Bad hours/mixed dates/unparsed records reject. Report shows precise IDX tag mode.
- On checkpoint reuse recheck BOTH GRIB object ETag/size and separate .idx SHA256, do not conflate them. Bump resume cache contract v3.
- Retain ORIGINAL source Gate, no relaxation: 24 dates × 5 members × 2 variables = **240** real HEAD/IDX; W3/W4 exactly 28 six-hourly leads each and 7 temporal leads. Physical sample 6 year-stratified valid dates × c00/p04 × 2 variables × f456/f786 = **48 actual ranged GRIB2** metadata/field checks; >=20/24 full five-member issues in >=5 years + 48/48 physical checks is research source sample Gate only.
- No weather model training, no official quintile RPSS, no claim of historical real-time operational GEFS availability.

## Deliverable and QA
- GitHub-source-loading light Colab Notebook (6504 bytes) and fully readable sources/mock QA/Chinese report in `CIDM_EARTH_A15_P11_FIX1_RealIDX_GitHub_LITE_FULL.zip`, SHA256 `6914c170b02caf4188f5f11c9dbcf44a8861d28ea66d5806ea545c30c656ed8d`, ZIP CRC+member SHA PASS.
- Notebook downloads both exact SHA-pinned FIX1 engine + historical GEFS reader at `bb1583fc7078e74f251ec101d0e11a522f2a9eeb` SHA `4e88bee38ccce563d9cfd75e457e9a35070439b972dc7fc742a9e4b23520a98e`. No past weather result uploads, no HF token needed. Browser attempts result ZIP + SHA download on success or BLOCKED.
- Simulated 24 issues, 240 valid real-format ten-digit .idx files, 48 mock GRIB byte records: PASS. Legacy 8-digit path accepted with documented hour limitation, wrong hour / missing f786 / changed .idx despite stable data ETag / missing reader blocked. Full 4-cell Notebook offline QA PASS and corrupt engine path produces accurate BLOCKED ZIP + SHA.
- **Real FIX1 cloud S3 has NOT yet executed**. Simulated QA is not a data source pass. Next step user runs FIX1 Notebook; only then choose multi-date full fields + independent train/validation/test weather benchmark and truly small CIDM-vs-CNN mechanism qualification.
