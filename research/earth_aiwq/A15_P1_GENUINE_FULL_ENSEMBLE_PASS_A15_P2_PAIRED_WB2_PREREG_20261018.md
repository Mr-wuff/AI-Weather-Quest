# A15-P1 real five-member complete-week data PASS → A15-P2 paired GEFS–ERA5 qualification
2026-10-18

## Verified uploaded user A15-P1 result
- Exact result ZIP SHA256: `a8025c19bc9109a639891c8526bbce1a7bff18a0c5451e6e3b4d15b486b1fa99` matching user sidecar; ZIP CRC PASS.
- User Colab actually completed 560/560 ranged GRIB2 records (five c00/p01–p04 members × MSLP/TAS × W3/W4 × 28 six-hour leads) for NOAA GEFSv12 retrospective 2018-01-03 00UTC. Ten variable/member group audits and source key, ETag, IDX SHA all saved.
- Derived five-member data shape `[5,2,2,121,240]` with verified exactly-consistent member mean and sample SD, hPa / degC. Median gridpoint ensemble SD: MSLP W3 3.3673 hPa, W4 4.7017 hPa; TAS W3 0.8142 C, W4 1.0923 C. **These measure ensemble disagreement, NOT skill against observation.**
- GEFS 0.5deg regular_ll was node-subsampled to 1.5deg, not conservative regridded. Output represents 2018 reforecast produced retrospectively, not contemporaneous operational forecasts.
- No true weekly ERA5 label evaluation, no official AI Weather Quest quintile RPSS, no original CIDM weights trained, no forecast issuance-time proof.

## A15-P2 preregistered paired evaluation (no neural training)
- Strict four Wed 00UTC initializations: `2017010400`, `2017070500`, `2018010300`, `2018070400`. Thursday issue reference; W3 Monday issue+18 through Sunday issue+24; W4 Monday issue+25 through Sunday issue+31, exactly 28 6-hour observations per week.
- Reuse the **actual A15-P1 user result ZIP embedded inside standalone Notebook with exact SHA verification** for the 2018-01-03 case. Other three dates require at most 3×560 additional GRIB2 ranges, using same proven S3 fields/index SHA/ETag/atomic per-four-record checkpoints and geometry/physics contract.
- Pair each weekly forecast with *real, time-exact* 6-hourly WB2 public ERA5 1.5deg conservative-regridded MSLP and T2m, not future ERA5 predictors. Strict coords, units, 28/28 samples and week UTC indices.
- Four-case area-weighted per-case MSLP/TAS W3/W4 ensemble-mean RMSE/MAE/bias vs verification, control RMSE, sample ensemble RMS SD, spread/RMSE, and empirical five-member CRPS (not official quintile RPSS). Preserve per-case forecasts, truth, metrics, provenance and source hash.
- Pass requires all four actual paired cases, no nonfinite fields, and verified report hashes; blocked cases fail closed with archived tracebacks, no silent lowering of N.
- Case count four is **not** a statistically meaningful generalization sample. Reforecast performance does not prove operational forecast availability. Before CIDM formal investment: enlarge independent dates, obtain train-only climatological quintile thresholds and official or validated RPSS, calibrate a strong ensemble baseline, and test CIDM as a causal residual/physics correction. No CIDM training authorized by this contract.
- Output Colab Notebook self-contained, no external private credentials, defaults to real run with automatic browser download of result ZIP + SHA; data volume and Colab runtime cost may exceed earlier source pilots.
- QA: entire 8-code-cell offline Notebook PASS; simulated four-case pairing PASS and simulated network interruption correctly returns BLOCKED ZIP. Actual A15-P2 cloud run **not yet performed**.
- Complete reproducible deliverable ZIP SHA256 `0bd128f4ee942b10196d5cb01063f9c9cdcd8da26751a166f71b718dda8e5834`, with exact original user input evidence embedded/pinned.

## Repository policy
GitHub research preregistration is a draft PR. HF `wuff-mann/AI_Weather_Quest` is unchanged; no checkpoints exist for A15-P2. No raw GEFS GRIB, large original ERA5 caches, secret tokens or passwords should be published.

## Data references
https://registry.opendata.aws/noaa-gefs-reforecast/
https://weatherbench2.readthedocs.io/en/latest/data-guide.html
https://aiweatherquest.ecmwf.int/evaluation-system/
