# A15-P0-FIX3 confirmed MARS ambiguity → A15-P0-FIX4 AWS S3 GEFSv12 pilot (2026-10-18)

User A15-P0-FIX3 ZIP SHA256: 4520d6f86be2f08ed5ce6432a5603e1a221794d524ab56663a2788e3b777f2e7; sidecar/CRC PASS.
Actual ECDS request a47a5589-c8f0-462e-9de9-3cbc78ee21f3 accepted then failed. Even official-style 2023-01-30 +24h control TAS request returned HTTP 400 MarsRuntimeError: class s2 ambiguous SWEDEN/SWITZERLAND; 2m_temperature ambiguous -1K/-2K anomaly. No delivered GRIB and no official RPSS. Support ticket is included in user-facing deliverable.

A15-P0-FIX4 source: anonymous AWS S3 bucket noaa-gefs-retrospective (distinct NOAA PSL/ECDS infrastructure). Genuine GEFSv12 **reforecasts** from 2000–2019, 5 members daily and 11 members once weekly, up to 35 days. THESE ARE RETROSPECTIVE AND DO NOT PROVE HISTORICAL OPERATIONS OR REAL-TIME AIWQ ELIGIBILITY.

Pilot checks real 2018 Wednesday-init 00UTC object directories for 35-day control forecast files pres_msl and tmp_2m. Following Thursday 00UTC issue yields W3 456..618h and W4 624..786h, each exactly 28 six-hour time points. Verify actual S3 inventory and index contents, and when byte offsets exist, download one max 8MB ranged GRIB2 record per variable, check header and optional ecCodes physical metadata; never infer completeness from filename alone. No full file downloads, model training or RPSS.

Offline self-contained Colab test: all 8 executable cells PASS, missing lead and missing 35-day folders do not falsely pass. No real S3 retrieval executed in assistant environment. Default notebook real online probe, automatic result ZIP and SHA256 local browser download, GitHub/HF optional lightweight sync only upon explicit user opt-in. Research contract and ECMWF support ticket archived inside the deliverable.

Deliverable ZIP SHA256: 0c1d86f2db1ea78aa81312d1be1a6a57f15b04349dfcada34773d21a5c3dd2b0.

Any S3 port pass is RESEARCH ONLY. Full W3/W4 member means, quintile calibration, operational availability audit and official RPSS with fresh dates remain mandatory before CIDM-v2 formal training.

Official NOAA references:
https://registry.opendata.aws/noaa-gefs-reforecast/
https://noaa-gefs-retrospective.s3.amazonaws.com/Description_of_reforecast_data.pdf
