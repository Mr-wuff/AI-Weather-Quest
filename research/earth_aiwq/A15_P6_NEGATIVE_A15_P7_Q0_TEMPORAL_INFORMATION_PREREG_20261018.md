# A15-P6 measured failure → A15-P7-Q0 real-trajectory temporal information necessity (preregistered 2026-10-18)

## Parent evidence
A15-P6 genuine six-case fixed edge-aware categorical smoothing failed to outperform equal-strength isotropic smoothing AND shuffled spatial-gate placebo on all MSLP/TAS × W3/W4 heads; no CIDM training and no official RPSS. Parent case archives are six **previously seen** dates; do not re-tune them to chase the Gate.

## Question and design
Are **forecaster-issued temporal GEFS member trajectories** informationally useful for forecasting spatial blocks of W3/W4 weekly-mean ensemble-forecast absolute error, after controlling frozen single-issue weekly-mean spread, seasonal climate displacement and spatial gradient?

- Six existing 2017/18/19 paired GEFS–WB2 ERA5 Wednesday 00UTC dates: 2017010400,2017070500,2018010300,2018070400,2019010200,2019070300.
- Exact parent ZIP SHA locks: P2 `dd380fb310546b080314ecb8ad78a2edbecad68413ac21f7696ea6a9feda486c` (actual previously uploaded bytes; originally paired sidecar mismatch disclosed); P3 `0a60d66483bdf81d9e7bdeacf72c40d7eb1e4cabd8f7d6cb6691cc61c903ebd9`; P4 `10e471b02b94583c0b6a59857d1f9f6339c3ab2a498845e429a621bb34a25871`.
- Existing paired weekly averages are **not temporal trajectories**. Source missing trajectory data must come from real NOAA S3 GEFSv12 reforecast: five members c00/p01..p04 × 2 variables MSLP/TAS × 7 exact forecast lead hours `f408,f456,f504,f600,f624,f696,f786` × 6 dates = <=420 real GRIB2 range messages. Same initialization, no future ERA5 predictors. Strict object ETag / IDX hash / GRIB2 time/units/spatial grid QC and atomic resumable message caches.
- Four-patch-step temporal features per target: path length, curvature and local cross-scale step contrast (3×4 area-weighted patches), matched static features: mean ensemble spread, train-period climate displacement, field gradient. MSLP and TAS are handled separately with exact physical units.
- Frozen closed-form ridge L2=8 in train-years only, 3 leave-one-year-out folds (2017/18/19), exact matched six-dimensional controls `ordered` / `middle-time-permuted` / `season-matched wrong-year trajectories`; baseline 3D static feature probe.
- Scientific metric = heldout-year patch-level RMSE **predicting absolute WEEKLY ENSEMBLE FORECAST ERROR MAGNITUDE**, NOT forecasting weather skill or official RPSS. Region patch correlation and 6 already-seen dates prevent generalization claims.
- Exploratory candidate only if one variable has BOTH lead weeks with >=1% mean reduction relative static, >0.5% vs both negative controls and a positive sign in every leave-year-out fold. Even positive outcome **only authorizes designing a new independent-date gate**, never CIDM formal long training.
- Retrospective GEFSv12 generated later: NOT proof of historic operational source availability or permission to use this source in live ECMWF AI Weather Quest.

## Deliverables tested and provided to user in ChatGPT
- `CIDM_EARTH_A15_P7_Q0_TemporalPath_LITE_Colab.ipynb`: ~20 KB (4 code cells; no historic report ZIP or forecast GRIB embedded). Uses pinned real GitHub A15-P1 S3 ingest source at `bb1583fc7078e74f251ec101d0e11a522f2a9eeb`, exact SHA256 `4e88bee38ccce563d9cfd75e457e9a35070439b972dc7fc742a9e4b23520a98e`. New 20KB P7 engine embedded compressed with exact SHA256 `56a80ad26bf88f2c90fdd4579b5e3d306aaa519767c3bd73fbf9a19b1fb4eaef`.
- Reuses prior archives from SHA-validated local/Drive/HF source (and optional one-time Colab upload). No assertion parent files already exist in HF.
- Automatic browser download ZIP+SHA for PASS/BLOCKED, no raw GRIB in final report. Optional Drive cache for Colab runtime reset.
- Entire Notebook executed in offline QA: `OFFLINE_QA_PASS_NO_REAL_S3_OR_GEFS_LABEL_EVAL`. Additional tests read real three previous parent archives, simulated range S3 7/7 records+resume, missing f786 correctly rejected, fake temporal features/year-fold eval. Corrupt parent SHA -> BLOCKED with fresh report, no stale QA.
- **Real S3 temporal retrieval NOT executed** as of preregistration; no measured P7 gain or neural training. First user run should upload the ZIP + SHA256.
- Downloadable compact ZIP including all readable source, Notebook, mock test, pinned P1 fallback and Chinese protocol is `CIDM_EARTH_A15_P7_LITE_Temporal_Information_Qualification_FULL.zip`, sha256 `feafd501c6383f51f9fb0d5145957396d02752b8789edacce4e376d532e56837`.
- GitHub PR contains preregistration and links to project work; readable P7 source shipped in user ZIP, not yet committed in this branch. HF remains unchanged.

## Follow-up
Depending on real P7-Q0 info Gate: if negative, stop simple fixed temporal features; if positive, first acquire truly unseen forecast dates, compare matched neural/conv/temporal/CIDM dynamics at equal budgets and evaluate official compatible calibrated probabilities with operational availability checks. No long training without mechanism qualification.
