# CIDM-Earth × ECMWF AI Weather Quest — competition-first mainline (2026-10-15)

## Strategic decision
Pivot CIDM-v2 from attempting universal PDEBench wins to real atmosphere-ocean Earth-system sub-seasonal prediction, scored externally by *official* AI Weather Quest RPSS. PDEBench B0/B2 and MHC A0/A1/A2 are mechanism research or regression tests, not the main optimization objective. The public competition does not directly score continuous 3D ocean variables.

## Official protocol checked on 2026-10-15
- ECMWF AI Weather Quest submissions: weekly Thursday 00UTC issue, lead targets days 19–25 and 26–32; 1.5-degree global quintile probabilities for mslp/tas/pr. Rank metric RPSS (with masks for TAS/PR), 0 climatology. MJO phase/Brier and tropical-storm days tercile/RPSS start public evaluation in DJF 2026, while SON 2026 remains a bridge for new targets. Official package AI-WQ-package evaluates and submits.
- Official annual ERA5-derived historical target data via `AI_WQ_package.retrieve_training_data.retrieve_annual_training_data`, requires registered AIWQ_PASSWORD. Additional S2S dynamical hindcast/issue-time ensemble predictors need availability-time provenance; **future ERA5 truth is never a predictor**.
- Documented in sources: https://aiweatherquest.ecmwf.int/competition-structure/ ; https://aiweatherquest.ecmwf.int/submitting-forecasts/ ; https://aiweatherquest.ecmwf.int/evaluation-system/ ; https://aiweatherquest.ecmwf.int/resources/ .

## Own archived previous weather baseline
The repository's `reports/20261010_L4_Flagship_v02_Q3_REVIEW.md` documents previous 1.018861M-parameter MSLP/TAS eight-week autonomous CIDM-S2S L4 v0.2: 20 historic 2024–25 official-scoring-mirror issues, **MSLP W3 mean RPSS 0.001480 and W4 0.007201**. Official-score retrospective already inspected; **not fresh future official leaderboard ranking**. Previous experiment classified STOP that specific input architecture / do not scale to RTX PRO6000 without new evidence. This does not rule out a new hybrid causal S2S forecast approach. Prior B2-R10-R14 SWE extreme long-adjoint amplification and MHC weak task-dependent signals do not qualify weather model training either.

## New preregistered research sequence
1. EARTH-AIWQ-P0, no optimizer or forecast submission: inspect verified official datasets using AI-WQ-package, 121×240 grid, year/date/units/source hashes; validate Thursday 00 UTC, eight completed historical weeks, targets issue+18/+25 Monday, 5-class probability simplex and basic RPS. Require password/official data; if absent BLOCKED, **do not claim real data PASS**.
2. P1 historical issue-time provenance: reproducible S2S ensemble/reforecast model/version/availability, ERA5 targets vs operational observations; introduce SST/Z500/MJO auxiliary input only if historically and operationally available at initialization; seal future outer periods. Compare climatology, unmodified S2S ensemble and calibration-only strong baselines.
3. P2 small-budget **CIDM-Earth hybrid**: slow/fast Markov representation, causal external S2S forecast port, learned anomaly/probability residual on appropriate probabilistic baseline; train latitude-weighted RPS and reliability; compare 3 weather lead/variable conditions, pretrained system/fresh seeds, fixed budget. MHC is an optional controlled ablation, not assumed beneficial.
4. P3 frozen outer validation and real weekly issuance/official package submission only on user authorization. This is the only path to actual AI Weather Quest ranking. Ocean forecasting (SST/SSH/currents) may become coupled auxiliary tasks/products but cannot count as current global mslp/tas/pr leaderboards.

## First research deliverables
Notebook `CIDM_Earth_AIWQ_P0_OfficialDataProtocol_Colab.ipynb`, driver `aiwq_p0_data_protocol_audit.py`, and detailed Chinese roadmap. Full standalone Notebook executed all 9 code cells in offline synthetic **smoke PASS**, including date/RPS simplex mathematical checks, source hash, ZIP/CRC/SHA256. Official ECMWF data/download **NOT RUN** because password/account is not provided to this chat. No score, model training or submission, no misleading synthetic result. Notebook optional HF Model and GitHub result sync with runtime write tokens; no RAW weather archive or passwords uploaded; no Google Drive. Deliverable bundle SHA256 `cd94262d7df36ee239fc441ef52bfeeb6761eb79bce683e4d552d0e327abae0d`.

## Gate
Do not invest in full-v2 global retraining or PRO6000 compute before operational predictors have trustworthy issue-time provenance and CIDM adds independently validated RPSS over a strong legal S2S/calibrated baseline. 2024–2025 previously observed dates cannot be relabeled as pristine outer blind test.
