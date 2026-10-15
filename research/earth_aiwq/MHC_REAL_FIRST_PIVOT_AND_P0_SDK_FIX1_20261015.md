# MHC-Real before CIDM-Earth — corrected research ordering, actual P0 blockage and FIX1 (2026-10-15)

## Critical decision
MHC is not finished or validated as a broadly effective Markov closure mechanism. Do not prematurely replace it with a separate CIDM-Earth backbone. Instead take ECMWF AI Weather Quest actual data as the development/test distribution: **MHC-Real A0 data → MHC-Real A1 fair statefulness training → MHC-Real A2 full predictor/physical consistency → CIDM-Earth integration → official AIWQ RPSS**. PDEBench remains a diagnostic not the optimization target. Current competition scoring is not an ocean-variable leaderboard.

## Uploaded actual P0 (independently inspected)
User file `CIDM_EARTH_AIWQ_P0_PROTOCOL_REPORTS.zip` **SHA256 dbf2060b94726dc539904863bf3fb11f29c6ae5c1685f33a13d3b423936d8667**, user sidecar EXACT, zip CRC PASS. Run `status=BLOCKED_NO_TRAINING`, data_status FAIL_CLOSED, **error ModuleNotFoundError: No module named 'sites'** when importing `AI_WQ_package.retrieve_evaluation_data`, via `from sites.sdk.sites import Site, Authenticator`. User Notebook package install succeeded and smoke contract passed but **no real weather field was downloaded**. This is SDK dependency rather than weather-data or MHC efficacy failure.

Official ECMWF documentation after 2026-08-10 ECBox transition prescribes:
`python -m pip install "sites-toolkit>=20261005.1" -i https://get.ecmwf.int/repository/pypi-all/simple`
in addition to `AI-WQ-package`; see https://ecmwf-ai-weather-quest.readthedocs.io/en/latest/installation.html.

## New FIX1 deliverable (created in ChatGPT; official NOT RUN)
- Self-contained `CIDM_MHC_REAL_A0_FIX1_OfficialDataBridge_Colab.ipynb`, separate original driver `mhc_real_a0_data_bridge.py`, Chinese roadmap, README, original P0 blocked evidence, internal per-file SHA manifest, complete ZIP **SHA256 a00b5a35d8be78a0f310cef629e84da66c0d88e0da086b2209f4a34fcd567c29**.
- Official 2012–2017 MSLP/TAS weekly 7-day ERA5-derivative annual files, validate 121×240 1.5° coordinates, dimensions, units, SHA256, weekday start labels. STRICT Thursday 00UTC issue samples, eight complete preceding weeks, W3 Monday issue+18 days, W4 issue+25 days. 2013–15 TRAIN, 2016 VAL, 2017 old historical audit (not independent AIWQ official RPSS), no future information in inputs, no derived ocean/operational sources falsely claimed.
- NO training, no evaluation score, no raw weather/credentials in outputs. Stops if missing authenticator/files/samples. Each downloaded ERA5 annual dataset is hindsight historical truth; additional provenance checks are required for live-issue legal inputs. Optional Colab secret AIWQ_PASSWORD.
- **Actual local tests**: every code cell in the complete Notebook executed in CPU smoke mode PASS; Python compilation PASS; synthetic 121×240 53-week NetCDF parser PASS; simulated Monday-date causal sample builder PASS (157 TRAIN, 52 VAL, 47 old-audit synthetic metadata examples). These are PROGRAM TESTS only, **not official data result**, and cannot authorize MHC training.
- Full data bridge code and Markdown are downloadable in the ChatGPT message; notebook defaults to official and will ask for authenticated official download in user Colab. No Google Drive, no raw data public sync.

## Future measured MHC-Real A1 after successful official data bridge
Compare **stateless closed state, persistent slow-fast Markov state, persistent without multi-horizon aux** with same original CIDM dynamic core and matched trainable params, RNG, budgets and issue-time inputs. Only after Q0 proves zero-correction replay, physically legal forward and finite useful gradient, run a small Q1. Score W3/W4 per-variable physical MSE as diagnostic and official RPSS only with valid official 20-yr quintile climatology and legal probabilistic forecast. Strong climatology plus available S2S ensemble/calibration baseline and future operational availability must be established before leaderboard claims. Previously inspected 2024–2025 historical flagship dates cannot be fresh holdout.

## Historical MHC evidence
A0-FIX1 RD and SWE gained less than 1% relative late pooled MSE in short pilot; persistent MHC is slightly worse than stateless in RD. A1 removal of memory with equal action RMS **improves RD 0.069122% (16/16 windows), harms SWE 0.020630% (15/16 windows)**. Thus neither multi-PDE superiority nor Markov sufficiency established, and A2 memory innovation test has not been returned with new user official evidence. No full training qualification.
