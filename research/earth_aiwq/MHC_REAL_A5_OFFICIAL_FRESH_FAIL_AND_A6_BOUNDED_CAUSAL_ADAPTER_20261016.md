# MHC-Real A5 official result: calendar gate fails on fresh 2018-19; preregister A6 causal bounded adapter

Date 2026-10-16. User A5 ZIP SHA256: 0c5c74bc58b5729dd99219305b58433a819b133f080648190336d8493b5117e7. Supplied SHA sidecar and sync receipt match, ZIP CRC PASS. Original report status FRESH_2018_2019_EVALUATION_COMPLETED_NO_WEATHER_TRAINING, 98 external historical weekly Thursday issue dates (49 each 2018 and 2019), original seed17 step64 head and 2016 replay verified by Colab, 2017 truth sealed, ZERO new weather-network training. HF new report sync SKIPPED_DEFAULT; raw 2018/19 reported to private HF dataset, not in uploaded ZIP.

All results are 3-zonal-belt/2-variable **coarse residual diagnostic MSE**, NOT operational forecasts, native grid skill, or ECMWF AI Weather Quest quintile RPSS.

| Method | W3 MSE | W3 gain over Instant | W4 MSE | W4 gain |
|---|---:|---:|---:|---:|
| Instant | .003176178747 | 0 | .002967883542 | 0 |
| Old MHC Memory | .003144659098 | +.992376% | .002939321020 | +.962387% |
| Always Physical | .003435304934 | -8.158426% | .003293855604 | -10.983317% |
| Q1 Physical | .003162472410 | +.431535% | .002971364403 | -.117284% |
| **Q1+Q3 Physical**, precommitted A5 primary | .003244170802 | **-2.140687%** | .003106747726 | **-4.678896%** |

Primary A5 gate FAIL; activated wins 20/52 W3 and 18/52 W4, yearly primary gain 2018 -6.700%/-17.668%, 2019 +3.438%/+8.224%. Previously attractive seasonal retrospect failed to transport. Always-on physical features also negative in both leads; reject date-based always-on injection.

Old Memory gain is small positive both years (2018 W3/W4 .510%/.491%, 2019 1.582%/1.431%) and 56/98,57/98 improving issues. Independent 5,000-draw four-week moving-block DESCRIPTIVE CIs for memory gain [-0.202%,1.972%] W3, [-0.301%,1.843%] W4, overlap zero; NOT proof of robust Markov sufficiency. Also does not mean original recurrent network itself gained 1%, only frozen learned ridge residual diagnostic.

New **A6/P0**: bounded simplex convex physical/history residual adapter, preserves Old Memory as meaningful baseline. y = (1-alpha)y_Instant+(alpha-beta*g)y_OldMHC+(beta*g)y_Physical; alpha=.5/1, beta=0/.05/.1/.25; g is causal preissue last-4-week/last-8-week volatility ratio, attenuation when recent 4wk volatility exceeds 8wk. No calendar/round labels, no future ground truth in inference. 2013-14 101 pre-embargo fit cases; 2015 49 lambda/blend tune cases; 2016 48 **reused development** safety check (not blind). Require >=.2% both leads over Old Memory in 2015 and >=.2% both leads over Old Memory in 2016, cvar tail ratio <=1.05 and positive over Instant; if any false **never access 2020 or 2021 target truth**. Only if ALL development gates pass, evaluate once on entirely fresh 2020-21, no 2017 label access, no 2018-19 target score reuse; 2019 past observation is causal context. External threshold relative Old Memory and Instant both >=.5% both W3,W4, >=60% winning dates over Old Memory, no year regression. Diagnostic only, not formal MHC training or AIWQ RPSS.

Source: mhc_real_a6_causal_bounded_adapter.py; standalone Colab CIDM_MHC_REAL_A6_P0_CausalBoundedAdapter_Colab_STANDALONE.ipynb. Exact A0-A5 historical evidence and source code embedded, all original SHAs verified, bit-exact original 2016 replay & historic ridge numerical replay required in official. 8/8 notebook code cells executed under CPU smoke with REAL archived evidence and synthetic math tests; real-weather A6 NOT RUN. Full research ZIP SHA256 **5fbcd51f34a43c494623837b61ca5f99169dff2d695de67429aec1b53eef0cf3**. Private HF Dataset for raw caches, HF Model at most one atomic output ZIP+SHA commit, optional GH source/status sync, no Google Drive; persistence receipts separately verified.