# CIDM-MHC-Real A12 official ECBox run — negative global linear MJO signal; A12-R1 protocol
2026-10-17

## User actual output (independently audited)
- User ZIP SHA256 `860b3c36ce7fa446275be8a0f34de28f499ebf7147d5fdf3ddf585ab196235d1`; matches sidecar, ZIP CRC PASS. Status `A12_OFFICIAL_MJO_CAUSAL_PROXY_DIAGNOSTIC_COMPLETE`, scientific information gate false; NO neural steps, no official RPSS.
- Four ECMWF ECBox official annual MJO files 2012–15 downloaded with per-year SHA, 12 pinned private HF MSLP/TAS/PR annual files restored; original real .nc bytes are NOT in uploaded result report, cannot separately rehash raw input bytes locally.
- Fit: 2013–14 101 issues; reused 2015 dev 49; training-only 2012–14 quarterly climate and scaling.

| Target | MJO6 gain vs quarter climate | Weather6 gain | MJO dates beating climate |
|---|---:|---:|---:|
| W3 MSLP | -4.9491% | -3.5799% |20/49|
| W3 TAS | -5.1028% | -6.6877% |24/49|
| W3 PR | -4.3231% | -2.6627% |9/49|
| W4 MSLP | -4.8284% | -4.4969% |14/49|
| W4 TAS | -8.3051% | -7.0215% |15/49|
| W4 PR | -5.0568% | -2.8200% |5/49|
All six MJO global results worse than quarter climate; none beats all 8 training-shuffled MJO baselines. Original A12 fixed scalar lambda=10 direct full-global-grid Ridge; no shrink / inner-year hyperparameter selection / conditional MJO test. Outcome refutes that setup, NOT intrinsic MJO physical predictability. Exploratory 4-issue circular block bootstrap: five six lead-var confidence intervals for MJO-vs-climate negative, W3 TAS spans zero; not independent and 2015 already reused.

## A12-R1 protocol (NO net weather model training)
- Train-year time crossvalidation: restrict 2013 training issues to W4 labels in 2013 (49 issues); 2014 inner validation issues' W4 labels remain in 2014 (49). Fit climate & scaling in CV on 2012–13 only. Select Ridge lambda among {10,100,1000} and α ∈ [0,1] via closed-form weighted quadratic on **2014** independently for two predeclared panels: global all3 W3/W4 and Indo-Pacific tropics PR W3/W4 (15S–15N, 60E–180E). No 2015 information used for parameter choice.
- Refit complete 2013–14 101 training issues, with climate+normalization 2012–14; fixed hyperparams applied to reused 2015 49 dev. Compared climate, Weather6, MJO6, Weather+MJO12, and 4 within-year circular RMM misalignment controls (7,13,19,29 weeks). Report per issue, lead, variable, region, win count and true paired Weather-vs-Combined gain. Conditional MJO gate: IndoPac tropics PR both leads >=0.5% better than climate AND Weather6, >=27/49 wins each, strictly beat all four temporal-shift controls; global PR both leads worse than climate no more than 2%. This is exploratory information qualification only; **even a positive result does NOT authorize long training** due previously reused 2015 and absent official RPSS.
- MJO 14 days ending issue-4 Sunday historical ERA5 retrospective proxy; not proven operationally timely; 2016+ forecast truth unused. On official rerun first retrieve A12 user-evidence-pinned MJO annual NetCDF, strict original SHA. ECMWF official `sites-toolkit` and AI-WQ-package authentication via Colab Secrets. No Google Drive; no upload by default.

## Verified A12-R1 artifacts
- Notebook SHA256 `2bd16585f00031dad6947e5ce474e4cfabf0bf769030bd25acbad08517b427b8`
- Standalone driver SHA256 `5e0292f21fec1b81056d99dfd6599b5f5ae145d3ffe38429b1d1b889402e8e06`
- Complete research ZIP SHA256 `9cb123699cfb6b3263e7bf190c4423339cdf2a00d46051a599e6c6989af06898`
- Original A12+R4 uploaded ZIP and exact prior source embedded with SHA lock. Whole Notebook **4/4 code cells CPU smoke pass**, actual full-size 121x240 synthetic weather mock executes full 2013/2014 CV + 101 train / 49 dev and all 7 models, four cyclic control models, correctly exports metrics. Forced no-HF auth failure generates traceback/status/report with CRC. **Official real-data A12-R1 has not yet run.**

Scientific decision: no new CIDM training until reliable conditional S2S weather signal. If A12-R1 negative, freeze MJO idea for now and prioritize initial Z500/SST / long historical training with true end-to-end causality and sufficient independent blocked years.