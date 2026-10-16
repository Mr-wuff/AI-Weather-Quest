# CIDM-MHC-Real A4 official duplicate verification → A5/P0 locked 2018–19 experiment

2026-10-16

## Original user-uploaded A4
- ZIP `CIDM_MHC_REAL_A4_P0_RISK_REGIME_REPORTS(1).zip`: SHA256 `8572ac348e4d722883f149f064ea5238d0de67a213de2db04fc5e6f7726ff269`; supplied sidecar matches, ZIP CRC PASS.
- Four entries in this user's ZIP are byte-for-byte identical to the four entries in the assistant's earlier A4 risk report; same `status.json`, original A3 source archive reference, 96 per-issue rows and frozen hypothesis. This is a successful *replay*, **not an additional independent trial**. Official status `COMPLETED_OFFICIAL_A3_RETROSPECTIVE_REANALYSIS_NO_TRAINING`; no optimizer updates, no new weather issues, 2017 target labels sealed, no AIWQ RPSS.
- Uploaded sync receipt shows HF `SKIPPED_DEFAULT` and GitHub `SKIPPED_DEFAULT`; do not claim archival sync occurred.
- A3 original 2016 reused validation data: Physical History relative Instant +9.9025% W3 and +6.7818% W4 pooled **coarse 3-latitude-band residual MSE**, win counts 27/48 and 25/48; Q2 loss regressions −17.18% W3, −43.67% W4. Retrospective fixed Q1+Q3 policy +12.72% W3, +13.99% W4 **same 2016**, not external proof.

## A5/P0 pre-registered test (not run on official data)
- Complete Colab `CIDM_MHC_REAL_A5_P0_FreshYearsSeasonGate_Colab_STANDALONE.ipynb`; independent driver `mhc_real_a5_fresh_year_season_gate.py` SHA256 `c8edeb5e5531dcf89110a25053e1787279e30ac2a54af235e77cd60d44aa4544`.
- 2018 and 2019 **both held as a single precommitted external experiment**, not separate chances to retune. 2017 forecast targets remain sealed; historical 2017 observations may appear as fully known 2018 causal inputs.
- Verify all original A0–A4 source and report SHAs, same original seed17/step64 MHC checkpoint (`e55f47fb3ecd1f1518e0ec629d1f24311b2e05f6a8fd07fe04ac7de1948528d6`) and 2016 12-issue replay. Never update weather backbone. Ridge heads fit 2013–14, hyperparameter tune 2015 and final refit 2013–15; no adaptation using 2018–19 labels.
- Compare Instant, always Physical, Old Memory, Q1-only Physical, predesignated **primary Q1+Q3 Physical**, all causal known-at-issue policies. Per-lead primary Gate: >=0.5% pooled MSE reduction, >=60% wins **among activation windows** (not all windows, since fallback windows necessarily tie Instant), 4-week moving-block gain CI lower bound >0, >=0% improvement in each of 2018/2019. Gate exploratory only, no permission for full MHC training or AIWQ submission.
- Original ECMWF weekly ERA5 data recovered from private HF Dataset; new annual 2018–2019 MSLP/TAS first downloaded only if missing and then uploaded in **one batch private commit** after SHA/grid/time checks. Source model in HF Model; result ZIP+SHA saved locally; optional one atomic report HF Model commit with opt-in, default no report upload to avoid 429. No Google Drive.
- Entire Notebook CPU smoke: **7/7 code cells successfully executed** in isolated kernel, pinned evidence A0–A4 rehydrated, synthetic calendar 2018–2019 issues, selection code and ZIP SHA/CRC PASS. **No actual new year dataset or model skill evaluated here.**
- Full research-delivery ZIP SHA256 `54e62b61519232992be79d509aead495c896bdf3eae7b3fa3f640408a348ae8c`.

## Scientific caution
The old physical-history probe is a low-dimensional *retrospective residual* correction (3 latitude belts × 2 variables), not a trained standalone global physical Markov model and not ECMWF AI Weather Quest official RPSS. All A5 skill must be reported specifically as new-year historical coarse diagnostic, not real-time/leaderboard skill. Once 2018–19 are evaluated they are no longer untouched for future research. If Gate fails, abandon date-only selector and move toward physically meaningful issue-time circulation/ocean drivers (Z500/SST and legally available S2S reforecasts), before any longer full training.