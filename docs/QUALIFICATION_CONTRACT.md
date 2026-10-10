# Q0–Q3 Qualification Contract

**Q0**: Verified official training dataset, Thursday issue dates, Day-19 / Day-26 Mondays, strict no-leak split with 32-day embargo, parameter/gradient audit.

**Q1**: Uniform quintile climatology, anomaly-persistence baseline. Compare validation/test RPSS *proxy* only, not leaderboard values.

**Q2**: 3 seeds each: CNN, CIDM-Local, CIDM-Generic-Rescue and CIDM-Structured-Flux. Same issue pool, batch, optimizer schedule, area-weighted RPS; CNN/Generic approximately parameter matched to Structured. Require candidate superiority beyond best validation control and positive independent test proxy before independent official mirror.

**Q3**: ECMWF official `retrieve_weekly_obs`, `retrieve_20yr_quantile_clim`, `conditional_obs_probs`, `work_out_RPSS` on held-out 2024–2025 forecast issue dates. Evaluate both Days 19–25 and 26–32. PRO 6000 investment permitted only on success, including year-wise and per-issue checks.

**Indicative gate** (project engineering decision, NOT ECMWF rule): official sample-week MSLP RPSS ≥0.25 in both windows, CIDM over matched CNN by ≥0.03 in each, sufficient fraction positive weeks and no underperforming years. These are historic mirror thresholds, not an official-period leaderboard rank prediction.

**Hardware**: L4 22.5GB for gate; RTX PRO 6000 Blackwell 95GB for separate formal global training only if Q3 passes.

**Artifacts**: GitHub stores code/reports; HF Model repo `wuff-mann/AI_Weather_Quest` stores checkpoints and weights. No auto competition submissions. Credentials only as runtime secrets.
