# A15-P3 real outcome and A15-P4 LITE frozen 2019 evaluation preregistration (2026-10-18)

## User-uploaded actual A15-P3 LITE verified
- Current uploaded report ZIP SHA256: `0a60d66483bdf81d9e7bdeacf72c40d7eb1e4cabd8f7d6cb6691cc61c903ebd9`. User sidecar matches, ZIP CRC PASS.
- Github A15-P3 engine provenance commit `4a2372187df3f7c006d4301457eeb82541aedaa0`, source SHA256 `4185e72d0c7da535f718adf56d4ab2c6dbd45df5297dcb6c5ea77f72d0369edd`.
- REAL ERA5 reference: 2008–2016, Jan and July, 27 historical weekly means per season; 4 train-only 20/40/60/80% thresholds per MSLP/TAS × W3/W4, `TRAIN_ONLY_PROXY_QUINTILES.npz`, each season `[4,2,2,121,240]`.
- Evaluated GEFSv12 reforecasts of 2017-01-04, 2017-07-05, 2018-01-03 and 2018-07-04. These four dates are already inspected and must NOT count as fresh holdout.
- Research proxy RPSS (pooled GEFS RPS against pooled uniform climatological RPS in order W3 MSLP, W4 MSLP, W3 TAS, W4 TAS):
  - Raw empirical five-member probabilities `-0.148631, -0.069050, -0.139621, -0.205797`.
  - Predeclared 0.5 half-count probability smoothing `-0.021058, +0.027598, +0.037053, -0.008613`.
- Interpretation: five-member bin-count probabilities insufficiently calibrated on this tiny four-case development set. Smoothing helps but does NOT uniformly beat the climate reference. No CIDM neural training, no official AIWQ RPSS, no operational availability.
- Historical quintile climate estimated from only 9 training years × 3 neighboring weeks per season (samples correlated); reference proxy not an official climatology.

## Next A15-P4 contract (freeze, no 2017/18 re-tuning)
1. Two **not previously evaluated** GEFSv12 reforecast initializations: Wednesday 2019-01-02 and 2019-07-03 at 00UTC; reference Thursday issue, 28 six-hour target times per W3 and W4.
2. Strictly read ONLY exact A15-P3 ZIP SHA above to recover original fixed 2008–16 Jan/Jul quintile thresholds. No 54-historical-case redownload.
3. Reuse audited A15-P1 and P2 S3/GRIB2/ERA5 reading libraries committed in `research/earth_aiwq/a15_p4_lite/`; preflight all ten member-variable objects and 56 leads per object prior to substantive S3 reads, then download up to 2 × 5 × 2 × 56 = 1120 ranged GRIB messages, checkpoint every four.
4. Pair forecasts with identical valid-time WB2 ERA5 2019 weekly-mean truth. Compute MSLP/TAS W3/W4 area-weighted RMSE, empirical CRPS and **RESEARCH-ONLY** five-bin RPS/RPSS; raw and fixed half-count 0.5. No outcome-based search.
5. **Eligibility**: genuine two new paired dates, physical units, times, spatial grid and SHA audit. Any source missing => BLOCKED rather than silently shrinking test. Optional all-four-channels, both-dates positive half-count proxy is a *strict two-case candidate*, NEVER authorization for full CIDM training.
6. GEFSv12 2019 data are retrospective **reforecasts** not 2019 historical operational forecasts, cannot claim AIWQ input compliance; proxy quintile thresholds and scores are NOT official RPSS. Need much larger independent years and historical operational provenance before model training.

## Lightweight reproducible delivery
- Standalone Colab LITE notebook approximately 13 KB; nothing from old P2 or P3 embedded.
- GitHub path `research/earth_aiwq/a15_p4_lite/`, pinned commit `5a46d39b51aa142359193ed4e680a24a90f11aef`; four modules checked by fixed Git blob SHA1 (each module's SHA256 also logged).
- The exact A15-P3 ZIP ~2.9MB is read from cached local/Drive or HF repo `wuff-mann/AI_Weather_Quest` IF present, else one-time user Colab upload; never assume previously uploaded to HF.
- Auto-download report ZIP plus SHA256 on success OR blocking failure. Persist checkpoint in Drive if user chooses.
- Offline 4-code-cell Notebook QA with local equivalent driver PASS. Deliberate mismatched parent-input SHA generated `BLOCKED_BOOTSTRAP_NOT_SKILL_RESULT` and verified ZIP CRC + sidecar. **The final remote P4 compact module was not executed locally because public raw GitHub internet is unavailable here; GitHub source contents and exact blob IDs were verified through connector.**
- Full ChatGPT deliverable research ZIP SHA256 `94a0b1a97eae5470f8a96cb0308782639f6a189df4edea4466f9cd75c32218b7` (~10 KB).
