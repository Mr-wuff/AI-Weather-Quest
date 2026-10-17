# A14 WB2 Q1 formal evidence → A14-R1 fixed-data alpha diagnostic (2026-10-17)

## Uploaded A14 results verified
- User-uploaded `CIDM_MHC_REAL_A14_WB2_CONDITIONAL_REPORTS.zip` SHA256 `7b2a5f1848c0a925dfcf96f4fb9c4f6de0a42e8602f0aeaf61dfa925b930df41`, ZIP CRC PASS; Notebook 11 cells executed.
- Status: `A14_REAL_WB2_EXPLORATORY_CONDITIONAL_Q1_COMPLETE_NO_CIDM_TRAINING`; 202 cases total; sampled 2008–2011 train 102, 2012 validation 24, 2013–14 reused development 50, 2015 input-only 26. Data from WB2 ERA5 6-hourly conservative 1.5°: 121×240 native input, 3×4 coarsened diagnostic outputs.
- Five ridge residual probes (surface / +Z500 / +SST / +both / wrong-time sham) *all* chose global `alpha=0` on validation. Thus all reported development errors equal the seasonal climatology baseline, not five equally performant learned prediction systems.
- Climate MSE (W3 MSLP, W3 TAS; W4 MSLP, W4 TAS): 3.920610 hPa² / 0.319479 °C²; 3.408862 hPa² / 0.336786 °C². Numerically ~0 conditional skill and 0 per-issue wins because predictions coincide.
- Time/coordinate/provenance checks and 2015 no-label contract completed. No official probabilistic RPSS, no formal CIDM training, and no proof of issue-time operational availability.
- Mechanism gate FAIL. Interpretation limited: fixed 3×4 coarse features, ~102 fits and one global shrinkage over four heads with discrete alpha {0,.25,.5,1} is too restrictive to settle physical informational value of Z500/SST.

## A14-R1 preregistered minimal hypothesis test
- Frozen same true WB2 A14 cases/splits, no new learning of CIDM weights, no fresh holdout labels.
- Exact original global-alpha replication plus alternative *only*: independently choose a ridge lambda (1/10/100/1000) and finer alpha (0,.01,.025,.05,.1,.15,.25,.4,.6,.8,1) for each of the four MSLP/TAS × W3/W4 heads using **2012 validation exclusively**, then refit on 2008–2012 and report 2013–14 development without retuning.
- Paired spatially area-weighted MSE versus frozen seasonal baseline, surface-only and wrong-time sham; inspect chosen alphas, correlations, per-year gains, and strictly input-only 2015.
- Exploratory candidate only if one variable on BOTH lead windows improves over surface >=0.5% and over sham by >=0.5%, and other heads degrade no more than 2%; even PASS **does not authorize formal long training** or official leaderboard claims.
- Full standalone Colab Notebook `CIDM_MHC_REAL_A14_R1_FineAlpha_ConditionalGate_Colab_STANDALONE.ipynb`; executable `a14_r1_conditional_audit.py`, A14 source bundled and SHA-locked.
- Verified CPU full 7/7 notebook code cells **offline** with live WB2 purposely disabled, positive synthetic Z500 signal test PASS, and fail-closed missing-cache test PASS. No A14-R1 real evaluation has been run.
- A14 upload *report ZIP excludes 202 raw cache NPZ*, so a new Colab runtime must explicitly refetch public WB2 or reuse persisted A14 cache. Automatic per-issue atomic write permits resume.
- GitHub/HF lightweight sync is opt-in at runtime using configured credentials. Never publish raw ERA5 cache/secrets/private data.

Decision: analyze A14-R1 before expanding causal history/SST/Z500 features or committing full CIDM-v2 formal training.
