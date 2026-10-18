# A15-P3 LITE: 10.8 MB → 15 KB self-contained Colab loader (2026-10-18)

## Repository-backed source (committed)
- Engine: `research/earth_aiwq/a15_p3_lite/a15_p3_quantile_baseline.py`.
- SHA256 of engine: `4185e72d0c7da535f718adf56d4ab2c6dbd45df5297dcb6c5ea77f72d0369edd`.
- Pinned engine commit `4a2372187df3f7c006d4301457eeb82541aedaa0`; connector re-fetch confirmed exact original source (including final newline).
- Original standalone A15-P3 Notebook: 10,828,088 bytes due to base64/zlib embedded prior 8 MB A15-P2 report ZIP and full source.
- New LITE Notebook: 15,159 bytes; 4 executable code cells and one markdown cell, with **no embedded parent ZIP and no embedded source** (roughly 99.86% smaller).

## Input rule
- Strict accepted user-uploaded A15-P2 actual ZIP SHA256 `dd380fb310546b080314ecb8ad78a2edbecad68413ac21f7696ea6a9feda486c`; originally supplied sidecar `91292d2cc1e6a47b5a0eb031cddb12b360967bb7fa221b28be8fe95602e5b75d` DID NOT MATCH. The mismatch remains disclosed.
- Prior result auto-resolution: 1) already SHA-verified file in persistent Colab workdir; 2) explicit local/Drive path; 3) auto-scan HF `wuff-mann/AI_Weather_Quest` for exact basename (repo path and model/dataset/private token can be overridden); 4) explicit GitHub artifact HTTPS download URL; 5) one-time Colab upload.
- **No claim that the expected prior results ZIP already exists on HF or GitHub.** Only engine source was actually committed to GitHub. Never silently replace or skip a mismatched archive.

## Preserved A15-P3 scientific contract
- 2008–2016 ERA5 historical train-only climate reference (Jan/Jul × 27 weekly samples each; 28 six-hourly truth times for each W3/W4).
- Forecasts/truth four previously seen 2017–2018 GEFSv12 reforecast cases; globally gridded probability RPS + research proxy RPSS, uncalibrated 5-member empirical bins and prespecified half-count sensitivity.
- No official AIWQ RPSS, no new independent holdout, no operational input availability proof, no CIDM neural training, no formal training authorization.
- User-run real data access pending. Reduced notebook parsing and upload overhead does not remove real ERA5/cloud computation time.

## Offline QA and safety
- Full 4-code-cell Notebook offline executed with exact original source and input P2 file: `OFFLINE_QA_ONLY_NOT_REAL_CLIMATOLOGY` PASS.
- Tampered parent ZIP: correctly refused in isolated runtime, `BLOCKED_BOOTSTRAP_NOT_SKILL_RESULT`, ZIP includes only appropriate diagnostics.
- On restart clear stale `reports/` while preserving atomic `cache/`; process logs and immutable source + input SHA; ZIP CRC and matching SHA256 sidecar. Attempts Colab automatic browser download on both PASS and BLOCKED.
- Full delivery ZIP SHA256: `5dfcff911614fd073a42b8bb8b6879b5ace866fcbe03091903420691eb737f4d`, 16,639 bytes.
- Entire runnable LITE Notebook + local source fallback + README/contract are delivered as downloadable files in ChatGPT. The branch has the canonical engine source; draft PR not merged.
