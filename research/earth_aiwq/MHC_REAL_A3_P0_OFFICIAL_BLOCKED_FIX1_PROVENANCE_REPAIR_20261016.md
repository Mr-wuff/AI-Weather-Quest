# CIDM-MHC-Real A3/P0 — actual user blocked run and FIX1 provenance repair
Date: 2026-10-16.

## Uploaded official A3 report
- `CIDM_MHC_REAL_A3_P0_PHYSICAL_STATE_REPORTS.zip`: SHA256 `98cf5e6912f1fabd72bc0a2072bf9192e9674875f3e44fae14858e5255260a05`; sidecar exact match, ZIP CRC PASS.
- ZIP contains only `reports/status.json` and `source/mhc_real_a3_physical_state_probe.py`. Final status `BLOCKED_NO_TRAINING` with `TypeError: argument should be a str or an os.PathLike object where __fspath__ returns a str, not 'module'`.
- Failure at original source `official`, status construction: `sha(parent_source())` where `parent_source()` returns imported Python **module** rather than source file path. A3 official W3/W4 physical-history and placebo metrics were **NOT archived**; neither a pass nor a failure of the scientific Gate can be concluded.
- User's executed Notebook outputs show restoration of all 12 private HF Dataset annual ERA5 MSLP/TAS files 2012–2017, successful HF checkpoint restore and A2 frozen feature extraction through 2016 validation. No forecast-model optimizer updates, no 2017 independent outer evaluation, no AIWQ RPSS.

## FIX1 without changing science contract
- Use `Path(parent.__file__).resolve()` as a valid path to hash the original imported A2 source and compare it against immutable source SHA `4d6edafa009d09813de15241b8c485b193578bd9782117855f69547f96bf5063`.
- Preflight provenance/file/JSON serialization *before* downloading the ~500MB original ERA5 from private HF to Colab. Embed user A3 blocked ZIP and its SHA as locked historical evidence.
- Keep 2013–14 train (101), 2015 tune (49), 2016 reused validation (48), sealed 2017; same frozen seed17 checkpoint, original MHC memory vs causal 8-week MSLP/TAS physical historical features, 16 globally randomized + 16 season-preserving random controls, ridge tuning, 4-week block diagnostics, original preregistered Gate. No new model training.
- Preserve separate immutable report ZIP and sidecar; optional one-time HF Model commit, GitHub light report only; no Google Drive.

## Verified FIX1 delivery
- Source `mhc_real_a3_fix1_physical_state_probe.py` SHA256 `b4b13d3ea1d95f18e745c9152c9766db8132184140df93759569fb77c2de4d60`.
- Full standalone 9-cell Notebook `CIDM_MHC_REAL_A3_FIX1_PhysicalHistoryState_Colab_STANDALONE.ipynb` all 5 code cells executed in isolated nbclient CPU SMOKE, including the exact provenance serialization preflight that blocked the official run. Exact A0/A1/A1-FIX1/A2 and A3 blocked evidence SHA256 all PASS. ZIP CRC and hash passed.
- Complete delivery ZIP SHA256 `0a4b4e7cbe761da8a2f42671e003ac93554e71f68dd9b4a3246e5755f4eb023b`. Official A3 FIX1 scientific run NOT YET performed.