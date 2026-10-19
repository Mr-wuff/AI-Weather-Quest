# A15-P13 real 16-case diagnostic FAIL → A15-P14 new-year frozen spatial residual transfer

Preregistered 2026-10-19. This is a strict **research/diagnostic stage**, not CIDM network training, official AI Weather Quest RPSS, or proof of historic operational forecast input eligibility.

## Parent, real evidence
- 16 real 2002/2004/2006/2010 train-year five-member GEFSv12 W3/W4 forecast–ERA5 pairs: actual Train16 ZIP SHA `957b66c3d3e04201d5ebcb6e077599844f755683863c47bc71f98c81ea527cc1`, complete 16/16 tensor/file SHA and source checks previously audited. 2014 validation and 2016 test years are untouched.
- A15-P13 real per-grid weighted 11×12 block moments `E[e],E[e²]`, authentic 65,632-byte NPZ SHA `cae9ce1c6dead8ac2a9a1a209c8ce1dd7329ed00582577bad9d32b1040f69d4a`. Exact weighted RMSE reconstruction verified on all old cases.
- P13 frozen static spatial shrink + no-polar-wrap 3×3 smoother, scale 0.5: 4-year LOYO MSLP W3 +3.906% (4/4 years), W4 +0.608% (0/4), TAS W3 +1.930% (2/4), W4 +0.955% (1/4); therefore joint two-week **FAIL**. These are calibrated GEFS spatial bias baselines, not CIDM weather skill.

## P14 true prospective data contract
- Train/freeze estimator using **only** old 16 cases, 2002/2004/2006/2010. Do not tune it on the new years or fit it on new ERA5 verification labels.
- New four distinct research years: 2003/2005/2008/2011, none part of the 2014/2016 holdouts. Stage `pilot4` issues `2003040200` (April), `2005070600` (July), `2008100100` (October), `2011010500` (January). Four cases × five members × MSLP/TAS × W3/W4 each 28 six-hourly messages = **2240** GRIB messages.
- Stage `expand16`: 4 first-Wednesday months Jan/Apr/Jul/Oct of each new year, total **16** cases; incremental cache reuse from pilot4; at most 8960 forecasts, 16 paired ERA5 weekly truths. 2014 and 2016 ERA5 verification truth MUST NOT be fetched.
- Reuse exact SHA-pinned public GitHub P1/P2/P11-FIX1/P12 sources. Validate 10/10 member×variable source HEAD/ETag+IDX SHA and exact 10-digit issue hours for every case, every paired-case forecast/truth tensor digest, and reconstruct original raw ensemble mean RMSE from moments.
- Frozen controls: zero correction, global mean-bias correction, spatial correction with 0.5 scale and 3×3 non-polar-wrap smoothing, 4 longitude-block shifted placebo. Area-weighted exact moment scoring. True future ERA5 only used as EVALUATION labels, never as a predictor input.

## Qualification gate
- `pilot4` is a *directional/engineering pilot only*, reports `candidate_gate_eligible=false` regardless of score.
- Only all **16/16** `expand16` real cases permit candidate Gate evaluation. For the same variable BOTH W3 and W4 must meet all:
  1. spatial bias vs raw ensemble-mean RMSE improves at least 1.0%;
  2. spatial bias vs frozen global correction and shifted-lon placebo improves at least 0.5% on each;
  3. these requirements pass independently in >=3/4 NEW years, with all four seasonal issues in each passing year.
- Candidate success would qualify a **strong static spatial calibration baseline**, not an advantage of the actual CIDM GeoPort neural dynamics. A neural CIDM pilot still requires more dates, matched convolution/EMOS baselines, shuffled/zero port & flux ablations, >=3 seeds and untouched 2014/2016 evaluation.

## Delivered artifact and tests
- Lightweight 109,224-byte self-contained Colab Notebook includes only SHA-locked compressed small P14 engine and the exact P13 65 KB derived statistics, **not old 31 MB dataset nor historical GRIBs**. Default user settings: `STAGE='pilot4'`, `USE_DRIVE=True`, automatic ZIP+SHA download on success/BLOCKED.
- Readable P14 engine SHA `90499e741d9247d55a6588c75c038f4d5e7476656b7376b8479d2a5e37532b34`, delivered in FULL ZIP. P1/P2/P11/P12 real stable sources pinned by Github commits+hashes, complete verified fallbacks included in FULL ZIP.
- Complete user-downloadable code/data/QA ZIP: `CIDM_EARTH_A15_P14_G2_E2_NewYear_Prospective_LITE_FULL.zip`, SHA `7cbb1228576aa84ecaf931db0db75e418e98e97d4fd8f7b8437f9a7e5fdc3fde`, 202,011 bytes, CRC+per-file manifest PASS.
- Genuine P13 stats reproduce the old genuine Train16 RMSE for MSLP W3 5.61343, W4 5.66493; TAS W3 2.24942, W4 2.36500. Independent genuine old forecast/truth pair reproduces area-weighted moments within 2e-4.
- Full Notebook offline QA PASS. Separate mock-only full pilot4, resume without redownload, and cached array corruption -> BLOCKED PASS, all mock weather results **deleted**. Tampered embedded source SHA in Notebook -> BLOCKED diagnostic ZIP+SHA PASS.
- **NEW 2003/2005/2008/2011 GEFS+ERA5 DATA NOT RETRIEVED YET.** No real P14 outcome, no CIDM neural training, and no official RPSS; the mock success must never count as scientific evidence.
- User-authorized optional private HF repo only, with SHA readback. Public HF writes disabled, no HF upload actually occurred. If not configured, completed pair archives remain in the persistent user's Google Drive.

Research source dependencies: `research/earth_aiwq/a15_p4_lite/a15_p1_gefs_weekly_ensemble.py`, `.../a15_p2_gefs_era5_paired.py` at `bb1583fc...`; `research/earth_aiwq/a15_p11_fix1/a15_p11_fix1_full_ensemble_gate.py` at `df938844...`; and `research/earth_aiwq/a15_p12_e1/a15_p12_paired_dataset.py` at `5c77ec2c...`. All checksummed in Notebook and delivery package.
