# A15-P16-D0-R1: deterministic north-positive wind / north-to-south latitude orientation repair

## Independent actual uploaded-user verification (2026-10-21)
- User real D0 ZIP `CIDM_EARTH_A15_P16_D0_DELIVERY_REPORTS(1).zip`: SHA256 `83d79313eb7c2b81e54c58980f95e28f7da7e5203a5890923bb375eff8277130`, sidecar exact match and ZIP CRC pass. Executed notebook shows `REAL_FROZEN_4CASE_DIAGNOSTIC_RESEARCH_ONLY` and 8 fitted old alpha parameters; no CIDM NN training.
- Frozen input real four-issue GEFSv12 retrospective wind10/wind850+MSLP/TAS+ERA5-label-only 11x12 panel: SHA256 `d399e91c974a8a1cb08e2b36b5a3baddcfea81f8b86cfe977de1ab89aad4a0f1`, actually reloaded and verified.
- Data P16-E0/P2 canonical order is **latitude 90 − 1.5 × original row**, i.e. array row 0 = 90°N and last row = 90°S; 11x12 blocks preserve this NORTH -> SOUTH row indexing.
- Original D0 upwind `v>0` donor `q[:-1]` and `change_mass[:-1]-=fn;change_mass[1:]+=fn` physically moves tracer south while v>0 indicates moving NORTH. Global discrete mass cancellation alone could not discover this bug.

## Fix contract and rigorous controls
- New R1 replaces **ONLY** the meridional flux upwind choice by `q[1:]` for northward v>0 and flux updates `change_mass[:-1]+=fn; change_mass[1:]-=fn`. All original D0 eastward flux and exact polar-cell face lengths, unchanged scoring patch weights, global-mean subtraction, six-hour lead pair selection, train-only alpha fit (2009/2012), audit (already viewed 2013/2015), ridge 0.5× power, alpha clamp ±1, shifted/reversed/time controls, Gate and source input are kept unchanged.
- Deterministic northward tracer pulse PASS, original wrong southward pulse reproduced, pure zonal v=0 calculation bitwise/tight 1e-13 invariant, constant-tracer delta zero, weighted mass residual max 8.14e-17 on authentic wind/pressure fields.
- Real old parent score reproduced to <1e-8 without altering original engine, then genuine same-data corrected branch independently executed.

## Genuine retrospective corrected exploratory results (mean RMSE improvement vs raw GEFS)
| Variable/port | Old | R1 corrected | R1 advantage vs +4 patch longitude-shifted wind | Diagnostic gate |
|---|---:|---:|---:|---|
| MSLP 10m | −0.0126% | **+0.2739%** | −0.2527% | FAIL |
| MSLP 850hPa | +0.0846% | **+0.3923%** | −0.4363% | FAIL |
| TAS 10m | +0.2326% | **+1.0566%** | +1.2223% | **PASS exploratory-only** |
| TAS 850hPa | +0.2278% | **+0.8099%** | +0.7985% | FAIL (relative raw <1%) |

## Essential scientific boundary
- Corrected TAS+10m signed fitted alpha W3 −0.183936 and W4 −0.185098, i.e. prediction benefits from **opposing** the signed transport feature, not proof of physically valid forward heat transport.
- Previously viewed 2013/2015 audit years invalidate fresh blind Generalization claims. Only 4 cases total, 2 development dates. Original D0 analysis needs engineering qualification but P16-R1 is NOT full CIDM multi-step neural training or unique Flux Necessity PASS. Keep 2014/2016 untouched; retrospective GEFS source availability at original issue time not proven; not official RPSS.
- Next: many independent weather episodes, closed/physical transport physics at native 121x240 resolution, 10m/850hPa port eligibility, true 6-hour rollout, matched CNN/no-flux/symmetric/directed controls, three seeds and prespecified fresh-year Gate, only then consider long training.

## Artifacts & reproducibility
- User self-contained 1,345,126-byte Colab Notebook SHA256 `2618c556177a966edecfb89fe3f5a307d80048ef9a025d39bea8b4ab05976589`, embeds SHA-pinned **genuine** panel+unchanged parent+R1 engine+old user score ZIP; no repeat NOAA downloads/Drive/HF/old ZIP uploads.
- Local actual paired old and corrected scientific result ZIP SHA256 `555565a7c86a3ee36d475ad50a1323a24e8d0b9acb2a9682636f7e1713597da7`, including full per-case controls and both JSONs.
- User research FULL ZIP 3,048,758 bytes SHA256 `975c28d87254efa8ddaff1aabf57916a3e0c757d20e5758ef38ae779fd640e2e`; 18 ZIP members, CRC and exact SHA manifest verified.
- Whole standalone notebook QA-only and full real-panel execution verified, deliberate ZIP_SHA corruption yields `BLOCKED_BOOTSTRAP_NO_SCIENCE` with valid ZIP+SHA. No new external NOAA data acquisition. No user weather-derived binary committed publicly; this PR records contract and hashes only; DRAFT, no automatic merge.