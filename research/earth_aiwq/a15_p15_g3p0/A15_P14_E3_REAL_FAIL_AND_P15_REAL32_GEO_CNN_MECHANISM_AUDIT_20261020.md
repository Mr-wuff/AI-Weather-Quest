# A15-P14-E3 real 16-case FAIL → A15-P15 G3-P0 actual GeoPort vs matched CNN learning probe
Date: 2026-10-20. This note archives scientific results and immutable provenance SHA hashes only. **No complete weather arrays were publicly uploaded.**

## A15-P14-E3 verified genuine user results
- Full actual user `CIDM_EARTH_A15_P14_EXPAND16_DELIVERY_REPORTS.zip` SHA256 `aaaf5f8cd5cc8c22818893b9682245489785b6fe6ab22572acd093efefb769d3`, user sidecar match, ZIP CRC PASS. Contains genuine 16/16 blocked new-year area moments and complete real research evaluation.
- User `P14_VOLATILE_CASE_CACHE_RECOVERY(2).zip`, SHA256 `98fb93a3c460cf31bdb43775308b46ae22a23976158d11e6bd4a5031fc2f5dd7`, sidecar/CRC PASS: all 12 added full GEFS–ERA5 pairs with signed per-file manifest. Prior Pilot4 recovery SHA `b57504b66b2f4645d389f19920e7c1e3163fce17200f633278c0eb2ccfdae7cd` contains four other full pairs.
- Frozen P13 spatial-bias effect relative to original GEFS at new years 2003/05/08/11 × four seasons: MSLP W3 **+0.4656%**, W4 **+1.2360%**, TAS W3 **+1.6323%**, W4 **+1.5032%**. Four-year complete head passes: **1/4, 1/4, 3/4, 2/4**, respectively. Joint W3+W4 gate **FAIL for both variables**. Stop retuning the P13/P14 static-bias mechanism on these viewed years.
- 2014 validation year and 2016 independent test year untouched. Global TAS RMSE research proxy, NOT official TAS land-only RPSS. GEFSv12 is historic retrospective reforecast; historical issue-time operational availability NOT proven.

## Independently executed NEW A15-P15 G3-P0 genuine 32-case patch neural probe
- From SHA-locked original Train16 ZIP `957b66c3d3e04201d5ebcb6e077599844f755683863c47bc71f98c81ea527cc1` and the two P14 signed recovery ZIPs above, independently load and SHA-verify **32/32 real** GEFS 5-member W3/W4 + ERA5 truth pairs. Recompute EVERY original case raw RMSE within ±0.0005 in original source units (hPa / °C); source arrays were ALREADY converted to those units.
- 16 training issues, 2002/04/06/10; 16 evaluation issues 2003/05/08/11 that were **already inspected in P14** (NOT unseen blind test). No 2014 or 2016 access. Area-weighted 11×12 patch first & second moments plus issue-time GEFS mean/spread, latitude/longitude, month sin/cos yield compact 255,840-byte user-downloadable dataset SHA `960373a8c893eeffe64a1899c9f01d4c612fffedda1ab4b6b89691ce0e268b71`; forecast-only inputs, ERA5 ONLY for target labels/eval; exact original weighted RMSE for a patchwise constant additive correction.
- Four approaches, all trained same data, AdamW 0.005, 120 fixed optimizer updates, seeds 11/23/37, no eval-based tuning or early stop: tiny P10-inspired GeoPort (~653 parameters), matched 3×3 convolutional baseline (~659 params), GeoPort no learned flux (~653), GeoPort no external GEFS-spread port (~653). Note the new patch model is NOT the full P10 CIDM encoder/decoder, original full global resolution, or physically validated W3→W4 time-rollout.
- Genuine THREE-SEED mean RMSE gain percent **relative original GEFS**:
| method | MSLP W3 | MSLP W4 | TAS W3 | TAS W4 |
|---|---:|---:|---:|---:|
| frozen P13 spatial | .466 | 1.236 | 1.632 | 1.503 |
| matched CNN | .642 | 1.491 | .746 | 1.361 |
| GeoPort | 1.286 | 2.686 | .537 | 1.444 |
| **GeoPort no_flux** | **1.430** | **2.891** | .608 | **1.561** |
| GeoPort no_port | 1.321 | 2.740 | .309 | 1.388 |

## Interpretation / decision
- Genuine residual learning has a *candidate* signal on MSLP, especially W4 where GeoPort exceeds matched CNN and static correction.
- But **removing the geometry flux IMPROVES MSLP W3 and W4**, and no-port MSLP results remain close. Therefore **unique CIDM dynamical flux necessity FAILS** at this scale. TAS W3 cannot exceed the frozen static benchmark. Do NOT approve longer formal training, call this official RPSS, or claim global 1.5-degree physical atmosphere conservation.
- These are only 16 independent training episodes in 4 years. All new-year evaluation outcomes had been inspected during preceding P14 analysis, thus over-selection risk; no fair unseen 2014/16 result.
- Subsequent Q1 should target *causal and numerical flux necessity* with fixed training/evaluation contracts (flux dissipation/port response) and more genuinely independent training seasons, not another tuned static bias smoother. Genuine multi-step dynamic weather trajectories and strict train-only climate reference are required before official probabilistic comparisons.

## User artifact SHA / reproducibility
- Self-contained ~333KB Colab `CIDM_EARTH_A15_P15_G3_P0_Real32_GeoPort_vs_CNN_Colab.ipynb` SHA256 `893c1db71486d67aadd4985fb24a879a51809227d2db643d0337c0452f6beccd`. No prior ZIP uploads, NOAA network, or HF token required. Byte-pinned embedded authentic 32-case compact feature dataset and source.
- Actual locally executed 3-seed results ZIP `CIDM_EARTH_A15_P15_G3_P0_DELIVERY_REPORTS.zip` SHA256 `6de0b50a2cd1acbdfecbb4f8f435433eafa316325d80976521db49b89eb2165f`, score by head/date/seed/year, no official RPSS.
- Full 554337-byte user-delivered research package `CIDM_EARTH_A15_P15_G3_P0_Real32_GeoPort_CNN_FULL.zip` SHA256 `b8786ba638f5e8ec25652be93cf49f00ca080c7b5057e6a460436884d540ec4d`; ZIP CRC + file SHA manifest validated. Includes readable engine, genuine patch data, extractor, notebook build, Chinese scientific review, independent computed report and log.
- Smoke-mode executed all Notebook code cells with actual real32 compact data, reports QA-only; hash corruption yielded BLOCKED ZIP+SHA, with no false pass. Full actual 3-seed engine training executed separately on the identical immutable data and budget.
- This public GitHub draft stores research protocol/results hashes ONLY; user full result or derived weather data is not committed, and no HF public upload, auto merge or authorized external binary transfer occurs.
