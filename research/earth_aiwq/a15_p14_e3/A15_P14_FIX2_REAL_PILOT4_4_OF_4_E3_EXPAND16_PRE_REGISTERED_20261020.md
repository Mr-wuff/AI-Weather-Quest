# A15-P14 FIX2 real Pilot4 4/4 complete -> E3 Expand16 frozen spatial transfer qualification
Date: 2026-10-20. Status: REAL Pilot4 completed; Expand16 new twelve dates NOT YET downloaded.

## Genuine user-source chain & independent integrity
- Actual `CIDM_EARTH_A15_P14_PILOT4_DELIVERY_REPORTS(2).zip` SHA256: `2c8c97ae02ef306829d8777feaf7714d2e2223ed0350abf47d894dd7f472e898`; exact user SHA sidecar, ZIP CRC PASS, `REAL_PILOT4_NEW_YEAR_TRANSFER_COMPLETE_RESEARCH_ONLY`, four distinct real issues 2003040200,2005070600,2008100100,2011010500.
- Actual `P14_VOLATILE_CASE_CACHE_RECOVERY(1).zip` SHA256: `b57504b66b2f4645d389f19920e7c1e3163fce17200f633278c0eb2ccfdae7cd`; SHA sidecar and ZIP CRC PASS; four signed per-date full forecast/truth NPZ + metadata, all per-file and tensor SHA verified (shape forecast 5×2×2×121×240, truth 2×2×121×240).
- Direct independently recomputed P14 area-weighted 11×12 E[e],E[e²] from all four **real full array pairs** equals the actual uploaded `NEW_YEAR_AREA_MOMENTS.npz` element by element, max difference 0.
- Embedded provenance-locked 17,261-byte *small sufficient moments only* SHA256: `169f6e4bfbdc8feed5f039d2e52974d1419530826323f9200db7010c454eb854`. Full raw 8MB old paired array ZIP is **not** embedded in Notebook, not reuploaded to public GitHub/HF, nor needed to recompute exact patch-constant RMSE Gate.

## Genuine frozen P13 correction on four unseen seasonal issues
Four-case arithmetic mean case-RMSE metrics:
| Head | Raw | Frozen spatial | Gain vs raw | Gain vs shifted lon placebo |
|---|---|---|---|---|
| MSLP W3 | 5.801242 hPa | 5.678353 hPa | +2.1183% | +1.3409% |
| MSLP W4 | 5.540583 hPa | 5.472263 hPa | +1.2331% | **+0.1523%** (below .5% threshold) |
| TAS W3 | 2.211678 C | 2.153922 C | +2.6114% | +0.6591% |
| TAS W4 | 2.367632 C | 2.308297 C | +2.5061% | +0.7178% |

All four heads **worsened** in the 2011 Jan issue (MSLP W3/W4 −1.253%/−1.054%, TAS W3/W4 −0.255%/−0.353%). 2005 Jul MSLP W3 −0.394%; 2008 Oct MSLP W4 loses to shifted placebos. Each new year currently has only one season, thus **no yearly Gate may count as passing**. `candidate_gate_eligible=false`, `variable_both_week_gate=false` by design. P14 remains **static spatial correction diagnostic, not CIDM model skill**.

## E3 Extend-to-all-seasons frozen contract
- Exact P13 train-only years 2002/04/06/10 and frozen original P14 `a15_p14_new_year_transfer.py` SHA `90499e741d9247d55a6588c75c038f4d5e7476656b7376b8479d2a5e37532b34`. P13 train reference SHA `cae9ce1c6dead8ac2a9a1a209c8ce1dd7329ed00582577bad9d32b1040f69d4a`.
- Same NEW years **2003/2005/2008/2011** (first Wednesday of Jan/Apr/Jul/Oct). Exactly 16 candidate cases; reuse only four already-audited Pilot4 *sufficient moments* and frozen paired-source hashes. Download **12 NEW full** 5-member GEFSv12 W3/W4 forecast + ERA5 truth pairs = at most 12×560 = **6,720** range GRIB forecast records. New full pairs use exact original P14 source validation/score and FIX2 guarded P1 write; all 16 regional moments passed unchanged to original P14 `eval_new` and `make_report`.
- Seed-only 4 pilot moments **cannot substitute for actual full forecast/truth arrays in future neural CIDM model training**; avoid calling them 4 newly downloaded full field pairs. Result status reports these separately. Keep the original signed 8 MB full array archive for later training; this E3 Notebook never requests user to reupload it.
- **Gate only on complete 16/16**: for SAME variable both W3/W4: ≥1% RMSE improvement vs raw and ≥0.5% vs frozen global-bias and longitude-shifted controls, ≥3/4 NEW years with each year's 4 seasons complete and matching all thresholds. Do not loosen/tune on this set. Validation 2014 and independent test 2016 strictly untouched. Even PASS qualifies only static-correction baseline; no CIDM neural training occurred.
- Colab default stage `expand16`; Drive preferred but nonfatal unless `REQUIRE_DRIVE=True`; atomic per-group GEFS write SHA `1512c8f1f847f734e70f6c247bdde911425aacafd6d8a045f9b9de7977a984bf`, unchanged physical data/score P1/P2/P11/P12 code pinned Git commits+SHA. Auto ZIP/SHA result and case cache backup on volatile fallback. Public HF automatic upload DISABLED.
- Research boundary: NOAA GEFSv12 **retrospective reforecast** is not verified historic issue-time-available operational input; TAS is globally evaluated here, not official land-only RPSS. Formal CIDM long training still forbidden until matched CNN/EMOS/actual GeoPort pilot, more training episodes, proper probability metrics and separate independent years.

## Artifact and QA
- User download `CIDM_EARTH_A15_P14_E3_Expand16_FrozenSpatialTransfer_Colab.ipynb` 147,484 bytes; E3 engine source SHA `60ddaea8a4ede49de77ab1c77ca55b5c3dfa6c10c45b64de736044f958b99d07`.
- Complete real-parent / Python / frozen source fallback / unit QA package `CIDM_EARTH_A15_P14_E3_Expand16_NewYears_LITE_FULL.zip` size 235,491 bytes SHA256 `7eb2a8faf186372c0a77445adfd9e3b56cfc34151226b1dff4b5e797e6619828`, CRC+per-file SHA manifest PASS.
- Actual user Pilot4 metrics independently recomputed exactly from seed; source authentic full arrays externally re-audited bytewise. Mock **synthetic** 12-case completion + real seed, interrupted/corrupted seed fail-closed PASS. Whole 6-cell Notebook offline QA PASS. Deliberately corrupted embedded seed SHA returns standalone BLOCKED ZIP and SHA, no fabricated new-year performance.
- **No real E3 12-date online weather data was retrieved in this assistant turn.** No CIDM training, no official RPSS, no HF private data upload. PR stays draft; do not merge.
