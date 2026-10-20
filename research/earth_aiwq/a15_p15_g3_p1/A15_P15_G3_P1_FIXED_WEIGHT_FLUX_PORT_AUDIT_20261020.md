# A15-P15 G3-P1: fixed-weight flux/port diagnosis and exact patch-area consistency

Date: 2026-10-20. Status: actual CPU execution completed; research-only mechanism diagnosis.

## Contract
- Real compact 32-case paired GEFSv12/ERA5 patch feature/error-moment dataset SHA256: `960373a8c893eeffe64a1899c9f01d4c612fffedda1ab4b6b89691ce0e268b71`.
- Train 16 issues from 2002/2004/2006/2010; exploratory evaluation 16 issues from 2003/2005/2008/2011, which were already viewed in P14. No blind-generalization claim. 2014 and 2016 untouched.
- 3 seeds [11,23,37], 120 fixed steps, AdamW .005, weight decay .001; no evaluation early stop or tuning.
- Small 11×12 block-constant residual model only; not full 121×240 CIDM, not time rollout, not official RPSS, no formal long training.

## Numerical finding
P0 used synthetic `lat ±7.5°` row areas although evaluation uses a separate actual 11×12 area-weight matrix. With same random probe and same learned edge-conductance weights, maximum relative weighted-mass residual under the *evaluation* weights is **0.5145%** (mean 0.1290%) for legacy geometry, versus **2.06e-8** using exact row sums of the scoring weights. Both operators' weighted energy dot was non-positive (no positive-energy cases). This is a discrete geometry/scoring consistency correction, not proof of atmospheric physical conservation.

## Port/geometry factorial and matched controls
2×2 training factorial:
- legacy geometry + old 6-channel external port (four spreads plus sin_lat/cos_lat);
- exact geometry + old 6-channel port;
- legacy geometry + spread-only 4-channel port;
- exact geometry + spread-only 4-channel port.
Geography/month remain available to the encoder in all models. Also trained corrected no-flux, corrected no-port and CNN (659 parameters vs corrected GeoPort 631; 4.3% difference), all 3 seeds × 120 steps.

Mean RMSE improvement over raw GEFS (%):
| Arm | MSLP W3 | MSLP W4 | TAS W3 | TAS W4 |
|---|---:|---:|---:|---:|
| legacy GeoPort | +1.286 | +2.686 | +0.537 | +1.444 |
| exact geometry + legacy port | +1.314 | +2.694 | +0.571 | +1.479 |
| legacy geometry + spread-only port | +1.225 | +2.844 | +0.396 | +1.409 |
| exact geometry + spread-only port | +1.226 | +2.854 | +0.393 | +1.406 |
| corrected no-flux | +1.500 | +2.835 | +0.587 | +1.503 |
| corrected no-port | +1.323 | +2.716 | +0.332 | +1.384 |
| matched CNN | +0.642 | +1.491 | +0.746 | +1.361 |

Geometry correction changes scores only about −0.003 to +0.035 percentage points, despite fixing the mass-weight mismatch. Port effects are head-dependent: the old 6-channel port included duplicated latitude features; spread-only port slightly helps MSLP W4 but not all other heads.

## Same-weight flux intervention (no retraining)
Mean gain (%) at fixed trained corrected GeoPort weights:
| Flux multiplier | MSLP W3 | MSLP W4 | TAS W3 | TAS W4 |
|---:|---:|---:|---:|---:|
| 0 | +1.217 | +2.867 | +0.302 | +1.274 |
| 0.5 | +1.222 | +2.864 | +0.355 | +1.349 |
| 1.0 | +1.226 | +2.854 | +0.393 | +1.406 |
| 2.0 | +1.233 | +2.821 | +0.438 | +1.487 |

No universal gain: TAS benefits slightly from stronger flux; MSLP W4 worsens slightly as flux rises; full-vs-zero flux MSLP effects vary by seed. The corrected model's actual latent flux is ~3–4% of latent-state norm on average. The independently trained no-flux arm remains better on MSLP W3 and TAS W3/W4; the full model's MSLP W4 edge over no-flux is only ~0.019 percentage points.

## Decision
- **Numerical consistency correction: PASS.**
- **Unique CIDM flux necessity: NOT ESTABLISHED / FAIL for qualification.**
- **Port value: variable-specific, not universal.**
- Freeze P14/P15 evaluation-year results; do not tune on 2003/2005/2008/2011 again. Next qualification requires more independent train years, genuine 6-hourly/trajectory state evolution, a pre-registered fresh-year gate and matched CNN/EMOS controls. No formal long training authorized.

## Artifact hashes
- Engine: `d3d95f88e952d5ff5115c737502098b7b35571d689258c11cc156d9d22f2049d`
- Notebook: `c815771b691e3ce273364163ec669bd40e90fca4daac4046e863953c3889147a`
- Real result report ZIP: `ff69bc25db3a526c1d87d41ce4d87815b9e7faed633fcea6726995ee327fe64e`
- Result bundle: `fe8823d5d111fa805dc6739c88430f355fc029e9f32d0c63b7310c5ee53c194a`
- Full user delivery ZIP: `338973e9ea1567eb6aa49d3b328a43ae262b1ed1e42a770edd760f71c27ed291`

Full notebook, compact derived data and results are delivered privately in the ChatGPT artifacts, not committed to public GitHub/HF. This note archives only the research contract and SHA fingerprints.
