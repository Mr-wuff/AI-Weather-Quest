# P16-D0-R1 real replay PASS → P16-D0-R2 flux / upwind-viscosity mechanism identifiability

Date 2026-10-21. This draft PR archives method and honest diagnostics ONLY. No full GEFS/ERA5 tensors, user-derived compact weather dataset, or self-contained notebook are published in this public GitHub PR.

## User uploaded R1 verification
- Real executed notebook 5 cells, no Python errors.
- User R1 ZIP SHA256 `cd14d4c89dc974cd0c2595a73545dc48e79702ed8cc040535b8b84164638f343`, upload .sha256 match, ZIP CRC PASS; status `REAL_4CASE_ORIENTATION_REPAIR_DIAGNOSTIC_COMPLETE_NOT_BLIND`.
- Real 11×12 four-case signed parent weather panel SHA256 `d399e91c974a8a1cb08e2b36b5a3baddcfea81f8b86cfe977de1ab89aad4a0f1`, 2009/2012 fit, already-inspected 2013/2015 audit. 2014/2016 untouched.
- Exact independent R1 RMSE gains relative original GEFS: MSLP wind10 +.273947%, wind850 +.392289%; TAS wind10 +1.056562%, wind850 +.809925%. Only TAS+wind10 passes prior diagnostic gate, but fitted W3/W4 alphas are **negative** −.183936/−.185098; not a physically forward advection benefit claim.

## R2 actual experiment (source SHA locked / executed 2026-10-21)
- Original R1 corrected north-positive/south-descending rows face orientation and scoring weights strictly fixed.
- Discretely decompose shared-face upwind flux into **odd centered signed transport** and **even upwind numerical dissipation**. Exact algebra and finite-volume area-weighted tracer exchange conservation tested. Reverse wind flips the odd component, not the even; **refitting an unconstrained signed centered coefficient after reversing wind produces exactly identical predictions**, so naive reversal/refit cannot prove cause.
- Fit 1 scalar per target/week/port/diagnostic arm using 2009/2012 only; exact same ridge 0.5×feature-power and alpha [-1,1]; special diagnostics alpha [0,1]; heldout labels 2013/2015 previously seen, exploratory only. ERA5 never weather input, only dev targets and offline scores.
- Mean RMSE gain (%) compared to raw GEFS across previously viewed 2013/2015 and both week windows:

| Mechanism | MSLP + 10m | MSLP + 850hPa | TAS + 10m | TAS + 850hPa |
|---|---:|---:|---:|---:|
| R1 full upwind | +.273947 | +.392289 | +1.056562 | +.809925 |
| odd centered advection | −.116718 | −.072793 | +.875901 | +.470245 |
| even numerical viscosity | +.783742 | +.963400 | +.697886 | +.799581 |
| full upwind, alpha>=0 | +.273947 | +.392289 | 0 | 0 |

- MSLP isolated viscosity beats signed full flux, odd advection loses. TAS+10m apparently depends on **negative alpha**; forbidding negative alpha removes gain entirely, so cannot claim physically positive transport. With retrained wind reversal, MSLP 10m/850hPa mean gains +.952999%/+1.134434% (can outperform aligned wind). Shifted-wind refit can also outperform aligned MSLP. This is a no-unique-port-necessity diagnosis.
- Numerical invariant: relative area-weighted tracer flux budget max 7.246651e−17, upwind decomposition relative <=1.37e−16, centered odd reversal exactly 0, even reversal exactly 0, viscosity weighted energy nonpositive on 30 random tracers. **These do NOT establish full atmospheric mass/energy conservation.**
- Crucial non-claims: only 4 initializations, 2 dev / 2 previously viewed, 11×12 block-constant, no true CIDM autoregressive learning, W3/W4 126h disjoint, GEFSv12 retrospective as-issued historical availability not proven, not official RPSS, no long training release. Gate is `NOT_ELIGIBLE` for fresh blind validation.

## User-local deliverables and verification (not public GitHub)
- Standalone Colab Notebook with exact signed real four-case panel and R1 source provenance, 1,398,012 bytes SHA256 `71cc7c4dd9c071c52af13de46b6c535b9db3a67577a1afc64aed4fda421b3a69`.
- Full source/data/test/Chinese-review package ZIP, 2,093,763 bytes SHA256 `de779ccdc5ad162ad2dab8852d617d5ff1b922dec9de9be4b43dbf9ad8818833`, 16 members, CRC and every member manifest digest verified.
- Actually executed real-panel R2 results ZIP, 8,933 bytes SHA256 `878ec18c255a3c79542ff855c15071239d939a9ce10185bb90f87fecf8175486`.
- Tests: original R1 gain replay within 2e−7 pp, the upwind-odd-even numerical audit, full five-cell notebook execution, QA-only mode, tampered embedded payload SHA → `BLOCKED`, source real panel mutation → rejection; all passed.
- Science next step: fixed physics with more independent training examples, 121×240 native grid, actual contiguous 6-hour state rollout rather than already-issued F_next postprocessing, physically relevant wind/target mapping and >=3 seeds matched CNN/no-flux/symmetric/directed baselines; new untouched independent years and preregistered skill/physics gate. No formal long training yet.

**Draft; never auto-merge.**
