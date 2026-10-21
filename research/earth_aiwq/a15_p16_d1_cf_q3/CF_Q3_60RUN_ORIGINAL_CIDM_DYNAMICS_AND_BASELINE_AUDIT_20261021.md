# CIDM-FR CF-Q3: original archived CIDM dynamics attribution, 60 actual seeded runs (2026-10-21)

**Decision: CF-Q3 numerical/source/short-training qualification PASS; distinctive CIDM latent-flux necessity FAIL; matched CNN superiority FAIL; no formal long training authorization.** Actual scores are a controlled synthetic CFD teacher family only, not original third-party CFD benchmark, user weather forecasts or official RPSS.

## Study and precise scientific limits

- Genuine SHA-locked archived `cidm_architectures.py` original CIDM Python class (SHA `f9b1cf4ff7630f5540bb3adde20e4c4b81fff3218d9325ea3105bf28b1d97665`) with SpaceEncoder, TimeHistoryEncoder, PhaseSpectralEncoder, ClosedInformationDynamicsCore, actual original AntisymmetricFlux, HiddenReservoirTransfer and two original decoders. The original CIDM is used for trainable neural feature backbone, but adapters are newly added; **NONE of DIRECT/BARE/PHYS is a byte-for-byte unmodified original user-facing CIDM predictor.**
- NumPy RK2-HLL synthetic independent reference `cidm_cf_q1_teacher_locked.py` SHA `a7d9029052515c85630b42b7b67e6854174c81a58c6500b0efce03c2a6c1ad6d`. Same Q1/Q2 underlying generator, NOT independent new published CFD or unseen physics family. Two 24×24 Cartesian periodic compressible Euler regimes: Q=0 and externally heated with explicitly known Q, DT=.0045, 8-step free trajectory. 48 distinct train, 12 val, 18 test initializations per regime. Q known to ALL models, injected structurally into E; do not present Q availability as learned discovery. 2014/2016 real weather evaluation years untouched.
- 3 seeds 11/23/37 × 2 regimes × 10 arms × 60 AdamW optimizer steps = **60 actual independent short trainings**. CNN registration within 1.9–2.9% of CIDM, but active parameters are lower for disabled CIDM submodules, FLOPs/time and convergence not equal. Eight-step *free* rollout; source energy, area/telescoping/positivity audited.
- DIRECT: original CIDM decoder or CNN directly corrected state `U_next=U_last+0.01*r_theta + dt*S_known` (NO numerical Euler/FV physics). BARE: `U_last-dt·div F_theta+dt*S_known` (known FV conservative face structure, NO Euler solver). PHYS: `Rusanov(U_last)-dt·div F_theta+dt*S_known` (known first-order Euler step). All models have trainable heads and known source, so non-physics vs physics is not an isolated pure architecture effect.

## Verified actual 60-run summary

Eight-step whole-state RMSE, 3-seed mean. Lower better:

| Arm | Closed | Heated |
|---|---:|---:|
| Zero-learned Rusanov + known Q | .006974 | .004790 |
| CIDM DIRECT | .063897 | .049827 |
| CNN DIRECT | .050902 | .038538 |
| CIDM BARE FV | .058746 | .053456 |
| CNN BARE FV | .018268 | .016107 |
| CIDM PHYS | .006846 | .004817 |
| CNN PHYS | **.003974** | **.003277** |
| CIDM PHYS no latent flux | .006828 | .004803 |
| CIDM PHYS no history | .006892 | .004750 |
| CIDM PHYS no phase | .006806 | .005640 |
| CIDM PHYS no reservoir | .006840 | .004713 |

**Critical**: In heated CFD, CIDM+known-physics .004817 is slightly **worse** than zero-learned Rusanov+known-Q .004790. CNN learns an additional useful residual .003277. Original latent antisymmetric flux ablation no worse; its unique necessity Gate FAIL. Turning off phase may harm on the heated synthetic family, yet active parameters change; cannot claim universal criticality.

Frozen trained-weight interventions (no refit) in both regimes: remove flux, history, phase, reservoir and re-score. E.g. heated CIDM-PHYS original .004817, remove flux .004804, remove history .004865, remove phase .004860, remove reservoir .004817; no robust flux necessity.

## Literal predeclared Gate results

- Shared **known physical Euler** assistance helps both CIDM and CNN relative BARE for 3/3 seeds in BOTH regimes: controlled positive evidence.
- CIDM BARE > same-budget CNN BARE 3/3 in both regimes: FAIL.
- CIDM PHYS > same-budget CNN PHYS 3/3 in both regimes: FAIL.
- Original CIDM antisymmetric latent flux necessity 3/3 in both regimes: FAIL.
- CIDM PHYS > zero-learning Rusanov+known Q 3/3 in both regimes: FAIL.
- CNN PHYS > zero-learning Rusanov+known Q 3/3 in both regimes: PASS in this limited task family.
- Original unadapted CIDM published CFD benchmark and full maturity: NOT TESTED. Full weather training authorization: FALSE.

## Full reproducibility evidence and downloads (private conversation artifacts, NOT uploaded publicly)

- Verified complete CRC/per-file-SHA package `CIDM_EARTH_P16_D1_CF_Q3_Original_CIDM_Dynamics_Ablations_FULL.zip` 202199 bytes SHA256 `fb8319db40b96084732de180e08ea5b07f77235c117f97832b5b125606694f56`, 21 signed file entries plus manifest.
- Standalone all-in-one Colab Notebook `CIDM_FR_CF_Q3_Direct_vs_Physics_Assisted_Ablations_Colab.ipynb` 33455 bytes SHA256 `48a75c402e707b518767368c0afad3bd2547d0b41e28866affbbfeccd82b8fc6`, contains internally SHA-locked source bundle. Dependencies are standard NumPy/PyTorch.
- Original 60-run actual results ZIP vs independently executed 60-run Notebook results ZIP each validate per case; **60/60 RMSE exact numerical agreement max difference 0.0**. Full Notebook notebook executed twice (FULL and QA); corruption of source bundle hash gives `BLOCKED_BOOTSTRAP_NOT_SCIENCE` valid report ZIP, not fake PASS. Full numerical/source signature and conservative-face direction/preflight passed.
- Actual 60-run verified result ZIP from standalone full notebook SHA `656cf960b60b9ea32d5c74b7b01ef90f06f9c689c2f69e004fd4405f854bd9ff818abd9`; contained in full package.
- Chinese audit, test script, full per-seed CSV, mean baseline CSV, fixed-weight module counterfactuals, full training curves included.
- No private actual weather arrays/personal user data are published in this public draft PR; results from synthetic data only. This PR is draft and must NOT auto-merge.

## Next experiment: CF-Q4 (qualification before any heavy train)

Run true external independent CFD benchmarks (not same synthetic Q1-HLL family), validation-selected 60/200/800-step budgets with strong matched-active-param/FLOPs CNN/ResNet and no-Rusanov CIDM; correct exact byte original architecture vs adapters; inspect per-regime Jacobians, latent flux use, phase/history gradient effectiveness and shock positivity/entropy, hard perturbations, actual multi-step stability; separate compressible Euler, low-Mach and Boussinesq governing equations. Use ≥3 seeds and unopened regimes/dates, and reserve genuine real-weather 2014/2016 heldout. Do not claim latent conservation == physical energy or treat known source Q as unknown future weather forcing.

**Research PR only: archive honest negative qualification result. Do not merge.**
