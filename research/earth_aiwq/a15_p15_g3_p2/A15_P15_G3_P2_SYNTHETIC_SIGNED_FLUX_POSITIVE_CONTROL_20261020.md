# CIDM-Earth A15-P15 G3-P2: controlled signed-transport flux + external port identifiability

**Date 2026-10-20. Status: actually executed CPU 3-seed 100-step controlled-SYNTHETIC experiment, NOT real weather forecast training.** Research contract and hashes only; no raw GEFS/ERA5 data made public.

## Motivation and nontrivial distinction
P15-G3-P0 and P1 showed positive-conductance, symmetric, dissipative flux has not proved its unique utility against no-flux models on the authentic 32-case GEFS/ERA5 patch residual benchmark. Simply retuning that operator on already-inspected 2003/05/08/11 exploratory years would not establish causality or generalization.

This G3-P2 Q0 tests a *different expressivity*: directional advection with signed velocity and a directly measurable external velocity/source port. Use 11x12 exact area weights obtained from the authentic real32 compact data (parent SHA `960373a8c893eeffe64a1899c9f01d4c612fffedda1ab4b6b89691ce0e268b71`). Only geometry comes from real weather; tracer evolution is entirely synthetic and generated from a published finite-volume-style formula.

## Frozen experiment and real execution
- Two regimes: (A) pure area-conservative symmetric diffusion + synthetic known source, (B) directed east-west shared-edge upwind transport + diffusion + synthetic known source. North-south no cross-pole exchange.
- Disjoint randomly seeded *synthetic* 128 training / 64 holdout sequences per regime, 8 time steps, train rollout 4 steps, test rollout 8. Fixed 3 seeds [11,23,37] × 100 optimization steps; all training and holdout diagnostic checkpoints printed; no chosen best seed/early stopping/holdout retuning.
- Arms: constrained symmetric diffusion, directional conservative flux with wind port, directional flux with missing wind port, no flux, unconstrained CNN 297 params (physics arms 4 scalars; **param and hypothesis family not matched**).
- Actual three-seed mean 8-step MSE:

| Arm | Synthetic pure diffusion | Synthetic directional+source |
|---|---:|---:|
| Symmetric diffusion | 3.907137e-7 | .07783688 |
| Directed flux + correct wind | 3.907137e-7 | 9.121144e-6 |
| No wind port | .01303392 | .07735166 |
| No flux | .13868886 | .41370624 |
| CNN | .00250314 | .06054651 |

- Fixed-weight wind-shuffle on correct directional model increases directional holdout MSE from 9.12e-6 to ~.12238, zero wind to ~.14708.
- Numerical synthetic scalar unit tests: area-weighted mass residual symmetric ~1.52e-8, signed directional ~1.17e-8; injected source budget residual ~7.63e-8, diffusion energy dot strictly non-positive; no across-polar edge.
- **Positive control PASS only**: the synthetic ground-truth generator belongs to the directional physical model's own function family, which accounts for its near-perfect fit. These tests are not an equal-parameter CNN comparison, not true 6-hourly GEFS/ERA5 forecasting skill, not global atmospheric conservation or official ECMWF RPSS.
- Older real-weather flux-necessity Gate continues **FAIL**. 2014 and 2016 true labels remained untouched. No formal long training authorized.

## Reproducibility and future work
- Readable Python engine SHA256: `5dd0eafbe4bc19bcf320c6909a4720da284a84c8ca4dd41de35593851dac0ce6`; Colab source/true-geometry SHA verified on bootstrap; no user historical ZIP, no NOAA/HF/Google Drive required.
- User-facing 14,255-byte full Colab Notebook, SHA256 `d3427c865534852a185939962b646df1fc78ade3d5b7bec6d2c665d83f0bd195`.
- Actually executed controls result ZIP SHA256 `04cfff30934bb9f94e1c8148b2cb22e93c776999b47fb259aa3df3f96360bfba`. Complete source + notebook + authentic-area JSON + QA + result ZIP downloadable as `CIDM_EARTH_A15_P15_G3_P2_DirectionalFlux_Controlled_FULL.zip`, SHA256 `b2b1fe394a2f2ed6935ecfa7c436ddb8585f340e6576376f92a7d848b9a3fdf1`. CRC/member SHA PASS.
- Whole 5-cell Notebook executed in offline smoke and full 3-seed mode; deliberate embedded source SHA corruption gives accurately labeled BLOCKED+matching SHA result ZIP; geometry SHA tamper blocked; standalone repeatability tests PASS.
- **Next eligible phase:** build actual GEFS 6-hour **issued-time forecast trajectories** + ERA5 *labels* (not future inputs) on genuinely fresh years, pre-register variable/year held-out tests, weather-appropriate advection/force fields, and compare signed+sym/no-flux and CNN under matched capacity/compute and fixed multistep loss. The retrospective GEFSv12 archive is not proof of operational availability at historical issue time.

No public weather tensors or automatic HF upload. PR stays draft; no merge.
