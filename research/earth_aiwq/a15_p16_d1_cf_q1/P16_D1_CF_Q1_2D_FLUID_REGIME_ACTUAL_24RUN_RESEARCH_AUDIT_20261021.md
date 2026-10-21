# CIDM-FR CF-Q1: 2D compressible Euler closure/source qualification (2026-10-21)

**Status**: Numerical QA PASS and 24 actual synthetic small-model trainings complete; source-aware physical closure qualified only in a deliberately controlled teacher-supplied-source setting. **Full CIDM neural architecture and real weather dynamic Gate remain NOT QUALIFIED; no formal training authorization.** Draft research PR, not merged.

## Background and causal research question

CF-Q0 1D ideal-gas Euler suggested local compressibility is compatible with control-volume global mass/energy budgets, whereas an external heat source requires explicit energy-source accounting. This does NOT yet prove original CIDM's prior underperformance on CFD was due to a universal latent information closure. CF-Q1 extends numeric + learning controls to 2D.

## Actual new experiment (no NOAA, no ERA5)

- Synthetic 2D ideal-gas Euler U=(rho,rho*u,rho*v,E), gamma 1.4, local **Cartesian periodic** N=24×24 grid, DT 0.0045, eight-step rollout.
- Independent NumPy **RK2 HLL** reference solver: randomized-direction discontinuities and smooth perturbations for train; independent 15 holdout initializations per regime include previously untrained low-Mach style. Closed Q=0 and externally heated *known Q*.
- All learners share **known one-step Rusanov physics baseline**, train only a tiny neural residual. Thus this is NOT physics-free CNN vs full CIDM comparison. Four arms: CNN free 1316 params, intentionally false local-density-locked CNN 1316, flux-conservative source-disabled 1306, same-parameter conserved flux with exact supplied-Q energy port 1306. Shared train/val sources per regime and seed, 42/12/15 independent episodes, each eight time steps.
- Three seeds 11/23/37, 100 fixed training optimizer steps, AdamW, no holdout tuning/early stop: **24 actually executed training jobs**. 8-step autonomous holdout evaluation and global budget checks, plus fixed-weight source shuffled intervention.

## Numerical gates independently passed (18 episodes per regime)

| Condition | max mass residual | max x-momentum | max y-momentum | max energy minus source | mean local rho 1step RMS |
|---|---:|---:|---:|---:|---:|
| closed | 1.77e−8 | 3.98e−9 | 2.52e−9 | 4.39e−8 | 0.00846 |
| heated | 2.24e−8 | 8.39e−9 | 4.91e−9 | 9.38e−8 | 0.00698 |

Heated mean absolute eight-step global energy change 0.04149. Reflective wall tests zero normal mass+energy flux and separate momentum pressure budget; outflow open-boundary flux ledger residual <1e−13. These boundary modes are **numeric tests only**; learning/eval periodic. Cartesian positive y index increases with row index, **NOT globally latitude north-positive P16 grid**. Never directly port to geodetic grids without new coordinate test.

## Actually trained 3-seed mean 8-step rollout

| model | closed RMSE | heated RMSE | heated energy budget RMSE |
|---|---:|---:|---:|
| common-Rusanov + CNN | .007721 | .004885 | .001330 |
| common-Rusanov + wrong local-density lock | .020304 | .020673 | .001411 |
| conservative learned flux without source | .006582 | .016013 | .049019 |
| **same conservative flux with known heat-source port** | **.006582** | **.003808** | **2.8848e−7** |

3-seed source-aware heated RMSE is 76.22% below source-disabled, with identical 1306 parameters. With Q=0 same flux architectures yield exactly equal results. Fixed-weights wrong source order yields **terminal t8** RMSE 0.027236; it is not directly comparable with 8-step average RMSE. All models' accepted density/pressure fraction 1 on this limited distribution. **This teacher provides the true Q**, favoring a physically matching explicit-source system; cannot extrapolate to unknown atmosphere forcing or general-model superiority.

## Causal bounds

- PROVEN in controlled settings: physical componentwise finite-volume closure + known-source terms matter; incorrect local density conservation hurts; 2D global budget diagnostics pass. Teacher (RK2-HLL) and learners (Rusanov baseline) use different schemes but the same known Euler physics.
- NOT proven: initial full CIDM CFD failure attribution, full CIDM vs strong matched CNN capacity and compute, moist/anelastic/buoyant ocean, real ERA5/GEFS improvement, usable unknown Q, geophysical sphere/coast and pole coordinate validity, entropy-stable shock handling at long horizon. No 2014/2016 heldout weather truth was accessed.
- Post-run diagnostic control checks passed across all 3 seeds; **not an unseen preregistered blind Gate**.

## User-downloadable source + reproducibility

- Self-contained Google Colab Notebook 14,918 bytes SHA256 `f8727d1687866da578db13b727b906e3fa09712bd0c6da16cb005b4541cdee45`.
- Source `cidm_cf_q1_2d.py` SHA256 `a7d9029052515c85630b42b7b67e6854174c81a58c6500b0efce03c2a6c1ad6d`. Byte SHA hardcoded and independently checked in Notebook before executing.
- Actual computed 24-run reports ZIP SHA256 `de052d492aad1a7cc0f486c43c35f8bee7bcfdd4795dc502f10864b2983b72bc`.
- Full user-downloadable complete source/Notebook/QA/Chinese audit/results ZIP 58,208 bytes SHA256 `bebfabbb809ce597bee1e15c315e2b50abefcd84cb98b66a625d4da9086289ea`, 24 members, CRC and exact per-file SHA manifest pass.
- Independently executed whole Notebook FULL 24 runs and checked all model RMSE reproduced within 1e−7; QA-only all numeric checks pass; corrupt embedded code SHA returns BOOTSTRAP_BLOCKED valid ZIP and SHA rather than claimed success. Full training curve, 3-seed cases, signed real generator outputs and docs are included in user-facing package. This draft PR only publishes scientific audit/hashes (no private weather tensors).

Next **CF-Q2** should directly test original CIDM core with causal architectural ablations against independent 2D CFD reference, low-Mach/Boussinesq as a distinct governing equation, genuinely trained wall/open boundaries, stronger shock and multistep checks, then P16 real-data with new dates and fair CNN/EMOS/no-flux comparison. Formal long-period training remains BLOCKED.
