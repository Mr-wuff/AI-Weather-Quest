# CIDM-FR CF-Q2: original archived CIDM core under 2D compressible Euler physical budgets

Date 2026-10-21. **Actually executed 30 independent training jobs and separately re-executed the whole standalone Notebook; 30-run per-seed RMSE numeric replay difference ZERO.** This is synthetic controlled CFD, not operational GEFS/ERA5 or the original CFD benchmark. Do not merge this draft automatically.

## Original user issue
Initial CIDM had poor results on certain compressible-fluid PDE benchmarks; user hypothesized general/absolute information-closure conflicts with physically compressible states, warranting domain-specific physical contracts. Earlier Q0/Q1 tested small conservation networks, not original CIDM. Here actual archived CIDM modules are involved.

## Exact model heritage and fairness
- User archived original `cidm_architectures.py` byte-for-byte copied, SHA256 `f9b1cf4ff7630f5540bb3adde20e4c4b81fff3218d9325ea3105bf28b1d97665`. Actual `CIDM(CIDMConfig(...))` with original SpaceEncoder, TimeHistoryEncoder, PhaseSpectralEncoder, `ClosedInformationDynamicsCore` (skew channel, original `AntisymmetricFlux`, hidden reservoir, local residual) plus field and spectral decoders, trained through gradients. No surrogate architecture replaces it. Experimental adapter maps original decoder features to physical finite-volume fluxes.
- Original full archived CIDM *unmodified architecture alone* NOT evaluated: each neural predictor is a small learned correction atop a common **known first-order Rusanov Euler predictor**. Ref truth built by distinct independent NumPy RK2-HLL implementation pinned SHA `a7d9029052515c85630b42b7b67e6854174c81a58c6500b0efce03c2a6c1ad6d`. This is not fair stand-alone general-purpose machine-learning leaderboard.
- 2D periodic Cartesian Euler gas U=(rho,rho*u,rho*v,E), gamma 1.4, 24x24, dt .0045, 8-step autonomous rollout. 48 train/12 validation/18 independent eval ICs per regime, smooth/shock train and extra low-Mach eval, fresh deterministic RNG. Known Q externally supplied only through explicitly authorized physical energy source dt Q, NEVER obtained from target future fields.
- Three seeds [11,23,37] x two regimes [closed,heated] x five arms, fixed 60 AdamW steps, same history and minibatch schedules: `cidm_free_source`, `cidm_cons_closed`, `cidm_cons_source`, `cidm_cons_source_noflux`, and `cnn_cons_source`. CIDM 10,887 registered params, noflux 10,579 approximately active (same registered), matched CNN 11,199 (+2.87%). Differences in active capacity, training speed, FLOPs and architectures are *not fully matched*, and 60-step is not full convergence.

## Actual 3-seed mean 8-step RMSE (lower better)

| Regime, arm | Closed RMSE | Known heat-source RMSE |
|---|---:|---:|
| CIDM free correction + exact known source | 0.0069443 | 0.0051128 |
| CIDM flux-conservative but Q forbidden | 0.0068884 | 0.0155769 |
| CIDM flux-conservative + true known Q | 0.0068884 | 0.0047221 |
| Same CIDM+Q, original latent antisymmetric flux **disabled** | 0.0068806 | 0.0047102 |
| Near-parameter CNN conservative + same known Q | **0.0049140** | **0.0035789** |

- Known-Q source-port physical budget positive control: CIDM closed→Q 0.0155769→0.0047221 (**69.69% lower RMSE**), mean energy budget RMSE .0456651→2.23e−7, and fixed-weight shuffled Q RMSE ~.01878. 3/3 training seeds show gain; Q is oracle-known (not new CIDM inference).
- **Original latent `AntisymmetricFlux` necessity NOT established**: among three heated seeds, original flux enabled beats the disabled version only **2/3**; mean disabled performance marginally *better* .0047102 vs .0047221. No proof original CIDM bottleneck is universal latent closure.
- CIDM-vs-CNN matched-parameter skill Gate fails: CNN wins all 3 heated seeds, with mean ~24.2% lower RMSE than active flux CIDM+Q, and wins all closed seeds too. Short training, physically assisted, one test family => not a universal architecture ranking.
- Synthetic local 24x24 Cartesian Y indexing, **not 90N→90S geophysical latitude indexing**. Neither atmospheric/moist/ocean physics nor original CFD published benchmark tested. No long training authorized. Protected weather years 2014/2016 not accessed. No official RPSS.

## Pre-run numeric and code audits
- Original class identity check, original core exact use, initialized source vs closed bitwise Q=0 equality, source-only E addition dt Q (no mass/mom forcing), periodic flux divergence telescopes exactly, north-positive Cartesian test, parameter-size audit, original and teacher SHA guards, all PASS.
- Independent full Notebook finished **51.87 seconds**, exact equality vs original 30-job engine RMSE across all 30 jobs (max numeric difference **0.0**). Source SHA tamper notebook QA returned `BLOCKED_BOOTSTRAP_NO_SCIENCE`, not a forged science PASS. Numerical-only Notebook smoke also passed.

## Actual completed files (delivered privately to user in conversation)
- Full research bundle `CIDM_EARTH_P16_D1_CF_Q2_OriginalCIDM_Regime_Ablation_FULL.zip`, 83,080 bytes SHA256 `8ac9282b3eac902535fab7c31808a8d224231b85343fb7e53874babde7836bbc`; ZIP CRC+22 individual content SHA manifest verified.
- Full standalone self-extracting Colab Notebook `CIDM_EARTH_P16_D1_CF_Q2_OriginalCIDM_Closure_Colab.ipynb` SHA256 `314768bc0f174bcb44e5dfe642afe0940a029adcdd074034b048290978e30265`; SHA locks both original model and teacher.
- Actual independent 30-job run result ZIP `CIDM_CF_Q2_DELIVERY_REPORTS.zip` SHA256 `431cd56c095236b5de2f01e6b4e62d61751b1b16db66cbd3d7ad679a9c87822f`, all 30 seed metrics, curves, scores, QA.
- Next **CF-Q3** must discriminate bare original CIDM without shared PDE predictor vs physically assisted CIDM, and test original latent flux/source/backbone encoder ablations with matched ACTIVE params and actual FLOPs, longer learning/rollout, shock entropy/positivity. Expand to unrelated benchmarks and independent fluid regime before real GEFS multiyear short training. No formal long training from Q2.

This PR contains **research text and source hashes only**, no private weather-derived tensors, Notebook payload, or user data. DRAFT; do not merge.
