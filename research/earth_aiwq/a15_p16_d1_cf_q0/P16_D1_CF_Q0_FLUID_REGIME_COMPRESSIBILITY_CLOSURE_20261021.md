# CIDM-v2 P16-D1 / CF-Q0 Fluid Regime Closure Qualification (2026-10-21)

**Actual numerical and three-seed training executed. Synthetic physics only. No real GEFS/ERA5 forecast skill or full CIDM-v2 neural model training.**

## User hypothesis and archived source re-audit
User proposed that legacy CIDM poor compressible-CFD benchmark performance might originate in indiscriminate information closure, and fluid-regime specific adaptations may be needed for atmosphere/ocean. This precise causal attribution remains a **hypothesis** because the original CFD benchmark failures are not re-executed in this study.

Independently inspect attached `cidm_architectures.py` (SHA256 `f9b1cf4ff7630f5540bb3adde20e4c4b81fff3218d9325ea3105bf28b1d97665`) and the CIDM v2 Chinese architecture archive:
- Archived `AntisymmetricFlux` pairwise exchange approximately sums to zero globally, but full `ClosedInformationDynamicsCore` also contains a skew channel rotation, nonconservative local convolution and hidden reservoir; its complete latent global sum is NOT an exactly conserved physical mass. Random example: flux sum about -2.86e-6 (float32 accumulation), full-field total delta about -3.02672.
- North/south `torch.roll` of archived flux links opposite poles, not physically correct for global meteorology. Numerical skew channel rotation != physical gas mass/energy conservation.
- CIDM-BER virtual latent reservoir is not a validated componentwise mass/momentum/energy budget.
- Local density variations do NOT violate total mass conservation of a closed compressible fluid. An external heat source breaks closed total energy invariance but not mass conservation.

## Frozen actual controlled CFD Q0-N
1D periodic ideal-gas Euler (gamma=1.4), N=64, DT=.0025, FV Rusanov, 8 steps; smooth and Riemann-type initial discontinuity, actual numerical simulation in float64, training data written float32. Closed `q=0`, heated with known exogenous `q(x)`.

On 24 independent physics QA episodes per regime:
- Closed max global mass/momentum/energy step budget error [1.12e-8, 3.00e-9, 2.61e-8] in saved float32 arrays, while **local density one-step RMSE changes 0.00655**.
- Heated mass/momentum/energy-minus-source max budget error [2.79e-8, 6.05e-9, 8.25e-8]. Eight-step mean absolute total energy shift ~.04004. A learned map enforcing globally constant total energy has a mathematical nonzero lower bound on its heated rollout global energy-mean RMSE ~.02631.
- Independent incompressible uniform-velocity passive-tracer control: divergence=0 and carrier density unchanged, local tracer changes 0.00443 but global integral unchanged.
- All **9/9 predeclared numerical gates PASS**. Positive pressure/density and face flux conservative construction tested. No claim on moist atmosphere or ocean circulation here.

## Frozen actual controlled three-seed learnability Q0-T
Separate RNG source streams: each regime 72 train/20 validation/24 holdout 8-step synthetic episodes. 3 seeds 11,23,37, 160 AdamW steps, one-step supervised fitting, **8-step free rollout holdout**; 4 architectures x 2 fluid regimes x 3 seeds = **24 actually executed training jobs**.

| Model | Trainable params | Closed overall 8-step RMSE | Known-heating overall 8-step RMSE |
|---|---:|---:|---:|
| CNN direct residual |1043|.039259|.031843|
| CNN density-locked (intentionally incorrect contract) |1043|.049037|.050350|
| Finite-volume learned conservative interface flux without source |1083|.023134|.024076|
| **Identical conservative interface flux with explicit known energy-source port** |1083|**.023134**|**.017235**|

Heated energy RMSE learned closed flux .040852 vs correctly open source port .028648 (**29.9% lower**). Port-shuffled with frozen trained weights yields energy RMSE .052289. Correct open port has ~5e-7 global energy source balance residual, closed-only flux ~.00746 energy source mismatch. Same flux model's results exactly match between port and closed when q=0. CNN vs flux parameter budget within 3.8%; different inductive physics still invalidates broad universal rank claims.

Important controlled-positive bias: experiment explicitly supplies the correct exogenous Q and structurally injects +dt*Q to the energy equation. The teacher itself is Rusanov FV Euler; a structure-matched physical mechanism model is not an independent general-purpose ML model ranking or proof of unexpected new knowledge. This is an expressivity/constraint qualification only.

## Suggested non-frozen architecture research branch: CIDM-FR
Shared CIDM space/time/frequency/latent dynamics backbone **plus optional explicit physical state U**, regime-specific conserved density and exact shared-face area-volume flux, explicit source/boundary physical budget, correct pole/coast topology and validity conditions, positivity/CFL/entropy controls when compressible/shocked:
- Dry compressible atmosphere: rho, rho*u, rho*v, total E; admissible heat/body/moist source terms only if causally available, no source inferred from future ERA5.
- Anelastic stratification: div(rho_0(z)*u)=0, not div(u)=0 blindly.
- Boussinesq ocean: div(u)≈0 under approximation, active buoyancy/temp/salt, surface heat/freshwater/stress and free-surface boundary accounting.
- Shallow water waves: h and depth momentum, depth positivity, wet/dry/open coast and topographic source balance.
- Never identify latent L2 norm/latent total sum with physical energy or information entropy without a proven observation map.

Next **CF-Q1**: independent 2D compressible Euler shock solver, low-Mach/Boussinesq and open boundaries, at least 3 seeds; explicit baseline ablations wrong universal closure/physical conservation head/known source-aware port/CNN/no-flux; independent generated regimes and long rollout. Only after causal mechanism gate and untouched real weather additional years should P16-D1 genuine GEO/WIND 121x240 trajectory be trained. Original protected 2014/2016 untouched. **No long training authorized.**

## Reproducible deliverables
- Self-contained 27,355-byte Colab `CIDM_EARTH_P16_D1_CF_Q0_FluidRegime_Qualification_Colab.ipynb` SHA256 `9dcec7e78c585d5b492b31c8f4e6387740183209089362f9362711de06ef91a6`; source SHA bootstrap; no historical weather archive or internet needed.
- Independent engine `p16_d1_cf_q0.py` SHA256 `0b612a5759d8861a48e9e72831032db47866f58ef97afcfb841ec99ef0d6f5b3`, conceptual task metadata `fluid_contracts.py` SHA256 `60eb4a8de3981c4533df64622a03d4924020a0e913b21699564fffcef33d170d`.
- Actually computed 24-job controlled result ZIP SHA256 `304b2d4297158b556c5470d135370be0c35056a8a25619953055b89a616e95ea`.
- Full 74,238-byte source/notebook/actual results/QA/Chinese-report ZIP SHA256 `601707001583932035e0cb901ad6e2f31daa63586801a828a2b790a7c331619b`; 17-member SHA manifest, ZIP CRC all PASS. All Notebook cells executed under full 160-step training and numeric-only smoke, source tampering produces BLOCKED ZIP+SHA, independent numeric and learned-gate test PASS.
- No user derived weather fields or individual private user data uploaded publicly. This GitHub draft contains ONLY method/quantitative results/hashes; source and notebook remain in downloadable ChatGPT ZIP. **Do not merge automatically.**
