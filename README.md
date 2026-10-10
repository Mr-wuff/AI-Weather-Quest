# CIDM-S2S for ECMWF AI Weather Quest

**Current experiment: L4 single-model investment qualification v0.2.** No extra CNN/Generic/FNO models are trained. Do not confuse the older v0.1 four-model architecture-ablation plan with the current competition-first plan.

## Current single candidate
- Name: CIDM-S2S Flagship
- Inputs: past 8 complete weeks of global MSLP and TAS; temporal, geographic, slow/fast contextual encoding
- Architecture: spatial/time/phase encoders + antisymmetric latent Flux + Skew + Hidden + optional spectral operator (SOC enabled); BER disabled without a real external port
- Trainable parameters: 1,018,861 in default L4 qualification configuration
- Output: global 121×240 five-quintile MSLP probabilities for days 19–25 and 26–32
- Hardware: NVIDIA L4 for qualification; RTX PRO 6000 Blackwell 95GB **only if** the investment decision warrants it

## Evidence-based decision
A: held-out **official** historical RPSS strongly positive and stable → consider PRO 6000 scale-up, while acknowledging competition-ranking uncertainty.  
B: skill exists but insufficient → improve data/architecture on development periods and retest with **fresh** held-out dates.  
C: no credible skill → stop **this candidate/input route**.  
BLOCKED: official evaluations unavailable → do not declare investment-ready.

The live SON 2026 MSLP leaderboard is **not directly comparable** to historical 2024–2025 issues. AI Weather Quest provides public recognition for high rankings but **does not award monetary or in-kind prizes**.

## Release assets
- Self-contained Notebook: `CIDM_AIWQ_Single_Flagship_L4_CompetitionGate_v02.ipynb`
- Python source files: `src/cidm_aiwq_single.py`, `src/cidm_architectures_reference.py`
- Run locally or in cloud with an ECBox `AIWQ_PASSWORD` secret.
- Source and lightweight reports sync to GitHub if `GITHUB_TOKEN` is configured in the cloud runtime.
- Training checkpoints, optimizers, schedulers, RNG and reports sync to HF **model** repository `wuff-mann/AI_Weather_Quest` if a write-capable `HF_TOKEN` is configured.

Raw ECMWF annual weather data is kept in persistent private storage; it is **not** uploaded into the public model repository. No automatic competition submission happens in the qualification notebook.

The notebook and full project ZIP were distributed in the ChatGPT project handoff; the actual cloud-trained source/reports will sync here once the L4 session is run with repository credentials.

## Official resources
- https://aiweatherquest.ecmwf.int/
- https://aiweatherquest.ecmwf.int/leaderboards/
- https://aiweatherquest.ecmwf.int/evaluation-system/
- https://github.com/ecmwf/AI_weather_quest/tree/src
