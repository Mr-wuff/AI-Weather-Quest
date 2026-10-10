# CIDM-v2: Predictability-Constrained Structured Information Dynamics (RESEARCH PROPOSAL)

**Status:** A0 prequalification only; does not replace the frozen CIDM-v1 paper or constitute an approved global weather model.

## Why revise CIDM now
- Foundation A3: Antisymmetric Flux had an average conditional ablation effect of +17.04% RMSE when removed, with weak generic-capacity rescue in tested cases. SOC / BER are task-dependent.
- Foundation A4: autonomous CIDM-Complete latent rollout became unstable in several PDE regimes; self-reencoding sometimes dramatically improved long forecasts, but excessive frequency also harmed accuracy.
- A5 initial scaling/saturation: parameter allocation and training contracts interact with apparent scaling gains; longer short-lead training is not a universal cure for rollout drift.
- CIDM-v4 South China Sea and CIDM-TF: target physical quantities (SSH, current, cyclone center/track, intensity) need task-sensitive readouts and causal drivers; aggregate RMSE cannot certify every critical variable.
- AIWQ L4 v0.2: official 2024–25 retrospective MSLP RPSS W3=0.001480 and W4=0.007201 (20 issue dates). The existing 8-week MSLP/TAS autonomous model **must not be scaled on PRO 6000**.

## Proposed differentiating principles (unverified)
1. **Predictability-aware latent state:** split predictable slow memory, fast perturbations, unobserved/unpredictable uncertainty; learn anomaly-conditioned probabilistic forecasts relative to climatology.
2. **Responsible structured dynamics:** retain Flux/Skew when justified, separate physically designated transported components from non-conservative forcing and residual terms. Use spherical metric-aware finite-volume operators for global grids; periodic-plane sum-zero is not global physical conservation.
3. **Causal exterior forcing:** future exogenous information only if sourced from a forecast initialized no later than issue time. Never use future final ERA5 analyses as if operationally available.
4. **State validity and synchronization:** bounded, learned correction against reencoding of model-generated predictions only; evaluate synchronization on/off on an identical trained model and prevent test-set frequency selection.
5. **Task-specific prediction constraints:** robust trajectory/track/intensity/target quantities and ranked-probability / Brier calibration targets; do not equate generic RMSE with scientific success.

Candidate abstraction:

    z = (z_slow, z_fast, z_uncertain)
    z_fast' = z_fast + dt * [F_flux + F_skew + F_global + F_issued_forecast - D]
    z_slow' = z_slow + eta*(slow_target-z_slow)
    z_sync = z_pred + bounded_gate*(E(model-predicted-history)-z_pred)
    p(quintile|issued inputs)=softmax(log(p_climatology)+delta_logits(z,lead))

These are ideas to test, not proven innovations. Similar partial concepts exist in Neural ODE, Koopman, conservative flux networks, PhyDNet, port-Hamiltonian modeling and neural weather/forecast postprocessing.

## Staged Gate (no unearned GPU investment)
- **V2-A0:** single-model synthetic forced transport mechanism preflight: Flux cancellation; finite gradient; external driver causal influence; self-sync on/off; OOD; proper synthetic RPS. Passing this does **not** qualify for PRO 6000.
- **V2-A1 CPU data feasibility:** real issue-time causal S2S reforecasts, SST, Z500, MJO availability; history depth; off-grid metadata; compare official score protocols; ensure live Thursday issuance.
- **V2-A2 L4 single-model AIWQ:** use one strong qualified hybrid/pure candidate on native 121×240 global mesh, legitimate input times, probabilistic W3/W4 scoring and fresh holdout. Compare with climatology and leaderboard only on comparable dates/windows.
- **V2-A3 detailed ablations for v2 paper after a working candidate is established** (not a competition gate).
- **V2-A4** PRO6000 global training *only after* actual independent evidence meets registered investment conditions.

Security: No tokens, data or weights in this research roadmap. Revoke any credentials pasted into chat and inject replacement write-scoped secrets in cloud; GitHub already connected through authorized app; HF app is read-only in this conversation. Continuous sync is a training-runtime process, not an active background chat.
