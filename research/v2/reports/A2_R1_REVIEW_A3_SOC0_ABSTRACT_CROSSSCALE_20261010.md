# CIDM-v2 A2-R1 result and A3-SOC0 architecture research — 2026-10-10

**CIDM is a general prediction architecture, not a weather/ocean-only forecast product.** SOC should be formulated as abstract *information evolution across latent representation scales*, not a correction designed around specified PDE variables.

## A2-R1 evidence (synthetic, uploaded experiment)
- One 36,361-param model, 10 epochs ×48 steps, best epoch 8, nine of nine synthetic mechanisms/score Gates passed.
- ID12 MSE 0.000061307; OOD12 both 0.000049913; OOD12 velocity-only 0.000050740; source-only 0.000049918; OOD24 0.000177272; OOD36 0.000475148.
- OOD velocity-only counterfactual skill 0.999561; OOD36 compound CF 0.995517. A2-R1 used simulator-known future velocities and source plus a prescribed upwind transport step, so these do **not** establish general learned dynamics.
- Long-horizon categorical RPS OOD36 0.196192 worse than the uniform-category RPS 0.186197. Deterministic success does not establish probability calibration.
- Implemented SOC is a low-pass pooled two-variable exchange +/-q; it is not a genuinely general scale-to-scale transfer operator. Its strong frozen-weight ablation effect is not sufficient evidence of architecture-wide universal cross-scale reasoning.

## Proposed A3-SOC0 (qualification hypothesis, not achievement)
- One shared model across four synthetic dynamical families: transport, reaction, wave, nonlinear cross-scale coupled.
- Training model sees **neither** family identity, PDE coefficients, simulator velocities, privileged future forcing, nor true future states.
- Latent **two-level orthonormal Haar** decomposition; state-dependent bounded pairwise rotations between coarse and fine detail bands at both levels. Operates independently of physical variable meanings; preserves only the SOC suboperator's encoded coefficient L2 norm, not physical energy, mass or Shannon information.
- Generic antisymmetric learned local Flux, channel Skew, bounded local dynamics, prediction-only reencoding synchronization.
- Frozen-model SOC on/off test; independent OOD 8/16/24 step forecasts; initial coarse-band and fine-band perturbations separately propagated by data generator to test causal coarse→fine and fine→coarse future responses.
- Decision cannot pass if only math checks pass. Model must beat simple persistence/linear-trend references in specified regimes and show **observable added cross-scale response** on independent cases.
- Code and standalone Colab notebook provided in the ChatGPT research package; small-budget full-notebook smoke passed while scientific Gate failed as expected.
- This stage is synthetic. RTX PRO 6000 AI Weather Quest training remains **BLOCKED** pending legitimate real-weather data and independent RPSS.

## Research progression
Abstraction of SOC is the current isolated research target. Later revisit BER as a causal external-boundary information mechanism with explicit ports and uncertainty, not a free latent residual.
Previously disclosed account tokens must not be reused; any cloud repository synchronization requires newly rotated Secret-based credentials.