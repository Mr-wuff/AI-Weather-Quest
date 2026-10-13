# CIDM-v2 B1-R1 actual qualified synthetic BER-route experiment and B1-R2/P0 pulse-memory audit (2026-10-13)

## Verified user-uploaded B1-R1 report
- Nine complete best model checkpoints, 3 seeds x 3 19,038-parameter variants; 150 updates each. Original 4/4 synthetic gates passed: `B1R1_STATE_PORT_ADVANTAGE_QUALIFIED_SYNTHETIC_ONLY`.
- Mean State-Port vs equal-parameter Readout-Port relative MSE gains: seen6 +30.07%, unseen6 +12.37%, seen12 +2.66%, unseen12 **only +1.12%**, unseen12-stress +14.56%. At unseen12 one of three seeds is slightly negative.
- State-Port heldout unseen12 CF skill ~0.589 vs Readout-Port ~0.509, indicating an identifiable contrast-sensitivity gain despite minimal overall MSE gain.
- Conditional future ports are GIVEN by the simulator at forecast issue; neither model predicts unobserved future forcing. Coupled family external force coefficients (+.23,-.08) are unchanged across seen/unseen internal dynamics, so external coupling-law transfer has not been proven.
- Best checkpoint for all nine training runs was selected at final step 150: there is no demonstrated full convergence.

## Completed independent B1-R2/P0 frozen diagnostic
- Exact nine checkpoints + B1-R1 source embedded in one standalone Colab Notebook; all 9 best.pt SHA256 verified. Original models NOT retrained.
- Issue-time matched history; baseline known future forcing is identical for both futures except a two-step positive/negative external event. After the event ends all issued ports are identical. Evaluate true vs model-generated persistent *post-event* response, full-field MSE, power ratio, and lead-resolved CF skill at 12/24 steps. Optional external force law shift labelled OOD stress.
- Local full execution: 3 seeds ×32 new pairs per dynamical family; model and case grids all completed. Gate `B1_R2_FROZEN_MEMORY_NOT_FULLY_ESTABLISHED`.
- For unseen family, early pulse 12-step State-vs-Readout field gain **-7.2%**, post-pulse CF-skill delta **-4.219**. Early pulse 24-step field gain **-34.4%**, post-pulse CF-skill delta **-34.726**. Real post-event response power ~1.64e-4, much smaller than original continuous forcing CF power ~.0144, so report denominators and amplitudes.
- In the 24-step unseen early-pulse case, mean post-event predicted/true response power for State-Port rises drastically; findings are input temporal-distribution-shift stress, not direct contradiction of B1-R1 on its train-distribution continuous forcing.
- Changing the external source coefficient in an OOD stress without informing the model is not a test of whether BER can infer unknown external physical laws.
- Full default inference core audited; standalone complete Colab notebook independently passed end-to-end CPU smoke with embedded weights, output ZIP/plot and hash checks.

## Decision
State-Port BER has genuine synthetic continuous-input short-horizon conditional forecasting advantages, but persistent and stable memory after sparse external events is **not yet established**. Keep BER as a candidate only; do not mandate integration into v2 PDEBench mainline. Next: preflight mixed continuous/pulse/cessation curriculum and compare against matched Readout-Port, with separately frozen law shifts, lead-resolved response power and at least 3 independent data seeds.
Main CIDM-v2 mandate unchanged: beat archived v1 across all eight PDE benchmarks, solve long rollout/latent drift, compare to leading models, then causally correct official AI Weather Quest hindcast RPSS; no RTX PRO6000 training authorization.

Chat artifacts: `CIDM_v2_B1_R2_P0_Frozen_BER_Pulse_Memory_Audit_Colab_STANDALONE.ipynb`, `CIDM_v2_B1_R2_P0_Frozen_BER_Memory_Full.zip`, local executed report ZIP and Chinese review.