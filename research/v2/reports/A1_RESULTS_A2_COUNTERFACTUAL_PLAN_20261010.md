# CIDM-v2 A1 review and A2 research plan

Results derived from the user's A1 synthetic experiment ZIP. These are not ECMWF weather forecast scores.

| Test | CIDM MSE | Persistence MSE |
|---|---:|---:|
| ID 12-step | 0.00115682 | 0.00752707 |
| OOD 12-step | 0.011810 | approximately 0.03431 |
| OOD 24-step | 0.0934875 | 0.0841103 |

A1 trained 42,413 parameters for 8 epochs, selected epoch 7. Mechanism Gate 4 of 8 passed, 4 failed.

OOD 12-step relative error changes when frozen mechanisms are disabled: no SOC +0.37%, no Flux +1.74%, no BER +0.33%, no Sync +0.11%. Setting velocity to zero improves error by 0.055%, indicating very weak velocity-conditioning. BER is more helpful in ID than OOD. Twenty-four-step OOD is worse than persistence.

A2 proposal: train only one conditional CIDM on paired synthetic counterfactual futures that share the exact same past observations, but differ in known future velocity, known external source, or both. Evaluate 12 and 24 steps, plus a 36-step extrapolation. Freeze model before diagnostic interventions. Explicit upwind transport is an inductive physical prior, not learned physics. Algebraic cross-scale transfer balance is not proof of physical conservation.

Hardware: Colab L4 for A2; RTX PRO 6000 remains unqualified until independent real AI Weather Quest skills are proven.

Chat handoff: CIDM_v2_A2_Counterfactual_Control_Qualification_Colab_STANDALONE.ipynb; CIDM_v2_A2_Counterfactual_Qualification_FULL.zip.