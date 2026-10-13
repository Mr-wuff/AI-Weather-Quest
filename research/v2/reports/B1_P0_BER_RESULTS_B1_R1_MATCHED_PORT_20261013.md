# CIDM-v2 B1/P0 actual BER experimental results and B1-R1 matched-port structural control

Date: 2026-10-13. Actual B1/P0 user-uploaded Colab outputs and result ZIP audited. This is the **CIDM-v2 research program**, not a new v3.

## Uploaded official user-run B1/P0 synthetic experiment
- 3 seeds (17,41,73) ×3 independently trained variants (no_BER, zero_port_BER, port_BER), 120 updates each, 6-step horizon and 16×16 two-channel synthetic states; nine `best.pt` files.
- Gate `B1P0_QUALIFIED_SYNTHETIC_ONLY`, 6/6. Train families internal transport/reaction, validation-unseen Coupled-family internal dynamics.
- Parameter counts: 11,009 no_BER, and 19,038 for both zero_port_BER and port_BER.
- **Seen MSE** means: no_BER 0.010429169, zero_port 0.010367604, port_BER 0.004136402. Mean per-seed relative gain port vs zero 60.10%.
- **Unseen MSE** means: no_BER 0.021573718, zero_port 0.021760486, port_BER 0.015066123. Mean per-seed gain 30.68%.
- Port BER paired counterfactual skill seen=0.69052, unseen=0.62771; true CF signal power seen=.01331, unseen=.01443.
- Masking correct issued future input produces MSE seen~.010486 vs .004136 and unseen~.021549 vs .015066; swapping issues further degrades. External port has real conditional information value.
- Crucial: zero-port and no-BER cannot distinguish same-history alternate ports BY CONSTRUCTION, thus CF skill 0 is trivial and not evidence of BER-specific superiority.
- Unseen family's external injection rule .23*q into channel0, -.08*q into channel1 is IDENTICAL to train families. This does not show out-of-family external-coupling transfer.
- This does not demonstrate prediction of future *unknown* forcing or benefit on real PDEBench/AIWQ. BER is absent from the B0 PDEBench mainline.

## B1-R1 single-file Colab: latent injection vs readout conditioning
- Provided user-facing `CIDM_v2_B1_R1_BER_Latent_vs_Readout_Port_Colab_STANDALONE.ipynb` and `CIDM_v2_B1_R1_BER_Latent_vs_Readout_Qualification_FULL.zip` (source, Chinese analysis and preregistered contract).
- Three EXACT same-parameter 19,038 models: (a) BER future port into recursively evolving latent state; (b) same issued port and network capacity into decoder readout only (hidden latent evolution port independent); (c) zero-port control. This isolates whether BER dynamics injection is better than merely having a causal issued future input.
- 3 seeds, 150 updates/model, 6-step training, 12-step extrapolation, validation-unseen Coupled family, out-of-distribution future forcing strength/spatial stress, masked/swapped port and paired counterfactual tests, separate gates for conditional-port usefulness and BER-specific latent-injection gain.
- Standalone Notebook **fully executed end-to-end in CPU SMOKE** with all 3 variants, parameter/gradient/path checks and results ZIP. No full official B1-R1 GPU experiment yet; smoke correctly remains diagnostic only.

## Research decisions
Flux and latent evolution remain core candidates based on B0-R2 frozen evidence. Skew and Local require trained capacity-matched PDE ablations. State Synchronization still not validated as a broadly useful repair. NoSOC remains provisional. BER port may be important, but do not add unproven residual network to v2 mainline until structural and real-data qualification.
No RTX PRO 6000 funding until eight-PDE fair benchmarking and issue-time-causal official AI Weather Quest RPSS.