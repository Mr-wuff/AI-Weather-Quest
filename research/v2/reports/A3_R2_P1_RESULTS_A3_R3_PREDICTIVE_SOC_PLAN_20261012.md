# CIDM-v2 A3-R2/P1: official user-run frozen predictive-scale probe; A3-R3 experiment (2026-10-12)

CIDM is a GENERAL prediction architecture. SOC is abstract cross-scale latent information evolution, not a PDE-specific discretization.

## Uploaded P1 run (frozen A3-R1 SOC1 model, no new CIDM training)
- 52,658 parameters; epoch-9 checkpoint SHA256: `8616737266efd776c1c95e63d440cc14dd8df02abe7bc901a240f22ff3381603`.
- Train local ridge probes only: train 24, val 8, test 12 paired cases per seen family/direction; Coupled blind 16 per direction. Fixed ridge alpha 0.05, local 3x3 features, leads 4/8/12/24.
- Colab report status: `DIAGNOSTIC_ONLY_NO_ARCHITECTURE_GATE`. The report has 280 score records and 24 diagnosis records.
- 12-step skills test: Wave coarse-to-fine raw=0.3720, latent=0.5874, immediate-post-SOC=0.5877; Wave fine-to-coarse raw=0.3378, latent=0.7654, post=0.7656.
- Reaction coarse-to-fine raw=-0.1664 and latent=-0.4361; Coupled coarse-to-fine raw=-3.2432 and latent=-9.5049; Coupled fine-to-coarse signal/source ratio only 0.00342, weak and ineligible.
- Wave fine-to-coarse lead12 *opposite* latent scale probe=0.844, compared with expected-band probe=0.716. Fixed Haar band identity may not align with predictive information.
- Initial SOC operation barely changes local linear decodability, which neither proves SOC helpful nor eliminates non-linear / repeated-rollout losses. Linear probes are not an information sufficiency proof; variable dimensions and paired-delta diagnostic restrict interpretation.

## A3-R3 preregistered follow-up
- ONE trainable domain-general CIDM (~53,740 params): same A3-R1 SOC1 structural backbone plus train-only latent predictive-scale auxiliary readout; no simulator coefficients, PDE labels or future observations passed into inference.
- Same-prefix issue-time intervention pairs; 4/8/12-step future opposite-band supervision and autoregressive field prediction with bounded normalized contrast. 8/12-step alternating train horizon.
- Train only Transport, Reaction, Wave; validation only on those families. Entire Coupled family withheld until checkpoint frozen.
- L4/T4 Colab standalone Notebook, all sources embedded and verified; 10 epochs x40 updates, batch 3. Persist workdir for resumability.
- Includes exact real A3-R1 epoch-9 model-only checkpoint, SHA256-verified, for same-test-seed descriptive comparison. This is NOT an equal-budget trained ablation; SOC on/off is a frozen-weight dependence diagnostic.
- Blind Coupled weak-signal directions are explicitly INCONCLUSIVE, not forced positive. If any science gate fails, RTX PRO6000 remains blocked.
- Full notebook execution smoke has completed on CPU using 1 epoch x2 updates and 1 eval seed/direction, including parent checkpoint loading, field and paired tests, Gate JSON, plot and ZIP packaging. Smoke scientific status failed; no claim of full experiment achievement.

Chat attachments for the experiment:
- `CIDM_v2_A3_R3_Predictive_Scale_Dynamics_Colab_STANDALONE.ipynb`
- `CIDM_v2_A3_R3_Predictive_Scale_Dynamics_Colab_FULL.zip`
- `A3_R2_P1_RESULT_REVIEW_CN.md` and `A3_R3_EXPERIMENT_CONTRACT_CN.md`.