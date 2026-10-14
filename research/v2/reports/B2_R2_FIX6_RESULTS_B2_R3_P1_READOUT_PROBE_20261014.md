# CIDM-v2 B2-R2 FIX6 real result and B2-R3/P1 frozen-backbone readout pilot

## Verified uploaded FIX6 evidence (2026-10-14)
Original user ZIP SHA256: `121f60b271ff7a970647c510c68dcd273a244b67267ce90cccdddd928d5390e7` (sidecar match, ZIP CRC PASS).
`report.json.status=FROZEN_DIAGNOSTIC_COMPLETED_NO_TRAINING`; 9/9 official model replay PASS, 45/45 task×seed×lead summaries, 270 raw batch-lead records, 54 forward parity checks (max abs difference 0), max additive residual `2.659548949283678e-07`. **Frozen audit only; not architecture improvement.**

Three-seed average at lead12:

| System | Flux RMS | Local RMS | predicted vs true increment cosine | predicted/true increment power | pre-tanh decoder >2 | self re-encoding latent cosine |
|---|---:|---:|---:|---:|---:|---:|
| Burgers-1D | 90.6658 | .0508 | .0888 | 2.4664 | 35.12% | .1523 |
| Reaction-Diffusion-2D | 2.5189 | .0376 | .4835 | 2.0683 | 0% | .4071 |
| Shallow-Water-2D | 24.7148 | .0227 | .4701 | .3803 | 6.64% | .0804 |

Burgers has severe late-direction degradation and some decoder saturation. SWE is under-responsive from the **first** step despite initially good direction. Single-step deletion counterfactual and latent coordinate metrics do not establish operator necessity or physical conservation. Reaction-Diffusion is a no-regression control, and failed B2-R1 Flux-Cap Q0 must not be repeated.

## B2-R3/P1 experiment released to user
Self-contained Colab Notebook: `CIDM_v2_B2_R3_P1_FrozenBackbone_ReadoutSufficiency_Colab_STANDALONE.ipynb` delivered in conversation with full ZIP, exact original source files, Chinese analysis and SHA256 manifest.

- Embedded exact FIX2 4 Python modules, hashed; original B2-R2 repo driver and new probe, hashed; embedded *actual FIX6 report + all 45 rows*, hashed.
- HF Model `wuff-mann/AI_Weather_Quest` / GitHub Releases first for complete original 18-run B0-R1 parent ZIP, SHA256 `69552b29e169ec22ee85d9c1b1a678fdb188f6f25dc18f33b927b88f4dddffd4`; one-time user upload only when absent. No Google Drive.
- Q0 frozen evidence and parent ZIP verified BEFORE any probe training. Q1 only **seed17** paired small readout probes, fixed 100 updates each for 3 PDEs, frozen B0 parent. Identical parameter count, batch indices, optimizer and schedule for OBS-control vs LATENT-readout. Readouts correct baseline outputs only; they do not modify or feedback into latent dynamics.
- Train on original B0 train split; first half val for choosing checkpoint every 20 updates, second half for independent *probe* audit. Original B0 parent was previously selected using validation and therefore this is not an untouched outer holdout.
- Pilot selection gate: Burgers and SWE LATENT vs baseline late-lead MSE improvement >=3% AND LATENT vs OBS >=2%, RD regression <=2%. Passing **only** authorizes subsequent 3-seed/fresh-holdout replication, not full CIDM-v2 training or competition.
- Both 1D and 2D new probe code supports original model shape and normalization. In-container end-to-end CPU synthetic program smoke executed successfully, including notebook all executable cells, exact SHA import, matched paired pilot, ZIP+CRC+SHA export. **Official B2-R3 trial not run**.

No statement about the full PDE v1 qualification or AI Weather Quest RPSS eligibility is made. Both remain unqualified.