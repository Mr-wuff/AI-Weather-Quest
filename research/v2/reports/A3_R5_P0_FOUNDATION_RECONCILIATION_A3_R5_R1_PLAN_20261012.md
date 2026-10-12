# CIDM-v2 A3-R5/P0 findings: Flux and State Synchronization are not disproved

2026-10-12 — actual uploaded R5/P0 report from the user's Colab + archived Foundation A3-R1 and A4-R2 result bundles.

## Three genuinely different experiment contracts

1. Foundation A3-R1: independently retrained conditional ablations across 5 PDE tasks x 3 seeds. Removing Flux raised average RMSE by 17.0378% (13/15 task/seed pairs worse, four of five tasks), Skew +6.5807%, BER +5.1764%. This evidence is not invalidated by a different v2 implementation.
2. Foundation A4-R2: 48-step CIDM-Complete advection pure-latent RMSE 0.599393. Every-six-step **hard replacement** by an encoding of the model's own predicted history reduces RMSE to about 0.0237 (mean gain 96.03% across three seeds), with task/frequency-dependent failures elsewhere. There is no use of future true observations.
3. A3-R5/P0: inference-only ablation of a 36,104-parameter A3-R4 NoSOC checkpoint, eight new samples per family and direction. For 12 seen family x horizon cells, Flux-off alters MSE between -0.02329% and +0.04955%, whereas masking the latent contribution to readout degrades MSE by 62.95% to 921.41%. Sync-off effects span -1.6678% to +2.7903%, and at 36 steps Sync improves all three seen families modestly. Status is DIAGNOSTIC_ONLY_NO_RETRAINED_ABLATION.

## Why results differ: executable mechanism contracts

- Archived CIDM core uses a single state step multiplier `tanh(0.18)` (initial ~0.178), with Flux entering the vector field directly. The A3-R4 path suppresses Flux **twice**: `0.06*sigmoid(dyn_gain)*0.08*sigmoid(flux.strength)`. Read from the actual epoch-9 checkpoint: dyn_gain=-0.9821485, flux.strength=-1.879066, net scalar = **0.000173283**. Different state spaces and flux magnitudes mean these numbers are not mathematically interchangeable, but they support a concrete double-gating hypothesis.
- A3-R4 also allows the shortcut `x_next=x+0.35*(x-prev)+0.17*sigmoid(out_gain)*readout([z,x,x-prev])`, so a useful encoded latent may be mostly static while the output trends forward; this weakens attribution to Flux evolution.
- A4 used `z := E(predicted rolling history)` with 100% replacement at selected 1/6/12-step intervals. A3-R4 instead inserts `z := z+beta*(E(predicted rolling history)-z)` every four steps; actual beta **0.0221835**, far from the hard-sync contract.
- A3/A4 tasks, capacity, optimization updates, horizons, and inference methods are not interchangeable. Parent A3 conditional ablations and current frozen-masking identify different causal questions.

## Corrected scientific statement
Current NoSOC implementation underutilizes Flux; the present soft Sync is task/horizon dependent. **This does not establish that Flux or self-reencoding is conceptually worthless.** Prior positive Foundation findings remain conditional evidence, not universal guarantees.

## New A3-R5/R1 single-candidate training experiment
Delivered Colab standalone Notebook and full project ZIP in ChatGPT.
- Restore RAW antisymmetric edge Flux with exactly one independent trainable strength; local and Flux update coefficients separated.
- Remove explicit history-trend output shortcut: latent state now exclusively generates field increments, while the current field enters as an identity update.
- One shared trainable model; mixed off/soft/hard self-sync training, select sync mode on seen-family validation ONLY; compare all three modes after freezing.
- Evaluate against strong persistence/linear naive, historical R4 frozen checkpoint on identical newly seeded samples, no-Flux and frozen-latent inference interventions, and issue-time causal pair probes.
- Train families Transport/Reaction/Wave, Coupled excluded from this model's training and model selection, but **not untouched in the research program**.
- A program smoke test ran end to end and failed the science Gate (2/8); the formal full experiment has NOT been run. This is synthetic mechanism research, not AI Weather Quest or RTX PRO 6000 qualification.

Runtime: 9 epochs × 40 updates by default, 34,664 model parameters, checkpoint/resume, model-only parent archive checksum verified, science results and best/latest optimizer-complete weights in generated ZIP.

Artifacts: `CIDM_v2_A3_R5_R1_Flux_Sync_Mechanism_Restoration_Colab_STANDALONE.ipynb` and `CIDM_v2_A3_R5_R1_Core_Restoration_FULL.zip`.