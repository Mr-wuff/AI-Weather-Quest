# CIDM-v2 A3-SOC0 audit and A3-R1 SOC1 plan (2026-10-12)

**Research role:** CIDM is a general prediction architecture. Cross-scale information evolution is an abstract latent mechanism, not a task-specific weather/ocean add-on.

## Verified A3-SOC0 results (synthetic only)
- Source: user-uploaded CIDM_v2_A3_SOC0_QUALIFICATION_REPORTS.zip and executed Colab Notebook; ZIP SHA256 manifest matches its contents.
- One 46,767-parameter shared CIDM, 6 epochs ×48 updates; best checkpoint epoch 5 (zero-based).
- Gate 5/6 passed, **cross-scale counterfactual benefit failed**.
- SOC counterfactual gain over frozen disabled-SOC for coarse-to-fine perturbations: Transport -0.0641, Reaction -0.1013, Wave -0.0672, Coupled -0.5444; all 6 seeds per family were negative.
- Fine-to-coarse gains: Transport +0.0045, Reaction +0.0436, Wave +0.0241, Coupled -0.2963. Only two of eight aggregate family/direction tests exceed +0.02.
- All four families beat persistence for 8/16/24 steps, but early horizon linear trend defeats the model on Transport and Reaction. Original 'long forecast stability' gate compares with max(persistence, linear_trend), which is relatively weak.
- Original scale perturbation was applied before generating all history frames; history already contains some propagated response. That prevents isolating propagation *after issue time*.
- The orthonormal Haar SOC suboperator conserving coefficient L2 is not physical mass/energy or information entropy conservation. Disabled SOC inference is not an equal-parameter trained ablation.

## New A3-R1 SOC1 qualification experiment
- **Single-model training** for a broad model, no CNN/FNO/Generic re-training and no task ID / known PDE coefficients / future forcing in model inputs.
- Two-level latent Haar representation with independent bounded learned coarse-to-fine and fine-to-coarse messages, whose output is not artificially forced to conserve coefficient L2; zero-activity identity check and transfer-gradient tests.
- A strict paired issue-time counterfactual: input histories match exactly before the last observation, and only the final frame is perturbed at one scale. The same physical simulator parameters produce two alternate futures.
- Training families: Transport, Reaction, Wave. **Entire Coupled family unseen** until blind assessment; do not use it for model selection.
- Frozen-model intervention experiments: SOC both / none / only coarse-to-fine / only fine-to-coarse, independently on 8/16/24-step OOD prediction and 12-step scale-causal response.
- Scientific Gate requires actual cross-scale predictive gain, separate directions, cross-family transfer and stronger baseline than persistence alone.
- Default standalone Colab: 9 epochs × 48 updates, native 24×32 synthetic grid, 52,658 parameters, checkpoint resume, optional Secret-only HF model and GitHub status sync.
- A complete mini-budget notebook smoke ran successfully, and science Gate remained failed as expected. **Full A3-R1 not yet run.**

Neither stage measures AI Weather Quest RPSS, qualifies for RTX PRO 6000 global training, nor proves universal physical conservation.

Deliverables attached in chat: CIDM_v2_A3_R1_SOC1_Causal_Scale_Exchange_Colab_STANDALONE.ipynb and CIDM_v2_A3_R1_SOC1_Causal_Scale_Exchange_FULL.zip.