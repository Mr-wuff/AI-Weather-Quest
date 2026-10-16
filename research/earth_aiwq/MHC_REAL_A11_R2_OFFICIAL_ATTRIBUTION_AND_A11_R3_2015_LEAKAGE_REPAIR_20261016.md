# CIDM-MHC-Real A11-R2 official frozen attribution and A11-R3 temporal leakage correction

Date: 2026-10-16. This is a scientifically mandatory correction to the old A11-R1/R2 development evaluation, NOT a change of competition scoring.

## Audited actual user R2
- Submitted report ZIP SHA256 `f130538c0578df882592e2c6a5f52f0086039f7c63e8840149f02eaa79e314f2`; supplied sidecar exact, ZIP CRC PASS.
- Original A11-R1 2 seeds × original 40-step frozen weights × 49 reused 2015 dates × W3/W4 × six interventions = **1,176** rows, no new model training, 2016+ prediction labels not scored, no official AIWQ RPSS.
- Across seeds and leads, relative to the original model's very small forecast anomalies, replacing memory changes outputs **0.27–0.57%**, no_flux **0.18–0.31%**, no_spectral **5.15–5.48%**, no_skew **0.74–1.32%**, no_cross **16.81–19.10%**. No_cross gives tiny apparent 2015 MSE gains ~0.0021–0.0172% in all four combinations. These are same-weights post-hoc development interventions and NOT independent skill.
- Existing trained `lat_scale` isn't used in forward, delta readout sigmoid about 0.118; operator existence or nonzero gradient is not proof of positive predictive impact.

## Critical *newly confirmed* implementation leakage
- In exact original R1 source, `Bank.__init__` quarterly climatology uses `int(d[:4]) <= 2015`, although the neural trainfit cases are only **2013–2014** (101) and **2015** is held out as development (49).
- Its inherited A9 `WeatherBank.__init__` feature normalization also uses `d[:4] <= '2015'`. Both computed statistics incorporate complete 2015 observations INCLUDING forecast targets and even weeks after issue dates. This contaminates original 2015 climatology and MSE comparison. A9's original 2015-in-training setup was not necessarily contaminated *in its original train/2016-dev split*; the leakage was introduced when A11-R1 repurposed 2015 as development without changing both source-year cutoffs.
- R1/R2 frozen output *response* under contaminated preprocessing remains a valid algebraic diagnostic, but its 2015 forecast-error ranking is not a time-clean hindcast and cannot qualify architecture. Earlier model-score claims for A11 must be explicitly demoted.

## A11-R3 preregistered fix
- Standalone Notebook `CIDM_MHC_REAL_A11_R3_StrictTemporal_Colab_STANDALONE.ipynb` SHA256 `c7d16c09ed9d6525b4d7e337b11598181850f5a621c67e34a6e74e62f529055c`; independent driver `a11_r3_strict_temporal_retrain.py` SHA256 `8b28f9820a7c876dcad24c33afe2b245d9a4853dea965b97d8947d169d78e589`.
- First lock and verify the user R1 ZIP, original 8 optimizer checkpoints and R2 report SHA, plus canonical original A0/A8/A9/A10/A11-FIX2 evidence.
- Compute **BOTH** mean/std and quarter-specific **121×240 maps** solely from **2012–2014** past data (2013–14 are training data). Q0 counterfactual perturbation of 2015+ fields must leave *all* trained stats and all four normalized climate maps **bit-identical**. Accessing 2015 for 2015 score is permitted, but not for fitting the normalization or baseline.
- Q0c: 2013–2014 only training-batch true task-gradient/one-step descent, transient discarded; fail-closed if no learning.
- Q1: run original 13,685-param MHC architecture, identical seeds 17/41, original four arms, paired initializations/schedules, **40 optimizer steps each** (320 updates total), same optimizer settings, real native 121×240 MSLP/TAS/PR, 2015 49 dev cases, W3/W4. Per-8-step diagnostics and local optimizer snapshots, report seed on all per-issue rows. Never directly compare absolute MSE on old contaminated normalization to new clean normalization.
- Hard-coded NO independent generalization claim, NO official AIWQ RPSS, NO long training authorization; do not score 2016+ targets, 2017 original sealed. Source files downloaded from SHA-pinned **private** HF Dataset via authorized Colab Secrets; NO Google Drive. Historical R1 weights upload paths are NOT overwritten; new complete checkpoint report ZIP+SHA downloaded locally.
- Whole notebook **5/5** code cells executed CPU smoke with SHA-verified embedded evidence and synthetic label-perturbation invariance; independent real-shape 121×240 four-arm forward/backward gradients PASS. Official private-HF weather R3 320-step retraining **NOT executed** here.
- Full research delivery ZIP SHA256 `e23253dc46a754dcfa3109dc47d394d7d9e8d7ede7cb5c6246604d663bc07ffe` with 8 SHA-manifest entries and ZIP CRC PASS.

## Scientific decision
Do **not** continue to tune the Flux/Skew/Cross operator against the contaminated 2015 scores. First establish a time-clean comparator. If A11-R3 shows meaningful intrinsic persistent-slow-state contribution after clean retraining, then test a direct causal slow-state readout and honest independent future-year qualifications. Otherwise investigate whether slow state is actually informative, and add Z500/SST/legally available S2S predictors before scaling.