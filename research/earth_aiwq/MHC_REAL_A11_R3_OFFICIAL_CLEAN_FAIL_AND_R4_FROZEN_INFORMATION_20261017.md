# MHC-Real A11-R3 strict-temporal actual result and preregistered A11-R4 frozen information audit

2026-10-17. **A11-R3 user result ZIP SHA256** `c1e4836327b319b541ad267aa6e79c451ec586c519a182387aa1675a21d62f9e` matched supplied sidecar; ZIP CRC PASS; user Colab log and status agree. Training climate + standardization computed from 2012-2014 ONLY. Mutating 2015 observed truth yielded bit-identical means/std and four quarter climate maps (Q0 PASS). Q0c train-batch one-step descent PASS. User ran 2 seeds x 4 arms x 40 gradient steps = 320 steps, training 2013-14 101 cases, 2015 **reused developer** 49 issues, no 2016+ forecast scoring or official AIWQ RPSS.

Clean native-grid standardized MSE averaged across variables (2015):
| arm | seed17 W3 | seed17 W4 | seed41 W3 | seed41 W4 |
|---|---:|---:|---:|---:|
| train-only climate | .2331867 | .2365019 | .2331867 | .2365019 |
| Stateless | .233167 | .236583 | .233470 | .236851 |
| Persistent | .233167 | .236583 | .233470 | .236849 |

Exact persistent gain over stateless: seed17 W3 +0.00005452%, W4 +0.00009659%; seed41 W3 +0.00020216%, W4 +0.00102076%. All < predeclared 0.2%. Seed17 persistent gain over climate +0.00828% W3 / -0.03453% W4; seed41 -0.12139% / -0.14686%. **Mechanism Gate FAILED. Long training not authorized.** Do not compare raw original R1/R2 scores against R3 due different normalization; R1/R2 2015 contaminated when selecting preprocessor statistics.

New frozen diagnostic A11-R4 (NO new neural optimizer steps): strict replay of user R3 2015 forecasts from pinned frozen seed17/41 persistent/stateless checkpoints; preserve latest week but replace first 7 historical weeks by train-only climate and quantify sensitivity; assess persistent output deltas; fit one bounded scalar output amplitude for each seed, lead, variable on **2013-14 training-only observations** and assess original vs calibrated vs climate on reused 2015 development. No access to 2016+ scored forecast truth, 2017 audit sealed, no AIWQ RPSS, no formal training eligibility granted. R4 explicit historic information deletion is a nondeployable diagnostic, not a causal deployment proof.

Research delivery notebook `CIDM_MHC_REAL_A11_R4_FrozenInformation_Colab_STANDALONE.ipynb` SHA256 `5f52a1be88b33d7e38ace1eea6ff34c7a5386cb79578fd5743214e89ea17112c`; driver `a11_r4_frozen_information_audit.py`; independent whole-notebook CPU 5/5 PASS, trained R3 checkpoint SHA replay, plus standalone **121x240** synthetic full-pipeline (2 seeds/2 arms/history deletion/scalar calibration) PASS, including discovery and repair of a dictionary-cleanup runtime bug. **No true R4 meteorology run yet.** Complete research ZIP SHA256 `4560f7b806a80c6b1ff45e06fbc6783b0766010b16c50ed5d83b3d875588ac13` with CRC + per-file hashes PASS. Private HF Dataset for weather, authorized HF_TOKEN, no Google Drive; download resulting ZIP, SHA and log. 

Research decision: if R4 shows poor time history utility or saturated/readout-unsalvageable anomaly projection, stop low-capacity A11 tuning, introduce properly causal Z500/SST/official S2S predictors and evaluate climate-relative skill per variable/lead with new held-out periods.