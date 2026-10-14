# CIDM-v2 B2-R7 actual frozen gain audit and B2-R8/P0 equal realized-RMS preregistration

Date: 2026-10-14.

## Actual, verified user run (not a training improvement claim)
- ZIP SHA256 **`9c7891b0784ce63c95dad6eb52c073ea69495f2e99adf38a4f187255397f987a`**, sidecar MATCH, ZIP CRC PASS.
- `report.json.status=FROZEN_CONTROLLER_SENSITIVITY_COMPLETED_NO_TRAINING`; 3/3 official seed17 parent test RMSE replay, raw original forward parity zero, 285 aggregate gain-lead curves, 4560 paired window MSE rows, three official PDE file MD5 pass. Frozen backbone and all six heads; zero updates. HF and GitHub source/release sync succeeded or unchanged per user status JSON.
- Lead12 natural **MSE/baseline** ratios by +gain 1/2/4: Burgers `0.998720 / 0.997391 / 0.994629`; Reaction-Diffusion `0.982770 / 0.967053 / 0.940418`; SWE `0.988533 / 0.977920 / 0.959911`.
- Lead12 gain4 rotated controls: Burgers `0.995340`, RD `0.995987`, SWE `0.991854`. Natural negative gain4: Burgers `1.005217`, RD `1.082663`, SWE `1.046082`.
- Natural lead12 +gain1 **actual latent relative RMS** Burgers 0.01632%, RD 0.06495%, SWE 0.18416%; +gain4 0.0654%, 0.2653%, 0.7358%. Rotated +gain4 RMS 0.2255%, 0.0719%, 0.6310% (amplitude NOT matched). No nonzero-gain case triggered cap. The zero-gain `cap_applied_fraction=1` is an edge-case reporting artifact, not clipping.
- Lead12 Natural +gain4 validation-window win counts: Burgers 12/16, RD 16/16, SWE 16/16. These windows **are not unseen outer holdouts** because B0 parent was val-selected.
- Scientific interpretation: under-activation of B2-R6 trained controllers materially contributes to small gains, and sign matters; cannot yet attribute Natural-v-Rotated difference solely to local operator direction because actual perturbation RMS is mismatched. B2-R6 single-seed training eligibility gate remains **FAIL**.

## Delivered B2-R8/P0
Full standalone Colab Notebook `CIDM_v2_B2_R8_P0_FrozenEqualExposure_Colab_STANDALONE.ipynb`, exact Python `b2_r8_equal_exposure_audit.py`, detailed Chinese audit, full ZIP and SHA manifest created and delivered in user chat. Includes exact upstream SHA-locked B2-R6 six trained heads, full REAL B2-R7 result ZIP with hash, and original eight source modules. Complete original B0-R1 18-run parent pinned SHA `69552b29e169ec22ee85d9c1b1a678fdb188f6f25dc18f33b927b88f4dddffd4`; HF **Model**/GitHub Release remote-first, strict official data MD5, test RMSE exact replay and archived validation score reproduction. No Google Drive.

Frozen-only, no new optimizers: predeclared relative latent RMS budgets **0.05%, 0.1%, 0.3%, 0.5%, 1%**, both ± signs, with **0.3% preregistered interpretation budget**. For each autoregressive issue-time step and sample, normalize learned Natural or Rotated controller direction to exactly that fraction of current model's own latent-state RMS; no future truth for any action. Score physical MSE leads 1,3,6,9,12 only after prediction. Pair individual window results (16–31, 3 PDE families, seed17), include win rates, p90/2% regression risk, early/late safety and 1x replay against B2-R7. Not a fresh outer test, and multiple-budget scans are diagnostics only, no deployed gain selection. Successful Q0 only licenses *considering* a new small-budget causal pilot, NOT full retraining, not AI Weather Quest RPSS qualification.

## Actual program tests
Whole **12-cell Notebook** CPU smoke executed every executable cell: exact sources/evidence recovered; real six trained head weights loaded in 1D and 2D, forward zero-correction parity=0, equal-RMS action assert PASS, output ZIP+CRC+SHA256 PASS. Offical 3-family B2-R8 experiment **NOT RUN**.
