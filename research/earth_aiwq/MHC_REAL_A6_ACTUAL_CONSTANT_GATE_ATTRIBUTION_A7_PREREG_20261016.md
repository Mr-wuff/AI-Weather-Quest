# CIDM-MHC Real A6/P0 actual 2020–2021 evidence and A7/P0 next qualification
2026-10-16 | Scope: **frozen, low-dimensional residual probe**; not ECMWF AI Weather Quest official RPSS.

## Audited original upload
- User `CIDM_MHC_REAL_A6_P0_BOUNDED_CAUSAL_ADAPTER_REPORTS.zip` SHA256: `6edbe611a04e63d183d7515e9009b27d0f453b4aee3742e90821139199071706`; user sidecar matches; ZIP CRC passed; exactly three payload files, including status, 194 per-issue rows, and original source SHA256 `99da9b0d9f1b911d0339a0310ba3ca1255a50847dd0b5d6031af4a2777c0a9d7`.
- Status `NEW_2020_2021_FROZEN_ADAPTER_EVALUATION_COMPLETED`; 2015/2016 development Gate passed, 2020–21 97 external issues (49/48 years), W3 external Gate **FAIL**, W4 PASS, joint FAIL; 2017 original evaluation sealed; no model optimizer updates. HF and GitHub sync receipt SUCCESS.
- Pooled gain of adapter vs old MHC: **+0.88744% W3** (50/97 improving, under 60% contract), **+3.94211% W4** (60/97 improving). Against Instant: +0.98974% W3 and +3.63076% W4. W3 2021 vs Instant -0.46493%. These are coarse lat-belt residual MSE, not full-grid skill.
- Four-week circular moving block 5000-draw **descriptive, non-independent** 95% ranges vs Old Memory: W3 [-1.286%, +3.216%], W4 [+1.583%, +6.402%].

## High-confidence new architectural finding: A6's adaptive gate is mathematically inert
- ALL 194 archived W3/W4 records have `preissue_risk_gate=1.0` (zero sample variance). Recomputed aggregate scores exactly from per_issue.csv match status JSON.
- Among original A6 2015 diagnostic candidate list, each comparable `(alpha,beta)` adaptive True and False pair had **bit-identical mean loss**.
- Actual code `risk_g` uses mean last-four-week std divided by mean eight-week std; for ratio <=1 it returns **1**. Under all observed 2020/21 issues, no attenuation happened. The selection function prefers a safe adaptive-labeled candidate even when its score is identical to constant; labeling it adaptive is not evidence of adaptation.
- Chosen alpha=1, beta=.25 makes all tested external predictions **0.75 old-MHC + 0.25 Physical**, a constant convex blend. W4 improvement is real for this frozen coarse probe but cannot be attributed to conditional state-aware gate or to a new Markov closure.

## A7/P0 preregistration: develop-only, no new external years
- Self-contained Notebook `CIDM_MHC_REAL_A7_P0_NondegeneratePhysicalGate_Colab_STANDALONE.ipynb`, source `mhc_real_a7_nondegenerate_gate_audit.py` SHA256 `a440a303d30e1c381d9c633250ddabdbfd64f985c2877442a2953e8a8d17bade`.
- Pin and embedded original A0–A6 evidence. First strictly reproduce original A6 ZIP, report metric means and gate collapse. Then recover original exact frozen real-weather A1 seed17 step64 checkpoint, causal 2012–17 annual netCDF, exact 2016 12-issue prediction and A6 constant-mix development MSE replay before any scoring.
- Keep alpha=1 and beta=.25 fixed. Compare constant `g=1`, legacy risk, training-only-q10/q90 calibrated `vol_quantile`, `trend_quantile`, and predeclared primary `hybrid=(vol_quantile+trend_quantile)/2`. Features only from already known eight-week MSLP/TAS. No phase/round labels, no future true data in inference.
- A7 joint qualifying developmental signal requires the hybrid gate to be **nondegenerate** on 2015 and 2016 (std >=.08 and >=15% of gates in [.05,.95]); improve constant-mixture W3 and W4 MSE by >=0.2% on BOTH 2015 and **reused** 2016, >=55% 2016 per-issue wins and tail CVaR<=1.05. If failed, stop; **no A7 official external test**, no full CIDM weather training and no RPSS claim.
- The model must not open 2017 forecast evaluation truth, revisit 2020–21 for parameter tuning, nor access 2022+ target labels. If a candidate passes only the exploratory development gate, prospectively reserve fresh 2022–23 for a future separate qualification after explicit new contractual freeze.
- Model weights/caches in private HF Model/Dataset; report ZIP+SHA at most one atomic HF commit when authorized; optional GitHub source + lightweight report sync; no Google Drive; failure statuses and separate sync receipts. All **10 Notebook executable cells** passed CPU smoke with embedded immutable A6 evidence, and independent synthetic ridge+5-gate mathematical end-to-end test passed. This is NOT an official full-data A7 result.
- Complete deliverable ZIP SHA256 `802c798920d0943a37ac245324d2bb3bf875dfec8a60297e3ee304903bec253e`.

## Decision
Do not call A6 physical-state-adaptive closure successful. Preserve its fixed 75/25 mixture as an A6 diagnostic comparator; demand measurable gate variation and causal action-value contribution before any full Markov architecture training. Next mainline after A7 is more information-rich Z500, SST, S2S hindcast inputs, and ultimately official quintile-probability RPSS.