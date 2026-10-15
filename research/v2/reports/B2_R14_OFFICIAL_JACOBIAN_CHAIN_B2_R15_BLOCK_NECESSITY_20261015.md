# CIDM-v2 official B2-R14 frozen Jacobian-chain results; B2-R15 block-Jacobian necessity preregistration (2026-10-15)

## User-provided official B2-R14, independently verified
ZIP `CIDM_v2_B2_R14_P0_JACOBIAN_CHAIN_REPORTS.zip` SHA256 **`091cf3e31fd77e98745f332b47ede2438d1ffb486cf42c1d5c5cd10543b6b772`**, user sidecar matches, CRC PASS, 18/18 internal SHA256 manifest entries independently verified. Final report status `FROZEN_JACOBIAN_CHAIN_COMPLETED_NO_TRAINING`; 6 controller-family groups, 288 chain comparisons, 864 local VJP step rows, parent B0 seed17 test RMSE replay 3/3. No training updates, `train_allowed=false`. Uploaded runtime sync JSON reports HF Model and GitHub source/release/status SUCCESS.

Actual original `lead=1` **median log10 cumulative gain** across 2 TRAIN windows × 2 probes:
| PDE | head | FP32 at FP32 states | FP64 same cast FP32 states | FP64 at FP64 native states |
|---|---|---:|---:|---:|
| Burgers | Natural |4.076974|4.079397|4.684807|
| Burgers | Rotated |2.981897|2.965894|3.819541|
| RD | Natural |0.607085|0.607085|0.607085|
| RD | Rotated |0.615966|0.615966|0.615768|
| SWE | Natural |**32.576525**|**32.576398**|**32.995365**|
| SWE | Rotated |**27.761135**|**27.760195**|**28.022100**|

SWE shows catastrophic direction-dependent *12-step recurrent-state linearized adjoint gain even at identical FP32 original states* in FP64, thus trajectory drift is NOT necessary for huge gain. State drift additionally changes direction. Values represent normalized local VJP chain, not Jacobian spectral norm, full-parameter gradient or forecast skill. R14 held history conditioning fixed and used 2 TRAIN windows × 2 probes/seed17 only. Original B0 v2-v1 superiority still fails on Burgers and SWE.

## New B2-R15/P0 frozen block-Jacobian audit
Delivered standalone `CIDM_v2_B2_R15_P0_BlockJacobianNecessity_Colab_STANDALONE.ipynb`, source `b2_r15_block_jacobian_audit.py`, detailed Chinese expert review, original exact actual B2-R14 report SHA-locked and all prior original source/trained head evidence, research archive ZIP. Output ZIP SHA256 **`182b733acf838e6917c9e0fc0366c07e295801fa6673a59a1e013a93fec6a2e9`**.

Exact original 12-step forward at FP32 states, FP64 local derivative on same frozen states, coupled z,p recurrence. VJP block split into Jzz, Jzp, Jpz, Jpp. At each timestep apply five gradient-only **counterfactual** chains: full, drop d(pnew)/d(zold), drop d(znew)/d(pold), drop both cross derivatives, replace d(znew)/d(zold) by Identity. Independent four-block sum vs joint autograd VJP <=1e-8 required; the new `full` chain MUST reproduce all **288** B2-R14 original official `fp64_fixed_state_chain_log10_gain` records within 1e-6. Any failure blocks. 3 PDE, 2 trained heads, 2 TRAIN windows, 2 terminal cotangents, 12 leads, 5 modes = **720** counterfactual step rows + **720** paired comparison rows. No optimizer, no gradient clipping, no changed architecture, no forward intervention, no newly generated target predictions. Intended attribution is **necessity on this linearized path only**, not proof of physical branch causing forecast error.

Standalone Colab restores SHA-locked exact original B0 18-run parent ZIP (SHA `69552b29e169ec22ee85d9c1b1a678fdb188f6f25dc18f33b927b88f4dddffd4`) from HF Model/GitHub Releases or audited cache, official PDEBench 3 HDF5 MD5 PASS, original checkpoint test RMSE replay, runtime dual-repo sync when write tokens available, local ZIP SHA download, no Google Drive.

All **16 executable notebook cells** executed in local CPU smoke (old original-historical preflights included), 1D Natural / 2D Rotated real R6 pretrained controller weights, full 12-step 5 Jacobian modes, two synthetic cases 120 steps+120 comparison rows, full block reconstruction numerical checks, archive ZIP CRC/SHA PASS. **Official B2-R15 has NOT been executed.** All full CIDM-v2 retraining, BPTT(3/12) pilots, unseen outer-test and AI Weather Quest RPSS gates remain CLOSED.
