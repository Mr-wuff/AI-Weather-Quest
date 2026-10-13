# CIDM-v2 B2-R2/P0 FIX3 source-input repair (2026-10-13)

## Actual user error
Previous B2-R2 notebook prompted for original B2-R1 FIX2 execution source `b2_r1_flux_control.py`. User uploaded `CIDM_v2_B2_R1_FLUX_CONTROL_QUALIFICATION_REPORTS.zip` (5.67 KB), which contains compact evaluation results rather than the FIX2 Python driver. The old loader wrongly treated any .zip/.ipynb/.py as source and called HF upload before checking source contents. The upload returned SUCCESS but the driver remained absent, causing `BLOCKED_MISSING_VERIFIED_FIX2_SOURCE`.

## FIX3 changes delivered as standalone ChatGPT artifacts
- Reject results-only ZIP (no .py/.ipynb inside) *before* any source publishing.
- Retry interactive Colab upload up to three times, with exact needed FIX2 source filenames.
- Statically extract notebooks/full source bundles (without executing unknown cells), check required `ControlledV2`, `assess`, `run` API, and publish only validated driver source.
- Retain checksum-pinned 18-run B0 parent and exact official replay; never reconstruct a different model or weaken the scientific gate.
- Keep HF **model** repository and GitHub dual-archive workflow; NO Google Drive.
- Report blocked precheck as an exported diagnostic ZIP, not a misleading successful experiment.

## Tested
Whole FIX3 notebook CPU PROGRAM_ONLY_SMOKE PASS, source notebook static extraction PASS, simulated upload of incorrect result ZIP followed by legitimate driver PASS. These are **program tests only**, not official PDEBench B2-R2 results.

## Remaining prerequisite
The actual FIX2 source is not in the compact 5.67 KB results ZIP, and the public GitHub tree contains only the result JSON. If original executable source is not available in the HF Model repo or GitHub Releases, the user must once provide the earlier FIX2 notebook/full source bundle/driver. The already archived parent B0 ZIP is not a substitute for FIX2 source. The source will then be published separately if runtime write credentials are configured.

## Scientific state
B2-R1 FIX2 remains Q0 FAIL / NO TRAINING. B2-R2 official nine-checkpoint diagnostic is not yet run.
