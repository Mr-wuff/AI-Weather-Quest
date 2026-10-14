# CIDM-v2 B2-R6/P1 FIX1 — Remote asset function NameError repair

Date: 2026-10-14.

## Error reported from user Colab execution
`NameError: name 'discover_assets' is not defined` in B2-R6 Notebook section 5 when searching HF Model / GitHub Release for the exact original 18-run parent B0-R1 ZIP. **No official pilot training was started.**

## Root cause
The notebook restored and verified the original SHA256-pinned `b2_r2_repo_and_driver.py` (original SHA256 `e196bc58b0f90faee5fa01805365e62c98651d8af20021f5cfdc1cd3714049f0`) but never imported its helper functions into the notebook namespace. Section 5 referenced five undefined symbols: `discover_assets`, `fetch_asset`, `verify_parent`, `publish_hf`, `publish_gh_release`.

## Fix (no change to scientific contract)
- After restoring exact eight prior Python sources, explicitly import all five helper functions from `b2_r2_repo_and_driver`.
- Validate actual imported module path, exact source SHA256, callability and signature of all five functions; write `asset_api_preflight.json`.
- Also repeat explicit import immediately at the section 5 entry point. Never fabricate weights or bypass the SHA-locked parent verification.
- Parent expected SHA256 remains `69552b29e169ec22ee85d9c1b1a678fdb188f6f25dc18f33b927b88f4dddffd4`. HF Model `wuff-mann/AI_Weather_Quest` and GitHub Releases remain the remote-first sources. No Google Drive.
- No change to original model, six/12-step definitions, three PDEs, training data, seed17, causal coefficient heads, paired equal-parameter controls, 60-step default pilot, stage gates or output-sync routines.

## Execution validation
The fully modified Notebook **all executable cells** ran on CPU with `CIDM_B2R6_MODE=smoke`. 1D and 2D original forward parity had max difference 0, paired probe smoke training passed, data-integrity patch and result ZIP/SHA256 packaging passed. An isolated official-mode discovery test with a mocked empty remote asset list passed without NameError and reported the missing verified parent; no real parent/model was claimed.

**Official B2-R6 3-PDE pilot not yet executed.**

## User delivery
`CIDM_v2_B2_R6_P1_FIX1_AssetDiscovery_Colab_STANDALONE.ipynb`
`CIDM_v2_B2_R6_P1_FIX1_Complete_Delivery.zip`
Additional `TEST_SUMMARY.json` and `SHA256SUMS.txt` in complete ZIP.

FIX1 is a harness-repair release, not an improved CIDM-v2 model nor AI Weather Quest qualification.
