# CIDM-v2 B2-R2/P0 FIX5 user-run failure and FIX6 colocated-source repair (2026-10-14)

## Exact uploaded evidence (not a mechanism result)
- User-uploaded `CIDM_v2_B2_R2_P0_FIX5_EXACT_OBSERVABLE_REPORTS.zip`: SHA256 `4b4adcebd5414c73538c293f20b44134844feade9eca66fecb486ad4c3d7e143`, matches `.sha256`. ZIP CRC PASS.
- 7 entries only; **no** `report.json`, `exact_replay.csv` or `lead_seed_summary.csv`.
- `run_status.json`: `BLOCKED_NO_TRAINING`, FileNotFoundError `/content/b2_r1_flux_control.py`, raised on observer line 149 before official replay.
- FIX5 observer uses `Path(__file__).with_name('b2_r1_flux_control.py')` to check driver SHA256. Embedded FIX2 sources had been verified under `/content/CIDM_v2_B2_R2/assets/FIX2_original_verified_source/`, **not** colocated with the observer in `/content/`.
- Source sync manifest reports HF/GitHub SUCCESS, but binary upload or remote source existence is different from being available at the required runtime path. No science inference or CIDM-v2 architecture failure from this run.

## FIX6 repair artifacts delivered in ChatGPT conversation
- `CIDM_v2_B2_R2_P0_FIX6_SourceCollocation_OfficialAudit_Colab.ipynb`: retained exact SHA256-pinned source bytes and all official B0-R1 strict replay and B2-R2 frozen diagnostic gates. Inserted a preflight cell to determine the observer module's *actual Python import path*, copy the four original SHA256-verified FIX2 modules to the observer's own directory, and verify loaded `benchmark` and `b2_r1_flux_control` hashes.
- Separate FIX6 work root prevents past blocked results from polluting the official run.
- `CIDM_v2_B2_R2_P0_FIX6_完整修复实验包.zip`: Notebook, preflight source, original observer source, original four FIX2 sources, Chinese report.
- No Google Drive, no test-set threshold fitting, no retraining, no bypass of parent hash `69552b29e169ec22ee85d9c1b1a678fdb188f6f25dc18f33b927b88f4dddffd4`.

## Verified program-only tests
- Entire FIX6 Notebook 10 code cells executed on CPU in `CIDM_B2R2_MODE=smoke`; output `NOTEBOOK_SMOKE_ALL_CODE_CELLS_PASS`.
- The earlier broken source-file location now verifies and co-location succeeds with all four upstream hashes.
- 1D and 2D original architecture forward-parity max absolute difference `0.0`; 10 smoke lead records; max additive reconstruction residual `4.010186066805963e-06`.
- Packaging generated its ZIP and sidecar successfully, CRC pass. **Official 9-checkpoint replay and 45 task-seed-lead frozen diagnostic groups remain unexecuted**.

## Scientific next step
Run FIX6 on official validated PDEBench assets. Only after success analyze Burgers/RD/SWE Flux, Local, Skew response geometry and model-only re-encoding to design the minimal B2-R3 pilot. B2-R1 prior Flux-Cap Q0 failure remains intact.