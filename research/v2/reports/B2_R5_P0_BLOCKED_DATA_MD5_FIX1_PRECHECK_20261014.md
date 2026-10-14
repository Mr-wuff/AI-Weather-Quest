# CIDM-v2 B2-R5/P0 actual blocked result; FIX1 integrity-safe rerun (2026-10-14)

## Actual user-run B2-R5 result
- Uploaded ZIP `CIDM_v2_B2_R5_P0_SPATIAL_OPERATOR_GEOMETRY_REPORTS.zip` SHA256 **`0aef20b94d84a861e6fcad14b782ca74df51f00f15f96684cce0940d4c754575`** matches uploaded sidecar; CRC PASS.
- ZIP contains exactly 4 entries: `B2_R4_UPSTREAM_FROZEN_REPORT.json`, `FILE_SHA256_MANIFEST.json`, `parent_discovery.json`, `run_status.json`. It includes **NO B2-R5 `report.json`, exact replay or Oracle sample CSV**.
- Parent discovery `local PASS`. 8 original Python sources restored and program smoke PASS.
- `run_status.status=BLOCKED_OFFICIAL_DIAGNOSTIC_NO_TRAINING`; traceback: `benchmark.fetch_official()` Burgers-1D official data hash mismatch **actual `600dfb4ce65e655401c485037b307a38` vs pinned `e6d9a4f62baf9a29121a816b919e2770`**. No model replay or B2-R5 official operator geometry started. It is a correct fail-closed data integrity block, **not an architectural failure**.
- Runtime sync JSON: HF zip/checksum/source SUCCESS, GitHub source + Release SUCCESS, GitHub status contents UPDATE `HTTP 409 Conflict`. The linked successful release may contain a blocked-results ZIP, not qualified science.
- The original `benchmark.py` resumes `.part` on HTTP 206 without checking `Content-Range` offsets and does not automatically recover from MD5 mismatch. The user-run report alone does **not** prove this was the causal reason for the corrupted bytes; could alternatively involve an incomplete/altered upstream download.
- DaRUS V8 page lists pinned Burgers MD5 `e6d9a4f62baf9a29121a816b919e2770`; older dataset versions may show different MD5 and must not be silently substituted.

## Next FIX1 delivered artifacts
Self-contained `CIDM_v2_B2_R5_P0_FIX1_VerifiedDataResume_Colab_STANDALONE.ipynb` and `CIDM_v2_B2_R5_P0_FIX1_Complete_Delivery.zip` created and locally validated. **Exact original eight upstream Python modules and SHA256 pinning remain unchanged**.
- Verified-data discovery from user's configured `PDEBENCH_DATA_ROOT` or specific old Colab B2-R4/B2-R3/B2-R2/B2-R5 directories. No Drive.
- Strict HTTP range-start/end, Content-Length, final total length checks, plus locked MD5 and HDF5 magic validation. If a completed .part fails MD5, delete bad part and allow one clean full retry; failure remains blocked. Data integrity JSON emitted.
- New isolated workspace, safe GitHub upsert with fresh remote SHA on HTTP 409/422, unique status/release names. HF target remains `wuff-mann/AI_Weather_Quest` **Model** repo.
- No optimizer or training; test only original nine-model test RMSE replay, then frozen truth-informed validation Oracle across 3 PDE x3 seeds x5 leads. Future truth used **only for retrospective Oracle diagnostics**.

## Verified FIX1 program tests
- Entire FIX1 notebook `CIDM_B2R5_MODE=smoke` CPU executed, PASS 20 sample records and 10 lead groups.
- Fake HTTP tests: clean file, correct 206 resume, corrupted prefix followed by clean redownload, incorrect range-offset rejected. PASS.
- ZIP/SHA256 built and CRC verified. **No official B2-R5 FIX1 nine-model data replay or 45-group diagnostics executed yet**.

## Science gates
B2-R1 Flux-Cap Q0 and B2-R3 uniform latent readout pilot failed. The valid B2-R4 full-gradient oracle is not issue-time deployable. Until B2-R5 genuine operator-vs-matched-random geometry results are available, no new dynamics mechanism qualification, no full CIDM-v2 formal training, no AI Weather Quest RPSS eligibility.
