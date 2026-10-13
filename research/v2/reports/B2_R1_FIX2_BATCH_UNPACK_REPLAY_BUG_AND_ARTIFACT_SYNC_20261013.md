# CIDM-v2 B2-R1 FIX2: baseline replay bug and repository/archive fix (2026-10-13)

## Confirmed ROOT CAUSE of the user's RuntimeError
Original B2-R1 `assess()` executed `p,tt=net(h,steps,return_trace=details)` even when details=False and `ControlledV2.forward` returned a single predictions Tensor of shape [B,12,C,...]. Because evaluation batch B=2, Python unpacked the first TWO examples into p and tt, then compared p to the full target batch by broadcasting, yielding a spurious Burgers seed17 12-step RMSE 0.454414 instead of archived B0-R1 RMSE 0.13152127036229722. THIS WAS NOT A MODEL ACCURACY REGRESSION.

Fix: `result=net(h,steps,return_trace=details); p,tt=result if details else (result,[])`. Restore original benchmark batch_1d=8, batch_2d=2. Validate both original `benchmark.assess` and controlled baseline metric plus exact per-element predictions BEFORE Q0 candidate screening. Verify official MD5, trajectory IDs, heldout windows, source SHA256; replay fail closed on >0.1%/1e-5 or prediction difference >2e-6. Never lower science Gate to accept faulty model replay.

## Tests performed
- Original user's 18-run B0-R1 archive verified, old 6-run Pilot correctly rejected.
- Loaded REAL archived Burgers and Shallow-Water v2 seed17 best checkpoints; tested 1D and 2D synthetic data at batch 1,2,8: wrapper baseline RMSE matches canonical `benchmark.assess` to ~1e-8, raw-to-wrapped prediction max_abs=0.
- Q1 new checkpoint includes optimizer, best state, step and train history; interruption/restart logic independently tested for 2 small updates; deterministic samples allow resume.
- Whole notebook CPU SMOKE executed successfully, status PROGRAM_ONLY_SMOKE. **The repaired official full PDEBench Q0/Q1 has NOT yet been executed.**

## Artifacts delivered in ChatGPT
- CIDM_v2_B2_R1_FIX2_ExactReplay_AutoRepoSync_Colab_STANDALONE.ipynb
- CIDM_v2_B2_R1_FIX2_ExactReplay_AutoArchive_FULL.zip
- FIX2_DIAGNOSIS_CN.md, README_CN.md, SHA256SUMS.txt

## Source and result synchronization contract
- Auto-resolve verified 18-run input ZIP from local/Google Drive, actual GitHub Release assets, or listing actual Hugging Face repository files; fallback to one-time Colab upload only when sources are missing. Do not assume a GitHub textual report contains checkpoint ZIP.
- Always write complete report ZIP, SHA256 and sidecar manifest to runtime; persist snapshots every evaluation node (Q1 every ~60 updates), copy to Drive when mounted, trigger Colab browser download.
- GitHub repo `Mr-wuff/AI-Weather-Quest`: report JSON/small progress with GitHub Contents API, result ZIP as GitHub Release. Hugging Face: result ZIP/checkpoints to an existing, explicitly configured writable Dataset repo. Required Colab Secrets `GH_TOKEN` and `HF_TOKEN` have appropriate write access. If missing, upload is SKIPPED with truthful status and local/Drive copies remain.
- The assistant checked the connected GitHub prior research report; Hugging Face connector identity is `wuff-mann` with READ scopes (not write). No remote upload of new binary archive is claimed in this turn.

CIDM-v2 internal closure and eight-PDE v1 superiority are NOT yet demonstrated. Keep the original frozen-validation preflight and trained Q1 Gate; AI Weather Quest real-data training still gated.