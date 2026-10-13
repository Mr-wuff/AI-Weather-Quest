# CIDM-v2 B2-R2 FIX4: exact original FIX2 source recovered (2026-10-13)

The user provided the original `CIDM_v2_B2_R1_FIX2_ExactReplay_AutoRepoSync_Colab_STANDALONE(3).ipynb` in this conversation. Its `SOURCE` assignments contain **all four actual upstream Python modules**, recovered byte for byte through static AST extraction. Original Notebook SHA256 assertions were independently checked:

- `cidm_complete_frozen.py`: `f9b1cf4ff7630f5540bb3adde20e4c4b81fff3218d9325ea3105bf28b1d97665`
- `benchmark.py`: `3f1ba062f14f3281859ef608f3ecc6ce394a231be3e9c03aebadfc874ae2d0ec`
- `b2_r1_flux_control.py`: `8b4fd32e3f3675fd23764506267020dc61aba31c9ad54cfdf3ce6ace5d7e2c98`
- `artifact_store.py`: `080edf39778531ccbf7217a87c2072c78d6d2aff6f4befbc57a11f095fc31129`

New self-contained B2-R2/P0 FIX4 Notebook embeds the exact modules in compressed form and checks their SHA256 **before any model import**. No user source upload or reconstruction is necessary. It still requires the original 18-run B0-R1 parent model ZIP (SHA256 `69552b29e169ec22ee85d9c1b1a678fdb188f6f25dc18f33b927b88f4dddffd4`) from HF model repo or GitHub Release; otherwise it explicitly requests that **parent ZIP only**. The uploaded compact `CIDM_v2_B2_R1_FLUX_CONTROL_QUALIFICATION_REPORTS(1).zip` contains only CSV/JSON report data, no `.pt` weights.

Testing: CPU *full notebook program-only smoke* executed and passed, including exact source hash verification, code imports and observer algebra. Separately evaluated original vs B2-R1 FIX2 ControlledV2 baseline on 1D and 2D synthetic inputs, prediction max absolute difference **zero** in both cases. Official nine-checkpoint diagnostic **has not been run**.

Notebook runtime attempts publishing original four source modules to HF model `wuff-mann/AI_Weather_Quest` and GitHub `Mr-wuff/AI-Weather-Quest` when write credentials permit; this report does not claim that binary source was already uploaded by ChatGPT. No Google Drive mount or scan. No training / science gate altered.
