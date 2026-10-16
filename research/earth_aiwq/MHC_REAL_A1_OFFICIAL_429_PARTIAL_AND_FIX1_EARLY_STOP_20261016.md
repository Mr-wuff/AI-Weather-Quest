# MHC-Real A1: actual blocked user results and early-stop FIX1 preregistration

2026-10-16; user ZIP SHA256 989de95a8bba3fdacc8de007a68aae4945f3198e310f6018d7cba8235cf7de4c, matching supplied sidecar; CRC PASS.
Actual STATUS: BLOCKED_NO_TRAINING due to HF Model HTTP429 128 commit creations/hour, not a numerical failure. Seed17 completed stateless, MHC and MHC-noaux, each 64 updates; seed41 stateless reached step24, with its remote checkpoint requiring actual SHA verification.
Observed seed17 W3 and W4 standardized validation MSE: stateless 0.11103641552229722 / 0.10791260438660781; MHC 0.11103636150558789 / 0.10791248766084512; MHC-noaux 0.1110341747601827 / 0.10790904238820076.
Seed17 MHC improvement vs stateless ≈0.00004865% W3 and ≈0.000108% W4. Original Gate requires >=0.5% for BOTH lead windows and BOTH seeds; logically FAILED regardless of seed41. No 2017 held-out audit.
Original A1 AdamW optimizes all ~19,954 parameters (including ~7,504 head weights), NOT frozen-head-only.

New standalone A1 FIX1 Notebook CIDM_MHC_REAL_A1_FIX1_EarlyStop_AtomicResume_Colab_STANDALONE.ipynb and source mhc_real_a1_fix1_atomic_resume.py sha256 9574866fc485c0868e3ac7b00c6d64db959f6353eaa3a569c60fd2b8ade84d9e.
DEFAULT gate_stop uses 3 SHA-verified real seed17 step64 model checkpoints and exact original 2016 12-issue replay, followed by 48-issue 2016 equal-actual-action-RMS frozen MHC-memory vs reset intervention. Zero new optimizer updates; 2017 remains sealed.
OPTIONAL resume_full restores legacy progress using immutable user uploaded 429 report and remote SHA-verified checkpoints (seed41 step24 if available else step16); fixes HF 429 by bundling each checkpoint and progress index in ONE atomic commit every 16 updates; local optimizer checkpoint every 8. Supports private HF Dataset caching with verified ECMWF netCDF and no Google Drive.
Full ZIP SHA256 6307c6060e733a3d27efb413be74653543d245bace4bbf726c398ae74462f5a2. Four notebook code cells CPU smoke PASS, original MHC three-arm bit-exact forecast parity, five-step equal correction RMS test PASS, provenance migration and one-commit two-file mock PASS.
New official FIX1 frozen scientific experiment not yet run; historical scores are continuous MSE, NOT AI Weather Quest RPSS or leaderboard results. Continue real-weather MHC state sufficiency and scientifically meaningful predictor checks before large-scale training.