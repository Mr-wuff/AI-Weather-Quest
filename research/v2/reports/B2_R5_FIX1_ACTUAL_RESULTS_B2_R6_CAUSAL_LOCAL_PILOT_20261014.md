# CIDM-v2 B2-R5 FIX1 official result and B2-R6/P1 causal coefficient qualification (2026-10-14)

## Actual user-supplied result; integrity verified
- `CIDM_v2_B2_R5_P0_FIX1_SPATIAL_OPERATOR_GEOMETRY_REPORTS.zip` SHA256 `0586af4b7f943b546637d828ae24cf38471b802f139138755132ce1e2784a196` equals uploaded sidecar, ZIP CRC PASS.
- Status `FROZEN_SPATIAL_SPAN_AUDIT_COMPLETED_NO_TRAINING`, official PDEBench three source MD5 checks PASS, 9/9 original parent model test-RMSE replays, 45 task-seed-lead groups, 360 samples.
- User's repository sync log reports HF source/ZIP/checksum and GitHub source/release/status SUCCESS.
- The B2-R5 posterior oracle is **future-truth-conditioned, NOT deployable forecast skill**.

## True lead12 1% fixed-budget pooled results (24 samples per family, 3 archived seeds)
| PDE | natural spatial squared gradient capture | equal-rank random capture | pooled spatial MSE ratio | pooled random spatial ratio | pooled global3 ratio |
|---|---:|---:|---:|---:|---:|
| Burgers-1D | .633423 | .032042 | .960996 | .983578 | .986739 |
| Reaction-Diffusion-2D | .103348 | .031234 | .981548 | .988762 | .996025 |
| Shallow-Water-2D | .156233 | .031678 | .987057 | .992708 | .998746 |

These pooled values were independently recomputed from 360 raw sample rows, using weighted SSE ratios, **not** a macro-average of per-sample ratios. The natural spatial-3 oracle is better than its same-effective-rank random oracle on each of the nine seed×task lead-12 paired groups at 1% budget. This supports meaningful operator basis geometry, not a guarantee a causal learner can infer correct coefficients.

## Preregistered B2-R6/P1 (new ChatGPT Notebook, no official execution yet)
- Delivered `CIDM_v2_B2_R6_P1_Causal_LocalCoefficient_Qualification_Colab_STANDALONE.ipynb`, `b2_r6_causal_spatial_controller.py`, detailed Chinese review, B2-R5 evidence CSV, and full ZIP bundle.
- Notebook embeds original exact 8 upstream files by verified hash and whole actual B2-R5 FIX1 evidence ZIP by SHA256. Original B0-R1 parent 18-run ZIP fixed SHA256 `69552b29e169ec22ee85d9c1b1a678fdb188f6f25dc18f33b927b88f4dddffd4`; HF Model/GitHub Releases first, one-time upload only if absent. Source MD5 and data/split/test replay fail closed.
- Q0 **frozen source/Oracle geometry verification only** before any pilot training. Q1 **only lightweight paired training of two identical small coefficient heads** in seed17 Burgers, RD, SWE; all original CIDM-v2 parameters frozen, no long training. Per-location Flux/Local/Skew direction coefficients predicted from **issued history + autonomous current prediction state**, no future truth during rollout. An orthogonally channel-rotated operator basis control matches rank, norms and local Gram at equal state. Both heads identical init, train windows/steps and optimizer; bounded 1% latent RMS intervention. Initial output parity with original v2 exactly zero.
- Train from original TRAIN only, step selection VAL 0-7 and separate new-probe audit VAL 8-15, default 60 optimizer steps each head/task, evaluate and store full optimizer/RNG snapshots each 20 updates. Truncated BPTT(1) to bound pilot memory, but full 12-step autoregressive rollout for scoring. Actual 1/3 early and 9/12 late lead physical MSE. Pilot success requires Burgers/SWE late >=3% gain over frozen baseline and >=1% against rotated control, no early >2% regress; RD late regress <=2%.
- Historical B0 model checkpoint was val-selected, therefore two halves of VAL are **not an untouched outer holdout**. Even if pass, mandatory fresh outer holdout and >=3 new seeds, full-v2 retraining still blocked.
- Program test: whole Notebook all code cells executed on CPU in smoke mode, 1D+2D pair probe 2 steps each, original parity max=0, exact eight source SHA verification, real B2-R5 evidence SHA verification, end-to-end ZIP + SHA successful. **Official Q1 has NOT run**.
- No Google Drive; after runtime execute, notebook attempts HF Model results/source, GitHub release/source/status, plus local browser ZIP/SHA/status download.

Current B2-R1 flux-cap Q0 and B2-R3 across-family latent-specific gate remain failed; no eight-PDE v1 superiority or AI Weather Quest operational RPSS qualification.
