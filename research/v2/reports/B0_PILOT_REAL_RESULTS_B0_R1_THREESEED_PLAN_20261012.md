# CIDM-v2 B0 — Actual user-uploaded PDEBench pilot and three-seed convergence gate (2026-10-12)

**CIDM-v2 remains the active development target; no CIDM-v3 is planned.**

## What actually ran

Source: user-uploaded completed `CIDM_v2_B0_PDE_FAIR_REPORTS.zip` and executed Notebook, including 6 trained checkpoints. All 6 runs completed. 3 official DaRUS/PDEBench datasets, 1 training seed (17), 120 optimizer updates each, about 500k parameters per model, history4, train6, evaluation6/12. `official_data=true` reported after MD5 validation in run script, though raw HDF5 not included in result package; independent raw file rehash not possible. Overall Gate: `DIAGNOSTIC_ONLY`, `all_v2_better_v1=false`.

| PDE | 6-step v1 RMSE | 6-step v2 RMSE | 12-step v1 RMSE | 12-step v2 RMSE |
|---|---:|---:|---:|---:|
| Burgers-1D | .101825 | .122146 | .182356 | .167513 |
| Reaction-Diffusion-2D | .013814 | .007029 | .017685 | .014427 |
| Shallow-Water-2D | .020802 | .023905 | .029418 | .035776 |

v2 gains in 6/12 steps: Burgers -19.96% / +8.14%, Reaction-Diffusion +49.12% / +18.42%, Shallow-Water -14.92% / -21.61%. 3/6 wins; no universal v1 superiority.

## Mechanism and fairness insights

- Disabling Flux at inference on the v2 checkpoint worsens 12-step RMSE by 10.07% (Burgers), 33.25% (Reaction-Diffusion), and 6.82% (Shallow-Water); dependency is real under this implementation, though masking is not retraining ablation.
- v2 `sync_logit` exists but sync **mode='off'** throughout fitting/eval. These runs cannot establish any drift or Sync repair.
- Inference 12-step on reported T4: Burgers v1 91.09ms vs v2 16.48ms (much faster v2); RD2D v1 109.96 vs v2 125.94; SWE2D v1 114.84 vs v2 124.29 (slower v2 on 2D).
- V2 is an explicitly ported R5/R1 architecture from 2-channel synthetic to PDEBench shapes. 500k matching does not equal matched FLOPs.
- One seed and 120 updates do not establish convergence. Validation picked checkpoints at 6 steps while long horizon 12 was also evaluated.

## Released B0-R1 three-seed convergence qualification

Self-contained Colab notebook and ZIP are delivered via ChatGPT attachments. Uses SHA256-verbatim archived v1 source and documented v2 port; 3 PDE families x seeds 17,41,73 x two architectures = 18 training runs, 600 updates/run, 60-update diagnostics, same official dataset trajectory splits and normalization, **12-step validation selection** for both, 1–12-step test RMSE and frozen Flux dependence, checkpoint/resume and ZIP. It reuses the existing B0 `official_data` root if available; new profile `pilot_r1` prevents overwriting previous results.

Full Notebook **smoke** ran end to end on CPU; official B0-R1 training has NOT been executed. The three-PDE experiment remains diagnostic, not an eight-PDE leaderboard qualifier. Next only if convergence and paired independent seed evidence justify more investment. No AI Weather Quest RPSS or RTX PRO 6000 authorization.