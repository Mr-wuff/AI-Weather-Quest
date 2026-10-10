# CIDM-v2 A0 · Colab Standalone FIX1 Results (2026-10-10)

## Grounded results
- Actual GPU: Tesla T4, not L4; model 37,678 trainable parameters; five epochs × 60 optimizer updates (300 total).
- Preflight: original Flux cancellation test and activated state-sync backward/gradient test PASSED.
- ID synced MSE 0.000582538 vs persistence 0.018420312 (3.162% of baseline); RPS 0.06262066 vs uniform 0.20353192.
- OOD synced MSE 0.012293749 vs persistence 0.010310178 (119.24% of baseline; +19.24% worse); RPS 0.08613722 vs uniform 0.21208904.
- Zeroing prescribed source on the four validation seeds increased MSE from 0.000582538 to approximately 0.000647875 (+11.22%). This is evidence of use in these cases, not a held-out causal proof.
- Sync changed ID MSE from 0.000581548 to 0.000582538 (+0.17%), OOD MSE from 0.012309363 to 0.012293749 (-0.127%): no meaningful long-horizon stability claim.
- synthetic_gate: 3/4 conditions true; OOD > persistence made passed_all=false.
- Investment gate remains permanently BLOCKED at synthetic A0 stage, regardless of synthetic metrics. HF/GitHub runtime credential sync was skipped and checkpoint is not in uploaded report ZIP.

## Critical protocol observation
Training uses fixed simulated transport (vx=0.16, vy=0.04), OOD reverses/changes it (vx=-0.09, vy=0.13). Original model receives known future source trajectory but not explicit speed fields, creating a confound between state identifiability and learned transport generalization. ID results are also on the four seeds monitored during validation, not a fresh final ID holdout.

## Next experiment: CIDM-v2 A1 (single-candidate architecture/mechanism qualification)
Self-contained Colab Notebook in ChatGPT delivery: CIDM_v2_A1_Causal_CrossScale_BER_OOD_Qualification_Colab_STANDALONE.ipynb.
- One learned v2 candidate, NO CNN/Generic/FNO training.
- Issue-known controls supply velocity components and known synthetic external source; ID train samples cover positive/negative velocities, blind OOD extrapolates beyond training ranges.
- Conditional antisymmetric Flux, explicit paired fine/coarse SOC transfer with weighted-exchange cancellation, zero-port-neutral BER, bounded model-prediction-only self-synchronization.
- Test one frozen checkpoint under full/no-SOC/no-sync/no-Flux/no-BER and zero/permuted velocity/source, plus a longer 24-step OOD case.
- Keep qualification criteria stringent; validation selects checkpoint, independent ID and OOD seeds do not tune hyperparameters. A1 cannot authorize PRO6000 or imply ECMWF competition quality.
- Optional HF model checkpoints and GitHub JSON sync from NEW runtime-secret tokens only. Do not reuse previously disclosed credentials.

Original A0 user-provided file: CIDM_v2_A0_Colab_STANDALONE_FIX1_REPORTS.zip.