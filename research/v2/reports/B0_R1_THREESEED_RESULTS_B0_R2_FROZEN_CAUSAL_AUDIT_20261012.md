# CIDM-v2 B0-R1 actual three-seed PDEBench pilot: 2/6 horizon groups wins; R2 frozen diagnostic

Date: 2026-10-12. User-uploaded completed B0-R1 report ZIP checked for integrity and 18 trained-checkpoint records, notebook source SHA verified.

## True paired experiment contract
- Official DaRUS/PDEBench MD5 checks declared in script and run manifest (source HDF5 bytes not included in ZIP for independent rehash).
- Three PDE systems: Burgers-1D, Reaction-Diffusion-2D and Shallow-Water-2D. Three seeds 17,41,73, two architectures, 600 updates/run (18 runs), similar ~500k params, train horizon6, **selection validation horizon12**; all 18 completed.
- Gate `DIAGNOSTIC_ONLY`, `all_v2_better_v1=false`. 3-seed results do not prove convergence or generalization.

## RMSE (average of three seeds; improvement computed as 1 - mean(v2)/mean(v1))

| Task | Horizon | Original CIDM-Complete | v2-R5 port | v2 gain | All three paired seed signs |
|---|---:|---:|---:|---:|---|
| Burgers 1D | 6 | 0.056367 | 0.091537 | -62.39% | v1 wins |
| Burgers 1D | 12 | 0.094384 | 0.130207 | -37.95% | v1 wins |
| Reaction-Diffusion 2D | 6 | 0.007144 | 0.003144 | +55.99% | v2 wins |
| Reaction-Diffusion 2D | 12 | 0.010051 | 0.005508 | +45.20% | v2 wins |
| Shallow-Water 2D | 6 | 0.004362 | 0.021037 | -382.31% | v1 wins |
| Shallow-Water 2D | 12 | 0.008519 | 0.030632 | -259.59% | v1 wins |

## Key mechanism hypotheses; not yet measured causally
- Burgers seed17 one-step v2 RMSE ~0.02956 vs v1 ~0.03841, but 6th-step v2 ~0.12577 vs v1 ~0.05806: degradation during rollout is central.
- Shallow-Water v2 is much worse already at first step, then the gap grows; could reflect representation, field increments, boundary behavior, or update structure.
- V1 and v2 generally have best checkpoint at/near final 600 steps, so further convergence is plausible, **but there is no evidence that longer training alone closes Shallow-Water gap**.
- Burgers and SWE are one-channel while Reaction-Diffusion is two-channel; channel count is confounded with task physics and grid dimension, hence no causal attribution yet.
- Same parameters are not same FLOPs. Current v2 sync remains OFF, so B0-R1 is not a valid Sync experiment.

## B0-R2 frozen diagnostic release
- Standalone Colab, full source embedded SHA256 checked; load exact user B0-R1 result ZIP containing all 18 checkpoints; official PDEBench data retrieved from previous cache (or downloaded and MD5 checked).
- Recompute 12-step test RMSE on identical cases and FAIL CLOSED on >0.1%/1e-5 relative/absolute replay discrepancy.
- Per-step curve, low/high spatial residual, boundary/interior error, autoregressive increment magnitude; frozen-v2 no-Flux / no-Local / no-Skew / frozen latent, per-seed complete reporting.
- NO NEW MODEL TRAINING, NO VALIDATION SELECTED TEST TUNING, NOT CAPACITY-MATCHED CAUSAL ABLATION.
- Whole single-file Notebook CPU synthetic 2-case smoke executed successfully end to end; real PDEBench 18-checkpoint frozen audit not yet run.
- Artifacts supplied in ChatGPT: `CIDM_v2_B0_R2_Frozen_Causal_Attribution_Colab_STANDALONE.ipynb` and `CIDM_v2_B0_R2_Frozen_Causal_Attribution_FULL.zip`.

CIDM-v2 remains active development target. PDE-wide v1 superiority gate and AI Weather Quest official RPSS gate NOT met; PRO 6000 spending still blocked.