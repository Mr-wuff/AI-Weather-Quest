# CIDM-MHC A0-FIX1 actual official result and A1/P0 frozen memory necessity preregistration

Date 2026-10-15. Original user-uploaded `CIDM_v2_MHC_A0_FIX1_Q0_Q1_CAUSAL_PILOT_REPORTS.zip` SHA256 **afde1a7f7face201108aaeb2b472fbc5a4f7cbd2c84945808a507ef387c1ff38**, uploaded sidecar exact match, ZIP CRC PASS, internal 27-entry SHA256 manifest PASS. HF Model/GitHub sync user log shows SUCCESS but report is correctly **`MHC_FIX1_Q0_OR_RESOURCE_GATE_FAIL_NO_FORMAL_TRAINING`**.

All three official original seed17 parent RMSE replays PASS and all Q0 zero-correction 12-step original forward max diff=0. Burgers Q0 remains FAILED for all 3 heads: gradients finite nonzero (~1.3e-5 to 1.6e-5) but initial AdamW increase and all preregistered negative gradient radii fail measurable relative 1e-5 loss reduction; **zero training** Burgers heads. RD and SWE Q0 PASS, three arms each complete exactly 24 optimizer steps with original CIDM-v2 backbone frozen. All Q0 witness initial weights restored before Q1.

Original preused VAL windows16–31, pooled **physical lead9+12 MSE percentage improvements**:
| Task | stateless closure | persistent MHC + aux | persistent MHC no aux |
|---|---:|---:|---:|
| Reaction–Diffusion-2D | +0.687255% | +0.684324% | +0.677818% |
| Shallow-Water-2D | +0.235863% | +0.270049% | +0.277390% |

RD 16/16 validation windows improve for every arm. SWE stateless 14/16 windows improve; hierarchical + aux and hierarchical no-aux only 10/16 each (pooled late window). No >2% pooled-late regression windows. MHC v stateless advantage on SWE is only **0.03419 percentage points**, RD MHC is **0.00293pp worse**. Auxiliary loss benefit not consistent. 1-seed,24-step, preused validation, not externally validated. **MHC-specific efficacy Gate FAIL.**

New **A1/P0** source `cidm_mhc_a1_frozen_memory_necessity.py` SHA256 **bce5f729c2c49864cb6389bc210be5722393798e811ebf44a78ebf12b7a93c36**, fully standalone Colab Notebook `CIDM_v2_MHC_A1_P0_FrozenMemoryNecessity_Colab_STANDALONE.ipynb` and engineering ZIP SHA256 **5e6079019d6a975cd2f659f4904dc6223acd3e2b142ecf6b47887a8bd044844e**. All original upstream scripts/evidence embedded and locked, exact original user A0-FIX1 result plus six real 24-step heads embedded, original B0 parent hash-pinned SHA256 `69552b29e169ec22ee85d9c1b1a678fdb188f6f25dc18f33b927b88f4dddffd4`. HF Model/GH Release parent priority, PDEBench official HDF5 MD5 and original RMSE/validation-per-window replay required, no Drive.

A1 **strictly zero training**, two tasks with real official trained heads only (RD/SWE, explicitly exclude Burgers), factual MHC persistent state vs memory reset to `lowpass(z)` each step, vs half-persistent memory; raw intervention and **per-sample per-step realized RMS-matched** counterfactual, fixed same original controller head and exact canonical CIDM update. Use preused VAL windows16–31 and 12-step causal rollouts; no future truth in actions; full original head replay must match archived all 320 per-task arm×window×lead MSE scores within 0.5% per row, full original MHC forecast <=2e-6 max absolute difference. Frozen diagnostic Gate only, no trained step or model replacement. If change in memory has no material equal-exposure benefit, do not proceed to larger Markov hierarchy training.

Full notebook all executable cells CPU smoke passed on actual original source+real prior archives, synthetic 1D and 2D, full MHC forecast parity, frozen memory and equal RMS interventions and ZIP SHA/CRC output. This **does not constitute official A1 science**. Full v2 retraining, >=3 fresh seed fair v1 comparison, new outer holdout and AI Weather Quest RPSS qualification remain closed.
