# CIDM-v2 B2-R2/P0 FIX4 audit and FIX5 direct-operator repair — 2026-10-13

## User-run FIX4 evidence
- Uploaded official report ZIP SHA256 `cb0e83f93b261ccd03829dc2eab55019c4497b09a5c2679795b45601f2f7b423`, sidecar MATCH; ZIP CRC PASS.
- Source `b2_r1_flux_control.py` SHA256 `8b4fd32e3f3675fd23764506267020dc61aba31c9ad54cfdf3ce6ace5d7e2c98` exact.
- Official B0-R1 full 18-run parent ZIP SHA256 `69552b29e169ec22ee85d9c1b1a678fdb188f6f25dc18f33b927b88f4dddffd4`; 9/9 original V2 checkpoint test RMSE replays pass and ControlledV2 output parity max=0.
- `observer_audit.json` has **300/300 MISSING_UNAMBIGUOUS_MODULES** observations. `report.json` has zero valid mechanism rows, status `INCOMPLETE_OR_INTEGRITY_FAIL_NO_TRAINING`. Do NOT treat source/results archival success as scientific pass.

## Verified cause
FIX4 observer searched `named_modules()` for `skew` and `decoder`. Original PortedRestoredCIDM defines `core.model.skew` as **nn.Parameter**, and the decoder is named `core.model.decode` (nn.Sequential), so neither requested hookable module was found. Genuine trained CIDM-v2 dynamics NOT disproved; the observer contract was incorrect.

## FIX5 next run (not executed on official data yet)
New fully self-contained Colab Notebook `CIDM_v2_B2_R2_P0_FIX5_ExactCore_Observable_Audit_Colab.ipynb` and direct probe `b2_r2_fix5_exact_dynamics.py` delivered to user in this conversation. It reuses embedded **exact** FIX2 four-module Python source (SHA256 locked), official DaRUS/PDEBench data MD5+trajectory IDs, pinned full 18-checkpoint parent. Directly probes exact effective updates:
- `F=coeff['flux']*mm.flux(z)`
- `L=coeff['local']*tanh(mm.local(z))`
- `S=0.035*einsum((mm.skew-mm.skew.T),z)`
- `z'=z+F+L+S`, `delta_x=coeff['out']*tanh(mm.decode(z'))`.
- Frozen local operator deletion counterfactuals in normalized observable increment units, lead 1,3,6,9,12; own-prediction-history re-encoding only; validation true increment is used **solely for posthoc diagnostic measurement**.
- Official test set serves only strict B0-R1 baseline RMSE replay. NO Q0 flux-cap retuning; NO optimizer/training; no Google Drive.
- Gate: all three families × 3 seeds ×5 leads = 45 aggregated groups, full baseline replay, exact wrapper parity ≤2e-6, additive residual ≤1e-4. Diagnostic completion NEVER unlocks long training by itself.
- Intermediate CSV/JSON per paired model, eventual ZIP+SHA256 browser download and HF model/GitHub Release sync (if runtime write tokens available); inaccurate or blocked evidence is labeled BLOCKED.

## Actual tests
The **entire FIX5 Notebook** executed with `CIDM_B2R2_MODE=smoke` on CPU, PASS with ten 1D/2D operator rows, original ControlledV2 parity max abs exactly 0.0, additive relative residual ≈4.0e-6, packaged ZIP generated and CRC verified. No claim of actual official nine-weight FIX5 results.

## Remaining decision
Wait for the real FIX5 nine-checkpoint val-lead observable audit. Only then choose a small-budget B2-R3 mechanism-necessity/decoder-compatibility training prequalification. Keep previous B2-R1 Q0 failure intact, CIDM-v2 has NOT passed all-PDE v1 superiority gate, and no AI Weather Quest RPSS qualification yet.
