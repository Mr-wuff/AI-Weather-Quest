# CIDM-v2 B1-R4/P0 external-event geometry results and B2/P0 internal closure audit

Date 2026-10-13. Continue CIDM-v2 research only; no v3 planned.

## User-uploaded B1-R4/P0 factual results
- ZIP status FROZEN_DIAGNOSTIC_ONLY_NO_ARCHITECTURE_QUALIFICATION; three seeds x3 frozen BER variants x5 new paired scenarios =45 summary rows. Synthetic BER model ~19k parameters; **not** same as the ~500k PDEBench v2 backbone.
- State-port CF skill vs same-port-readout on seen early pulse H12: +.72862 vs +.55066; on seen H24 +.49023 vs +.43236.
- Reused-family early pulse H12 state +.04057 vs readout +.12565; early pulse H24 state -1.03710 vs readout -.61900; state response cosine -.06007 despite true/pred power ratio .92154. Reused-family middle pulse H24 state -.86149 vs readout -.21701, cosine +.15315.
- Teacher-forced oracle one-step positive results are confounded by providing true future observation history; oracle persistence is stronger in all scenarios and is NOT a deployable repair.
- The reused Coupled family is not pristine blind research and source coupling laws remain same. Signal normalized CF power is small. No attribution of defect solely to BER because readout branch also fails on the long stress.

## CIDM-v2 internal-closure scientific assessment
- PDE-ported R5 mainline equation: z_{k+1}=z_k + a_F*F(z_k)+ a_L*tanh(Local(z_k)) + .035*A_skew*z_k. x_{k+1}=x_k + a_out*tanh(Decoder(z_{k+1})). Sync=off, BER/SOC/old hidden reservoir absent.
- Mechanisms are useful but closed evolution is not proven. Flux's periodic latent sum cancelling is not conservation of decoded physical mass/momentum/energy. Skew explicit Euler has norm inflation h²||Az||² even when zᵀAz=0. Local may compensate/cancel Flux. Initial encoded history may not yield a sufficient Markov state. Autonomous latent and incremental output can drift functionally. Direct latent re-encode cosine is gauge-dependent.
- V1-Complete vs v2 fair B0-R1 three-seed 12-step RMSE: Burgers .094384 vs .130207, Reaction Diffusion .010051 vs .005508, Shallow-Water .008519 vs .030632; no across-PDE qualification.

## Released new B2/P0 trained-PDE backbone frozen operator audit
- Full standalone Colab Notebook `CIDM_v2_B2_P0_Internal_Closed_Dynamics_Audit_Colab_STANDALONE.ipynb`; complete package `CIDM_v2_B2_P0_Internal_Closure_Full.zip` in ChatGPT artifacts.
- Use user's previous B0-R1 official PDEBench 18-checkpoint result ZIP; reuse officially MD5 verified data/cache; fail closed on 12-step replay mismatch (>1e-5 or .1%). Three PDEs x3 seeds of v2 (9 model checks); no new training.
- Lead-resolved observed-increment direction/power vs truth; Flux latent spatial telescoping; isolated Skew Euler norm growth; Flux/Local/Skew vector field RMS and pairwise alignment; bounded latent perturbation growth; **output-level functional** discrepancy using self-predicted-history re-encoding, no oracle future observations as inputs.
- Scientific output descriptive only, first sixteen test windows for operator diagnostics and full test windows for strict RMSE replay; no hyperparameter tuning on test, no mechanism-necessity qualification from frozen ablations.
- Executed whole Colab Notebook successfully with actual pretrained SWE checkpoint and synthetic two-case inputs, status PROGRAM_ONLY_SMOKE. Official PDEBench B2/P0 audit NOT run and must not be portrayed as new PDE evidence. Archive SHA256 checks included.

## Research roadmap
First run B2/P0 freeze audit. If diagnostic localizes failure, B2-R1 prequalification compares matched-capacity/fixed-budget changes one at a time: e.g. time integration improvement, decoder/dynamics compatibility constraint, or true latent sufficient-memory test (not automatically re-add old SOC or Hidden). Only after direct gain and no regression should do three-seed full PDE v1 comparison. BER temporal/event stability remains an independent open problem; do not wait to begin internal audit and do not merge BER without causal qualification. No current issue-time official AI Weather Quest RPSS readiness or costly PRO6000 qualification.