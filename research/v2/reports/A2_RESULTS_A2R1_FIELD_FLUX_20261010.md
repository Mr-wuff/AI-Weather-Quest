# CIDM-v2 A2 result audit and A2-R1 controlled follow-up (2026-10-10)

User's actual A2 results: one 41,356-parameter model; 10 epochs ×48 optimization steps; best epoch 9; 7 of 9 gates pass. The two failures are OOD 12-step compound counterfactual skill and velocity-only conditional response. All results are synthetic; no real ECMWF RPSS.

| Benchmark | MSE | Persistence | CF skill | Predicted/true delta power |
|---|---:|---:|---:|---:|
| ID12 | .011009 | .024680 | .5948 | .4389 |
| OOD12 both | .039893 | .054675 | .2851 | .2644 |
| OOD12 velocity-only | .040318 | .055010 | .0816 | .0127 |
| OOD12 source-only | .039844 | .054384 | .7497 | .7000 |
| OOD24 | .099935 | .123646 | .2849 | .5099 |
| OOD36 | .137498 | .156394 | .0206 | .6340 |

Freeze-model ablations: OOD12 no-SOC +1.81% MSE, no-Flux +7.68%, no-Sync +3.41%; OOD36 no-SOC **improves** MSE 8.17%, no-Flux worsens 42.28%, no-Sync worsens 25.42%. Exchange algebra is not automatically a physical conservation theorem, and frozen inference ablation is not retrained capacity-equivalent attribution.

## Next v2-A2-R1 single-model test
Colab self-contained: `CIDM_v2_A2_R1_Field_Transport_Causal_Qualification_Colab_STANDALONE.ipynb`.
Goal: repair control-underresponse, **without** claiming that explicit upwind physics is a learned CIDM discovery. Candidate: explicit observable-field velocity upwind transport, coupled to trainable latent corrections, paired inter-channel cross-scale exchange (SOC candidate), known-source BER port, prediction-only reencoding synchronization.
Only one neural model trained. Evaluate against persistence, fixed upwind+diffusion, and a **stronger fixed upwind+diffusion+issued-source** reference. Separate OOD velocity/source/both counterfactual responses and 36-step extrapolation; require performance above strong fixed source-physics reference to claim learned incremental value.
Preflight validates causal twin history, genuine different counterfactual future, numerical flux cancellation and gradients. Offline smoke pass; short budget intentionally **failed** strong-baseline gate, demonstrating gate does not rubber-stamp success. No PRO6000 training approval from synthetic tests.

Artifacts delivered via ChatGPT: `CIDM_v2_A2_R1_Field_Transport_Qualification_FULL.zip` and standalone Notebook. Cloud runtime sync only from NEW Secret-based HF_TOKEN and GITHUB_TOKEN; previously published tokens must not be reused.