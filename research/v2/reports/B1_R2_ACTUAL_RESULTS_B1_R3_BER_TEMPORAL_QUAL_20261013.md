# CIDM-v2 BER B1-R2/P0 completed result and B1-R3 temporal-generalization experiment

Date 2026-10-13. Actual user-uploaded ZIP and executed Notebook checked; ZIP integrity valid, nine R1 checkpoints verified, three seeds (17,41,73), 5 stress scenarios.

## Frozen BER pulse-memory results (actual uploaded report)

| Test | State-Port vs Readout-Port field MSE improvement | State-Port minus Readout post-event CF skill | True post-pulse power |
|---|---:|---:|---:|
| Seen early pulse H12 | -39.6103% | -2.78667 | 2.01469e-4 |
| Unseen early pulse H12 | -7.1653% | -4.21857 | 1.86916e-4 |
| Unseen early pulse H24 | -34.3783% | -34.72649 | 1.64283e-4 |
| Unseen mid pulse H24 | -35.1394% | -5.41483 | 1.70584e-4 |
| Unseen early pulse H24 hidden coupling-law shift | -34.3171% | -66.21099 | 8.68107e-5 |

Gates: post-event signal sufficient TRUE; State-Port pulse H12/H24 advantage FALSE. Original report `B1_R2_FROZEN_MEMORY_NOT_FULLY_ESTABLISHED`. All 15 seed-by-case comparisons of State-vs-Readout field gain negative. True causal response power is weak (~1e-4), so large negative normalized CF must be interpreted alongside field MSE and amplitude. No architecture retraining; pulse forcing time distribution differs from original continuous R1. A hidden coupling-law change is not disclosed to either model, so its failure is OOD stress, not proof of no law inference.

## Next B1-R3: test training distribution before adding a new mechanism

**Hypothesis:** temporal training distribution mismatch can explain some of pulse-memory failure. Three original B1-R1 equal-parameter 19,038 networks are unchanged (State-Port, Readout-Port, Zero-Port), allowing targeted examination of mixed continuous/pulse/mid-pulse/cessation forcing training.

One standalone Colab Notebook with all source embedded and separate Chinese protocol. **Q0 preregistered mandatory stage before any formal training**: train seed17 x3 variants for 90 steps; validate only seen families. Require both port-using variants >5% gain vs zero on continuous inputs; State-Port early pulse MSE <=1.30x Readout; pulse post-event response power ratio <=8; true post-event signal >=1e-6. If any fails, stop and write report; never use unseen family to select. Passing allows three seeds x three variants x240 updates, mixed 6/12-step objectives. Blind-for-this-run family2 12/24 step event tests, ability to retain post-cessation response; stricter formal State-vs-Readout multi-seed gates.

Whole Colab Notebook successfully executed end-to-end under CPU **SMOKE**, status `SMOKE_DIAGNOSTIC_ONLY`. Formal Q0 and GPU training NOT yet executed. No claim of BER mechanism repair or PDEBench/AIWQ qualification. Coupled family has been revisited repeatedly during the research program and isn't a newly blind research family; external coupling formula remains shared.

Deliverables in ChatGPT: `CIDM_v2_B1_R3_BER_Temporal_Qualification_Colab_STANDALONE.ipynb` and `CIDM_v2_B1_R3_BER_Temporal_Qualification_FULL.zip`.

CIDM-v2 remains under development; do not jump to v3 or invest PRO6000 without PDE-wide v1 superiority and official issue-time-causal AI Weather Quest RPSS qualifications.