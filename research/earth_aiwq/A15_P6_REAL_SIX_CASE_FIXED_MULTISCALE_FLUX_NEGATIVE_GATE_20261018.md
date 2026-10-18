# A15-P6: Zero-training fixed multiscale probability flux — TRUE six-case audit (2026-10-18)

## Input evidence (actual user archives; all exact SHA256 locks verified)
- P2 four real GEFS-ERA5 cases 2017–2018: `dd380fb310546b080314ecb8ad78a2edbecad68413ac21f7696ea6a9feda486c` (the original P2 .sha256 sidecar had mismatched bytes; current uploaded bytes have valid internal case checksums).
- P3 2008–2016 train-only ERA5 climatological quintiles: `0a60d66483bdf81d9e7bdeacf72c40d7eb1e4cabd8f7d6cb6691cc61c903ebd9`.
- P4 new 2019 January/July paired GEFS-ERA5: `10e471b02b94583c0b6a59857d1f9f6339c3ab2a498845e429a621bb34a25871`.
- All SIX case dates already inspected. No new weather download, no training or honest fresh holdout.

## Prespecified intervention — numerical information-flux prototype ONLY
- Base empirical five-category GEFS ensemble probability uses frozen half-count (count+.5)/7.5.
- Benchmark `isotropic` is a two-step ordinary neighborhood probability diffusion at shifts 1,4 grid cells and fixed coefficients .125,.05.
- `edge_flux` has identical spatial scales and update coefficients; symmetric positive edge conductance derived exclusively from the forecast ensemble mean gradients / spread (no ERA5 labels), and applies to category probability simplex.
- `shifted_gate` moves the gate by 31 lat cells and 83 lon cells, preserving the probabilistic forecast itself, for spatial alignment placebo.
- Uniform five-bin climatology is a separate reference.
- This is NOT a trained CIDM latent state, physical atmospheric mass/energy-conserving solver, true wind advection, nor an official RPSS.

## Actual computed six-case global RPS (lower=better)
| Task | Half-count | Isotropic | Edge-gated | Shifted edge | Gated vs isotropic |
|---|---:|---:|---:|---:|---:|
| MSLP W3 | 0.867979 | 0.863789 | 0.864235 | 0.863884 | −0.0516% |
| MSLP W4 | 0.809895 | 0.805736 | 0.806193 | 0.805837 | −0.0567% |
| TAS W3 | 0.795647 | 0.782701 | 0.789474 | 0.785757 | −0.8653% |
| TAS W4 | 0.826443 | 0.813144 | 0.819869 | 0.816082 | −0.8271% |

Gated flux improves over half-count in all four tasks but loses to **the equal-cost isotropic control AND the deliberately misaligned gate** in all four. Per-year January/July means for 2017, 2018, 2019 likewise lose. Therefore incremental edge-informed cross-scale candidate Gate **FAIL**. No parameter retuning is authorized on these repeated six cases.

## Deliverables verified
- Real historic case output generated NOW: `CIDM_EARTH_A15_P6_Real_FixedFlux_Attribution_REPORTS.zip`, SHA256 `0d861561e9c9ca24fc7fc5d564283449e19452bf3addb77b0532c6df89f50a75`, ZIP CRC PASS; includes per-case RPS, per-year breakdown and exact intervention contract.
- Small full reproducibility ZIP in ChatGPT: `CIDM_EARTH_A15_P6_LITE_Qualification_FULL.zip`, SHA256 `feb86c3c29e57bd22ce5d2e7b83b30c0b6325a648f3b1c47d0ff2ab189bad6f0`.
- Source `a15_p6_cross_scale_flux_qualification.py`, SHA256 `806cb715fb954ba1e2fd8a1154e0416842d8294a83e520365e4ef9095a142a39` in user ZIP.
- Reproducibility Notebook: 19,231 bytes and 5 code cells. Lightweight: ~8KB compressed exact-source embedded, **no historical weather ZIP embedded**; searches local/Drive/HF or lets user upload only absent archives. Colab browser auto-download ZIP/SHA in success or blocked paths.
- Whole notebook independently executed using all genuine P2/P3/P4 archives; RPS matches standalone engine. Corrupted P2 SHA forces BLOCKED, no stale success files. Offline math positive/negative checks pass.
- Note: unlike earlier experiments, source is *distributed with the ZIP*, not yet committed as a raw Python file to GitHub; this PR is a traceable research contract and measured result. No HF write performed.

## Next gate
1. Freeze this negative mechanism result; do not rescue it by tuning step sizes or gate on reused six cases.
2. Acquire genuinely untouched forecast issue dates, maintain season/leads and check issue-time input availability.
3. Compare strong probabilistic calibration, ordinary convolution/diffusion, shuffled physics gate, and actual CIDM temporal/latent-information mechanism at matched compute/budget.
4. Only if a small, fresh-seed/fresh-date, physics-audited CIDM mechanism adds reproducible skill over these controls should formal long training begin.
5. GEFS historical reforecasts are retrospectively generated, not verified legal archived operational AI Weather Quest inputs. No official RPSS scored.
