# A15-P10 frozen G0 geo-port CIDM prototype and G1 24-date control-index preregistration
Date: 2026-10-19.

## Motivation
A15-P6 learned no incremental value of state-gated probability diffusion compared with isotropic or displaced-placebo diffusion. Real A15-P7 read 420/420 GEFS temporal GRIB records, but a fixed engineered temporal Ridge probe lost to static on every MSLP/TAS x W3/W4 task. A15-P9 audited three concrete implementation problems: north/south wrapping in AntisymmetricFlux, ignored CIDM.step external port, and explicit Euler skew mixing only approximately preserving latent L2. **Neither P6 nor P7 tested or disproved an actually trained CIDM-S2S model.**

## Real G0 engineering qualification — already executed offline with PyTorch CPU
- Frozen original architecture copied without mutation to `research/earth_aiwq/a15_p10/frozen_cidm_architectures.py` (commit `b5a9d07b55b1b55e30057daf399f85a6d572d801`, SHA256 `f9b1cf4ff7630f5540bb3adde20e4c4b81fff3218d9325ea3105bf28b1d97665`). Frozen proven S3 reader at `bb1583fc7078e74f251ec101d0e11a522f2a9eeb`, SHA256 `4e88bee38ccce563d9cfd75e457e9a35070439b972dc7fc742a9e4b23520a98e`.
- Candidate `CIDMGeoPortModel` reuses real archived CIDM's space/time encoders, fusion and decoder. The *dynamic core alone* is replaced by area-weighted pairwise positive symmetric conductance, zonal periodic/meridional closed edges, Cayley skew-channel mixing, and explicit nonzero causal external port with separate internal/external budgets.
- Actual tests: area-exchange relative budget 1.01e-17, polar cross-edge transfer zero, Cayley L2 relative drift 4.51e-16, port core perturbation max 0.00340 and complete CIDM output 0.00511, original CIDM port output zero. Both learned flux/port parameter gradients nonzero; no-port 32-step weighted energy ratio 0.04233; physical unit test checks PASS.
- **Strong scientific caveat:** latent tracer area-exchange accounting is NOT rigorous conservation of actual atmospheric MSLP/temperature/momentum/energy. Prototype has NOT trained on weather data. Parameter counts are NOT equal-budget forecast comparisons. G0 means numerical/unit-mechanism readiness only, not general architecture maturity or formal-training permission.

## G1 preregistration and no-claim boundary
- New 24 source-inventory dates: years 2002, 2004, 2006, 2010, 2014, 2016; first Wednesday 00UTC of Jan, Apr, Jul, Oct. Only `c00` control for MSLP/T2m: 24×2=48 GRIB metadata objects. HEAD + IDX only; no GRIB download.
- Frozen forecast leads: 408,456,504,600,624,696,786 h; validate init-date in index, seven exact entries, byte ranges, ETag and IDX SHA.
- Threshold at least 20/24 valid full two-variable control issues in >=5 different years -> `G1_CONTROL_INDEX_PREFLIGHT_PASS` only. **Not full five-member G1 dataset**.
- Real online S3 preflight NOT executed in this assistant environment. No A15-P10 actual weather forecast scores, calibration, official RPSS or CIDM training. Historic GEFSv12 is retrospective reforecast, not demonstrated legal issue-time operational input for ECMWF competition.

## Deliverable evidence and QA
- Standalone 16,208-byte Colab Notebook, complete independent Python engine, original frozen sources, mock S3 tests, Chinese README and preregistration contract, and locally computed actual G0 report all in small user-facing ZIP.
- Full ZIP `CIDM_EARTH_A15_P10_GeoPort_G0_G1_LITE_FULL.zip`, SHA256 `5a8e9a7c0d5088e79714e7a56354564a2c3287723364af60e3fee6355424f69a`; full CRC and file manifests verified.
- Whole 4-code-cell Notebook executed: `OFFLINE_QA_ONLY_NO_REAL_S3`, G0 PASS. Corrupted inline engine SHA makes the whole Notebook return a BLOCKED ZIP with an accurate sidecar rather than falsely succeeding. Fake S3 index positive and missing-lead negative tests pass.
- Notebook uses GitHub pinned real archived CIDM and S3 reader and no prior results ZIP. New tiny P10 engine is compressed in the 16KB Notebook and delivered separately as readable source; **this PR does not claim the P10 engine was separately pushed as a raw .py file**.
- A15-P8 HF P1–P7 binary upload status remains unconfirmed, but P10 requires no historic ZIP at all.

## Progress gate
G0 numerical candidate unit proof: PASS; G1 online control index: PENDING. Must still qualify **real five-member dataset**, official-compatible TAS land mask/global MSLP and training-only climatology, strong EMOS/CNN comparator, truly independent dates/seeds and model incremental value before formal long training. Do not use earlier hindsight oracles as achievable model gains.
