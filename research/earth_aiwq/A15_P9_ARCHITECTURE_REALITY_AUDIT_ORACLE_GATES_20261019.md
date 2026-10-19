# CIDM-Earth A15-P9 architecture-readiness & genuine six-case oracle budget (2026-10-19)

## Why A15 has stalled
- P1/P2 built a genuine GEFSv12 retrospective five-member weekly-forecast and WB2 ERA5 verification pipeline; P3/P4 constructed train-only approximate quintile probabilities but frozen 2019 probability Gate failed.
- P6 fixed state-gated multiscale diffusion loses to isotropic and spatially shifted placebo; P7 genuine 420/420 GRIB traces with six *previously seen* issues yielded no incremental time-path error-magnitude skill over static Ridge (−1.84/−3.92% for MSLP W3/W4, −2.11/−0.75% TAS). Neither P6 nor P7 trained the archived CIDM core. They do NOT establish original CIDM neural architecture is invalid, but fail to qualify long S2S training.
- P8 HF P1–P7 remote upload and untouched 2005–07 source inventory still await a user-supplied Colab run; a GitHub SHA register is not proof of actual HF binary archive.

## Archived reference code audit
The project reference `cidm_architectures.py` declares SpaceEncoder, TimeHistoryEncoder, PhaseSpectralEncoder, AntisymmetricFlux, skew-channel rotation, latent local/hidden transfer, spectral decoder, BER.
- `AntisymmetricFlux` uses north-south `torch.roll`, creating inappropriate spherical north/south polar wrap. Flat periodic flux sum cancellation ≠ spherical area-weighted global physical budget.
- `CIDM.step` ignores external port; the port is used only by the BER variant. True sub-seasonal atmosphere/ocean is forced, not physically closed.
- Explicit Euler update of skew-symmetric rotation does not exactly preserve L2; latent L2 energy drift is not atmospheric energy conservation.
- Avoid adding more router/reservoir before simple geometry, forcing provenance and authentic official metric gates pass.

## A15-P9 diagnostics run on genuine archived P2/P3/P4 data NOW
- Parent exact SHA: P2 `dd380fb310546b080314ecb8ad78a2edbecad68413ac21f7696ea6a9feda486c`; P3 `0a60d66483bdf81d9e7bdeacf72c40d7eb1e4cabd8f7d6cb6691cc61c903ebd9`; P4 `10e471b02b94583c0b6a59857d1f9f6339c3ab2a498845e429a621bb34a25871`.
- Separate area-weighted **toy scalar tracer** finite-volume operator with zonally periodic/polar no-flux lateral exchange passes 1e-12 area-mass, source-budget and energy-dissipation unit tests; actual observed relative mass drift ~1.43e-16. This is NOT a production CIDM repair or real atmospheric physical conservation proof.
- Euler skew generator dt=.2 produces L2 drift .04, Cayley exactly orthogonal within 1.1e-16; CIDM main port sensitivity = zero.
- Six-case genuine retrospective research RPS, frozen half-count base:
  - MSLP W3 .867979, frozen ×2 spread improves +1.62%; W4 .809895, ×2 spread **worsens** −1.86%.
  - TAS W3 .795647, ×2 spread improves +1.34%; W4 .826443, ×2 improves +2.45%.
  - Global scalar bias correction using future ERA5 truth **worsens every target**, while per-grid mean shift using future ERA5 truth yields +46–56% apparent RPS reduction. Both use FUTURE truth: **non-deployable hindsight oracles**, never forecast achievements. These results suggest location/variable/lead dependence and do not imply CIDM itself can achieve oracle results.
- Real report already computed: `CIDM_EARTH_A15_P9_REAL_SIX_CASE_ARCHITECTURE_REPORTS.zip`, SHA256 `2b004366df7e19ef4506f57847f31ce79101a31770a58a84ac5e7fffe6622c09`.
- Complete local reproducibility package (includes ~23KB Colab Notebook, readable engine, frozen CIDM architecture, mock tests, long Chinese audit): `CIDM_EARTH_A15_P9_Architecture_Audit_LITE_FULL.zip`, SHA256 `abaadde87f8115bb32cf7d018ea00ccee2fdbe435481f02719d7e4cfaa8b0fba`. CRC and independent parent SHA tests pass; Colab 4 executable cells offline PASS; malformed parent ZIP fails closed.

## Mandatory correction of official metric framing
ECMWF AI Weather Quest says MSLP globally scored, TAS **land-only**, W3 days 19–25 and W4 days 26–32, quintile grid 1.5°. Earlier six-case TAS all-global proxy RPS is NOT official RPSS. GEFSv12 retrospective archive is NOT historical real-time forecast availability proof. Current 2026 competition rules preserve issue-time data restriction.

## Next QUALIFICATION gates; currently no long-training authorization
- G0: implement/verify geometry-aware conservative internal transport and *audited physical* budget, external port causality and gradients in actual candidate CIDM (P9 tests only a standalone reference positive control).
- G1: source inventory for 24 genuinely uninspected Wednesday issues 2005/07/09/11/13/15 Jan/Apr/Jul/Oct, 2 variables c00, 7 leads each, S3 HEAD/IDX only; P9 online preflight NOT yet executed.
- G2: construct genuinely disjoint train/validation/test years (suggest minimum 60/20/20 candidate issues subject to learning curve), strong calibrated GEFS/climate/EMOS/CNN baselines with official land mask and no future input leakage.
- G3: <=0.5M-parameter CIDM-residual pilot vs same-budget CNN, no-flux and shuffled information ports, minimum 3 seeds, independent-year confidence bounds. **Only** measurable uniquely attributable incremental RPS/RPSS with physical audit can open formal long training.
- G4: audit truly issue-time-available operational forecast input and actual competition submission format. No current model deployment eligibility.

## Delivery and remote re-use policy
P9 LITE Colab does not contain old large archives and does NOT demand manual reuploads. It tries the A15-P8 HF remote manifest by SHA and gracefully skips historical score recomputation if not present, while running new architecture audit and S3 inventory. Its already-computed six-case real report is delivered separately. P9 engine/source are supplied with the ChatGPT downloadable ZIP. GitHub PR currently preserves research contract and SHA fingerprints, not a new code commit nor binary report, pending review.

## Sources
- https://www.ecmwf.int/en/about/media-centre/science-blog/2026/ai-weather-quest-2026
- https://aiweatherquest.ecmwf.int/terms-of-ai-weather-quest/
- https://registry.opendata.aws/noaa-gefs-reforecast/
