# CIDM-Earth A15-P11 FIX1 real full five-member qualification PASS → A15-P12 G2-E0 paired-data pilot
Date: 2026-10-19

## A15-P11-FIX1 actual user results, independently audited
- Actual submitted ZIP sha256 **1780713d2a7f4f35148fe1d51f6c833f3ef314788f6a4b6f72be6916cbe876b9**; user SHA sidecar exact match, ZIP CRC PASS; 3 JSON member files.
- Real report status **G1_R2_FIX1_FULL_ENSEMBLE_INDEX_AND_SAMPLE_PASS_RESEARCH_ONLY**.
- 24/24 first-Wednesday issues all six years 2002, 2004, 2006, 2010, 2014, 2016; 240/240 objects across c00+p01..p04, MSLP/TAS, all 56 six-hour W3/W4 lead index entries and 7 chosen trajectory leads; source IDX init dates all `YYYYMMDDHH`, exact UTC hour verified, source ETags/IDX SHA recorded.
- 48/48 genuine ranged GRIB2 ecCodes spot samples PASS, errors={} (year-stratified c00/p04 × MSLP/TAS × f456/f786), origin (361,720) 0.5° global output (121,240) 1.5° node-subsampling.
- This is real source *index plus stratified physical-sample qualification*, NOT a 24-issue fully decoded paired weather dataset, NOT official RPSS, NOT proof GEFSv12 retrospective reforecasts existed before historic issue time.

## A15-P12 G2-E0 preregistration
- Goal: qualify truthful end-to-end GEFSv12 five-member 28+28 six-hour **full** W3/W4 weekly forecast fields paired to *future-label-only* WeatherBench2 ERA5 (1959–2023 conservative 1.5°). No CIDM/CNN model training.
- Frozen split (only training labels accessed at P12):
  * Train 2002/2004/2006/2010 Jan/Apr/Jul/Oct first Wednesday, 16 dates.
  * Validation 2014 first Wednesday Jan/Apr/Jul/Oct, 4 dates; STRICTLY UNTOUCHED BY P12.
  * Test 2016 likewise 4 dates; STRICTLY UNTOUCHED BY P12.
- Default `STAGE=pilot4` uses EXACT train-only dates `2002010200,2004040700,2006070500,2010100600`; each requires 5×2×(28+28)=560 actual full ecCodes ranged GRIB messages = **2240** records total across 4.
- On pilot4 success, change `STAGE=train16` to extend via atomic per-member-variable progress caches to 16 train issues (8960 records total) before undertaking actual small-model validation. No test-year scores or model selection during data building.
- Every real-case source preflight uses **true P11-FIX1 strict ten-digit IDX hour check**, GRIB object ETag and IDX SHA; P1 whole weekly finite/physics checks, P2 exact 112 time stamps to WB2 ERA5 and physical units, truth as LABELS only. Cache integrity includes source fingerprints and derived-pair SHA. Derived arrays forecast shape [5,2,2,121,240], truth [2,2,121,240]. Record case IDs and provenance.
- Research deterministic ensemble RMSE/CRPS only, **NOT official RPSS**. Data grid difference: GEFS source 0.5° point-sampled to 1.5°, WB2 ERA5 conservative 1.5°; no claim exact conservative remapping equivalence.
- OLD A15-P3 2008–2016 climate thresholds overlap 2014 validation and 2016 test. DO NOT re-use them in this new split; fit a strictly prior train-only climatology before future probability benchmark, use appropriate TAS land-only mask.
- RETROSPECTIVE archive NOT true issue-time-available operational GEFS, cannot claim ECMWF AI Weather Quest deployability.

## Delivered and qualified code
- User-facing `CIDM_EARTH_A15_P12_G2_E0_PairedData_LITE_Colab.ipynb` **15012 bytes**. Includes compressed ~10.9KB P12 source without any historic ZIP or weather tensors. Three dependent readers fetched from pinned GitHub SHA:
  * P1 `bb1583fc7078e74f251ec101d0e11a522f2a9eeb` SHA `4e88bee38ccce563d9cfd75e457e9a35070439b972dc7fc742a9e4b23520a98e`.
  * P2 same commit SHA `27d5cec94ff9f2eb7464a1fb8a8707339a875f2636282b4187725f7788e85659`.
  * P11-FIX1 `df938844f1facd3b8d2c211751fae457fd911ec1` SHA `3afd20ea9cdab5ceb8025580cadbd148a3cc5e10fbad8b77c85bf495e2e217fb`.
- Source engine SHA `183f3c72dee268768674e29d1992c8041a178bc43230bb240e193f153f8f3f12`.
- Full source + Notebook + mock tests + Chinese protocol in user-facing ZIP `CIDM_EARTH_A15_P12_G2_E0_PairedData_LITE_FULL.zip`, SHA256 `aa98a20b7dfded11554527e5749152657a3872b71949fb8267deadf3bba3aea8`; zip CRC and per-file manifest verification PASS. Readable P12 engine is in this downloadable ZIP, **not yet separately committed to GitHub**.
- Entire four-cell Notebook executed offline: explicit `OFFLINE_QA_ONLY_NO_REMOTE_WEATHER`, no fabricated result. Four-case fully synthetic pipeline mock PASS; resume PASS; tampered saved paired cache BLOCKED; deliberately incorrect engine SHA -> standalone BLOCKED diagnostic ZIP. **Real P12 GRIB/ERA5 download has NOT run yet.**
- Colab output ZIP+sha auto-download both completion and blocked. Optional HF upload of SHA-named derived report ZIP only after user explicitly sets `HF_UPLOAD_DERIVED_RESULTS=True`, authenticates in Colab Secrets and reviews repo privacy; no HF upload by assistant at this step.

## Training gate status
G0 geometric/port unit test PASS; G1 c00 24/24 PASS; G1-R2 full ensemble index and sample **PASS**. G2 complete paired years/strong official-compatible probabilistic benchmark PENDING; G3 <=0.5M GeoPort CIDM versus equal-compute CNN and ablations, >=3 seeds, heldout years PENDING; long training NOT AUTHORIZED.
