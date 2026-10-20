# CIDM-Earth A15-P16-WIND-E1 real four-year result and P850-E0 next-step contract

Date: 2026-10-20. This draft only commits transparent weather-source research code; it does NOT publish user-derived GEFS arrays or claim real 850hPa qualification yet.

## What the provided executed notebook supports
- User's executed `CIDM_EARTH_A15_P16_WIND_E1_3NewYears_Real10mUV_Colab(1).ipynb` prints all three authentic new-year completions:
  - `2012040400`: `CASE PASS`, 160 wind records (five ensemble members × U10/V10 × 16 six-hour leads).
  - `2013070300`: `CASE PASS`, 160.
  - `2015100700`: `CASE PASS`, 160.
  - Terminal state: `REAL_PILOT4_SIGNED2009_PATCH_PLUS_3_NEW_FULL_10M_WIND_PASS_RESEARCH_ONLY`, `new_full_n=3`.
- Prior 2009 genuine signed patch was restored, **not a newly downloaded 121×240 full tensor** in this run. W3 and W4 consist of separate 8-point six-hour windows, with 126h gap between f498 and f624; no cross-gap one-step rollout.
- The execution log shows Google Drive credential propagation failed, local volatile fallback occurred and 62,025,830-byte `P16_WIND_E1_VOLATILE_RECOVERY.zip` was generated. The **current user upload includes ONLY its sidecar**, SHA256 `850b10fc3349640e1c5b6eacfbaa1ac83e36cb9eef2d01d18afbbc26583fbb06`. Backing ZIP bytes were **not supplied**, so cannot independently certify cache-file content or use the 62MB raw weather backup for downstream training in this model environment.
- Result ZIP sidecar claims SHA256 `f678979dd37aadf502e59ec692d94159d5d9c5f3436a1fab12e16b2341e2547c`. Executed Colab terminal status supports completion, but byte-level ZIP CRC/SHA could not be independently audited in this assistant environment because the file-reading container service was unavailable. Do not claim SHA pass.
- These are genuine GEFSv12 *retrospective reforecast* inputs, NOT certified historically operationally available, and no learned CIDM forecast skill.

## P16-WIND-P850-E0 FROZEN NEXT CONTRACT
- NOAA public GEFSv12 documentation specifies `ugrd_pres` and `vgrd_pres` pressure-level winds, including **850 hPa**, at 0.5 degree beyond day 10. References:
  - https://noaa-gefs-retrospective.s3.amazonaws.com/Description_of_reforecast_data.pdf
  - https://www.emc.ncep.noaa.gov/gmb/wd20hg/Guan_GEFSv12_refcst_revision.pdf
- Independent high-altitude source audit; existing true U10/V10 arrays from previous results are NOT necessary to run source qualification. Never use 10m wind as a substitute if 850hPa missing; never use 700, 800, 925 or 1000hPa instead of 850hPa. Existing W3/W4 MSLP/TAS ERA5 labels are not passed to the source loader.
- `pilot1` 2009010700, `pilot4` 2009010700+2012040400+2013070300+2015100700. Five GEFS members c00/p01..p04, `ugrd_pres/vgrd_pres`, 16 leads [456:498:6] and [624:666:6], 160 ranged 850hPa GRIB2 samples/issue, 640 max in full pilot4. Preserve exact 126h gap.
- A new 850-pressure custom **multi-level GRIB IDX** parser checks 10-digit issue date, strictly increasing offsets and one and only one 850hPa GRIB per lead, parameter category 2 and parameter number U 2 / V 3, `typeOfLevel=isobaricInhPa,level=850` (or equivalent Pa encoding), m/s, grid 361×720 → strict 121×240, no pole-roll. Every group HEAD ETag/content length/IDX SHA and per-array tensor SHA at beginning/end and on cache reuse.
- Preserve full five-member, two-component, two-week×eight-steps 121×240 arrays in workdir, plus 11×12 area-weighted patches and compact result ZIP. Atomic cache and SHA sidecar + signed volatile export, Drive preferred but optional. No automatic historical upload, HF, long training, or official RPSS.
- **Next source gate is PENDING**, Notebook created but neither synthetic QA nor NOAA real online processing were actually run in this assistant turn (container unavailable). Tool-generated Github source/Notebook are deliverables to be executed and debugged, not pre-certified successful software.
- Scientific continuation only after 850 source data real PASS and much larger independent episodes: multi-step state increments, directional vs symmetric flux, zero/shifted-wind and equal-budget CNN controls, zero ERA5 input leakage, >=3 seeds and previously unseen years. Both 10m and 850hPa wind should be compared as candidate ports; 850hPa is NOT automatically the physically correct velocity for MSLP/TAS.

## Deliverables
- Editable readable Python engine: `research/earth_aiwq/a15_p16_p850_e0/p16_850hpa_source_gate.py`.
- Complete self-contained Colab: `research/earth_aiwq/a15_p16_p850_e0/CIDM_EARTH_A15_P16_P850_E0_RealWindSource_Colab.ipynb`.
- User run: open Colab from GitHub, execute all cells with `STAGE='pilot1'`; inspect generated `P16_WIND_P850_E0_DELIVERY_REPORTS.zip` and SHA256; then switch to `pilot4` if source gate passes. Source file handles `--offline-qa` for pure IDX negative controls.
- No source hashes or success status are claimed beyond GitHub accepted commit; no binary weather output is uploaded in this PR. Draft, do not merge automatically.
