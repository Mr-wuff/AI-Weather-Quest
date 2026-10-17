# CIDM-MHC-Real A13-FIX2 NOAA exhaustion → A13-FIX3 independent WeatherBench2 ERA5 data bridge

Date: 2026-10-17.

## Actual uploaded failures
User's uploaded `CIDM_MHC_REAL_A13_FIX2_DIRECT_HTTPS_REPORTS.zip` exact SHA256 **`4da73c87a9f90bac54e70420b7b39da0b44fa54b8e58da10b94e76bc69316288`**, sidecar exact match, ZIP CRC PASS. Its own `status.json` states **A13_FIX2_EXPLICIT_APPROVAL_REQUIRED**, NOT a final network result. The user explicitly subsequently enabled `MHC_A13_ALLOW_FULL_YEAR_HTTP=1`; their pasted later logs then show `downloads.psl.noaa.gov DIRECT_HTTPS_HTTP_403` on 2012 Z500 and NOAA THREDDS FileServer `HTTP_503` for all retries. No later terminal result ZIP supplied: cannot claim independently verified final-report checksum. Earlier real FIX1 `A13_FIX1_STOPPED_REMOTE_OR_SOURCE_Q0_FAIL` had both NOAA OPeNDAP domains and NCSS HTTP503. These are **source/service failures, not learned-model failures**. Do not recommend more NOAA User-Agent spoofing as established remedy.

## A13-FIX3 explicitly switches dataset
To prevent repeated failure and huge NOAA full-year downloads, source switched deliberately to **Google Research WeatherBench 2 ERA5 6-hourly 1.5° conservative global regrid**. This is not the original NOAA NCEP/NCAR Reanalysis-1 Z500 and NOAA OISST. Authoritative data guide: https://weatherbench2.readthedocs.io/en/latest/data-guide.html ; public GCS source URI:

`gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr`

Fields `geopotential(level=500)`, `sea_surface_temperature`, `land_sea_mask`. Convert geopotential m²/s² to geopotential metres (/9.80665), SST K -> Celsius. Force latitude 90 to -90 and longitude 0..358.5 degrees in original 121x240 grid, accept underlying longitude -180..180 and reverse latitude when required. For coastal SST conservative-regridding uncertainty, mask cells with WB2 land fraction >=.5 and save separate finite-ocean mask. **This does not reconstruct NOAA SST coastal boundaries.**

For 2012–15 default n=4 evenly sampled UTC Sundays/year, select EXACT Sunday 12:00 UTC and associate following Thursday +4 calendar days. First verify one real remote 2012-01-01 12UTC Z500/SST/LSM before continuing. Save 4 real 121x240 year NPZ files, per-year SHA and source metadata in ZIP. No 2015 forecast label access, no 2016+ labels, no new training, no competition RPSS or historical operational release-timing proof. Increase 4 to 16 after actual Q0 succeeds. GCS is public anonymous and **not Google Drive**; no private HF/ECMWF credentials needed. Remote GCS accessibility must be confirmed by user Colab; do NOT call mock success real source acquisition.

## Delivered experiment and validation
- Notebook SHA256: `1c0f912a374fbff7e77600d6155d98cd0910311d828e97a422d4ccd80c07d688`.
- Script SHA256: `bac8d5bacac89dd3b57fea98be9a43a0841790c3a51d95eb0ab8e0860f4ec52b`.
- Complete ZIP SHA256: **`96c12dc5622ea1ab7f23759b89c823085afd1d7d5174cf95ab62f28f9d3e8a34`**, ZIP CRC + internal manifest per-file SHA PASS, contains immutable user FIX1/FIX2 negative archive proof.
- Whole standalone Notebook **4/4 code cells executed isolated CPU Smoke PASS**; independent fake WB2 public cloud 121x240 x2 variables x4 years x4 Sundays saved four year NPZ with SHA and complete ZIP PASS; forced remote source outage generated traceback/fail-closed status ZIP PASS. **No actual public cloud access tested by assistant local runtime**.
- Limitations: WB2 ERA5 retrospective rather than operational input; regridded coastal SST can differ, so output explicitly flags mask. This Q0 data bridge is not model skill or formal training authorization.
