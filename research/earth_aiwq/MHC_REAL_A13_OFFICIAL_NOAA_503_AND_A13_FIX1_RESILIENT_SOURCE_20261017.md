# CIDM-MHC A13/P0 official NOAA HTTP 503 → A13-FIX1 multi-backend physical data bridge (2026-10-17)

## Exact user evidence (independently checked)
- User upload `CIDM_MHC_REAL_A13_P0_PHYSICAL_SOURCE_REPORTS.zip` exact SHA256 **`b7da5c3869b9e99cda51a4d3f1db650c676794733554bf95d36ecde01f97b2de`**, matches sidecar and ZIP CRC PASS.
- User executed standalone Notebook stopped at `A13_REMOTE_OPENDAP_NOAA_SOURCE_FETCH`. Opening `https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/Dailies/pressure/hgt.2012.nc` returned **HTTP 503 Service Unavailable HTML**, inducing `OSError: NetCDF DAP server error -70`. SST extraction not attempted, NO legitimate Z500/SST NPZ produced, NO forecast labels, model updates or official RPSS.
- Scientific status is **SOURCE Q0 BLOCKED**, neither physics nor model negative result.
- NOAA source catalog separately documents OPeNDAP, HTTPServer and NetCDF Subset Service for the same year NCEP original dataset: https://www.psl.noaa.gov/thredds/catalog/Datasets/ncep.reanalysis/Dailies/pressure/catalog.html?dataset=Datasets%2Fncep.reanalysis%2FDailies%2Fpressure%2Fhgt.2012.nc ; highres SST is https://psl.noaa.gov/thredds/catalog/Datasets/noaa.oisst.v2.highres/catalog.html . HTTP full SST annual file ~476 MB per year.

## FIX1 (original A13 scientific contract unchanged)
- Notebook `CIDM_MHC_REAL_A13_P0_FIX1_ResilientSource_Colab_STANDALONE.ipynb` SHA256 `9d070015aed1c49b3401d5ee3951ea405317a419b2f4e3036b2f6aab8e5ce673`.
- Driver `a13_fix1_resilient_noaa.py` SHA256 `38517f4d967f1aee997d9f8a29bc04584d07cf7f427f259f6e5a69dd872729f3`; inherits original verified A13 source and original exact SHA historical A12-R1 user report.
- Trusted NOAA retrieval hierarchy: both PSL official OPeNDAP hosts with bounded retries → NCSS single exact Sunday NetCDF subset, Z500 500 hPa and SST horizontal stride6, cache daily subset → optional (explicit `MHC_A13_ALLOW_FULL_YEAR_HTTP=1` only) official NOAA annual HTTP fileServer of *same product*, ~0.56 GB/year, >2GB across four years. No silent product substitution, no overlarge download default.
- Source dates exact calendar Sunday, next Thursday issue +4 days; allow 12UTC nominal source timestamp as same calendar date, NO nearest-date substitution. Regrid Z500 longitude periodic to 121×240; preserve SST land NaNs; verify source variable/time level/coverage, source-derived SHA256, fail-closed when source inaccessible. DAP slices single day and level before remote materialization. Success ZIP now includes actual per-year derived NPZ + provenance and hashes, not only report metadata.
- 2012–2015 input only (16 Sundays/year default); no target labels, no neural updates, no climatology fit, no validation/official AIWQ RPSS and no operational release-latency proof. NOAA reanalysis/OISST are **retrospective** proxies, not ERA5-matched operational products.

## Independent execution proof and limits
- Entire Notebook **4/4** executable code cells ran real nbclient isolated CPU smoke with verified original user evidence. Z500/SST/Q0 math path passed.
- Offline fake HTTP503 → NCSS fallback tested with native 121×240 grid, correct 500hPa and noon timestamp for 2 Sunday dates x2 variables PASS.
- End-to-end fake NCSS source across 2012–2015 output 4 actual 121×240 annual NPZ, archive includes 4 data files + exact SHA provenance, PASS.
- Forced both endpoints fail: explicit `A13_FIX1_STOPPED_REMOTE_OR_SOURCE_Q0_FAIL`, saved traceback, source errors and CRC-valid output ZIP, PASS.
- **Actual authenticated/public NOAA 2026-10-17 remote rerun has NOT been performed by assistant**, internet inaccessible in local runtime; all end-to-end source tests use synthetic mock NetCDF data.
- Complete research ZIP SHA256 **`08e2e68db715758064788ce5abb6c7afc4d213e9a8b10208bc1e98497a0c82ab`**, manifest per-file SHA and CRC PASS.
- No Google Drive and no HF tokens necessary. Next user action: run FIX1 Colab (optionally `MHC_A13_WEEKS_PER_YEAR=4` quick public-source Q0 then 16 full), upload reports ZIP+SHA. If NOAA source stays 503, stop and report, or opt explicitly into full-year NOAA HTTP fallback; do not claim science-qualified input until actual source checks pass.
