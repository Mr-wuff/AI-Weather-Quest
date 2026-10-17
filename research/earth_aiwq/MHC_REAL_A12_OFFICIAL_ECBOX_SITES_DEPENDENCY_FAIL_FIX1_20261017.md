# CIDM-MHC-Real A12 official user run blocked at ECBox sites-toolkit dependency, FIX1 repair

2026-10-17. Uploaded user `CIDM_MHC_REAL_A12_P0_MJO_INFORMATION_REPORTS.zip`: SHA256 **`315a789e62524cf8022d8ad255c7bc8f5e0f50099916f5ce8c476c372e754132`**, matching original .sha256, ZIP CRC PASS. Uploaded notebook runtime logs status `A12_BLOCKED_EXPORT_DIAGNOSTIC`. Stage `LOAD_OFFICIAL_DAILY_MJO`, no forecast skill, zero neural optimizer updates, no 2015 MJO test results.

Root cause: expected first-use 404 from private HF cache `raw/MJO/2012/MJO_DAILY_2012.nc`, then installed `AI_WQ_package` transitively imports `from sites.sdk.sites import Site,Authenticator` but ECMWF ECBox `sites-toolkit` is absent, raising `ModuleNotFoundError: No module named 'sites'`. The old code incorrectly reclassified this as `AI_WQ_PACKAGE_NOT_INSTALLED`. Do not mistake this infrastructure block for a negative scientific MJO result.

ECMWF official installation documentation requires after 2026-08-10 `python -m pip install 'sites-toolkit>=20261005.1' -i https://get.ecmwf.int/repository/pypi-all/simple` and installation of `AI-WQ-package`. Verified documentation: https://ecmwf-ai-weather-quest.readthedocs.io/en/latest/installation.html and https://ecmwf-ai-weather-quest.readthedocs.io/en/latest/training_data.html .

New A12/P0-FIX1 unchanged science, upgraded bootstrap and explicit error classification:
- Standalone Colab `CIDM_MHC_REAL_A12_P0_FIX1_ECBox_Colab_STANDALONE.ipynb` SHA256 `8e839ae0bb284a5f5f17781ecc016bfe48f885af76a60bb4097a99008b4bd584`.
- Updated independent source `a12_mjo_incremental_information_FIX1.py` SHA256 `3f13f705966f3313b553c27963cc258a9aae3fa17c5b1580aa14e35b0153a1c1`.
- Self-contained research ZIP SHA256 **`04c4821603182ac329a4267c212d48383072359b2816f2ab4ce0979df68aa48b`**, CRC and per-file manifest SHA verified, embeds prior failed user report.
- Notebook preflight installs/updates ECMWF sites-toolkit if below 20261005.1 and installs AI-WQ-package if missing; **imports actual `sites.sdk.sites` and `AI_WQ_package.retrieve_training_data` before launching script**. Secret values from authorized Colab Secrets or env, never printed. Script distinguishes private HF entry-not-found from permission/network/429, and missing training API vs missing sites module; validates downloaded daily MJO RMM NetCDF and SHA before study.
- Full Notebook CPU smoke **4/4** code cells pass in isolated nbclient; synthetic stub test of private HF 404 -> official MJO API success -> annual 366 day NetCDF and provenance hash PASS; pre-cached local reuse PASS; missing official credential path PASS.
- **No actual authenticated ECMWF download tested** in assistant container; the user's authorized Colab remains necessary.
- Preserves old 2012–14 fit preprocessing; 2013–14 101 fit issue dates, 2015 reused 49 dev, six MJO features, same-dim surface-weather and eight shuffled control, no 2016+ forecast target scoring, no AIWQ RPSS, long training unqualified; causal Sunday input caveat and ERA5 retrospective MJO reference years caveat. Optional one private HF Dataset commit for 2012–15 official MJO files, default OFF; no Google Drive.
