# ECMWF ECBox SDK installation fix (2026-10-10)

**Affected Colab notebook:** CIDM-S2S Flagship L4 single-model qualification v0.2.

## Symptom
```
ModuleNotFoundError: No module named 'sites.sdk'
```
Import path: `AI_WQ_package.retrieve_training_data` → `retrieve_evaluation_data` → `from sites.sdk.sites import Site, Authenticator`.

## Official solution
From August 2026 the AI-WQ package uses ECMWF ECBox, requiring **`sites-toolkit` from the ECMWF Python package index**, not the unrelated PyPI `sites` package or `sites-sdk`.

In Colab:
```python
%pip uninstall -y sites
%pip install -U AI-WQ-package
%pip install -U sites-toolkit -i https://get.ecmwf.int/repository/pypi-all/simple
```
Validate immediately:
```python
from sites.sdk.sites import Site, Authenticator
from AI_WQ_package import retrieve_training_data, retrieve_evaluation_data
print("ECBox SDK import: PASS")
```

If importing still fails after pip succeeds, restart the Colab kernel once and run again from the first cell. If installation from ECMWF's index fails, diagnose network access to `get.ecmwf.int` instead of installing an unrelated package.

## Authentication
`AIWQ_PASSWORD` is the AI Weather Quest *forecast submission password*, supplied in the welcome or latest technical-update email (especially during the Aug 2026 ECBox transition). This value is used for bearer-token authentication by the official client. Do **not** put it in code or GitHub; store it in Colab Secrets or cloud environment variables.

## Cost-safe data preflight
Before transferring 84 large annual NetCDF files, use the official `retrieve_land_sea_mask(password, local_destination=...)` method to test one small authenticated ECBox download. Full data should be cached on a persistent volume. Colab `/content` is ephemeral.

## Source
- ECMWF official package installation: https://ecmwf-ai-weather-quest.readthedocs.io/en/latest/installation.html
- ECMWF Sites SDK docs: https://sites.ecmwf.int/docs/sites/sdk/source/index.html
- ECMWF data access: https://ecmwf-ai-weather-quest.readthedocs.io/en/latest/training_data.html

**Release**: `CIDM_AIWQ_Single_Flagship_L4_CompetitionGate_v02_FIX1_ECBox.ipynb` (delivered in ChatGPT).
