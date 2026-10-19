# A15-P12-E1 True Colab failure → FIX1 environment bootstrap (2026-10-19)

## Evidence from actual user run
- User ZIP `CIDM_EARTH_A15_P12_E1_TRAIN16_DELIVERY_REPORTS.zip`, SHA256 `e5e77fba2ace9b5404271ed89a76069f4003c128e523a487a27b01039ce9368f`, exact uploaded sha256 sidecar match and ZIP CRC PASS.
- The ZIP contains **ONLY** `reports/STATUS.json` (2027 bytes): `BLOCKED_TRAIN16_SOURCE_OR_PAIRING`, `completed_n=0`, 16 preselected training issues, no result NPZs and no model training.
- Colab first code cell printed `PINNED_SOURCES_PASS offline_qa False`; HF Pilot4 bootstrap failed `RemoteEntryNotFoundError` and fell back to NOAA/ERA5 retrieval.
- First actual error: `ModuleNotFoundError: No module named 'boto3'` at frozen P1 `anonymous_client()` before any source GRIB2 range retrieval.
- **Confirmed root cause:** The E1 Notebook had the pip dependency installer erroneously nested under `if OFFLINE_QA_ONLY:`; therefore real Colab (`False`) skipped all installation. This is a notebook bootstrap bug, NOT a GEFS index failure or failed CIDM physics.

## Delivered exact reproducible fix
- User-facing **18,127-byte** `CIDM_EARTH_A15_P12_E1_FIX1_Train16_Colab.ipynb`, SHA256 `81c714d141ea2bafd81884ee749164c0faee1c777a34e29b6e79f99ac3b0e0d6`.
- Precisely move dependency installation under `if not OFFLINE_QA_ONLY`, detect missing `boto3,eccodes,gcsfs,xarray,zarr,numpy,huggingface_hub`, keep `zarr>=2.16,<3` as required by the historical WeatherBench2 pairing code, then import all required modules, verify ecCodes binding, print installed versions before anonymous S3. Any installer error becomes explicit bootstrap BLOCKED report and SHA256 sidecar. Offline QA never pip installs.
- Full downloaded reproducibility pack `CIDM_EARTH_A15_P12_E1_FIX1_RuntimeDependencies_LITE_FULL.zip`, SHA256 `1182383892d6b2fbc8ff4b8a4ef13d80a2efc65dfec9b7bd6fcfdd28f0ce2fba`. Contains corrected Notebook, readable pinned P1/P2/P11-FIX1/P12/E1 engines, offline test and Chinese root cause/recovery explanation. ZIP CRC and per-file source hash manifest verified.
- Whole corrected Notebook executed locally with `CIDM_A15_P12_E1_OFFLINE_QA=1`: all four code cells PASS, `OFFLINE_QA_ONLY_NOT_REAL_TRAIN16`, no network pip invocation.
- Additional REAL-BRANCH **non-network mock**: missing eccodes/gcsfs/zarr detected; pip invocation deliberately intercepted and exception correctly became `BOOT_ERROR` and valid BLOCKED ZIP+matching sidecar SHA. This proves branch selection/error propagation, but not actual Colab package installation or NOAA retrieval.
- Original source modules and every scientific contract identical. Only Colab dependency startup and diagnostic stdout are changed. Preserve 2002/04/06/10 TRAIN16 first-Wednesdays, 2014 validation untouched, 2016 test untouched. HF Pilot4 archive existence NOT confirmed. Reuse same `/content/drive/MyDrive/CIDM_EARTH_A15_P12` workdir and no re-upload.
- Colab **real Train16 has not been rerun**. Still G2-E0 real four-case PASS, G2-E1 16-case PENDING. No official RPSS, no neural learning, no long-training authorization.

## Stable source
Original A15-P12 paired engine remains pinned to GitHub `research/a15-p12-e1-train16-20261019` at commit `5c77ec2cb06217304adf8335e6e68cbf7e7963f9`. FIX1 Notebook and ZIP are attached in ChatGPT. This PR commits the reproducible dependency installer and failure evidence, not user weather datasets or HF results.
