#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime dependency guard for Colab A15-P12 E1 and future GEFS-ERA5 stages.

No network dependency installation in offline QA. Real weather mode only:
detect missing imports or Zarr major-v3, install a compatible runtime
environment before creating the anonymous NOAA S3 client. Never hides failures.
"""
import importlib
import importlib.metadata
import importlib.util
import subprocess
import sys

REQUIRED = ("boto3","eccodes","gcsfs","xarray","zarr","numpy","huggingface_hub")
INSTALL = ("boto3>=1.34,<2","eccodes>=2.37,<3","gcsfs","xarray","zarr>=2.16,<3","huggingface_hub")

def ensure_real_weather_dependencies(offline_qa=False):
    if offline_qa:
        print("[DEPENDENCY_PREFLIGHT] OFFLINE_QA: no pip/network installation")
        return {"status":"SKIPPED_OFFLINE_QA"}
    missing = [p for p in REQUIRED if importlib.util.find_spec(p) is None]
    try:
        major = int(importlib.metadata.version("zarr").split(".")[0])
    except importlib.metadata.PackageNotFoundError:
        major = None
    if missing or (major is not None and major >= 3):
        print("[DEPENDENCY_PREFLIGHT] installing missing packages:",missing,"zarr_major:",major,flush=True)
        subprocess.check_call([sys.executable,"-m","pip","-q","install",*INSTALL])
        importlib.invalidate_caches()
    for p in REQUIRED:
        importlib.import_module(p)
    if int(importlib.metadata.version("zarr").split(".")[0]) >= 3:
        raise RuntimeError("UNSUPPORTED_ZARR_MAJOR")
    import eccodes
    if not hasattr(eccodes,"codes_new_from_message"):
        raise RuntimeError("ECCODES_PYTHON_API_MISSING")
    versions = {p:importlib.metadata.version(p) for p in ("boto3","eccodes","zarr")}
    print("[DEPENDENCY_PREFLIGHT] PASS",versions,flush=True)
    return {"status":"PASS","versions":versions}
