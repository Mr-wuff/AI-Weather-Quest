# A15-P14-FIX1 genuine partial prospective result → FIX2 atomic GEFS checkpoint and signed 2003 recovery

Date: 2026-10-20; independent audit of user-provided result and recovery ZIPs. This GitHub branch contains **only code/protocol**, not the user's full paired weather data.

## Verified actual uploaded evidence

1. `CIDM_EARTH_A15_P14_PILOT4_DELIVERY_REPORTS(1).zip`: SHA256 `2be663870d3aa018f66d6d1cf0ae38b7b3365f5c80d4f364809690e98bb4c34e`, user sidecar exact match, CRC PASS; status **BLOCKED_NEW_YEAR_SOURCE_OR_MOMENT_VALIDATION**, 1/4 real new dates complete, `2003040200` ONLY.
2. `P14_VOLATILE_CASE_CACHE_RECOVERY.zip`: SHA256 `6d4f5615f89d54c1e4ad6c352b66780721601f0dd08f596def61c4ec878c2b9e`, sidecar/CRC PASS. Contains signed manifest and exactly `cases/paired_2003040200.npz` (~2.04 MB) and matching `.json`. Paired SHA `c5df7eb16cb28c18d962ff96022b08b0164cda0cf484182233ec955c759b3c77`. Genuine forecast shape (5,2,2,121,240), ERA5 truth (2,2,121,240), tensor SHA, source ETag/IDX and 4 raw RMSE recomputed against the unchanged P14 engine; all 4 head result RMSE methods reproduce the independent user report to <1e-7.
3. Genuine 2003 APR frozen-P13 spatial correction vs raw research RMSE: MSLP W3 +4.13245%, W4 +1.85522%, TAS W3 +3.15332%, W4 +2.69510%. Against four-longitude-patch shifted placebo: +1.11139%, +1.06762%, +0.87903%, **−0.17028%**, respectively. Neither W3/W4 joint year Gate nor even 4-date Pilot4 completeness has been established. These are NOT CIDM scores.
4. FIX1 Drive credential propagation error fell back to volatile local successfully. On `2005070600` member p02/tas, original P1 `write_part` raised `FileNotFoundError` for `partial_2005070600_p02_tas.npz.tmp -> .npz`. Log does **not** establish a unique causal mechanism (could be same-name temp conflict/external cleanup/file-system transient). Source P1 uses `with open(tmp,'wb')`, so do NOT ascribe this to numpy suffix auto-extension.

## FIX2 narrow guarded I/O patch — completed and tested

- Exact UTF-8 source `research/earth_aiwq/a15_p14_fix2/a15_p14_fix2_atomic_p1_adapter.py`; source SHA256 `1512c8f1f847f734e70f6c247bdde911425aacafd6d8a045f9b9de7977a984bf`, committed at `ae62448428f24a7d77516a25bb5d85d5392c015f` and fetched/byte-length + FNV match independently checked.
- Frozen original P1/P2/P11/P12 and P14 engine all remain **unchanged**, SHA verified before loading. Runtime patch applies only P1's checkpoint write and group I/O: `tempfile.mkstemp` unique per attempt in target directory, fsync, reopen/readback count/data/key/IDX SHA/ETag, then atomic `os.replace`, cleanup unique temp; serialize per (date,member,var) with POSIX `flock` to guard against stale-writer counts. Unrelated physics/truth/Gate paths unchanged.
- **No user reupload** for 2003 restored pair: user-facing complete Colab Notebook carries SHA-locked true 2,039,117-byte recovery ZIP in base64, automatically restores after verifying CRC, manifest+two file SHAs. All source/patch byte digests independently checked before real model IO. Do not commit the ~2MB user data or Notebook (contains derived fields) to PUBLIC GitHub/HF. This is intentionally ~2.84 MB notebook instead of ~100KB to avoid repeat uploads.
- `STAGE=pilot4` still four NEW 2003/2005/2008/2011 seasonal cases, 2003 complete signed pair **reused**, leaving three real cases (~1680 6h GRIB records) if source is unchanged. `STAGE=expand16` remains frozen four seasons each 2003/2005/2008/2011, 2014 validation and 2016 test strictly untouched.
- Scientific Gate unchanged: Pilot4 exploratory ONLY. True 16/16 prospective episodes from 4 new years required; variable W3 **AND** W4 outperform raw >=1%, global mean-bias + shifted-longitude placebo >=0.5%, with >=3/4 complete new-year folds. Even passing qualifies a static spatial residual benchmark candidate, NOT CIDM neural dynamics or official RPSS; no formal long training authorization.

## Independent QA
- Whole Notebook 5 code cells executed `CIDM_A15_P14_OFFLINE_QA=1`: PASS, state `OFFLINE_QA_ONLY_NOT_REAL_NEW_YEAR_RESULT`; no network, no fabricated weather.
- True 2003 pair and moments loaded from user-supplied bytes: exact field/NPZ/metadata SHA verified, frozen four-head RMSE reproduced to 1e-7. Fake modification rejected.
- Real-mode bootstrap emulated **without S3/Colab network**: signed recovery persisted, original source SHA guarded, P1/P2 identity checked, patch installed; second run is idempotent; corrupted weather source hash is BLOCKED.
- Contention tests confirm unique temporary files + verified replace under two writers, and per-member group serialization under four worker requests.
- New FIX2 **ONLINE** 2005/2008/2011 actual download/test **NOT RUN**. Runtime fix is qualified offline only.
- Standalone Notebook SHA256 `49f5f461295adc5283d5d03e13d1f3d7ce71b227ca1bbdd277529c02263f63eb`, full source QA ZIP `CIDM_EARTH_A15_P14_FIX2_SignedRecovery_AtomicCheckpoint_FULL.zip` SHA256 `f601f3cc6f02c30052406c701d21c5655ae84c4b24907bf7b976147cba7f9cc4`, ~2.26 MB. All ZIP CRC, per-file SHA manifest checked. Notebook can auto-export results ZIP+SHA and volatile paired-case recovery ZIP+SHA when needed.

## Persistent research boundaries
GEFSv12 archive is retrospective and is not proof of legal pre-issue-time input for operational ECMWF AI Weather Quest; current results are research global RMSE, not official MSLP-global/TAS-land RPSS. No neural CIDM small-pilot trained; no long training. Private HF archival requires explicit user credentials and private repository; this work makes no HF upload.
