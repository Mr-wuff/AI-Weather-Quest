#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A15-P14 FIX2: narrow, auditable atomic GEFS intermediate-cache write adapter.

Original frozen P1/P2/P11/P12/P14 file bytes remain unchanged and SHA-pinned.
This adapter is installed only after verifying the old P1 source digest.
The adapter does NOT alter the forecast, averaging, truth, scoring or Gate math.
"""
from __future__ import annotations
import fcntl
import os
import pathlib
import tempfile
import numpy as np

VERSION = 'A15_P14_FIX2_UNIQUE_TEMPFILE_AND_PER_GROUP_LOCK_V1'


def install(p1, p2):
    """Patch only P1 checkpoint I/O; fail closed if the P2 parent is unexpected."""
    if getattr(p2, 'parent', None) is not p1:
        raise RuntimeError('FIX2_READER_IDENTITY_MISMATCH')
    if getattr(p1, '_p14_fix2_installed', False):
        if getattr(p1, '_p14_fix2_version', None) != VERSION:
            raise RuntimeError('FIX2_PATCH_VERSION_MISMATCH')
        return {'status': 'ALREADY_INSTALLED', 'version': VERSION}

    original_group = p1.group_aggregate

    def unique_atomic_write_part(path, sums, counts, key, idx_sha, etag):
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix='.' + path.name + '.pid' + str(os.getpid()) + '.',
                                    suffix='.npz-tmp', dir=str(path.parent))
        temporary = pathlib.Path(name)
        try:
            with os.fdopen(fd, 'wb') as out:
                np.savez_compressed(out, sums=sums, counts=counts,
                                    key=key, idx_sha=idx_sha, etag=etag)
                out.flush()
                os.fsync(out.fileno())
            # Before replacing a previous good checkpoint, read back every key.
            with np.load(temporary, allow_pickle=False) as written:
                if not np.array_equal(written['counts'], np.asarray(counts)):
                    raise RuntimeError('FIX2_TEMP_COUNTS_MISMATCH')
                if not np.array_equal(written['sums'], np.asarray(sums)):
                    raise RuntimeError('FIX2_TEMP_SUMS_MISMATCH')
                for k, expected in (('key', key), ('idx_sha', idx_sha), ('etag', etag)):
                    if str(written[k]) != str(expected):
                        raise RuntimeError('FIX2_TEMP_SOURCE_FINGERPRINT_MISMATCH_' + k)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def serialized_group_aggregate(client, out, date, member, var, spec, required,
                                   save_every=4, decode_fn=None):
        out = pathlib.Path(out)
        lock_path = out / 'cache' / f'partial_{date}_{member}_{var}.npz.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        # A group-level advisory lock also prevents two simultaneous writers
        # from reading a stale earlier count and overwriting newer progress.
        with open(lock_path, 'a+b') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                if decode_fn is None:
                    return original_group(client, out, date, member, var,
                                          spec, required, save_every=save_every)
                return original_group(client, out, date, member, var,
                                      spec, required, save_every=save_every, decode_fn=decode_fn)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    p1.write_part = unique_atomic_write_part
    p1.group_aggregate = serialized_group_aggregate
    p1._p14_fix2_installed = True
    p1._p14_fix2_version = VERSION
    if p2.parent.group_aggregate is not serialized_group_aggregate:
        raise RuntimeError('FIX2_LIVE_GROUP_REFERENCE_UNPATCHED')
    return {'status': 'INSTALLED', 'version': VERSION,
            'changes': ['unique_same-directory_temporary', 'fsync_before_rename',
                        'source_and_counts_readback', 'per_member_variable_lock']}
