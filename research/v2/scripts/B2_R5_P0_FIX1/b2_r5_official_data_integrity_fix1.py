"""CIDM-v2 B2-R5/P0 FIX1: verifiable DaRUS file resolver and safe downloads.
This is data IO only. It NEVER changes benchmark TASKS, dataset MD5, model
weights, validation split, model forward, oracle gradients, or train budget.
"""
from __future__ import annotations
import os, re, json, time, hashlib, shutil
from pathlib import Path
import requests

HDF5_MAGIC=b'\x89HDF\r\n\x1a\n'

def file_md5(path):
    h=hashlib.md5()
    with open(path,'rb') as f:
        for piece in iter(lambda:f.read(4<<20),b''):
            h.update(piece)
    return h.hexdigest()

def log_event(audit, record):
    if audit is None:return
    audit.append(record)

def search_verified_previous(task, name, expected, dest, old_dirs, audit):
    seen=set()
    for directory in old_dirs:
        p=Path(directory).expanduser()
        candidates=[p/name]
        if p.is_file() and p.name==name:candidates=[p]
        for candidate in candidates:
            if str(candidate) in seen:continue
            seen.add(str(candidate))
            if not candidate.is_file():continue
            actual=file_md5(candidate)
            log_event(audit,dict(task=task,stage='candidate',file=str(candidate),md5=actual,expected=expected,bytes=candidate.stat().st_size,pass_md5=(actual==expected)))
            if actual!=expected:continue
            if candidate.resolve()!=dest.resolve():
                dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists() or dest.is_symlink():
                    dest.unlink()
                try:
                    os.link(candidate,dest)
                    linktype='hardlink'
                except OSError:
                    dest.symlink_to(candidate.resolve())
                    linktype='symlink'
            else:linktype='same'
            log_event(audit,dict(task=task,stage='verified_reuse',file=str(candidate),reuse=linktype))
            return dest
    return None

def strict_download(url, part, *, session, allow_resume=True, audit=None, task='unknown'):
    part=Path(part);pos=part.stat().st_size if (allow_resume and part.exists()) else 0
    headers={'Accept-Encoding':'identity','User-Agent':'CIDM-v2-PDEBench-integrity/1.0'}
    if pos:headers['Range']=f'bytes={pos}-'
    with session.get(url,stream=True,timeout=(30,180),headers=headers,allow_redirects=True) as r:
        r.raise_for_status()
        code=r.status_code
        if code not in (200,206):raise RuntimeError(f'UNEXPECTED_HTTP_STATUS_{code}')
        cr=r.headers.get('Content-Range','')
        if code==206:
            match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+|\*)',cr)
            if not match:raise RuntimeError(f'INVALID_HTTP_CONTENT_RANGE:{cr}')
            start,end,total=match.groups()
            if int(start)!=pos or int(end)<int(start):
                raise RuntimeError(f'RANGE_OFFSET_MISMATCH expected={pos} got={cr}')
            expected_chunk=int(end)-int(start)+1
        elif code==200:
            # Server ignored Range. Reset before writing rather than appending!
            pos=0;expected_chunk=None
        clen=r.headers.get('Content-Length')
        if clen and int(clen)!=expected_chunk and expected_chunk is not None:
            raise RuntimeError(f'HTTP_CONTENT_LENGTH_RANGE_MISMATCH {clen}/{expected_chunk}')
        written=0
        with open(part,'ab' if pos else 'wb') as f:
            for buf in r.iter_content(chunk_size=4<<20):
                if buf:
                    f.write(buf);written+=len(buf)
        if clen and written!=int(clen):
            raise RuntimeError(f'INCOMPLETE_DOWNLOAD_BYTES {written}/{clen}')
        if expected_chunk is not None and written!=expected_chunk:
            raise RuntimeError(f'INCOMPLETE_HTTP_RANGE {written}/{expected_chunk}')
        if code==206 and total!='*' and part.stat().st_size!=int(total):
            raise RuntimeError(f'HTTP_RANGE_FINAL_SIZE_MISMATCH {part.stat().st_size}/{total}')
        log_event(audit,dict(task=task,stage='http_download',status=code,range=cr,content_length=clen,bytes_written=written,total_bytes=part.stat().st_size,url=url))
    return part

def fetch_official_verified(root,task,*, tasks_spec,previous_dirs=(),audit=None,session=None,max_fresh_attempts=1):
    if task not in tasks_spec:raise RuntimeError(f'UNKNOWN_OFFICIAL_TASK {task}')
    name,fileid,expected,_=tasks_spec[task]
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    dest=root/name
    dirs=[root,*[Path(p) for p in previous_dirs]]
    existing=search_verified_previous(task,name,expected,dest,dirs,audit)
    if existing:return existing
    # Discard corrupted dest only after recording original checksum; no false positives.
    if dest.exists() or dest.is_symlink():
        log_event(audit,dict(task=task,stage='reject_invalid_destination',file=str(dest)))
        dest.unlink()
    part=Path(str(dest)+'.part')
    url=f'https://darus.uni-stuttgart.de/api/access/datafile/{fileid}'
    if session is None:session=requests.Session()
    # Pass 0 resumes existing .part if applicable; on MD5 mismatch delete and do ONE fresh pass.
    last=None
    for pass_index in range(1+max_fresh_attempts):
        if pass_index>0:
            part.unlink(missing_ok=True)
        for attempt in range(3):
            try:
                strict_download(url,part,session=session,allow_resume=(pass_index==0),audit=audit,task=task)
                break
            except (requests.RequestException,OSError,RuntimeError) as e:
                last=repr(e)
                log_event(audit,dict(task=task,stage='retry_http',pass_index=pass_index,attempt=attempt,error=last))
                if attempt==2:raise RuntimeError(f'DOWNLOAD_INTEGRITY_BLOCKED_{task}: {last}') from e
                time.sleep(min(2*(attempt+1),5))
        if not part.is_file():raise RuntimeError('NO_DOWNLOADED_FILE_'+task)
        actual=file_md5(part)
        magic=part.open('rb').read(8)
        good=(actual==expected and magic==HDF5_MAGIC)
        log_event(audit,dict(task=task,stage='final_md5',pass_index=pass_index,bytes=part.stat().st_size,md5=actual,expected=expected,hdf5_magic=magic.hex(),pass_integrity=good))
        if good:
            part.replace(dest)
            return dest
        last=f'MD5_OR_HDF5_MISMATCH actual={actual} expected={expected}, size={part.stat().st_size}, magic={magic.hex()}'
    part.unlink(missing_ok=True)
    raise RuntimeError(f'BLOCKED_OFFICIAL_DATA_INTEGRITY_{task}: {last}; no training or oracle permitted')
