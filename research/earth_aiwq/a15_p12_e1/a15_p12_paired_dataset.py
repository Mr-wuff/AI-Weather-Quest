#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A15-P12 G2-E0: verified real GEFSv12 -> WB2 ERA5 paired weekly data.

RESEARCH ONLY, NOT a real-time operational input, competition eligibility, official
RPSS, or a trained CIDM experiment. ERA5 is used strictly as verification label.
Build 4 distinct TRAIN-only pilot dates first; optionally extend to 16 train
issues. 2014 validation and 2016 untouched test cannot be accessed in this stage.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
import sys
import traceback
import zipfile
from pathlib import Path
from collections import Counter
import numpy as np

PROJECT='CIDM_EARTH_A15_P12_G2_E0'
STATUS_NAME='CIDM_EARTH_A15_P12_DELIVERY_REPORTS.zip'
YEARS_TRAIN=(2002,2004,2006,2010)
YEARS_VALID=(2014,)
YEARS_TEST=(2016,)
MONTHS=(1,4,7,10)
MEMBERS=('c00','p01','p02','p03','p04')
VARIABLES=('mslp','tas')
P1_SHA='4e88bee38ccce563d9cfd75e457e9a35070439b972dc7fc742a9e4b23520a98e'
P2_SHA='27d5cec94ff9f2eb7464a1fb8a8707339a875f2636282b4187725f7788e85659'
P11_SHA='3afd20ea9cdab5ceb8025580cadbd148a3cc5e10fbad8b77c85bf495e2e217fb'

class GateError(RuntimeError):pass

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf8')
    os.replace(tmp,path)

def first_wed(year,month):
    d=dt.date(year,month,1)
    return (d+dt.timedelta(days=(2-d.weekday())%7)).strftime('%Y%m%d')+'00'

def split_contract():
    return {'train':[first_wed(y,m) for y in YEARS_TRAIN for m in MONTHS],
            'validation':[first_wed(y,m) for y in YEARS_VALID for m in MONTHS],
            'test':[first_wed(y,m) for y in YEARS_TEST for m in MONTHS]}

def selected(stage):
    splits=split_contract()
    if stage=='pilot4':
        return [first_wed(2002,1),first_wed(2004,4),first_wed(2006,7),first_wed(2010,10)]
    if stage=='train16':return splits['train']
    raise GateError('INVALID_STAGE_MUST_BE_pilot4_OR_train16')

def load_deps(source_dir):
    from importlib import util
    source_dir=Path(source_dir)
    for name,expected in [('a15_p1_gefs_weekly_ensemble.py',P1_SHA),
                          ('a15_p2_gefs_era5_paired.py',P2_SHA),
                          ('a15_p11_fix1_full_ensemble_gate.py',P11_SHA)]:
        p=source_dir/name
        if not p.is_file() or sha(p)!=expected:raise GateError('PINNED_DEPENDENCY_MISSING_OR_SHA_MISMATCH:'+name)
    if str(source_dir) not in sys.path:sys.path.insert(0,str(source_dir))
    import a15_p1_gefs_weekly_ensemble as p1
    import a15_p2_gefs_era5_paired as p2
    spec=util.spec_from_file_location('a15_p11_fixed_for_p12',source_dir/'a15_p11_fix1_full_ensemble_gate.py')
    m=util.module_from_spec(spec);spec.loader.exec_module(m)
    return p1,p2,m

def check_live_source(p1,p11,date,client):
    verified={}
    for member in MEMBERS:
        for var in VARIABLES:
            source=p11.check_one(client,p1,date,member,var)
            if source.get('status')!='PASS' or source['idx_init']['mode']!='YYYYMMDDHH':
                raise GateError('P11_SOURCE_HOUR_OR_IDX_NOT_VERIFIED:'+date+'/'+member+'/'+var)
            verified[(member,var)]=source
    return verified

def check_forecast_sources(source_meta,verified):
    got={(o['member'],o['variable']):o for o in source_meta.get('provenance',[])}
    if set(got)!=set(verified):raise GateError('FORECAST_PROVENANCE_GROUPS_MISMATCH')
    for k,spec in verified.items():
        item=got[k]
        if item.get('key')!=spec['key'] or item.get('idx_sha256')!=spec['idx_sha256'] or item.get('etag')!=spec['etag']:
            raise GateError('CACHED_FORECAST_SOURCE_DRIFT:'+str(k))

def validate_case(pred,truth,p2):
    pred=p2.validate_pred(pred)
    truth=np.asarray(truth,dtype=np.float32)
    if truth.shape!=(2,2,121,240) or not np.isfinite(truth).all():raise GateError('TRUTH_BAD_SHAPE_OR_NAN')
    if not np.isfinite(pred).all() or pred.shape!=(5,2,2,121,240):raise GateError('FORECAST_BAD_SHAPE')
    return pred,truth

def save_case(root,date,pred,truth,meta,score,p2,source_meta):
    root=Path(root);cases=root/'cases';cases.mkdir(parents=True,exist_ok=True)
    path=cases/f'paired_{date}.npz'
    if path.exists():
        existing=cases/f'paired_{date}.json'
        if existing.exists() and sha(path)!=json.loads(existing.read_text())['paired_sha256']:
            raise GateError('EXISTING_PAIRED_CACHE_DRIFT:'+date)
        # A crash between atomic NPZ write and JSON write leaves an orphan NPZ.
        # Compare every array with the freshly provenance-validated case before resuming.
        with np.load(path,allow_pickle=False) as data:
            if str(data['init'])!=date or str(data['source_truth'])!=p2.SOURCE:
                raise GateError('EXISTING_PAIRED_CONTRACT_DRIFT')
            if not np.array_equal(data['forecast'],pred) or not np.array_equal(data['truth'],truth):
                raise GateError('EXISTING_PAIRED_ARRAY_DRIFT')
    else:
        tmp=path.with_suffix('.npz.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f,forecast=pred,truth=truth,init=date,
                                units=np.array(['hPa','degC']),member=np.array(MEMBERS),
                                var=np.array(VARIABLES),week=np.array([3,4]),
                                source_truth=p2.SOURCE,
                                source_gefs='NOAA GEFSv12 later-produced retrospective reforecast')
        os.replace(tmp,path)
    card={'init':date,'pair_file':path.name,'paired_sha256':sha(path),
          'forecast_sha256':hashlib.sha256(pred.tobytes()).hexdigest(),
          'truth_sha256':hashlib.sha256(truth.tobytes()).hexdigest(),
          'source':meta,'source_checks':source_meta,'scores':score,
          'warning':'Forecast reforecast NOT operational, ERA5 future LABEL only, all-land MSLP/TAS research metrics not official RPSS'}
    atomic_json(cases/f'paired_{date}.json',card)
    return card

def build_result(root,stage,done,notes,status,error=None):
    root=Path(root);rep=root/'reports';rep.mkdir(parents=True,exist_ok=True)
    for p in rep.iterdir():
        if p.is_file():p.unlink()
    all_dates=split_contract()
    state={'phase':PROJECT,'status':status,'stage':stage,'selected_dates':selected(stage),
           'completed_dates':sorted(done),'completed_n':len(done),'split_contract':all_dates,
           'date_manifest':done,'diagnostic':notes,'error':error,
           'real_gefs_era5_paired':status.startswith('REAL_'),
           'qualification_scope':'research real field pairing only; not S2S probabilistic training',
           'complete_training_dataset':False,
           'long_training_authorized':False,
           'official_rpss':None,
           'competition_operational_source_verified':False}
    atomic_json(rep/'STATUS.json',state)
    zpath=root/STATUS_NAME
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(rep.rglob('*')):z.write(f,f.relative_to(root).as_posix())
        for date in sorted(done):
            for name in (f'paired_{date}.npz',f'paired_{date}.json'):
                file=root/'cases'/name
                if not file.is_file():raise GateError('PACKAGING_PAIRED_CASE_MISSING')
                z.write(file,'cases/'+name)
    with zipfile.ZipFile(zpath) as z:
        if z.testzip() is not None:raise GateError('OUTPUT_CRC_FAILED')
    side=zpath.with_suffix(zpath.suffix+'.sha256')
    side.write_text(sha(zpath)+'  '+zpath.name+'\n',encoding='utf8')
    return zpath,side,state

def real_run(root,src,stage='pilot4',offline=False,save_every=4):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    done={};notes={};err=None
    targets=selected(stage)
    try:
        p1,p2,p11=load_deps(src)
        if offline:
            notes['offline_qa_only']=True
            return build_result(root,stage,done,notes,'OFFLINE_QA_ONLY_NO_REMOTE_WEATHER')
        client=p1.anonymous_client()
        ds=None
        for j,date in enumerate(targets,1):
            print(f'[P12] {j}/{len(targets)} {date} source strict-preflight',flush=True)
            live=check_live_source(p1,p11,date,client)
            print(f'[P12] {date} 10/10 strict GEFS IDX PASS; decoding W3/W4',flush=True)
            pred,meta=p2.new_s3_case(date,root,save_every=save_every)
            check_forecast_sources(meta,live)
            if ds is None:ds=p2.open_wb2()
            truth=p2.truth_cache(ds,date,root)
            pred,truth=validate_case(pred,truth,p2)
            score=p2.score_case(pred,truth)
            source_ref={m+'_'+v:{'key':x['key'],'idx_sha256':x['idx_sha256'],'etag':x['etag']}
                        for (m,v),x in live.items()}
            done[date]=save_case(root,date,pred,truth,meta,score,p2,source_ref)
            print(f'[P12] {date} PAIR_PASS {j}/{len(targets)}',flush=True)
            atomic_json(root/'progress.json',{'stage':stage,'completed_dates':sorted(done),'last_date':date})
        status='REAL_PILOT4_PAIRED_PASS_RESEARCH_ONLY' if stage=='pilot4' else 'REAL_TRAIN16_PAIRED_PASS_RESEARCH_ONLY'
        return build_result(root,stage,done,notes,status)
    except Exception:
        err={'type':'EXCEPTION','traceback':traceback.format_exc()[-9000:]}
        status='BLOCKED_SOURCE_OR_ERA5_PAIRING_RESEARCH_ONLY'
        return build_result(root,stage,done,notes,status,err)

def selftest(src):
    p1,p2,p11=load_deps(src)
    splits=split_contract()
    assert len(splits['train'])==16 and len(splits['validation'])==4 and len(splits['test'])==4
    assert not(set(splits['train'])&set(splits['validation'])) and not(set(splits['validation'])&set(splits['test']))
    assert all(x.endswith('00') for x in splits['train']+splits['validation']+splits['test'])
    assert len(selected('pilot4'))==4 and set(selected('pilot4'))<=set(splits['train'])
    assert all(p1.required_steps(x,3)==list(range(456,624,6)) for x in selected('pilot4'))
    for x in selected('pilot4'):
        assert len(p2.valid_hours(x,3))==28 and len(p2.valid_hours(x,4))==28
    print('[P12] OFFLINE_SELFTEST_SPLITS_AND_TIME_CONTRACT_PASS',flush=True)
    return True

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='./CIDM_EARTH_A15_P12')
    parser.add_argument('--sources',default='.')
    parser.add_argument('--stage',choices=('pilot4','train16'),default='pilot4')
    parser.add_argument('--offline',action='store_true')
    args=parser.parse_args()
    selftest(args.sources)
    out=real_run(args.root,args.sources,stage=args.stage,offline=args.offline)
    print('[P12]',out[2]['status'],out[0],out[1])
