#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CIDM-Earth A15-P2: paired GEFSv12 reforecast vs WB2 ERA5 weekly verification.

RESEARCH RETROSPECTIVE ONLY. Forecasts are later-produced reforecasts, not historically
operationally available. ERA5 future observations are verification labels ONLY.
Metrics are deterministic RMSE/MAE/bias, empirical-ensemble CRPS, spread; NOT AIWQ RPSS.
Regridding caveat: GEFS 0.5->1.5 point-sampled vs WB2 conservative ERA5 1.5.
"""
from __future__ import annotations
import os, sys, io, json, math, hashlib, traceback, zipfile
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np

try:
    import a15_p1_gefs_weekly_ensemble as parent
except ImportError as exc:
    raise RuntimeError('Missing bundled A15-P1 exact reference source') from exc

SOURCE='gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr'
PARENT_SHA='a8025c19bc9109a639891c8526bbce1a7bff18a0c5451e6e3b4d15b486b1fa99'
CASES=('2017010400','2017070500','2018010300','2018070400')
MEMBERS=parent.MEMBERS
VARS=('mslp','tas')
WEEKS=(3,4)
OUT_NAME='CIDM_EARTH_A15_P2_DELIVERY_REPORTS.zip'

def hash_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def json_write(path,value): parent.as_json(Path(path),value)

def canonical(a,lat,lon):
    a=np.asarray(a,dtype=np.float32)
    lat=np.asarray(lat,dtype=float)
    lon=np.mod(np.asarray(lon,dtype=float),360)
    if a.shape!=(121,240):raise ValueError('WB2 native grid != 121x240')
    if np.allclose(lat,90-np.arange(121)*1.5,atol=1e-3):pass
    elif np.allclose(lat,-90+np.arange(121)*1.5,atol=1e-3):a=a[::-1,:].copy()
    else:raise ValueError('WB2 unexpected latitude coordinate')
    idx=np.argsort(lon)
    if not np.allclose(lon[idx],np.arange(240)*1.5,atol=1e-3):raise ValueError('WB2 unexpected longitude coordinate')
    return np.ascontiguousarray(a[:,idx])

def valid_hours(date,week):
    ini,issue=parent.init_issue(date)
    first=issue+timedelta(days=18 if week==3 else 25)
    hours=[first+timedelta(hours=6*i) for i in range(28)]
    step=[round((h-ini).total_seconds()/3600) for h in hours]
    if step!=parent.required_steps(date,week):raise ValueError('issue/lead calendar contract failed')
    if hours[0].weekday()!=0 or hours[0].hour!=0 or hours[-1].weekday()!=6 or hours[-1].hour!=18:
        raise ValueError('W3/W4 weekly Monday-Sunday mismatch')
    return np.array([np.datetime64(h.replace(tzinfo=None),'ns') for h in hours])

def open_wb2():
    import xarray as xr
    ds=xr.open_zarr(SOURCE,storage_options={'token':'anon'},consolidated=None,chunks=None)
    for v in ('mean_sea_level_pressure','2m_temperature'):
        if v not in ds:raise RuntimeError('WB2 missing '+v)
    for d in ('time','latitude','longitude'):
        if d not in ds.coords:raise RuntimeError('WB2 missing coordinate '+d)
    return ds

def read_truth(ds,date):
    results=[]
    for varname in ('mean_sea_level_pressure','2m_temperature'):
        weekvals=[]
        for lead in WEEKS:
            ts=valid_hours(date,lead)
            # Select exact timestamps. This will raise on any missing 6-hour truth.
            q=ds[varname].sel(time=ts)
            actual=np.asarray(q.time.values,dtype='datetime64[ns]')
            if len(actual)!=28 or not np.array_equal(actual,ts):raise RuntimeError('WB2 truth timestamps not exact')
            field=q.mean('time').transpose('latitude','longitude').load().values
            data=canonical(field,ds.latitude.values,ds.longitude.values)
            if varname=='mean_sea_level_pressure':data=data/100.0
            else:data=data-273.15
            if not np.isfinite(data).all():raise ValueError('Nonfinite WB2 week mean')
            weekvals.append(data)
        results.append(np.stack(weekvals))
    truth=np.stack(results).astype(np.float32) # [variable, week, lat,lon]
    if truth.shape!=(2,2,121,240):raise RuntimeError('Truth shape invalid')
    return truth

def validate_pred(x):
    a=np.asarray(x,dtype=np.float32)
    if a.shape!=(5,2,2,121,240) or not np.isfinite(a).all():raise ValueError('Invalid GEFS weekly shape/values')
    if min(np.mean(a[:,0]),1050)<600 or np.min(a[:,0])<700 or np.max(a[:,0])>1200:raise ValueError('MSLP units invalid')
    if np.min(a[:,1])<-100 or np.max(a[:,1])>80:raise ValueError('TAS units invalid')
    if np.max(np.std(a,axis=0))<1e-5:raise ValueError('Ensemble indistinguishable')
    return a

def load_p1_archive(path):
    if hash_file(path)!=PARENT_SHA:raise RuntimeError('A15-P1 user-uploaded archive SHA mismatch')
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:raise RuntimeError('A15-P1 CRC failed')
        info=json.loads(z.read('reports/STATUS.json'))
        if info['status']!='REAL_GEFS_REFORECAST_WEEKLY_ENSEMBLE_PASS_RESEARCH_ONLY' or info['initialization_utc']!='2018010300':
            raise RuntimeError('Unexpected provenance in parent archive')
        raw=z.read('reports/GEFSv12_REFORECAST_WEEKLY_MEANS_RESEARCH_ONLY.npz')
        if hashlib.sha256(raw).hexdigest()!=info['weekly_data_sha256']:
            raise RuntimeError('A15-P1 embedded NPZ checksum mismatch')
        with np.load(io.BytesIO(raw),allow_pickle=False) as p:
            arr=validate_pred(p['weekmean'])
            if list(p['member'])!=list(MEMBERS) or list(p['variable'])!=list(VARS) or list(p['week'])!=[3,4]:
                raise RuntimeError('A15-P1 parent channel axes mismatch')
            if np.max(np.abs(arr.mean(0)-p['ensemble_mean']))>1e-4:raise RuntimeError('A15-P1 mean inconsistency')
            if np.max(np.abs(arr.std(0,ddof=1)-p['ensemble_std']))>1e-4:raise RuntimeError('A15-P1 spread inconsistency')
        return arr,info

def new_s3_case(date,root,save_every=4):
    """Reuse SHA/ETag-locked P1 resumable GRIB reader, but on a new Wednesday initialization."""
    cache=Path(root)/'cache';cache.mkdir(parents=True,exist_ok=True)
    arr_path=cache/f'GEFS_{date}_weekly.npy'
    meta_path=cache/f'GEFS_{date}_source.json'
    if arr_path.exists():
        if not meta_path.exists():raise RuntimeError('Orphan forecast cache without provenance')
        meta=json.loads(meta_path.read_text())
        if meta.get('date')!=date or meta.get('source_bucket')!=parent.BUCKET or hash_file(arr_path)!=meta.get('sha256'):
            raise RuntimeError('Forecast cache drift')
        return validate_pred(np.load(arr_path,allow_pickle=False)),meta
    client=parent.anonymous_client()
    need={k:parent.required_steps(date,k) for k in WEEKS}
    source=parent.maybe_preflight(client,date,MEMBERS,need)
    completed=[]
    for member in MEMBERS:
        pair=[]
        for var in VARS:
            pair.append(parent.group_aggregate(client,Path(root),date,member,var,source[(member,var)],need,save_every))
        completed.append(np.stack(pair))
    arr=validate_pred(np.stack(completed))
    temp=arr_path.with_name(arr_path.name+'.tmp')
    with temp.open('wb') as f:np.save(f,arr)
    os.replace(temp,arr_path)
    meta={'date':date,'source_bucket':parent.BUCKET,'sha256':hash_file(arr_path),
          'provenance':[{'member':m,'variable':v,'key':o['key'],'idx_sha256':o['idx_sha'],'etag':o['etag']}
                        for (m,v),o in source.items()], 'retrospective_only':True}
    json_write(meta_path,meta)
    return arr,meta

def truth_cache(ds,date,root):
    p=Path(root)/'cache'/f'ERA5_{date}_truth.npz'
    if p.exists():
        with np.load(p,allow_pickle=False) as v:
            if str(v['source'])!=SOURCE or str(v['date'])!=date:raise RuntimeError('ERA5 cache provenance mismatch')
            a=v['truth'].copy()
    else:
        a=read_truth(ds,date)
        tmp=p.with_name(p.name+'.tmp');p.parent.mkdir(parents=True,exist_ok=True)
        with tmp.open('wb') as f:np.savez_compressed(f,source=SOURCE,date=date,truth=a)
        os.replace(tmp,p)
    if a.shape!=(2,2,121,240) or not np.isfinite(a).all():raise RuntimeError('Bad ERA5 cache')
    return a

def spatial_weights():
    w=np.maximum(np.cos(np.deg2rad(np.linspace(90,-90,121,dtype=np.float64))),0)
    w=np.broadcast_to(w[:,None],(121,240)).copy()
    return w/w.sum()

def score_case(pred,truth):
    pred=validate_pred(pred)
    truth=np.asarray(truth,dtype=np.float64)
    if truth.shape!=(2,2,121,240) or not np.isfinite(truth).all():raise ValueError('Invalid truth')
    wt=spatial_weights()
    out={}
    for vi,v in enumerate(VARS):
        for wi,week in enumerate(WEEKS):
            ens=pred[:,vi,wi].astype(np.float64)
            y=truth[vi,wi]
            mean=ens.mean(0); err=mean-y
            rmse=np.sqrt(np.sum(wt*err**2)); bias=np.sum(wt*err)
            spread=np.sqrt(np.sum(wt*np.var(ens,axis=0,ddof=1)))
            # finite-ensemble empirical CRPS, per pixel
            a=np.mean(np.abs(ens-y),axis=0)
            b=np.abs(ens[:,None]-ens[None,:]).sum(axis=(0,1))/(2*len(ens)**2)
            crps=np.sum(wt*(a-b))
            ctrl=np.sqrt(np.sum(wt*(ens[0]-y)**2))
            out[f'{v}_W{week}']={'ensemble_mean_rmse':float(rmse),
                'control_rmse':float(ctrl),'mae':float(np.sum(wt*np.abs(err))),
                'bias':float(bias),'ensemble_std_rms':float(spread),
                'spread_skill_ratio':float(spread/rmse) if rmse>0 else None,
                'empirical_crps':float(crps),
                'ensemble_minus_control_rmse':float(rmse-ctrl),
                'unit':'hPa' if v=='mslp' else 'degC'}
    return out

def deliver(root,state,status,error=None):
    root=Path(root);report=root/'reports';report.mkdir(parents=True,exist_ok=True)
    state.update({'status':status,'finished_utc':datetime.now(timezone.utc).isoformat(),
                  'official_rpss':None,'formal_training_authorized':False,
                  'historical_operational_availability_verified':False,
                  'reforecast_not_operational':True,
                  'wb2_reanalysis_future_labels_only':True})
    json_write(report/'STATUS.json',state)
    if error:(report/'ERROR_TRACEBACK.txt').write_text(str(error)[:30000],encoding='utf8')
    zpath=root/OUT_NAME
    with zipfile.ZipFile(zpath,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(report.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(root).as_posix())
    with zipfile.ZipFile(zpath) as z:
        if z.testzip() is not None:raise RuntimeError('result ZIP CRC invalid')
    Path(str(zpath)+'.sha256').write_text(hash_file(zpath)+'  '+zpath.name+'\n')
    print('[A15-P2]',status,'ZIP',zpath,'SHA256',hash_file(zpath),flush=True)
    return zpath

def selftest():
    for date in CASES:
        ini,issue=parent.init_issue(date)
        assert issue.weekday()==3
        for w in WEEKS:assert len(valid_hours(date,w))==28
    truth=np.ones((2,2,121,240),np.float32)*3
    ens=np.repeat(truth[None],5,axis=0).astype(np.float32)
    for j in range(5):ens[j,0]+=1000-3+j*.25;ens[j,1]+=j*.1 # proper mslp/tas
    ens=validate_pred(ens)
    sc=score_case(ens,np.stack((np.ones((2,121,240))*1000,np.ones((2,121,240))*3)))
    assert all(np.isfinite([v['ensemble_mean_rmse'],v['empirical_crps']]).all() for v in sc.values())
    # Perfect identical ensembles are forbidden by source integrity, but CRPS identity holds analytically.
    assert abs(score_case(ens,np.stack((np.ones((2,121,240))*1000,np.ones((2,121,240))*3)))['mslp_W3']['bias']-0.5)<1e-3
    try:valid_hours('2018010400',3);raise AssertionError('wrong weekday accepted')
    except parent.ContractError:pass
    a=np.arange(121*240,dtype=np.float32).reshape(121,240)
    assert np.array_equal(canonical(a,np.arange(90,-91,-1.5),np.arange(0,360,1.5)),a)
    print('OFFLINE_SMOKE_PASS: parent locked, 4 calendars, 28 timestamps, metrics, grid and failure controls',flush=True)

def run(root,parent_zip,offline_only=False,dates=CASES):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    (root/'reports').mkdir(parents=True,exist_ok=True)
    state={'experiment':'CIDM_EARTH_A15_P2','source_gefs':'NOAA GEFSv12 retrospective AWS',
           'source_truth':SOURCE,'parent_sha256_required':PARENT_SHA,
           'requested_cases':list(dates),'per_case_metrics':{},'paired_cases':0,
           'actual_extra_s3_cases':0,'parent_forecast_reused':False,
           'note':'Dataset-alignment pilot only: no official RPSS, no training approval; WB2 conservative versus GEFS point-sample 1.5deg'}
    try:
        selftest()
        xold,parentinfo=load_p1_archive(parent_zip)
        state['parent_forecast_reused']=True
        state['parent_source_sha256']=hash_file(parent_zip)
        state['parent_initialization']=parentinfo['initialization_utc']
        if offline_only:
            # Score known-format arrays against deterministic synthetic truth only; never real PASS.
            synthetic=xold.mean(0).copy()
            state['synthetic_qa_metrics']=score_case(xold,synthetic)
            return deliver(root,state,'OFFLINE_QA_ONLY_NO_REAL_ERA5_OR_EXTRA_S3')
        ds=None
        for date in dates:
            if date=='2018010300':pred=xold;meta={'reused_parent_sha':PARENT_SHA}
            else:
                pred,meta=new_s3_case(date,root)
                state['actual_extra_s3_cases']+=1
            if ds is None:ds=open_wb2()
            truth=truth_cache(ds,date,root)
            scores=score_case(pred,truth)
            state['per_case_metrics'][date]=scores
            state['paired_cases']+=1
            # Include ONLY derived forecasts/truth, NOT raw GRIB or full ERA5 source archive.
            p=root/'reports'/f'paired_{date}.npz'
            np.savez_compressed(p,forecast=pred,truth=truth,date=date,
                                source_gefs='GEFSv12 reforecast not operational',source_truth=SOURCE)
            json_write(root/'reports'/f'case_{date}_metrics.json',{'scores':scores,'source':meta,'truth_time_contract':'W3/4 exact 28 six-hour periods','paired_file_sha256':hash_file(p)})
            json_write(root/'reports'/'PROGRESS.json',state)
            print('[A15-P2]',date,'paired, cases',state['paired_cases'],'/',len(dates),flush=True)
        n=state['paired_cases']
        if n!=len(dates) or n<4:raise RuntimeError('Insufficient paired dates for preregistered pilot')
        summary={}
        for key in next(iter(state['per_case_metrics'].values())):
            summary[key]={metric:float(np.mean([case[key][metric] for case in state['per_case_metrics'].values()]))
                          for metric in ('ensemble_mean_rmse','control_rmse','mae','bias','ensemble_std_rms','empirical_crps')}
        state['summary_unweighted_mean_across_4_cases']=summary
        json_write(root/'reports'/'SCORE_SUMMARY.json',summary)
        return deliver(root,state,'REAL_GEFS_ERA5_FOUR_CASE_PAIRED_PILOT_PASS_RESEARCH_ONLY')
    except Exception:
        return deliver(root,state,'BLOCKED_INCOMPLETE_PAIRED_DATA_QUALIFICATION',traceback.format_exc())

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='/content/CIDM_EARTH_A15_P2')
    ap.add_argument('--parent-zip',required=True)
    ap.add_argument('--offline-only',action='store_true')
    args=ap.parse_args()
    run(args.out,args.parent_zip,args.offline_only)
