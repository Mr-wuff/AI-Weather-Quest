#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CIDM-Earth A15-P3: frozen GEFS–ERA5 four-case research probabilistic baseline.

No CIDM training; historic GEFS reforecasts are NOT 2017/18 operational forecasts.
No claim of official ECMWF RPSS: historical proxy bins are separately trained.
"""
from __future__ import annotations
import io, os, json, math, zipfile, hashlib, traceback, argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
import numpy as np

INPUT_SHA='dd380fb310546b080314ecb8ad78a2edbecad68413ac21f7696ea6a9feda486c'
SUBMITTED_SIDECAR_SHA='91292d2cc1e6a47b5a0eb031cddb12b360967bb7fa221b28be8fe95602e5b75d'
SOURCE='gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr'
TRAIN_YEARS=tuple(range(2008,2017))
WEEK_OFFSETS=(-1,0,1)
TEST_CASES=('2017010400','2017070500','2018010300','2018070400')
SEASONS=('jan','jul')
VARIABLES=('mslp','tas')
LEADS=(3,4)
N_HIST=len(TRAIN_YEARS)*len(WEEK_OFFSETS)
OUT_NAME='CIDM_EARTH_A15_P3_DELIVERY_REPORTS.zip'

def sha(b):return hashlib.sha256(b).hexdigest()
def sha_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write_json(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8');os.replace(tmp,p)
def issue_datetime(date):return datetime.strptime(date,'%Y%m%d%H')+timedelta(days=1)
def first_wed(year,month):
    t=datetime(year,month,1)
    return t+timedelta(days=(2-t.weekday())%7)
def sample_init(year,season,week_offset=0):
    return first_wed(year,1 if season=='jan' else 7)+timedelta(weeks=week_offset)
def valid_times(initialization,week):
    issue=initialization+timedelta(days=1)
    start=issue+timedelta(days=18 if week==3 else 25)
    ans=[start+timedelta(hours=6*i) for i in range(28)]
    assert ans[0].weekday()==0 and ans[-1].weekday()==6 and ans[-1].hour==18
    return np.array(ans,dtype='datetime64[ns]')
def weights_mask(lo=0,hi=90):
    lat=90-np.arange(121)*1.5
    w=np.broadcast_to(np.maximum(0,np.cos(np.deg2rad(lat)))[:,None],(121,240)).astype(np.float64).copy()
    sel=(np.abs(lat)>=lo)&(np.abs(lat)<hi)
    if hi==90:sel=sel|(np.abs(lat)==90)
    w[~sel,:]=0
    if w.sum()<=0:raise ValueError('empty zone')
    return w/w.sum()
def latitude_zone_scores(err):
    return {name:float(np.sqrt(np.sum(weights_mask(*limit)*err*err))) for name,limit in
            {'tropics':(0,23.5),'midlatitudes':(23.5,60),'polar':(60,90)}.items()}
def load_input(p):
    p=Path(p)
    real=sha_file(p)
    if real!=INPUT_SHA:raise RuntimeError('Uploaded input byte digest changed: '+real)
    with zipfile.ZipFile(p) as z:
        if z.testzip():raise RuntimeError('CRC invalid')
        s=json.loads(z.read('reports/STATUS.json'))
        if s['status']!='REAL_GEFS_ERA5_FOUR_CASE_PAIRED_PILOT_PASS_RESEARCH_ONLY':raise RuntimeError('Not genuine completed P2 data')
        if s['paired_cases']!=4 or set(s['requested_cases'])!=set(TEST_CASES):raise RuntimeError('Case contract mismatch')
        results={}
        for d in TEST_CASES:
            body=z.read(f'reports/paired_{d}.npz');manifest=json.loads(z.read(f'reports/case_{d}_metrics.json'))
            if sha(body)!=manifest['paired_file_sha256']:raise RuntimeError('Per-case digest invalid '+d)
            with np.load(io.BytesIO(body),allow_pickle=False) as a:
                pred=a['forecast'].copy();truth=a['truth'].copy();date=str(a['date'])
            if date!=d or pred.shape!=(5,2,2,121,240) or truth.shape!=(2,2,121,240):raise ValueError('Shape or date invalid')
            if not np.isfinite(pred).all() or not np.isfinite(truth).all():raise ValueError('Not finite')
            results[d]=(pred,truth)
        return results

def canonical(a,lat,lon):
    a=np.asarray(a,dtype=np.float32);lat=np.asarray(lat,dtype=np.float64);lon=np.mod(np.asarray(lon,dtype=np.float64),360)
    if a.shape!=(121,240):raise ValueError('WB2 grid is not 121x240')
    if np.allclose(lat,90-1.5*np.arange(121),atol=1e-4):pass
    elif np.allclose(lat,-90+1.5*np.arange(121),atol=1e-4):a=a[::-1]
    else:raise ValueError('WB2 lat orientation invalid')
    idx=np.argsort(lon)
    if not np.allclose(lon[idx],1.5*np.arange(240),atol=1e-4):raise ValueError('WB2 lon mismatch')
    return np.ascontiguousarray(a[:,idx])
def open_wb2():
    import xarray as xr
    ds=xr.open_zarr(SOURCE,storage_options={'token':'anon'},consolidated=None,chunks=None)
    for v in ('mean_sea_level_pressure','2m_temperature'):
        if v not in ds:raise RuntimeError('WB2 missing var '+v)
    return ds

def read_weekly_historical(ds,initialization):
    fields=np.empty((2,2,121,240),np.float32)
    for vi,var in enumerate(('mean_sea_level_pressure','2m_temperature')):
        for wi,w in enumerate(LEADS):
            tt=valid_times(initialization,w)
            q=ds[var].sel(time=tt)
            if not np.array_equal(q.time.values.astype('datetime64[ns]'),tt):raise RuntimeError('Historical time mismatch')
            a=canonical(q.mean('time').transpose('latitude','longitude').load().values,ds.latitude.values,ds.longitude.values)
            if vi==0:a=a/100
            else:a=a-273.15
            if not np.isfinite(a).all():raise ValueError('Nonfinite climate training truth')
            fields[vi,wi]=a
    return fields

def get_climatology(root,offline=False):
    root=Path(root);cache=root/'cache';cache.mkdir(parents=True,exist_ok=True)
    ds=None
    allvalues={}
    provenance=[]
    for season in SEASONS:
        items=[]
        for year in TRAIN_YEARS:
            for offset in WEEK_OFFSETS:
                init=sample_init(year,season,offset)
                date=init.strftime('%Y%m%d%H')
                f=cache/f'clim_{season}_{date}.npz'
                if f.exists():
                    with np.load(f,allow_pickle=False) as a:
                        if str(a['source'])!=SOURCE or str(a['initialization'])!=date:raise RuntimeError('climate cache drift')
                        fields=a['means'].copy()
                else:
                    if offline:
                        # A *deliberately fake* climate distribution, ONLY for QA and never a real score.
                        yy=year-2008;shift=0 if season=='jan' else 8
                        fields=np.empty((2,2,121,240),dtype=np.float32)
                        fields[0]=1008+(yy-4)*.9+offset*.4+shift
                        fields[1]=5+(yy-4)*.65+offset*.55+shift
                    else:
                        if ds is None:ds=open_wb2()
                        fields=read_weekly_historical(ds,init)
                    tmp=f.with_suffix('.tmp');
                    with tmp.open('wb') as o:np.savez_compressed(o,source=SOURCE,initialization=date,means=fields)
                    os.replace(tmp,f)
                if fields.shape!=(2,2,121,240) or not np.isfinite(fields).all():raise ValueError('Invalid historical fields')
                items.append(fields)
                provenance.append({'initialization':date,'season':season,'training_only':True,'sha256':sha_file(f)})
        values=np.stack(items)
        if values.shape!=(N_HIST,2,2,121,240):raise ValueError('Climatology sample count')
        q=np.quantile(values,[.2,.4,.6,.8],axis=0).astype(np.float32)
        if q.shape!=(4,2,2,121,240) or np.any(np.diff(q,axis=0)<-1e-4):raise RuntimeError('Nonmonotonic quantile thresholds')
        allvalues[season]=q
        print(f'[clim] {season} {N_HIST} train-only weekly means PASS',flush=True)
    return allvalues,provenance

def rps(probs,truth_cat):
    if probs.shape!=(5,121,240):raise ValueError('5-bin probability shape required')
    if np.max(np.abs(probs.sum(0)-1))>1e-6 or not np.isfinite(probs).all() or np.min(probs)<-1e-8:raise ValueError('Bad probabilities')
    if truth_cat.shape!=(121,240) or not np.isin(truth_cat,range(5)).all():raise ValueError('Truth quintile invalid')
    pcdf=np.cumsum(probs,axis=0)[:4]
    ocdf=(truth_cat[None,:,:]<=np.arange(4)[:,None,None]).astype(np.float64)
    return np.sum((pcdf-ocdf)**2,axis=0)

def score_prob_case(pred,truth,thresholds,smoothing=0):
    if smoothing<0:raise ValueError('No negative smoothing')
    wt=weights_mask(0,90)
    results={}
    for vi,v in enumerate(VARIABLES):
        for wi,lead in enumerate(LEADS):
            thresholds4=thresholds[:,vi,wi]
            if np.any(np.diff(thresholds4,axis=0)<0):raise ValueError('Wrong thresholds')
            # 5 quintile bins from train-only, 5-member GEFS empirical forecast probabilities.
            fc=(pred[:,vi,wi,None,:,:]>thresholds4[None,:,:,:]).sum(axis=1).astype(np.int64)
            obs=(truth[vi,wi,None,:,:]>thresholds4).sum(axis=0).astype(np.int64)
            counts=np.stack([(fc==k).sum(axis=0) for k in range(5)],axis=0)
            probs=(counts+smoothing)/(5+5*smoothing)
            base=np.full_like(probs,.2,dtype=np.float64)
            fs=float(np.sum(wt*rps(probs,obs)))
            rs=float(np.sum(wt*rps(base,obs)))
            if rs<=0:raise RuntimeError('Reference RPS zero')
            results[f'{v}_W{lead}']={
                'research_proxy_rps_gefs':fs,'research_proxy_rps_climatology':rs,
                'research_proxy_rpss':1-fs/rs,
                'member_probability_smoothing':smoothing,
                'observed_bin_frequencies':[float(np.sum(wt*(obs==k))) for k in range(5)],
                'note':'TRAIN-ONLY EXPERIMENTAL QUINTILES; NOT OFFICIAL AI WEATHER QUEST RPSS'}
    return results

def independent_numeric_audit(cases):
    # Independent weighted scores recomputed from original arrays.
    w=weights_mask(0,90);result={}
    for d,(f,y) in cases.items():
        r={}
        for vi,v in enumerate(VARIABLES):
            for wi,lead in enumerate(LEADS):
                ens=f[:,vi,wi].astype(np.float64);obs=y[vi,wi].astype(np.float64)
                err=ens.mean(0)-obs
                rmse=float(np.sqrt((w*err**2).sum()))
                ctrl=float(np.sqrt((w*(ens[0]-obs)**2).sum()))
                bias=float(np.sum(w*err))
                sp=float(np.sqrt((w*ens.var(axis=0,ddof=1)).sum()))
                r[f'{v}_W{lead}']={'ensemble_rmse':rmse,'control_rmse':ctrl,'bias':bias,
                        'spread_rms':sp,'latitude_band_rmse':latitude_zone_scores(err)}
        result[d]=r
    return result

def add_report_zip(root,status,details=None):
    root=Path(root);report=root/'reports';report.mkdir(parents=True,exist_ok=True)
    state={'experiment':'CIDM_EARTH_A15_P3','status':status,'finished_utc':datetime.now(timezone.utc).isoformat(),
           'input_uploaded_sha256':INPUT_SHA,'input_original_sidecar_sha256':SUBMITTED_SIDECAR_SHA,
           'input_sidecar_matches_uploaded':False,'official_rpss':None,'formal_training_authorized':False,
           'historical_operational_availability_verified':False,'retrospective_reforecast_only':True,
           'evaluation_cases_already_inspected':True,'climate_train_years':list(TRAIN_YEARS),
           'climate_offsets_weeks':list(WEEK_OFFSETS),'reference_quantiles':'0.2,0.4,0.6,0.8; 27 training weekly means per season and lead',
           'notes':'Diagnostic/proxy baseline only. No official RPSS and no independent holdout.'}
    if details:state.update(details)
    write_json(report/'STATUS.json',state)
    dst=root/OUT_NAME
    with zipfile.ZipFile(dst,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(report.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(root).as_posix())
    with zipfile.ZipFile(dst) as z:
        if z.testzip():raise RuntimeError('ZIP CRC invalid')
    side=Path(str(dst)+'.sha256');side.write_text(sha_file(dst)+'  '+dst.name+'\n')
    if sha_file(dst)!=side.read_text().split()[0]:raise RuntimeError('Self-hash mismatch')
    print('[DELIVER]',status,dst,'sha256',sha_file(dst),flush=True)
    return dst

def run(root,input_zip,offline=False):
    root=Path(root);(root/'reports').mkdir(parents=True,exist_ok=True)
    try:
        cases=load_input(input_zip)
        score=independent_numeric_audit(cases)
        write_json(root/'reports'/'P2_INDEPENDENT_RESCORE.json',score)
        if len(cases)!=4:raise RuntimeError('Missing P2 cases')
        climates,prov=get_climatology(root,offline=offline)
        # Preserve threshold arrays (derived-only; may be sizeable but no raw six-hour fields)
        np.savez_compressed(root/'reports'/'TRAIN_ONLY_PROXY_QUINTILES.npz',
                            january=climates['jan'],july=climates['jul'],
                            train_years=np.array(TRAIN_YEARS),week_offsets=np.array(WEEK_OFFSETS))
        write_json(root/'reports'/'TRAIN_ONLY_ERA5_PROVENANCE.json',prov)
        scored={}
        for date,(f,y) in cases.items():
            season='jan' if date[4:6]=='01' else 'jul'
            q=climates[season]
            scored[date]={'raw_ensemble':score_prob_case(f,y,q,0),
                          'prespecified_half_count_smoothing_sensitivity':score_prob_case(f,y,q,.5)}
        write_json(root/'reports'/'PER_CASE_PROXY_RPS_RPSS.json',scored)
        agg={}
        for kind in ('raw_ensemble','prespecified_half_count_smoothing_sensitivity'):
            agg[kind]={}
            for key in ('mslp_W3','mslp_W4','tas_W3','tas_W4'):
                rps=np.mean([scored[d][kind][key]['research_proxy_rps_gefs'] for d in TEST_CASES])
                rps_ref=np.mean([scored[d][kind][key]['research_proxy_rps_climatology'] for d in TEST_CASES])
                agg[kind][key]={'mean_gefs_rps':float(rps),'mean_reference_rps':float(rps_ref),
                                'proxy_rpss_from_pooled_rps':float(1-rps/rps_ref),
                                'per_case_proxy_rpss':[scored[d][kind][key]['research_proxy_rpss'] for d in TEST_CASES]}
        write_json(root/'reports'/'PROXY_SCORE_SUMMARY.json',agg)
        return add_report_zip(root,'OFFLINE_QA_ONLY_NOT_REAL_CLIMATOLOGY' if offline else 'REAL_TRAIN_ONLY_ERA5_QUINTILE_PROXY_BASELINE_FOUR_CASES_PASS_RESEARCH_ONLY',
                 {'historical_samples_per_season':N_HIST,'evaluated_cases':4,'real_climatology':not offline})
    except Exception:
        (root/'reports'/'ERROR_TRACEBACK.txt').write_text(traceback.format_exc()[-30000:],encoding='utf8')
        return add_report_zip(root,'BLOCKED_QUALIFICATION_NOT_SKILL_RESULT',{'real_climatology':False,'error':str(traceback.format_exc().splitlines()[-1])})

def selftest():
    assert N_HIST==27
    for s in SEASONS:
        for y in TRAIN_YEARS:
            for off in WEEK_OFFSETS:
                for lead in LEADS:assert len(valid_times(sample_init(y,s,off),lead))==28
    z=np.zeros((5,121,240),np.float64);z[0]=1
    assert abs(rps(z,np.zeros((121,240),dtype=int)).mean())<1e-12
    cl=np.ones((5,121,240))/5
    assert np.allclose(rps(cl,np.zeros((121,240),dtype=int)),1.2)
    # Correct all-members-in-last-bin and truth last bin => zero.
    z=np.zeros((5,121,240));z[4]=1
    assert rps(z,np.full((121,240),4,dtype=int)).max()==0
    w=weights_mask(0,90)
    assert abs(w.sum()-1)<1e-12
    print('A15_P3_OFFLINE_MATH_AND_TIME_QA_PASS',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-zip',required=True)
    ap.add_argument('--out',default='/content/CIDM_EARTH_A15_P3')
    ap.add_argument('--offline-smoke',action='store_true')
    args=ap.parse_args()
    selftest();run(args.out,args.input_zip,offline=args.offline_smoke)