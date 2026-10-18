#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CIDM-Earth A15-P1: GEFSv12 retrospective full-weekly multi-member source qualification.

WARNING: A historical reforecast produced later is NOT an archived operational forecast.
No CIDM training, no AIWQ RPSS, no usage of future verification truth.

Data contract: Wednesday 00UTC GEFS initialization, following Thursday 00UTC
research-only reference issue; W3 issue+18..24 days and W4 issue+25..31 days,
28 six-hourly instantaneous MSLP / T2m GRIB records each.
0.5-degree 361x720 source sampled at *exact matching* 1.5-degree grid nodes.
This is coordinate alignment, NOT area-conservative remapping.

Google Colab defaults to reading public anonymous S3 and saving progressive state.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
import traceback
import zipfile
from pathlib import Path
import numpy as np

BUCKET = 'noaa-gefs-retrospective'
ROOT = 'GEFSv12/reforecast'
INIT = '2018010300'
MEMBERS = ('c00', 'p01', 'p02', 'p03', 'p04')
VARS = {'mslp': ('pres_msl', 'msl', 'Pa', 151), 'tas': ('tmp_2m', '2t', 'K', 167)}
MAX_RECORD_BYTES = 8 * 1024 * 1024
DOCS = ['https://registry.opendata.aws/noaa-gefs-reforecast/',
        'https://noaa-gefs-retrospective.s3.amazonaws.com/Description_of_reforecast_data.pdf']

class ContractError(RuntimeError): pass

def sha(raw): return hashlib.sha256(raw).hexdigest()
def as_json(path, val):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(val,ensure_ascii=False,indent=2,default=str),encoding='utf8')
    os.replace(tmp,path)

def init_issue(date):
    ini = dt.datetime.strptime(date, '%Y%m%d%H').replace(tzinfo=dt.timezone.utc)
    if ini.weekday()!=2 or ini.hour!=0: raise ContractError('Expected Wednesday 00UTC init')
    return ini,ini+dt.timedelta(days=1)

def required_steps(date, lead):
    ini,issue=init_issue(date)
    if lead not in (3,4): raise ContractError('Only W3/W4 are contracted')
    first=issue+dt.timedelta(days=18 if lead==3 else 25)
    steps=[int((first+dt.timedelta(hours=6*i)-ini).total_seconds()//3600) for i in range(28)]
    if steps != list(range(456 if lead==3 else 624, 624 if lead==3 else 792, 6)):
        raise ContractError('Unexpected target-relative time axis')
    return steps

def parse_idx(raw):
    """Parse object sidecar: record:offset:metadata:fHHH:; reject ambiguity."""
    rows=[]
    for line in raw.decode('utf8','replace').splitlines():
        cols=line.split(':')
        if len(cols)<2 or not (cols[0].isdigit() and cols[1].isdigit()):continue
        found=re.findall(r'\bf(\d{1,4})\b',line,re.I)
        if not found:found=re.findall(r'\b(\d{1,4})\s+hour\s+(?:fcst|forecast)\b',line,re.I)
        if len(found)!=1:raise ContractError('IDX lead missing or ambiguous: '+line[:100])
        rows.append({'offset':int(cols[1]),'step':int(found[0])})
    if not rows:raise ContractError('IDX empty/unrecognized')
    if len({r['offset'] for r in rows})!=len(rows):raise ContractError('IDX duplicate byte offsets')
    rows.sort(key=lambda r:r['offset'])
    if rows[0]['offset']!=0 or any(rows[i+1]['offset']<=rows[i]['offset'] for i in range(len(rows)-1)):
        raise ContractError('IDX ordering/first offset invalid')
    leadmap={}
    for i,row in enumerate(rows):
        if row['step'] in leadmap:raise ContractError('IDX duplicate forecast lead')
        leadmap[row['step']]=row['offset']
    return rows,leadmap

def record_span(rows,steps,length,step):
    offsets={row['step']:i for i,row in enumerate(rows)}
    if step not in offsets:raise ContractError('IDX missing step '+str(step))
    i=offsets[step];start=rows[i]['offset']
    end=(rows[i+1]['offset'] if i+1<len(rows) else length)-1
    if not (0<=start<=end<length):raise ContractError('IDX invalid bounds')
    if end-start+1>MAX_RECORD_BYTES:raise ContractError('Safety limit per GRIB record')
    return start,end

def validate_full_index(rows,length,req):
    keys={r['step'] for r in rows}
    for lead,needed in req.items():
        missing=set(needed)-keys
        if missing:raise ContractError(f'W{lead}: {len(missing)} missing six-hour leads e.g. {sorted(missing)[:3]}')
        for step in needed:record_span(rows,needed,length,step)
    return True

def object_key(date,member,var):
    fname,_,_,_=VARS[var]
    return f'{ROOT}/{date[:4]}/{date}/{member}/Days:10-35/{fname}_{date}_{member}.grib2'

def anonymous_client():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    return boto3.client('s3',region_name='us-east-1',
        config=Config(signature_version=UNSIGNED,connect_timeout=15,read_timeout=45,
                      retries={'max_attempts':3,'mode':'standard'}))

def get_range(client,key,start,end):
    res=client.get_object(Bucket=BUCKET,Key=key,Range=f'bytes={start}-{end}')
    if not res.get('ContentRange'):raise ContractError('Remote S3 did not honor byte range')
    raw=res['Body'].read()
    if len(raw)!=end-start+1 or len(raw)<16 or raw[:4]!=b'GRIB' or raw[7]!=2 or raw[-4:]!=b'7777':
        raise ContractError('Truncated/corrupt GRIB2 single-record byte span')
    if int.from_bytes(raw[8:16],'big')!=len(raw):raise ContractError('GRIB2 encoded length mismatch')
    return raw

def read_grid_ec(raw,date,step,var):
    """Real ecCodes GRIB2 decode plus physical time/grid/variable audits."""
    import eccodes
    gid=eccodes.codes_new_from_message(raw)
    if gid is None:raise ContractError('ecCodes cannot open record')
    try:
        def val(k):return eccodes.codes_get(gid,k)
        if int(val('edition'))!=2:raise ContractError('Wrong GRIB edition')
        if (int(val('dataDate')),int(val('dataTime')))!=(int(date[:8]),int(date[8:])*100):
            raise ContractError('GRIB initialization mismatch')
        if int(val('endStep'))!=int(step):raise ContractError('Forecast lead mismatch')
        if str(val('shortName'))!=VARS[var][1] or str(val('units'))!=VARS[var][2] or int(val('paramId'))!=VARS[var][3]:
            raise ContractError('Physical variable/units/paramId mismatch')
        if str(val('gridType'))!='regular_ll' or (int(val('Ni')),int(val('Nj')))!=(720,361):
            raise ContractError('Expected global 0.5 degree 361x720 regular_ll')
        if float(val('iDirectionIncrementInDegrees'))!=0.5 or float(val('jDirectionIncrementInDegrees'))!=0.5:
            raise ContractError('Unexpected grid increments')
        if int(val('jPointsAreConsecutive'))!=0 or int(val('iScansNegatively'))!=0:
            raise ContractError('Unsupported GRIB scanning convention')
        first_lat=float(val('latitudeOfFirstGridPointInDegrees'))
        first_lon=float(val('longitudeOfFirstGridPointInDegrees'))
        if abs(first_lon)>1e-4 or min(abs(first_lat-90),abs(first_lat+90))>1e-4:
            raise ContractError('Grid origin mismatch')
        a=np.asarray(eccodes.codes_get_values(gid), dtype=np.float32).reshape(361,720)
        if first_lat<0:a=a[::-1]
        out=np.ascontiguousarray(a[::3,::3])
        if out.shape!=(121,240) or not np.isfinite(out).all():raise ContractError('Invalid or nonfinite spatial field')
        lo,hi=(70000,120000) if var=='mslp' else (160,350)
        if out.min()<lo or out.max()>hi:raise ContractError('Field physical bounds violated')
        return out,{'shortName':str(val('shortName')),'paramId':int(val('paramId')),
                    'unit':str(val('units')),'step':int(step),'grid_src':[361,720],
                    'grid_dst':[121,240], 'min':float(out.min()),'max':float(out.max())}
    finally:eccodes.codes_release(gid)

def checked_cache(path,key,idx_sha,etag):
    if not path.exists():return None
    try:
        with np.load(path,allow_pickle=False) as z:
            if str(z['key'])!=key or str(z['idx_sha'])!=idx_sha or str(z['etag'])!=etag:
                raise ContractError('PROVENANCE_DRIFT_EXISTING_CACHE')
            s=np.asarray(z['sums'],dtype=np.float64);counts=np.asarray(z['counts'],dtype=np.int32)
            if s.shape!=(2,121,240) or counts.shape!=(2,) or np.any(counts<0) or np.any(counts>28):
                raise ContractError('INVALID_EXISTING_CACHE_ARRAY')
            return s,counts
    except ContractError:raise
    except Exception as e:raise ContractError('Corrupted resume cache '+str(e))

def write_part(path,sums,counts,key,idx_sha,etag):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with open(tmp,'wb') as f:
        np.savez_compressed(f,sums=sums,counts=counts,key=key,idx_sha=idx_sha,etag=etag)
    os.replace(tmp,path)

def maybe_preflight(client,date,members,required):
    """No downloads before all member/variable files and indexes have valid coverage."""
    source={}
    for member in members:
        for var in VARS:
            key=object_key(date,member,var)
            head=client.head_object(Bucket=BUCKET,Key=key)
            length=int(head['ContentLength'])
            if length<1000:raise ContractError('Missing/tiny object '+key)
            etag=str(head.get('ETag',''))
            ix=client.get_object(Bucket=BUCKET,Key=key+'.idx')['Body'].read()
            if len(ix)>1024*1024:raise ContractError('Index too large')
            rows,_=parse_idx(ix)
            validate_full_index(rows,length,required)
            source[(member,var)]={'key':key,'size':length,'etag':etag,'idx_sha':sha(ix),'rows':rows,
                                   'idx_entries':len(rows),'coverage':{k:len(v) for k,v in required.items()}}
            print(f'[A15-P1 inventory] {member}/{var} verified {len(rows)} IDX messages',flush=True)
    return source

def group_aggregate(client, out, date, member, var, spec, required, save_every=4,decode_fn=read_grid_ec):
    path=out/'cache'/f'partial_{date}_{member}_{var}.npz'
    saved=checked_cache(path,spec['key'],spec['idx_sha'],spec['etag'])
    sums,counts=saved if saved else (np.zeros((2,121,240),dtype=np.float64),np.zeros(2,dtype=np.int32))
    stats=[];downloads=0
    for widx,lead in enumerate((3,4)):
        steps=required[lead]
        for i in range(int(counts[widx]),28):
            step=steps[i]
            st,en=record_span(spec['rows'],steps,spec['size'],step)
            raw=get_range(client,spec['key'],st,en)
            field,meta=decode_fn(raw,date,step,var)
            if field.shape!=(121,240) or not np.isfinite(field).all():raise ContractError('Decoded invalid field')
            sums[widx]+=field
            counts[widx]+=1;downloads+=1
            stats.append({'lead':lead,'step':step,'bytes':len(raw),'sha256':sha(raw),'min':meta.get('min'),'max':meta.get('max')})
            if (counts.sum()%save_every==0) or (counts[widx]==28):
                write_part(path,sums,counts,spec['key'],spec['idx_sha'],spec['etag'])
            if (i+1)%7==0:
                print(f'[A15-P1 read] {member}/{var} W{lead} {i+1}/28',flush=True)
    if counts.tolist()!=[28,28]:raise ContractError('Incomplete source record aggregation')
    mean=(sums/28.0).astype(np.float32)
    if var=='mslp':mean=(mean/100).astype(np.float32)  # hPa
    else:mean=(mean-273.15).astype(np.float32)          # Celsius
    as_json(out/'reports'/f'{member}_{var}_audit.json', {
        'key':spec['key'],'source_etag':spec['etag'],'source_idx_sha256':spec['idx_sha'],
        'number_of_6h_fields_per_week':counts.tolist(), 'fresh_range_reads':downloads,
        'mean_minmax':[[float(a.min()),float(a.max())] for a in mean],
        'units_out':'hPa' if var=='mslp' else 'degC','records_this_session':stats,
        's3_reforecast_not_operational':True})
    return mean

def status_package(out, status, info, error=None):
    rep=out/'reports';rep.mkdir(exist_ok=True,parents=True)
    payload={'experiment':'CIDM_EARTH_A15_P1','status':status,'run_time_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
             'source_type':'GEFSv12 reforecast RETROSPECTIVE not operational',
             'training_authorized':False,'official_rpss':None,
             'operational_issue_time_verified':False,'documents':DOCS,**info}
    as_json(rep/'STATUS.json',payload)
    if error:(rep/'ERROR_TRACEBACK.txt').write_text(str(error)[:24000],encoding='utf8')
    z=out/'CIDM_EARTH_A15_P1_DELIVERY_REPORTS.zip';hpath=Path(str(z)+'.sha256')
    with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(rep.rglob('*')):
            if f.is_file():archive.write(f, f.relative_to(out).as_posix())
    with zipfile.ZipFile(z) as f:
        if f.testzip() is not None:raise RuntimeError('ZIP CRC check failed')
    hpath.write_text(sha(z.read_bytes())+'  '+z.name+'\n',encoding='ascii')
    print('[A15-P1]',status,'ZIP=',z,'SHA256=',sha(z.read_bytes()),flush=True)
    return z

def offline_selftest():
    ini,iss=init_issue(INIT)
    req={3:required_steps(INIT,3),4:required_steps(INIT,4)}
    assert req[3]==list(range(456,624,6)) and req[4]==list(range(624,792,6))
    rows=[{'offset':i*50,'step':x} for i,x in enumerate(list(range(240,840,6)))]
    validate_full_index(rows,len(rows)*50,req)
    missing=rows[:]
    missing=[r for r in missing if r['step']!=768]
    try:validate_full_index(missing,len(rows)*50,req);raise AssertionError('missing step was accepted')
    except ContractError:pass
    a,b=parse_idx(('\n'.join(f'{i+1}:{i*100}:d=20180103:MSL:f{x}:0' for i,x in enumerate(range(240,840,6)))).encode())
    assert len(a)==100 and 624 in b
    for junk in (b'1:0:d=20180103:MSL:invalid', b'1:0:d=20180103:MSL:f456\n2:25:d=20180103:MSL:f456'):
        try:parse_idx(junk);raise AssertionError('bad index accepted')
        except ContractError:pass
    ens=np.stack([np.ones((2,121,240),np.float32)*k for k in range(5)])
    assert np.allclose(ens.mean(axis=0),2) and np.allclose(ens.std(axis=0,ddof=1),np.sqrt(2.5))
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        d=Path(d);tmp=d/'cache.npz';s=np.ones((2,121,240));c=np.array([8,4])
        write_part(tmp,s,c,'key','sha','etag')
        sr,ct=checked_cache(tmp,'key','sha','etag')
        assert np.all(sr==s) and ct.tolist()==[8,4]
        try:checked_cache(tmp,'key','DIFFERENT','etag');raise AssertionError('cache drift accepted')
        except ContractError:pass
        status_package(d,'OFFLINE_QA_ONLY_NOT_REAL_SOURCE',{'remote_data_retrieved':False})
        assert zipfile.ZipFile(d/'CIDM_EARTH_A15_P1_DELIVERY_REPORTS.zip').testzip() is None
    print('OFFLINE_QA_PASS: date/lead, IDX, missing step, ensemble moments, resume/SHA, ZIP',flush=True)

def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    offline_selftest()
    if args.offline_only:
        return status_package(out,'OFFLINE_QA_ONLY_NOT_REAL_SOURCE',{'remote_data_retrieved':False,'real_source_gate_pass':False})
    report={'source_bucket':BUCKET,'initialization_utc':INIT,
            'research_reference_issue_utc':init_issue(INIT)[1].isoformat(),
            'members_requested':list(MEMBERS[:args.members]),'variables':list(VARS),'grid_target':[121,240],
            'source_grid':[361,720],'temporal_windows_hours':{'W3':required_steps(INIT,3),'W4':required_steps(INIT,4)},
            'source_preflight_pass':False,'full_week_fields_decoded':False,
            'remote_data_retrieved':False,'real_source_gate_pass':False,'case_count':1}
    as_json(out/'reports'/'PROGRESS.json',report)
    try:
        client=anonymous_client()
        required={3:required_steps(INIT,3),4:required_steps(INIT,4)}
        members=MEMBERS[:args.members]
        source=maybe_preflight(client,INIT,members,required)
        report['source_preflight_pass']=True
        report['source_manifest']=[{'member':m,'variable':v,'key':s['key'],'size_bytes':s['size'],
                                   'etag':s['etag'],'idx_sha256':s['idx_sha'],'index_records':s['idx_entries']}
                                  for (m,v),s in source.items()]
        as_json(out/'reports'/'PROGRESS.json',report)
        result=[]
        for m in members:
            pair=[]
            for v in VARS:
                val=group_aggregate(client,out,INIT,m,v,source[(m,v)],required,
                                    save_every=args.save_every)
                pair.append(val)
                report['completed_groups']=report.get('completed_groups',0)+1
                as_json(out/'reports'/'PROGRESS.json',report)
            result.append(np.stack(pair,axis=0))  # C=2, lead=2, H,W
        allmeans=np.stack(result,axis=0)  # Member,Variable,Week,H,W
        if not np.isfinite(allmeans).all():raise ContractError('Non-finite derived forecasts')
        probemax=float(np.max(allmeans.max(axis=0)-allmeans.min(axis=0)))
        if len(members)>1 and probemax<0.001:raise ContractError('Ensemble members are unexpectedly identical')
        # Derived weekly means are portable but are not ERA5 observations or competition probabilities.
        derived=out/'reports'/'GEFSv12_REFORECAST_WEEKLY_MEANS_RESEARCH_ONLY.npz'
        np.savez_compressed(derived,weekmean=allmeans.astype(np.float32),
                            ensemble_mean=allmeans.mean(0).astype(np.float32),
                            ensemble_std=(allmeans.std(0,ddof=1) if len(members)>1 else np.zeros_like(allmeans[0])).astype(np.float32),
                            member=np.array(members), variable=np.array(list(VARS)),
                            week=np.array([3,4]), lat=np.linspace(90,-90,121).astype(np.float32),
                            lon=np.arange(240,dtype=np.float32)*1.5)
        report['weekly_array_shape']=list(allmeans.shape)
        report['weekly_data_sha256']=sha(derived.read_bytes())
        report['forecast_mean_range']={v:[[float(allmeans[:,ci,li].mean(0).min()),float(allmeans[:,ci,li].mean(0).max())] for li in range(2)] for ci,v in enumerate(VARS)}
        report['ensemble_member_spread_max']=probemax
        report['remote_data_retrieved']=True
        report['full_week_fields_decoded']=True
        report['real_source_gate_pass']=True
        as_json(out/'reports'/'PROGRESS.json',report)
        return status_package(out,'REAL_GEFS_REFORECAST_WEEKLY_ENSEMBLE_PASS_RESEARCH_ONLY',report)
    except Exception as e:
        report['error_type']=type(e).__name__
        as_json(out/'reports'/'PROGRESS.json',report)
        return status_package(out,'BLOCKED_INCOMPLETE_OR_INVALID_REAL_GEFS_WEEKLY_ENSEMBLE',report,traceback.format_exc())

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='./CIDM_EARTH_A15_P1')
    p.add_argument('--offline-only',action='store_true')
    p.add_argument('--members',type=int,default=5,choices=range(1,6))
    p.add_argument('--save-every',type=int,default=4)
    args=p.parse_args()
    run(args)
