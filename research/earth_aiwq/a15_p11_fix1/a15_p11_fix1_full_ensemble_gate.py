#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A15-P11 G1-R2 full ensemble S3 qualification: research-only retrospective GEFSv12.
No weather model fit; no scores; no claims of availability at historic forecast issue time.
"""
from __future__ import annotations
import datetime as dt
import hashlib,importlib.util,json,os,re,traceback,zipfile
from pathlib import Path
from collections import Counter

PHASE='A15_P11_FIX1_G1_R2_FIVE_MEMBER_PHYSICAL_SOURCE_QUALIFICATION'
YEARS=(2002,2004,2006,2010,2014,2016)
MONTHS=(1,4,7,10)
MEMBERS=('c00','p01','p02','p03','p04')
VARS=('mslp','tas')
LEADS=(408,456,504,600,624,696,786)
SAMPLE_LEADS=(456,786)
SAMPLE_MEMBERS=('c00','p04')
EXACT_DATE_COUNT=24

class QualificationError(RuntimeError):pass

def sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf8')
    os.replace(tmp,path)

def first_wed(year,month):
    day=dt.date(year,month,1)
    day+=dt.timedelta(days=(2-day.weekday())%7)
    return day.strftime('%Y%m%d')+'00'

def dates():return [first_wed(y,m) for y in YEARS for m in MONTHS]

def import_reader(path):
    if not Path(path).exists():raise QualificationError('FROZEN_GEFS_READER_NOT_PRESENT')
    spec=importlib.util.spec_from_file_location('cidm_a15_p11_pinned_reader',str(path))
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj

def validate_idx_initialization(raw: bytes, requested_date: str):
    """Parse GRIB sidecar d=YYYYMMDDHH or legacy d=YYYYMMDD.

    10-digit timestamps must match the requested issue *hour*, not only the
    calendar date. 8-digit sidecars retain an explicit lower-assurance marker;
    physical GRIB sample audit subsequently verifies date+time using ecCodes.
    Every parsable IDX row must contain exactly one d= token, and rows must
    agree. This is intentionally stricter than substring scanning.
    """
    if not re.fullmatch(r'\d{10}',requested_date):
        raise QualificationError('REQUESTED_DATE_NOT_TEN_DIGITS')
    modes=[];samples=set();n=0
    for line in raw.decode('utf-8','replace').splitlines():
        cols=line.split(':')
        if len(cols)<2 or not(cols[0].isdigit() and cols[1].isdigit()):
            raise QualificationError('IDX_RECORD_UNRECOGNIZED')
        tags=[token.strip()[2:] for token in cols if re.fullmatch(r'd=\d{8}(?:\d{2})?',token.strip())]
        if len(tags)!=1:
            raise QualificationError('IDX_DATE_TAG_ABSENT_MULTIPLE_OR_INVALID')
        got=tags[0];samples.add(got)
        if len(got)==10:
            if got!=requested_date:raise QualificationError('IDX_INIT_DATE_HOUR_MISMATCH')
            modes.append('YYYYMMDDHH')
        elif len(got)==8:
            if got!=requested_date[:8]:raise QualificationError('IDX_INIT_DATE_MISMATCH')
            modes.append('YYYYMMDD_LEGACY_UNVERIFIED_HOUR')
        else:raise QualificationError('IDX_DATE_LENGTH_UNSUPPORTED')
        n+=1
    if not n:raise QualificationError('EMPTY_IDX')
    if len(samples)!=1:raise QualificationError('MIXED_INITIALIZATION_TAGS')
    return {'mode':modes[0], 'tag':next(iter(samples)), 'n_lines':n,
            'strict_utc_hour_verified_by_idx':modes[0]=='YYYYMMDDHH'}

def check_one(cli,reader,date,member,var):
    key=reader.object_key(date,member,var)
    head=cli.head_object(Bucket=reader.BUCKET,Key=key)
    size=int(head['ContentLength']);etag=str(head.get('ETag',''))
    if size<1000 or not etag:raise QualificationError('HEAD_SIZE_OR_ETAG_INVALID')
    remote=cli.get_object(Bucket=reader.BUCKET,Key=key+'.idx')
    raw=remote['Body'].read(1024*1024+1)
    if len(raw)>1024*1024:raise QualificationError('INDEX_SIZE_LIMIT_EXCEEDED')
    init_audit=validate_idx_initialization(raw,date)
    rows,mp=reader.parse_idx(raw)
    if len(rows)!=init_audit['n_lines']:
        raise QualificationError('IDX_UNPARSED_ROWS')
    req={3:reader.required_steps(date,3),4:reader.required_steps(date,4)}
    reader.validate_full_index(rows,size,req)
    for lead in LEADS:
        reader.record_span(rows,LEADS,size,lead)
    return {'status':'PASS','key':key,'size':size,'etag':etag,
            'idx_sha256':hashlib.sha256(raw).hexdigest(),'idx_count':len(rows),'idx_init':init_audit,
            'time_coverage':{'W3':len(req[3]),'W4':len(req[4]),'trajectory':len(LEADS)},
            'sample_spans':{str(h):reader.record_span(rows,LEADS,size,h) for h in SAMPLE_LEADS}}

def inventory(cli,reader,root,daylist=None,members=MEMBERS,vars=VARS,cache=True):
    root=Path(root); cp=root/'cache'/'P11_SOURCE_CHECKPOINT.json'
    daylist=list(daylist if daylist is not None else dates())
    old={}
    if cp.exists() and cache:
        data=json.loads(cp.read_text(encoding='utf8'))
        if data.get('contract')=={'dates':daylist,'members':list(members),'vars':list(vars),'version':3}:
            old={r['init']:r for r in data.get('rows',[])}
    res=[]
    for k,date in enumerate(daylist):
        if date in old and old[date].get('ready') and len(old[date].get('objects',{}))==len(members)*len(vars):
            # Cached index is not a verification of current remote bytes. Check HEAD ETag and size.
            preserved=old[date];clean=True
            for keyid,spec in preserved['objects'].items():
                if spec.get('status')!='PASS':clean=False;break
                try:
                    m=cli.head_object(Bucket=reader.BUCKET,Key=spec['key'])
                    if int(m['ContentLength'])!=spec['size'] or str(m.get('ETag',''))!=spec['etag']:
                        clean=False;break
                    # A GRIB object's ETag does not protect a separate .idx object.
                    idx_now=cli.get_object(Bucket=reader.BUCKET,Key=spec['key']+'.idx')['Body'].read(1024*1024+1)
                    if hashlib.sha256(idx_now).hexdigest()!=spec.get('idx_sha256'):
                        clean=False;break
                    validate_idx_initialization(idx_now,date)
                except Exception:clean=False;break
            if clean:
                res.append(preserved)
                print(f'[P11 inventory] {k+1}/{len(daylist)} {date} CACHE_ETAG_PASS',flush=True)
                atomic_json(cp,{'contract':{'dates':daylist,'members':list(members),'vars':list(vars),'version':3},'rows':res})
                continue
        row={'init':date,'ready':True,'objects':{}}
        for m in members:
            for v in vars:
                try:s=check_one(cli,reader,date,m,v)
                except Exception as e:
                    s={'status':'BLOCKED','error':type(e).__name__+': '+str(e)[:350],
                       'key':reader.object_key(date,m,v)}
                    row['ready']=False
                row['objects'][m+'_'+v]=s
        res.append(row)
        atomic_json(cp,{'contract':{'dates':daylist,'members':list(members),'vars':list(vars),'version':3},'rows':res})
        print(f'[P11 inventory] {k+1}/{len(daylist)} {date}: '+('PASS' if row['ready'] else 'BLOCKED'),flush=True)
    return res

def sample_physical(cli,reader,root,rows,sample_leads=SAMPLE_LEADS,decode=None):
    """Physical checks only from 6 fixed year-stratified preselected dates. Reforecast record range fetch.
    Never use verification truth, never archive GRIB bytes.
    """
    root=Path(root); cp=root/'cache'/'P11_SAMPLE_CHECKPOINT.json'
    previous={}
    if cp.exists():
        try:previous=json.loads(cp.read_text(encoding='utf8')).get('records',{})
        except Exception:previous={}
    by_year={}
    for r in rows:
        if r['ready'] and r['init'][:4] not in by_year:by_year[r['init'][:4]]=r
    samples={};errors={}
    for year in map(str,YEARS):
        if year not in by_year:
            errors[year]='NO_FULL_ENSEMBLE_DATE_IN_YEAR';continue
        row=by_year[year];date=row['init']
        for m in SAMPLE_MEMBERS:
            for v in VARS:
                spec=row['objects'].get(m+'_'+v)
                if not spec or spec.get('status')!='PASS':
                    errors[f'{date}/{m}/{v}']='SOURCE_INDEX_MISSING';continue
                for step in sample_leads:
                    key=f'{date}/{m}/{v}/{step}'
                    source_id=spec['idx_sha256']+'|'+spec['etag']
                    if key in previous and previous[key].get('source_id')==source_id and previous[key].get('status')=='PASS':
                        samples[key]=previous[key];continue
                    try:
                        # Refetch idx to guarantee the byte spans remain faithful to the committed fingerprint.
                        head=cli.head_object(Bucket=reader.BUCKET,Key=spec['key'])
                        if str(head.get('ETag',''))!=spec['etag'] or int(head['ContentLength'])!=spec['size']:
                            raise QualificationError('OBJECT_CHANGED_SINCE_INDEX')
                        raw_idx=cli.get_object(Bucket=reader.BUCKET,Key=spec['key']+'.idx')['Body'].read(1024*1024+1)
                        if hashlib.sha256(raw_idx).hexdigest()!=spec['idx_sha256']:
                            raise QualificationError('IDX_CHANGED_SINCE_INDEX')
                        parsed,_=reader.parse_idx(raw_idx)
                        st,en=reader.record_span(parsed,[step],spec['size'],step)
                        grib=reader.get_range(cli,spec['key'],st,en)
                        field,meta=(decode or reader.read_grid_ec)(grib,date,step,v)
                        if tuple(field.shape)!=(121,240):raise QualificationError('INVALID_OUTPUT_GRID')
                        import numpy as np
                        if not np.isfinite(field).all():raise QualificationError('NONFINITE_GRID')
                        samples[key]={'status':'PASS','source_id':source_id,'bytes':len(grib),
                         'sha256':hashlib.sha256(grib).hexdigest(),'metadata':meta}
                    except Exception as e:
                        errors[key]=type(e).__name__+': '+str(e)[:350]
                        samples[key]={'status':'BLOCKED','source_id':source_id,'error':errors[key]}
                    atomic_json(cp,{'records':samples,'errors':errors,'sample_method':'real GRIB byte ranges; no raw GRIB in result'})
                    print('[P11 GRIB]',key,samples[key]['status'],flush=True)
    return samples,errors

def summarize(rows,samples,errors,required_dates=EXACT_DATE_COUNT):
    full=[r for r in rows if r['ready']]
    years=sorted(set(r['init'][:4] for r in full))
    object_count=sum(s.get('status')=='PASS' for r in rows for s in r['objects'].values())
    planned=len(rows)*len(MEMBERS)*len(VARS)
    idx_gate=len(rows)==required_dates and len(full)>=20 and len(years)>=5
    sample_good=sum(s['status']=='PASS' for s in samples.values())
    physical_gate=sample_good==48 and not errors and len(rows)==required_dates
    # Distinguish '24/24' (complete inventory) from relaxed 20/24 source pilot.
    complete=len(full)==required_dates and object_count==planned
    return {'G1_R2_FULL_ENSEMBLE_INDEX_GATE_PASS':bool(idx_gate),
        'G1_R2_ALL_24_DATES_FULL_COVERAGE':bool(complete),
        'G1_R2_PHYSICAL_SAMPLE_GATE_PASS':bool(physical_gate),
        'init_total':len(rows),'init_full':len(full),'eligible_years':years,
        'objects_pass':object_count,'objects_planned':planned,
        'samples_pass':sample_good,'samples_planned':48,
        'idx_date_modes':dict(Counter(spec['idx_init']['mode'] for r in rows for spec in r['objects'].values() if spec.get('status')=='PASS' and 'idx_init' in spec)),
        'errors':errors,
        'gate_note':'Index >=20/24 in >=5 years is research source pilot only; physical spot checks are NOT full field QC of all dates; YYYYMMDD legacy sidecar lacks hour and requires real ecCodes sample to validate hour'}

def package(root,report,status,error=None):
    root=Path(root);r=root/'reports';r.mkdir(parents=True,exist_ok=True)
    atomic_json(r/'P11_STATUS.json',{'phase':PHASE,'status':status,'report':report,'fix':'INDEX_DATE_8_OR_10_DIGIT_WITH_EXACT_HOUR_AUDIT',
        'error':error,'real_complete_weekly_field_data':False,'formal_training_authorized':False,
        'official_rpss':None,'operational_issue_time_verified':False,
        'source':'NOAA GEFSv12 retrospective reforecast; not historical operational data'})
    zp=root/'CIDM_EARTH_A15_P11_FIX1_DELIVERY_REPORTS.zip'
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED,compresslevel=8) as z:
        for f in sorted(r.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(root).as_posix())
    with zipfile.ZipFile(zp) as z:
        fail=z.testzip()
        if fail:raise QualificationError('OUTPUT_CRC_FAIL '+fail)
    side=root/(zp.name+'.sha256')
    side.write_text(sha_file(zp)+'  '+zp.name+'\n',encoding='ascii')
    return {'status':status,'zip':str(zp),'sha':str(side)}

def run(root,reader_path,online=True,client=None,decode=None,dates_override=None):
    root=Path(root);(root/'reports').mkdir(exist_ok=True,parents=True)
    for p in (root/'reports').glob('*'): 
        if p.is_file():p.unlink() # prevent stale success reports, never delete resume cache
    report={};error=None;status='BLOCKED_UNKNOWN'
    try:
        reader=import_reader(reader_path)
        if not online:
            # Exercises all offline mathematics without misrepresenting mock results.
            report={'source':'OFFLINE_QA_NO_REAL_S3','qualification':False}
            status='OFFLINE_QA_ONLY_NO_REAL_S3'
        else:
            cli=client if client is not None else reader.anonymous_client()
            rows=inventory(cli,reader,root,daylist=dates_override)
            atomic_json(root/'reports'/'P11_FULL_SOURCE_INVENTORY.json',{'rows':rows,'sources_are_reforecasts':True})
            samples,errors=sample_physical(cli,reader,root,rows,decode=decode)
            atomic_json(root/'reports'/'P11_SAMPLED_PHYSICAL_GRIB.json',{'records':samples,'errors':errors})
            report=summarize(rows,samples,errors)
            if report['G1_R2_FULL_ENSEMBLE_INDEX_GATE_PASS'] and report['G1_R2_PHYSICAL_SAMPLE_GATE_PASS']:
                status='G1_R2_FIX1_FULL_ENSEMBLE_INDEX_AND_SAMPLE_PASS_RESEARCH_ONLY'
            else:status='G1_R2_FIX1_PARTIAL_OR_BLOCKED_RESEARCH_ONLY'
    except Exception as exc:
        error={'error_type':type(exc).__name__,'message':str(exc), 'traceback':traceback.format_exc()[-5000:]}
        status='BLOCKED_EXCEPTION_NO_TRAINING'
    return package(root,report,status,error)
