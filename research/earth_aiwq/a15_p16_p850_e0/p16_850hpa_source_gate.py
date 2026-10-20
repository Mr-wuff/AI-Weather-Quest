# -*- coding: utf-8 -*-
"""A15-P16-WIND-P850-E0: strict REAL GEFSv12 retrospective 850-hPa U/V source gate.
Research-source qualification only. No CIDM training, ERA5 input, or official RPSS.
"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, io, json, os, re, tempfile, traceback, zipfile
from pathlib import Path
import numpy as np

CONTRACT="A15_P16_P850_20261020_FROZEN_V1"
DATES=("2009010700","2012040400","2013070300","2015100700")
MEMBERS=("c00","p01","p02","p03","p04")
LEADS=tuple(range(456,504,6))+tuple(range(624,672,6))
VARS={"u850":("ugrd_pres",2),"v850":("vgrd_pres",3)}
BUCKET="noaa-gefs-retrospective"
PREFIX="GEFSv12/reforecast"
GRID=(121,240)
PATCH=(11,12)
MAX_GRIB=8*1024*1024

class Blocked(RuntimeError): pass
def sha(data): return hashlib.sha256(data).hexdigest()
def a_sha(arr): return sha(np.ascontiguousarray(arr).tobytes())
def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix="."+path.name+".",suffix=".tmp",dir=str(path.parent))
    try:
        with os.fdopen(fd,"wb") as f:
            f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
def jsonsave(path,obj):
    atomic(path,(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True,default=str)+"\n").encode())
def npzsave(path,**data):
    with io.BytesIO() as bf:
        np.savez_compressed(bf,**data)
        atomic(path,bf.getvalue())
def src_key(date,mem,var):
    nm=VARS[var][0]
    return f"{PREFIX}/{date[:4]}/{date}/{mem}/Days:10-35/{nm}_{date}_{mem}.grib2"

def parse_index(raw,date,content_length,var):
    text=raw.decode("utf8","strict")
    entries=[]
    for line in text.splitlines():
        cols=line.split(":")
        if len(cols)<5 or not cols[0].isdigit() or not cols[1].isdigit():
            raise Blocked("GRIB_INDEX_ROW_MALFORMED")
        tag=re.findall(r"\bd=(\d{10})\b",line)
        if tag!=[date]: raise Blocked("GRIB_INDEX_INIT_NOT_EXACT_10_DIGITS")
        lh=re.findall(r"\bf(\d{1,4})\b",line,re.I)
        if len(lh)!=1: raise Blocked("GRIB_INDEX_FORECAST_LEAD_AMBIGUOUS")
        entries.append(dict(offset=int(cols[1]),lead=int(lh[0]),line=line))
    if not entries or entries[0]["offset"]!=0: raise Blocked("IDX_ORIGIN_NOT_ZERO")
    offsets=[e["offset"] for e in entries]
    if len(set(offsets))!=len(offsets) or sorted(offsets)!=offsets:
        raise Blocked("IDX_OFFSETS_NOT_STRICT_INCREASING")
    target="UGRD" if var=="u850" else "VGRD"
    matches={}
    other_levels=0
    for i,e in enumerate(entries):
        line=e["line"]
        nxt=entries[i+1]["offset"] if i+1<len(entries) else content_length
        if not(0<=e["offset"]<nxt<=content_length): raise Blocked("IDX_BYTE_SPAN_INVALID")
        parts=line.split(":")
        correct_parameter=target in [p.strip().upper() for p in parts]
        # NOAA wgrib2 IDX pressure records conventionally use "850 mb".
        correct_level=any(re.fullmatch(r"\s*850\s*(?:mb|hpa)\s*",p,re.I) for p in parts)
        if correct_parameter and not correct_level: other_levels+=1
        if correct_parameter and correct_level and e["lead"] in LEADS:
            if e["lead"] in matches: raise Blocked("DUPLICATE_850_HPA_LEAD")
            if nxt-e["offset"]>MAX_GRIB: raise Blocked("GRIB_MESSAGE_TOO_LARGE")
            matches[e["lead"]]=(e["offset"],nxt-1)
    if set(matches)!=set(LEADS):
        raise Blocked("850_HPA_MISSING_LEADS:"+str(sorted(set(LEADS)-set(matches))))
    return matches,{"total_index_records":len(entries),"other_level_records":other_levels}

def ec_decode(msg,date,lead,var):
    import eccodes as ec
    h=ec.codes_new_from_message(msg)
    if h is None: raise Blocked("ECCODES_NO_MESSAGE")
    try:
        def geti(k): return int(ec.codes_get(h,k))
        def gets(k): return str(ec.codes_get(h,k))
        if geti("edition")!=2 or geti("dataDate")!=int(date[:8]) or geti("dataTime")!=int(date[8:])*100:
            raise Blocked("WRONG_GRIB_INIT")
        if geti("endStep")!=lead: raise Blocked("WRONG_GRIB_LEAD")
        if geti("parameterCategory")!=2 or geti("parameterNumber")!=VARS[var][1]:
            raise Blocked("WRONG_GRIB_UV_PARAMETER")
        typ=gets("typeOfLevel");level=geti("level")
        if not((typ=="isobaricInhPa" and level==850) or
               (typ=="isobaricInPa" and level==85000)):
            raise Blocked("WRONG_PRESSURE_LEVEL:"+typ+":"+str(level))
        if geti("Ni")!=720 or geti("Nj")!=361 or gets("gridType")!="regular_ll":
            raise Blocked("WRONG_GEFS_GRID")
        if geti("jPointsAreConsecutive") or geti("iScansNegatively"):
            raise Blocked("UNSUPPORTED_GRID_SCAN")
        if abs(float(ec.codes_get(h,"iDirectionIncrementInDegrees"))-0.5)>1e-5 or abs(float(ec.codes_get(h,"jDirectionIncrementInDegrees"))-0.5)>1e-5:
            raise Blocked("WRONG_GRID_RESOLUTION")
        unit=gets("units").lower().replace(" ","")
        if unit not in ("m/s","ms-1","ms**-1","ms^-1"):
            raise Blocked("WIND_UNITS_NOT_M_PER_SECOND:"+unit)
        origin=float(ec.codes_get(h,"latitudeOfFirstGridPointInDegrees"))
        lon=float(ec.codes_get(h,"longitudeOfFirstGridPointInDegrees"))
        if min(abs(origin-90),abs(origin+90))>1e-4 or abs(lon)>1e-4:
            raise Blocked("WRONG_GLOBAL_GRID_ORIGIN")
        full=np.asarray(ec.codes_get_values(h),dtype=np.float32).reshape(361,720)
        if origin<0: full=full[::-1]
        arr=np.ascontiguousarray(full[::3,::3])
        if arr.shape!=GRID or not np.isfinite(arr).all() or np.max(np.abs(arr))>=200:
            raise Blocked("WIND_FIELD_BAD_VALUES")
        return arr,{"typeOfLevel":typ,"level":level,"units":gets("units"),
                    "max_abs":float(np.max(np.abs(arr)))}
    finally: ec.codes_release(h)

def get_source(cli,date,mem,var):
    key=src_key(date,mem,var)
    head=cli.head_object(Bucket=BUCKET,Key=key)
    size=int(head["ContentLength"]); etag=str(head["ETag"])
    if not(1024<size<4*1024**3) or not etag: raise Blocked("INVALID_SOURCE_HEAD")
    raw=cli.get_object(Bucket=BUCKET,Key=key+".idx")["Body"].read(4*1024*1024+1)
    if len(raw)>4*1024*1024: raise Blocked("IDX_TOO_LARGE")
    spans,meta=parse_index(raw,date,size,var)
    return dict(key=key,size=size,etag=etag,idx_sha=sha(raw),spans=spans,idx_meta=meta)

def check_origin(cli,src):
    h=cli.head_object(Bucket=BUCKET,Key=src["key"])
    if int(h["ContentLength"])!=src["size"] or str(h["ETag"])!=src["etag"]:
        raise Blocked("SOURCE_ETAG_OR_LENGTH_DRIFT")
    raw=cli.get_object(Bucket=BUCKET,Key=src["key"]+".idx")["Body"].read(4*1024*1024+1)
    if sha(raw)!=src["idx_sha"]: raise Blocked("SOURCE_INDEX_SHA_CHANGED")

def load_group(cli,root,date,mem,var):
    spec=get_source(cli,date,mem,var)
    out=root/"pressure_wind_groups"/f"{date}_{mem}_{var}.npz"
    arr=np.zeros((16,*GRID),dtype=np.float32);done=np.zeros((16,),np.uint8)
    if out.is_file():
        with np.load(out,allow_pickle=False) as z:
            if str(z["key"])!=spec["key"] or str(z["etag"])!=spec["etag"] or str(z["idx_sha"])!=spec["idx_sha"]:
                raise Blocked("CACHE_PROVENANCE_DRIFT")
            arr=z["field"].copy();done=z["done"].copy()
            if a_sha(arr)!=str(z["field_sha"]): raise Blocked("CACHED_FIELD_HASH_MISMATCH")
        if arr.shape!=(16,*GRID) or done.shape!=(16,) or not np.isin(done,[0,1]).all() or not np.isfinite(arr).all():
            raise Blocked("INVALID_CACHED_FIELDS")
    net=0; metas=[]
    for i,lead in enumerate(LEADS):
        if done[i]:continue
        lo,hi=spec["spans"][lead]
        r=cli.get_object(Bucket=BUCKET,Key=spec["key"],Range=f"bytes={lo}-{hi}")
        payload=r["Body"].read(MAX_GRIB+1)
        if len(payload)!=hi-lo+1 or payload[:4]!=b"GRIB":
            raise Blocked("SOURCE_GRIB_RANGE_BAD")
        field,meta=ec_decode(payload,date,lead,var)
        arr[i]=field;done[i]=1;net+=1
        if len(metas)<2: metas.append(meta)
        if (i+1)%4==0 or i==15:
            npzsave(out,field=arr,done=done,key=np.array(spec["key"]),
                    etag=np.array(spec["etag"]),idx_sha=np.array(spec["idx_sha"]),
                    field_sha=np.array(a_sha(arr)))
            print(f"[P850] {date} {mem}/{var} {done.sum()}/16",flush=True)
    check_origin(cli,spec)
    if not done.all(): raise Blocked("INCOMPLETE_UV_GROUP")
    return arr,{"key":spec["key"],"etag":spec["etag"],"idx_sha256":spec["idx_sha"],
                "source_bytes":spec["size"],"idx":spec["idx_meta"],"field_sha256":a_sha(arr),
                "cache_sha256":sha(out.read_bytes()),"records_downloaded":net,"decode_sample":metas}

def patch_weights():
    lat=np.deg2rad(np.linspace(90,-90,121))
    row=np.cos(lat).clip(0)
    # Fixed 11x12 nonoverlapping latitude/longitude bins.
    ybins=np.array_split(np.arange(121),11)
    xbins=np.array_split(np.arange(240),12)
    w=np.array([[row[yy].sum()*len(xx) for xx in xbins] for yy in ybins],dtype=np.float64)
    return (w/w.sum()).astype(np.float32),ybins,xbins

def make_patch(full,ys,xs):
    # full [member,var,week,time,lat,lon]
    row=np.cos(np.deg2rad(np.linspace(90,-90,121))).clip(0)
    out=np.empty((*full.shape[:-2],len(ys),len(xs)),dtype=np.float32)
    for i,yy in enumerate(ys):
        wt=row[yy]
        if wt.sum()<=0: raise Blocked("EMPTY_AREA_BIN")
        for j,xx in enumerate(xs):
            block=np.take(np.take(full,yy,axis=-2),xx,axis=-1)
            out[...,i,j]=(block*wt.reshape((1,)*(block.ndim-2)+(len(yy),1))).sum(axis=(-2,-1))/(wt.sum()*len(xx))
    return out

def save_report(root,status):
    root.mkdir(parents=True,exist_ok=True)
    jsonsave(root/"P850_STATUS.json",status)
    dest=root/"P16_WIND_P850_E0_DELIVERY_REPORTS.zip"
    with zipfile.ZipFile(dest,"w",zipfile.ZIP_DEFLATED) as z:
        z.write(root/"P850_STATUS.json","reports/STATUS.json")
        for f in sorted((root/"pressure_wind_cases").glob("*.json")):
            z.write(f,"reports/cases/"+f.name)
        for f in sorted((root/"pressure_wind_portable").glob("*")):
            if f.suffix in (".npz",".json"):z.write(f,"portable/"+f.name)
    digest=sha(dest.read_bytes())
    atomic(root/(dest.name+".sha256"),(digest+"  "+dest.name+"\n").encode())
    return dest

def offline_qa():
    def mk(n,off,lev,lead):
        return f"{n}:{off}:d=2009010700:UGRD:{lev}:f{lead}:\n"
    rows=[];off=0;n=1
    for lead in LEADS:
        rows.append(mk(n,off,"850 mb",lead));off+=1100;n+=1
        rows.append(mk(n,off,"700 mb",lead));off+=1100;n+=1
    sel,meta=parse_index("".join(rows).encode(),"2009010700",off,"u850")
    assert len(sel)==16 and meta["other_level_records"]==16
    try:parse_index("".join(rows).replace("850 mb","700 mb").encode(),"2009010700",off,"u850")
    except Blocked:pass
    else:raise AssertionError("700 hPa must not be accepted as 850")
    print("[OFFLINE_QA] MULTILEVEL_850_SELECTION_PASS; NO REMOTE DATA")
    return {"status":"OFFLINE_QA_ONLY_NOT_REAL_850_HPA","contract":CONTRACT}

def run(root,stage="pilot1",offline=False):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    status={"contract":CONTRACT,"stage":stage,"dates":list(DATES),
            "type":"GEFSv12_RETROSPECTIVE_850_HPA_RESEARCH_INPUT_ONLY",
            "era5_future_used_as_input":False,"trained_model":False,
            "official_RPSS":False,"historical_operational_availability_verified":False}
    try:
        if offline:
            status.update(offline_qa());return save_report(root,status),status
        import boto3
        from botocore import UNSIGNED
        from botocore.client import Config
        cli=boto3.client("s3",config=Config(signature_version=UNSIGNED,
                         retries={"max_attempts":7,"mode":"adaptive"},
                         connect_timeout=30,read_timeout=180))
        dates=[DATES[0]] if stage=="pilot1" else list(DATES) if stage=="pilot4" else None
        if dates is None:raise Blocked("INVALID_STAGE")
        ws,ys,xs=patch_weights()
        items=[]
        for date in dates:
            dest=root/"pressure_wind_cases"/f"{date}.npz"
            jdest=dest.with_suffix(".json")
            if dest.exists() and jdest.exists():
                m=json.loads(jdest.read_text())
                if sha(dest.read_bytes())!=m["file_sha256"]:
                    raise Blocked("CASE_FILE_SHA_MISMATCH")
                # Cached case is only accepted when underlying GEFS object sources remain identical.
                for rec in m["provenance"]:
                    spec=get_source(cli,date,rec["member"],rec["variable"])
                    if spec["key"]!=rec["key"] or spec["etag"]!=rec["etag"] or spec["idx_sha"]!=rec["idx_sha256"]:
                        raise Blocked("CACHED_CASE_SOURCE_FINGERPRINT_CHANGED")
                with np.load(dest,allow_pickle=False) as z:
                    f=z["wind"]; leads=z["leads"]
                    if f.shape!=(5,2,2,8,*GRID) or tuple(map(int,leads))!=LEADS or a_sha(f)!=m["wind_sha256"]:
                        raise Blocked("CACHED_CASE_TENSOR_BAD")
                print("[P850 CASE] SHA_VERIFIED_REUSE",date)
            else:
                chunks=[];provenance=[]
                for member in MEMBERS:
                    uv=[]
                    for var in VARS:
                        a,meta=load_group(cli,root,date,member,var)
                        uv.append(a.reshape(2,8,*GRID))
                        provenance.append(dict(member=member,variable=var,**meta))
                    chunks.append(np.stack(uv,axis=0))
                f=np.stack(chunks,axis=0) # 5,2,2,8,121,240
                if f.shape!=(5,2,2,8,*GRID) or not np.isfinite(f).all():
                    raise Blocked("CASE_WIND_SHAPE")
                npzsave(dest,wind=f,leads=np.array(LEADS),init=np.array(date))
                meta={"date":date,"wind_sha256":a_sha(f),"file_sha256":sha(dest.read_bytes()),
                      "provenance":provenance,"real_records":len(MEMBERS)*len(VARS)*len(LEADS),
                      "stage":stage}
                jsonsave(jdest,meta);m=meta
                print("[P850 CASE] REAL_FULL_PASS",date)
            pf=root/"pressure_wind_portable"/f"patch_{date}.npz"
            wind=make_patch(f,ys,xs)
            npzsave(pf,wind_patch=wind,weights=ws,leads=np.array(LEADS),init=np.array(date))
            jsonsave(pf.with_suffix(".json"),{"date":date,"file_sha256":sha(pf.read_bytes()),
                    "wind_patch_sha256":a_sha(wind),"parent_file_sha256":m["file_sha256"]})
            items.append({"date":date,"wind_sha256":m["wind_sha256"],"full_case_sha256":m["file_sha256"]})
        status.update(status="REAL_P850_SOURCE_GATE_PASS_RESEARCH_ONLY",
                      completed=len(items),required=len(dates),items=items)
    except Exception as e:
        status.update(status="BLOCKED_P850_SOURCE_OR_GRIB",exception=repr(e),
                      traceback=traceback.format_exc()[-4500:])
    return save_report(root,status),status

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",choices=["pilot1","pilot4"],default="pilot1")
    ap.add_argument("--root",default="P16_WIND_P850_E0")
    ap.add_argument("--offline-qa",action="store_true")
    a=ap.parse_args()
    p,s=run(a.root,a.stage,a.offline_qa)
    print("[P850 RESULT]",s["status"],p)
