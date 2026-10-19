#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A15-P7-Q0: retrospective GEFS temporal-information necessity diagnostic.

NOT a trained CIDM, not official RPSS, and not a fresh-date generalization claim.
Diagnostics use six ALREADY INSPECTED paired cases. Forecast trajectories are from
single-initialization GEFSv12 reforecasts, not future observations. Labels are
ERA5 weekly verification ONLY. Tiny closed-form ridge is diagnostic, not a CIDM.
"""
from __future__ import annotations
import os, io, json, math, hashlib, argparse, datetime, traceback, zipfile, time
from pathlib import Path
import numpy as np

import a15_p1_gefs_weekly_ensemble as src

PARENT_SHAS = {
 'P2':'dd380fb310546b080314ecb8ad78a2edbecad68413ac21f7696ea6a9feda486c',
 'P3':'0a60d66483bdf81d9e7bdeacf72c40d7eb1e4cabd8f7d6cb6691cc61c903ebd9',
 'P4':'10e471b02b94583c0b6a59857d1f9f6339c3ab2a498845e429a621bb34a25871'
}
DATES=('2017010400','2017070500','2018010300','2018070400','2019010200','2019070300')
VARS=('mslp','tas')
LEADS=(3,4)
MEMBERS=('c00','p01','p02','p03','p04')
# Forecast lead hours from the SAME GEFS initialization; no future ERA5 predictor.
STEPS=(408,456,504,600,624,696,786)
SEGMENTS={3:(0,1,2,3),4:(3,4,5,6)}
MODEL_NAMES=('static','time_ordered','time_shuffled','season_matched_wrong_year')
OUTPUT_ZIP='CIDM_EARTH_A15_P7_DELIVERY_REPORTS.zip'
STRENGTH_L2=8.0  # frozen exploratory ridge penalty
MIN_DELTA=0.01   # preregistered >=1% improvement in diagnostic RMSE

class GateError(RuntimeError):pass

def sha_file(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def sha_bytes(b):return hashlib.sha256(b).hexdigest()

def save_json(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
 os.replace(tmp,p)

def locked_zip(path,name):
 p=Path(path)
 if not p.is_file():raise GateError(f'MISSING_{name}: {p}')
 actual=sha_file(p)
 if actual!=PARENT_SHAS[name]:raise GateError(f'INPUT_SHA_MISMATCH_{name} expected={PARENT_SHAS[name]} got={actual}')
 z=zipfile.ZipFile(p,'r')
 corrupt=z.testzip()
 if corrupt:raise GateError(f'INPUT_CRC_{name}: {corrupt}')
 return z

def read_parents(paths):
 zs={n:locked_zip(paths[n],n) for n in PARENT_SHAS}
 qs={}
 try:
  raw=zs['P3'].read('reports/TRAIN_ONLY_PROXY_QUINTILES.npz')
  with np.load(io.BytesIO(raw),allow_pickle=False) as z:
   # Detect schema as actually recorded, not assume one key.
   data={k:z[k].copy() for k in z.files}
  for season in ('01','07'):
   k=next((x for x in data if x in (f'quantiles_{season}',f'q_{season}',season,{'01':'january','07':'july'}[season])),None)
   if k is None:
    # Support original A15-P3 published schema from provenance source.
    k=next((x for x in data if season in x and np.shape(data[x])==(4,2,2,121,240)),None)
   if k is None:raise GateError('UNKNOWN_FROZEN_P3_QUANTILES_SCHEMA: '+str([(x,data[x].shape) for x in data]))
   q=np.asarray(data[k],dtype=np.float64)
   if q.shape!=(4,2,2,121,240) or not np.isfinite(q).all() or np.any(np.diff(q,axis=0)<-1e-4):
    raise GateError('INVALID_FROZEN_QUINTILE_TENSOR_'+season)
   qs[season]=q
  return zs,qs
 except Exception:
  for z in zs.values():z.close()
  raise

def read_old_pair(zs,date):
 if date.startswith('2019'):
  z=zs['P4'];entry=f'reports/2019_paired_{date}.npz'
  status=json.loads(z.read('reports/STATUS.json'))
  expected=status['case_npz_sha256'][date]
 else:
  z=zs['P2'];entry=f'reports/paired_{date}.npz'
  status=json.loads(z.read(f'reports/case_{date}_metrics.json'))
  expected=status['paired_file_sha256']
 raw=z.read(entry)
 if sha_bytes(raw)!=expected:raise GateError('INNER_PAIRED_CASE_HASH_DRIFT_'+date)
 with np.load(io.BytesIO(raw),allow_pickle=False) as d:
  pred=np.asarray(d['forecast'],dtype=np.float32)
  truth=np.asarray(d['truth'],dtype=np.float32)
  stamp=str(d['date'])
 if pred.shape!=(5,2,2,121,240) or truth.shape!=(2,2,121,240) or stamp!=date:
  raise GateError('OLD_PAIRED_SHAPE_OR_DATE_'+date)
 if not (np.isfinite(pred).all() and np.isfinite(truth).all()):raise GateError('NONFINITE_PAIRED_'+date)
 return pred,truth

def mask_grid():
 lat=90-np.arange(121)*1.5
 w=np.maximum(np.cos(np.deg2rad(lat)),0)
 cells=[]
 for li in range(3):
  r0=40*li;r1=121 if li==2 else 40*(li+1)
  for lj in range(4):
   c0=60*lj;c1=60*(lj+1)
   weights=np.broadcast_to(w[r0:r1,None],(r1-r0,c1-c0)).copy()
   weights/=np.sum(weights)
   cells.append((slice(r0,r1),slice(c0,c1),weights))
 return cells
PATCHES=mask_grid()

def patch_stat(a,func='mean'):
 a=np.asarray(a,dtype=np.float64)
 vals=[]
 for s0,s1,w in PATCHES:
  x=a[...,s0,s1]
  if func=='mean':val=np.sum(x*w,axis=(-2,-1))
  elif func=='rms':val=np.sqrt(np.sum(x*x*w,axis=(-2,-1)))
  else:raise ValueError(func)
  vals.append(val)
 return np.stack(vals,axis=-1)

def forecast_features(week_forecast,quant,temporal=None,sham=False):
 """12 patch rows; predictor columns known at forecast issuance; never use ERA5."""
 fc=np.asarray(week_forecast,np.float64)  # [5,121,240]
 mean=np.mean(fc,axis=0)
 spread=np.std(fc,axis=0,ddof=1)
 qmean=np.mean(quant[1:3],axis=0)   # frozen training-period climate center
 if quant.shape!=(4,121,240):raise GateError('QUANT_SHAPE')
 gradient=(np.abs(np.roll(mean,1,-1)-mean)+np.abs(np.roll(mean,1,-2)-mean))/2
 bas=np.stack([patch_stat(spread),patch_stat(np.abs(mean-qmean)),patch_stat(gradient)],axis=-1)
 # np.stack above yields [12,3]
 if temporal is None:return bas
 tr=np.asarray(temporal,dtype=np.float64)
 if tr.shape!=(5,4,121,240):raise GateError('PATH_SHAPE')
 if sham:tr=tr[:,(0,2,1,3)]    # preserve temporal endpoints; flip two intermediate steps
 tp=tr.mean(axis=0)           # time x lat x lon
 d=np.diff(tp,axis=0)
 norm=1e-4+np.maximum(np.std(fc,axis=0,ddof=1),0.01)
 path_length=np.mean(np.abs(d),axis=0) / norm
 path_curve=np.mean(np.abs(np.diff(d,axis=0)),axis=0) / norm
 # Cross-scale local temporal-flow contrast (original field differences only)
 coarse=(np.roll(d,4,axis=-1)+np.roll(d,-4,axis=-1)+np.roll(d,4,axis=-2)+np.roll(d,-4,axis=-2))/4
 flux=np.mean(np.abs(d-coarse),axis=0)/norm
 extra=np.stack([patch_stat(path_length),patch_stat(path_curve),patch_stat(flux)],axis=-1)
 if not np.isfinite(extra).all():raise GateError('NONFINITE_TEMPORAL_FEATURES')
 return np.concatenate([bas,extra],axis=1)

def read_old_data(zs,qs):
 parent={}
 for date in DATES:
  fc,yy=read_old_pair(zs,date)
  data={}
  for vi,v in enumerate(VARS):
   for wi,w in enumerate(LEADS):
    baseline=forecast_features(fc[:,vi,wi],qs[date[4:6]][:,vi,wi])
    # target: forecast-mean absolute verification ERROR, area average per 3x4 block
    error=np.abs(fc[:,vi,wi].mean(axis=0)-yy[vi,wi])
    target=patch_stat(error)
    data[(v,w)]={'baseline':baseline,'target':target,'truth_only_for_label':True}
  parent[date]=data
 return parent

def load_trace_cache(path,date,member,var,idx_sha,etag):
 if not path.exists():return np.full((len(STEPS),121,240),np.nan,dtype=np.float32)
 with np.load(path,allow_pickle=False) as d:
  keys=(str(d['date']),str(d['member']),str(d['var']),str(d['idx_sha']),str(d['etag']))
  if keys!=(date,member,var,idx_sha,etag):raise GateError('CACHE_PROVENANCE_DRIFT_'+path.name)
  a=np.asarray(d['trace'],dtype=np.float32)
 if a.shape!=(len(STEPS),121,240):raise GateError('CACHE_SHAPE_DRIFT')
 mask=np.isfinite(a).all(axis=(1,2))
 if np.any(np.isfinite(a[~mask])):raise GateError('PARTIAL_NONFINITE_STEP')
 return a

def write_trace_cache(path,a,date,member,var,idx_sha,etag):
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.tmp')
 with open(tmp,'wb') as f:np.savez_compressed(f,trace=a,date=date,member=member,var=var,idx_sha=idx_sha,etag=etag)
 os.replace(tmp,path)

def read_trace_group(client,root,date,member,var):
 key=src.object_key(date,member,var)
 head=client.head_object(Bucket=src.BUCKET,Key=key)
 length=int(head['ContentLength']);etag=str(head.get('ETag',''))
 idx=client.get_object(Bucket=src.BUCKET,Key=key+'.idx')['Body'].read()
 if not 0<len(idx)<(1<<20):raise GateError('INVALID_IDX_SIZE')
 rows,_=src.parse_idx(idx)
 for step in STEPS:src.record_span(rows,STEPS,length,step)
 idx_sha=sha_bytes(idx)
 path=root/'cache'/f'path_{date}_{member}_{var}.npz'
 trace=load_trace_cache(path,date,member,var,idx_sha,etag)
 fresh=0
 for i,step in enumerate(STEPS):
  if np.isfinite(trace[i]).all():continue
  st,en=src.record_span(rows,STEPS,length,step)
  raw=src.get_range(client,key,st,en)
  field,meta=src.read_grid_ec(raw,date,step,var)
  field=np.asarray(field,dtype=np.float32)
  if field.shape!=(121,240) or not np.isfinite(field).all():raise GateError('INVALID_GRIB_FIELD')
  trace[i]=field/100 if var=='mslp' else field-273.15
  write_trace_cache(path,trace,date,member,var,idx_sha,etag)
  fresh+=1
 if not np.isfinite(trace).all():raise GateError('INCOMPLETE_GROUP')
 return trace,{'key':key,'idx_sha256':idx_sha,'etag':etag,'fresh_ranges':fresh,'steps':list(STEPS)}

def gather_path_features(client,root):
 allfeats={};audits={};downloads=0
 for date in DATES:
  values=[];per=[]
  for member in MEMBERS:
   fields=[]
   for var in VARS:
    path,metadata=read_trace_group(client,root,date,member,var)
    fields.append(path)
    per.append({'date':date,'member':member,'var':var,**metadata});downloads+=metadata['fresh_ranges']
    print(f'[P7] {date}/{member}/{var} {len(STEPS)}/{len(STEPS)}, fresh={metadata["fresh_ranges"]}',flush=True)
   values.append(np.stack(fields,axis=0))
  t=np.stack(values) # 5, 2, 7, 121, 240
  if t.shape!=(5,2,len(STEPS),121,240):raise GateError('TRAJECTORY_SHAPE')
  feats={}
  for vi,v in enumerate(VARS):
   for w in LEADS:
    ix=SEGMENTS[w]
    sub=t[:,vi,list(ix)]
    # only small patch features are exported, not raw GRIB or 5x2x7 full trajectories
    feats[(v,w)]={'trace':sub.astype(np.float32)}
  allfeats[date]=feats;audits[date]=per
  print(f'[P7] date {date} temporal states assembled, {downloads} fresh GRIB records so far',flush=True)
 return allfeats,audits,downloads

def feature_matrix(parent,temporal):
 rows={}
 for date in DATES:
  rows[date]={}
  for task in parent[date]:
   basic=parent[date][task]['baseline'];trace=temporal[date][task]['trace']
   fc_feature=parent[date][task]['baseline']
   # quantile-free forecast paths are sufficient for temporal extras, but baselines are frozen old arrays.
   # construct temporal-only feature using original 4-step path and member forecast final field.
   # Forecast mean and spread normalization derives from the trajectory's final sample.
   surrogate=trace[:,-1]
   extra=forecast_features(surrogate,np.stack([surrogate.mean(axis=0)]*4),trace)[:,3:]
   ordered=np.concatenate([basic,extra],axis=1)
   sham=forecast_features(surrogate,np.stack([surrogate.mean(axis=0)]*4),trace,sham=True)[:,3:]
   rows[date][task]={'static':basic,'time_ordered':ordered,
                    'time_shuffled':np.concatenate([basic,sham],axis=1),
                    'target':parent[date][task]['target']}
 # Cross-date placebo: each Jan/Jul case gets exactly one different-year same-month trace.
 for date in DATES:
  candidates=[d for d in DATES if d[4:6]==date[4:6] and d!=date]
  other=sorted(candidates)[0] if sorted(candidates)[0]!=date else sorted(candidates)[-1]
  for task in rows[date]:
   rows[date][task]['season_matched_wrong_year']=np.concatenate([
      rows[date][task]['static'],rows[other][task]['time_ordered'][:,3:]],axis=1)
 return rows

def ridge_predict(xtr,ytr,xte,lam=STRENGTH_L2):
 # Frozen ridge hyperparameter and train-only scaler, with unpenalized intercept.
 mu=xtr.mean(0);sd=np.maximum(xtr.std(0),1e-6)
 a=(xtr-mu)/sd;b=(xte-mu)/sd
 ym=float(ytr.mean());sigma=max(float(ytr.std()),1e-6)
 yy=(ytr-ym)/sigma
 beta=np.linalg.solve(a.T@a+lam*np.eye(a.shape[1]),a.T@yy)
 return ym+sigma*(b@beta)

def eval_folds(rows):
 res={};pairs=[(v,w) for v in VARS for w in LEADS]
 for task in pairs:
  years={};all_obs=[]
  for year in ('2017','2018','2019'):
   train=[d for d in DATES if not d.startswith(year)]
   test=[d for d in DATES if d.startswith(year)]
   ytrain=np.concatenate([rows[d][task]['target'] for d in train]);ytest=np.concatenate([rows[d][task]['target'] for d in test]);
   scores={}
   for m in MODEL_NAMES:
    xtr=np.concatenate([rows[d][task][m] for d in train]);xte=np.concatenate([rows[d][task][m] for d in test])
    pred=ridge_predict(xtr,ytrain,xte)
    scores[m]=float(np.sqrt(np.mean((pred-ytest)**2)))
   years[year]={'heldout_cases':test,'n_patches':len(ytest),'rmse':scores,
      'gain_vs_static_pct':100*(scores['static']-scores['time_ordered'])/max(scores['static'],1e-8),
      'gain_vs_time_shuffled_pct':100*(scores['time_shuffled']-scores['time_ordered'])/max(scores['time_shuffled'],1e-8),
      'gain_vs_wrong_year_pct':100*(scores['season_matched_wrong_year']-scores['time_ordered'])/max(scores['season_matched_wrong_year'],1e-8)}
   all_obs.append(scores)
  mean={m:float(np.mean([r[m] for r in all_obs])) for m in MODEL_NAMES}
  gains={'vs_static_pct':100*(mean['static']-mean['time_ordered'])/mean['static'],
         'vs_time_shuffled_pct':100*(mean['time_shuffled']-mean['time_ordered'])/mean['time_shuffled'],
         'vs_wrong_year_pct':100*(mean['season_matched_wrong_year']-mean['time_ordered'])/mean['season_matched_wrong_year']}
  positive=all(years[y][key]>0 for y in years for key in ('gain_vs_static_pct','gain_vs_time_shuffled_pct','gain_vs_wrong_year_pct'))
  res[f'{task[0]}_W{task[1]}']={'by_heldout_year':years,'fold_mean_rmse':mean,'fold_mean_improvements':gains,
   'exploratory_candidate':bool(positive and gains['vs_static_pct']>=MIN_DELTA*100 and gains['vs_time_shuffled_pct']>0.5 and gains['vs_wrong_year_pct']>0.5)}
 res['decision']={'candidate_both_leads_any_variable':any(
     res[f'{v}_W3']['exploratory_candidate'] and res[f'{v}_W4']['exploratory_candidate'] for v in VARS),
   'formal_training_authorized':False,'official_rpss':None,'cidm_neural_optimization_steps':0,
   'new_unseen_case_count':0,
   'interpretation':'Exploratory year-fold diagnostic on SIX ALREADY SEEN dates; tiny ridge regressors are diagnostic, not CIDM temporal latent model. PASS does not authorize long training.'}
 return res

def fake_offline(root):
 rng=np.random.default_rng(2317)
 # A causally controlled toy mechanism with temporal-order-sensitive feature.
 rows={}
 for i,date in enumerate(DATES):
  rows[date]={}
  for v in VARS:
   for w in LEADS:
    b=rng.normal(size=(12,3))
    s=rng.normal(size=(12,3))
    # Labels deliberately depend on a temporal feature not in baselines.
    y=2+np.maximum(0,0.3*b[:,0]+1.1*s[:,0]+0.1*rng.normal(size=12))
    rows[date][(v,w)]={'static':b,'time_ordered':np.concatenate([b,s],axis=1),
     'time_shuffled':np.concatenate([b,rng.normal(size=(12,3))],axis=1),
     'season_matched_wrong_year':np.concatenate([b,rng.normal(size=(12,3))],axis=1),
     'target':y}
 result=eval_folds(rows)
 for v in VARS:
  for w in LEADS:
   obj=result[f'{v}_W{w}']
   if obj['fold_mean_improvements']['vs_static_pct']<3:raise GateError('OFFLINE_SIGNAL_POSITIVE_CONTROL_FAILED')
 # Calendar and feature-shape tests; no fake data can enter real status.
 for date in DATES:
  for lead in LEADS:
   hours=src.required_steps(date,lead)
   if len(hours)!=28 or hours[0]!=(456 if lead==3 else 624):raise GateError('CALENDAR_CONTRACT')
 original=np.arange(5*4*121*240,dtype=np.float32).reshape(5,4,121,240)/200000
 dummy=np.ones((5,121,240));q=np.stack([np.ones((121,240))*i for i in range(4)])
 a=forecast_features(dummy,q,original,False)
 b=forecast_features(dummy,q,original,True)
 if a.shape!=(12,6) or b.shape!=(12,6):raise GateError('FEATURE_MATRIX_SHAPE')
 if np.max(np.abs(a-b))<1e-8:raise GateError('TIME_SHAM_IS_IDENTICAL')
 save_json(root/'reports'/'OFFLINE_QA.json',{'status':'OFFLINE_QA_PASS_NO_REAL_S3_OR_GEFS_LABEL_EVAL',
    'calendar_contract':True,'feature_shape':list(a.shape),
    'fake_positive_control_pass':True,'fake_metric':result,'no_weather_data_used':True})
 return {'experiment':'CIDM_EARTH_A15_P7_Q0','status':'OFFLINE_QA_PASS_NO_REAL_S3_OR_GEFS_LABEL_EVAL',
          'training_authorized':False,'official_rpss':None}

def make_delivery(root,status,error=None):
 rep=root/'reports';rep.mkdir(exist_ok=True,parents=True)
 save_json(rep/'STATUS.json',status)
 if error is not None:save_json(rep/'ERROR.json',{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()[-16000:]})
 out=root/OUTPUT_ZIP
 tmp=out.with_name(out.name+'.tmp')
 with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(rep.rglob('*')):
   if p.is_file():z.write(p,p.relative_to(root).as_posix())
 os.replace(tmp,out)
 with zipfile.ZipFile(out) as z:
  if z.testzip():raise RuntimeError('DELIVERY_ZIP_CRC_FAILURE')
 digest=sha_file(out)
 (root/(OUTPUT_ZIP+'.sha256')).write_text(digest+'  '+OUTPUT_ZIP+'\n',encoding='ascii')
 return out,digest

def run(args):
 root=Path(args.out).resolve();(root/'reports').mkdir(exist_ok=True,parents=True)
 # Remove STALE status/results from earlier runs while keeping resumable cache untouched.
 for p in (root/'reports').iterdir():
  if p.is_file():p.unlink()
 status={'experiment':'CIDM_EARTH_A15_P7_Q0','status':'BLOCKED_NOT_REAL_QUALIFICATION',
   'source':'NOAA GEFSv12 retrospective S3 reforecast, not operational 2017-2019',
   'parent_sha256':PARENT_SHAS,'forecast_step_hours':STEPS,'dates':list(DATES),
   'training_authorized':False,'official_rpss':None,'cidm_neural_training_steps':0,
   'issue_time_source_eligibility_proven':False,'data_leakage_guard':'ERA5 verification only labels, never temporal forecast predictors',
   'runtime_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 try:
  if args.offline_smoke:
   status.update(fake_offline(root));return 0
  paths={'P2':args.p2,'P3':args.p3,'P4':args.p4}
  zs,qs=read_parents(paths)
  try:
   parent=read_old_data(zs,qs)
   cli=src.anonymous_client()
   traces,audits,nfresh=gather_path_features(cli,root)
   features=feature_matrix(parent,traces)
   outcome=eval_folds(features)
   save_json(root/'reports'/'BY_DATE_S3_SOURCE_AUDIT.json',audits)
   save_json(root/'reports'/'TEMPORAL_PATH_FEATURES.json', {
      date:{f'{v}_W{w}':{'static':features[date][(v,w)]['static'].tolist(),
                    'ordered':features[date][(v,w)]['time_ordered'].tolist(),
                    'shuffled':features[date][(v,w)]['time_shuffled'].tolist(),
                    'wrong_year':features[date][(v,w)]['season_matched_wrong_year'].tolist(),
                    'target_error_patch':features[date][(v,w)]['target'].tolist()}
         for v in VARS for w in LEADS} for date in DATES})
   save_json(root/'reports'/'YEAR_BLOCKED_EXPLORATORY_MECHANISM_GATE.json',outcome)
   status.update({'status':'REAL_SIX_SEEN_CASE_TEMPORAL_Q0_EXPLORATORY_COMPLETE',
      'fresh_s3_grib_ranges':nfresh,'expected_total_records':len(DATES)*len(MEMBERS)*len(VARS)*len(STEPS),
      'genuine_weather_data_used':True,'exploratory_candidate_gate':outcome['decision']['candidate_both_leads_any_variable'],
      'formal_training_authorized':False})
   print('[P7] six-date temporal path experiment COMPLETED; no formal training authorization',flush=True)
  finally:
   for z in zs.values():z.close()
  return 0
 except Exception as e:
  status.update({'status':'BLOCKED_NOT_MECHANISM_RESULT','failure_type':type(e).__name__})
  save_json(root/'reports'/'ERROR.json',{'message':str(e),'traceback':traceback.format_exc()[-12000:]})
  print('[P7] BLOCKED:',type(e).__name__,str(e)[:1500],flush=True)
  return 2
 finally:
  pack,digest=make_delivery(root,status)
  print('[P7] report_zip:',pack,'sha256:',digest,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',default='./CIDM_A15_P7_RUN')
 p.add_argument('--p2',default='CIDM_EARTH_A15_P2_DELIVERY_REPORTS.zip')
 p.add_argument('--p3',default='CIDM_EARTH_A15_P3_DELIVERY_REPORTS.zip')
 p.add_argument('--p4',default='CIDM_EARTH_A15_P4_DELIVERY_REPORTS.zip')
 p.add_argument('--offline-smoke',action='store_true')
 exit(run(p.parse_args()))
