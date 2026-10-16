"""CIDM-MHC-Real A1: real AIWQ historical weekly data, matched MHC/Stateless pilot.
No Google Drive; verified private HF dataset cache; HF model checkpoints; GitHub light metadata.
Only small paired qualification. NEVER official leaderboard RPSS, never operational issue-time guarantee.
"""
from __future__ import annotations
import os,json,hashlib,zipfile,datetime as dt,random,math,traceback,csv,sys,copy,time
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

BASE=Path(os.getenv('MHC_A1_HOME','/content/MHC_REAL_A1')).resolve()
DATA_REPO=os.getenv('MHC_A1_HF_DATASET','wuff-mann/CIDM-MHC-AIWQ-ERA5-Cache')
MODEL_REPO=os.getenv('MHC_A1_HF_MODEL','wuff-mann/AI_Weather_Quest')
GH_REPO=os.getenv('MHC_A1_GITHUB','Mr-wuff/AI-Weather-Quest')
YEARS=range(2012,2018); VARS=('mslp','tas'); GRID=(121,240)
SPLITS={'train':('2013','2014','2015'),'validation':('2016',),'audit_not_fresh_official':('2017',)}
SEEDS=(17,41); ARMS=('stateless','mhc','mhc_noaux')
STEPS=int(os.getenv('MHC_A1_STEPS','64'))
LEGACY_SOURCE_SHA='498fa31f672759981d3039c03705cb720cd0ac799f78fd78e33acbcc71e5ee18'
LEGACY_RESULT_SHA='989de95a8bba3fdacc8de007a68aae4945f3198e310f6018d7cba8235cf7de4c'
NEW_PREFIX='experiments/MHC_REAL_A1_FIX1'
HF_SYNC_STEPS=int(os.getenv('MHC_A1_HF_SYNC_STEPS','16'))
if HF_SYNC_STEPS < 8 or HF_SYNC_STEPS % 8:raise RuntimeError('HF_SYNC_STEPS_MUST_BE_8_MULTIPLE')
EVAL_AT=tuple(sorted(set([0]+list(range(8,STEPS+1,8))+[STEPS])))
VAL_N=12; TRAIN_HISTORY=8
LEADS=(4,5)  # latest known Monday = Thursday-10, W3/W4 Mondays are +4/+5 weeks
SOURCE_A0_SHA='c4c45e8c0e48ee5f29b78006710f4ab17de537ac0c76ef3e8bb9b332b6a37fe3'
BASE.mkdir(parents=True,exist_ok=True)


def sha_file(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
 return h.hexdigest()

def jdump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False,default=str),encoding='utf8')

def get_secret(key):
 val=os.environ.get(key,'')
 if val:return val.strip()
 try:
  from google.colab import userdata
  return (userdata.get(key) or '').strip()
 except Exception:return ''

def log(*parts):print('[MHC-A1]',*parts,flush=True)

def get_hf(required=True):
 token=get_secret('HF_TOKEN')
 if required and not token:raise RuntimeError('MISSING_HF_TOKEN: Google Drive disabled; private HF persistence is required')
 from huggingface_hub import HfApi
 return HfApi(token=token),token

# IMPORTANT: no official password is ever printed, saved, archived, or uploaded.
def official_password():
 p=get_secret('AIWQ_PASSWORD')
 if not p:raise RuntimeError('MISSING_AIWQ_PASSWORD')
 return p

def ensure_private_dataset(api):
 try:api.create_repo(repo_id=DATA_REPO,repo_type='dataset',private=True,exist_ok=True)
 except Exception as e:
  # Existing private repo may be accessible despite lacking create rights.
  log('HF dataset create/reuse:',type(e).__name__,str(e)[:150])
 info=api.repo_info(repo_id=DATA_REPO,repo_type='dataset')
 if not info.private:raise RuntimeError('RAW_CACHE_REPO_MUST_BE_PRIVATE: '+DATA_REPO)

def ensure_model_repo(api):
 info=api.repo_info(repo_id=MODEL_REPO,repo_type='model')
 if info is None:raise RuntimeError('HF_MODEL_REPO_UNAVAILABLE')

def validate_raw_file(p,year,var,expect=None):
 import xarray as xr,pandas as pd
 with xr.open_dataset(p,decode_times=True) as ds:
  tm=next((x for x in ('time','date','valid_time') if x in ds.coords),None)
  lat=next((x for x in ('latitude','lat') if x in ds.coords),None)
  lon=next((x for x in ('longitude','lon') if x in ds.coords),None)
  if not all([tm,lat,lon]):raise RuntimeError('MISSING_AXES '+str(p))
  if (len(ds[lat]),len(ds[lon])) != GRID:raise RuntimeError('BAD_GRID '+str(p))
  if not np.allclose(np.abs(np.diff(ds[lat].values)),1.5) or not np.allclose(np.abs(np.diff(ds[lon].values)),1.5):
   raise RuntimeError('BAD_SPACING')
  fields=[f for f in ds.data_vars if ds[f].ndim==3 and set((tm,lat,lon))==set(ds[f].dims)]
  if len(fields)!=1:raise RuntimeError('AMBIGUOUS_DATA_FIELD')
  times=pd.DatetimeIndex(ds[tm].values)
  dates=[t.date() for t in times]
  if any(d.year!=year for d in dates) or len(dates) not in (365,366):raise RuntimeError('BAD_TIME_AXIS')
  if any((dates[i+1]-dates[i]).days!=1 for i in range(len(dates)-1)):raise RuntimeError('NOT_DAILY_7DAY_MEAN')
  field=ds[fields[0]]
  sel=[i for i,d in enumerate(dates) if d.weekday()==0]
  x=np.asarray(field.transpose(tm,lat,lon).isel({tm:sel}).values,dtype=np.float32)
  if not np.isfinite(x).all():raise RuntimeError('NONFINITE_FIELD')
  if np.std(x)==0:raise RuntimeError('CONSTANT_FIELD')
  monday=[dates[i].isoformat() for i in sel]
  fingerprint=sha_file(p)
  if expect and fingerprint!=expect:raise RuntimeError('RAW_FILE_SHA_MISMATCH '+str(p))
  return {'year':year,'var':var,'sha256':fingerprint,'size_bytes':p.stat().st_size,'field_name':fields[0],'units':str(field.attrs.get('units','')),
          'monday_dates':monday,'data':x,'filename':p.name}

def restore_or_download(year,var,manifest,api,token):
 """Always verify pinned A0 raw sha before using; upload private snapshot only."""
 info=next(x for x in manifest if x['year']==year and x['variable']==var)
 cache=BASE/'raw'/var/str(year);cache.mkdir(parents=True,exist_ok=True)
 candidates=list(cache.glob('*.nc'))
 # Reuse the exact A0 raw files from the SAME transient Colab runtime, if still present.
 if not candidates:
  prev=Path('/content/CIDM_MHC_REAL_A0')/'raw'/var/str(year)
  previous=list(prev.glob('*.nc')) if prev.is_dir() else []
  if not previous:
   old_a1=Path('/content/MHC_REAL_A1')/'raw'/var/str(year)
   previous=list(old_a1.glob('*.nc')) if old_a1.is_dir() else []
  if len(previous)==1 and sha_file(previous[0])==info['file_sha256']:
   import shutil
   shutil.copy2(previous[0],cache/previous[0].name)
   candidates=list(cache.glob('*.nc'))
 if len(candidates)==1 and sha_file(candidates[0])==info['file_sha256']:
  p=candidates[0];src='local'
 else:
  for q in candidates:q.unlink()
  try:
   from huggingface_hub import hf_hub_download
   repos=api.list_repo_files(repo_id=DATA_REPO,repo_type='dataset')
   matches=[r for r in repos if r.startswith('raw/'+var+'/'+str(year)+'/') and r.endswith('.nc')]
   if len(matches)!=1:raise FileNotFoundError('HF_CACHE_MISS')
   p0=Path(hf_hub_download(repo_id=DATA_REPO,repo_type='dataset',filename=matches[0],token=token))
   if sha_file(p0)!=info['file_sha256']:raise RuntimeError('HF_CACHE_SHA_MISMATCH')
   p=cache/p0.name
   import shutil;shutil.copyfile(p0,p);src='hf_private'
  except Exception as ex:
   log('Cache miss',year,var,type(ex).__name__,str(ex)[:120]);src='ecmwf'
   pwd=official_password()
   from AI_WQ_package import retrieve_training_data
   fn=retrieve_training_data.retrieve_annual_training_data
   output=fn(year,var,pwd,local_destination=str(cache))
   candidates=list(cache.rglob('*.nc'))
   if isinstance(output,(str,os.PathLike)) and Path(output).exists():candidates += [Path(output)]
   cand=list({q.resolve() for q in candidates if q.is_file()})
   if len(cand)!=1:raise RuntimeError(f'OFFICIAL_DOWNLOAD_AMBIGUOUS_{year}_{var} {len(cand)}')
   p=cand[0]
 data=validate_raw_file(p,year,var,info['file_sha256'])
 if src in ('local','ecmwf'):
  # The cache must stay private. No raw meteorology goes to GitHub or public model repo.
  remote=f'raw/{var}/{year}/{p.name}'
  existing=api.list_repo_files(repo_id=DATA_REPO,repo_type='dataset')
  if remote in existing:
   log('SKIP REDUNDANT RAW CACHE COMMIT; verified local SHA',remote)
  else:
   api.upload_file(path_or_fileobj=str(p),path_in_repo=remote,repo_id=DATA_REPO,repo_type='dataset',commit_message=f'Private verified ECMWF AIWQ cache: {year} {var}')
   info_remote=api.list_repo_files(repo_id=DATA_REPO,repo_type='dataset')
   if remote not in info_remote:raise RuntimeError('HF_RAW_UPLOAD_UNVERIFIED '+remote)
 log('REAL DATA READY',year,var,src, len(data['monday_dates']),'Mondays',data['sha256'][:12])
 return data,src


def read_embedded_manifest(a0_report_zip):
 with zipfile.ZipFile(a0_report_zip) as z:
  assert z.testzip() is None
  manifest=json.loads(z.read('reports/official_file_hashes.json'))
  case_csv=z.read('reports/causal_issue_samples.csv').decode()
  cases=list(csv.DictReader(case_csv.splitlines()))
  stat=json.loads(z.read('reports/status.json'))
 if stat['status']!='Q0_OFFICIAL_REAL_DATA_BRIDGE_PASS_NO_TRAINING' or len(manifest)!=12 or len(cases)!=257:
  raise RuntimeError('A0_REAL_BRIDGE_CONTRACT_FAIL')
 return manifest,cases,stat

class FieldBank:
 def __init__(self,raw):
  self.x={}
  for r in raw:
   for date,arr in zip(r['monday_dates'],r['data']):self.x[(r['var'],date)]=arr
  # normalise on 2012–15 observations only; never 2016/2017
  self.stats={}
  for v in VARS:
   xs=np.stack([x for (vv,d),x in self.x.items() if vv==v and int(d[:4])<=2015])
   mu=float(xs.mean(dtype=np.float64)); sd=float(xs.std(dtype=np.float64))
   if sd<=0:raise RuntimeError('ZERO_STANDARD_DEVIATION')
   self.stats[v]=(mu,sd)
  self.lat=np.deg2rad(np.linspace(90,-90,121,dtype=np.float32))
 def grab(self,v,date):
  arr=self.x.get((v,date))
  if arr is None:raise RuntimeError('MISSING_LABEL '+v+' '+date)
  mu,sd=self.stats[v];return ((arr-mu)/sd).astype(np.float32)
 def case(self,case):
  history=case['history_mondays'].split(';')
  starts=[case['W3_monday'],case['W4_monday']]
  arr=np.stack([np.stack([self.grab(v,d) for v in VARS]) for d in history])
  truth=np.stack([np.stack([self.grab(v,d) for v in VARS]) for d in starts])
  issue=dt.date.fromisoformat(case['issue_thursday_00utc'])
  phase=2*math.pi*(issue.timetuple().tm_yday-1)/365.25
  return torch.from_numpy(arr),torch.from_numpy(truth),torch.tensor([math.sin(phase),math.cos(phase)],dtype=torch.float32)


def low(x):return F.avg_pool2d(x,kernel_size=5,stride=1,padding=2,count_include_pad=False)
def rms(x):return torch.sqrt(torch.clamp(x.square().mean(dim=(1,2,3),keepdim=True),min=1.e-20))

class MHCWeather(nn.Module):
 """Same parameters for all 3 arms; MHC memories causal. This is not a transplanted PDEBench weight set."""
 def __init__(self,d=16,alpha=.25,limit=.01):
  super().__init__();self.d=d;self.alpha=alpha;self.limit=limit
  self.encoder=nn.Sequential(nn.Conv2d(20,d,3,padding=1),nn.GELU(),nn.Conv2d(d,d,3,padding=1))
  self.flux=nn.Conv2d(d,d,3,padding=1,bias=False)
  self.local=nn.Conv2d(d,d,3,padding=1)
  self.skewweight=nn.Parameter(torch.randn(d,d)*.002)
  self.head=nn.Sequential(nn.Conv2d(2*d+2,d,3,padding=1),nn.GELU(),nn.Conv2d(d,d,3,padding=1),nn.GELU(),nn.Conv2d(d,d,1))
  nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
  self.decoder=nn.Sequential(nn.Conv2d(d,16,3,padding=1),nn.GELU(),nn.Conv2d(16,2,1))
  nn.init.normal_(self.decoder[-1].weight,mean=0,std=.005)
  nn.init.zeros_(self.decoder[-1].bias)
 def forward(self,h,phase,arm='mhc',collect=False,memory_policy='full',reference_action_rms=None,return_exposure=False):
  assert arm in ARMS or arm=='baseline'
  if memory_policy not in ('full','reset'):raise RuntimeError('UNKNOWN_FROZEN_MEMORY_POLICY')
  if reference_action_rms is not None and len(reference_action_rms)!=5:raise RuntimeError('BAD_EXPOSURE_LENGTH')
  B,T,C,H,W=h.shape
  ll=torch.linspace(math.pi/2,-math.pi/2,H,device=h.device,dtype=h.dtype)
  coslat=torch.cos(ll).view(1,1,H,1).expand(B,1,H,W)
  sinlat=torch.sin(ll).view(1,1,H,1).expand(B,1,H,W)
  ph=phase.view(B,2,1,1).expand(B,2,H,W)
  z=self.encoder(torch.cat((h.reshape(B,16,H,W),coslat,sinlat,ph),dim=1))
  prev=h[:,-1]
  slow=low(z)
  values=[];lowtargets=[];fasttargets=[];acts=[];exposure=[]
  skew=self.skewweight-self.skewweight.T
  for step in range(5):
   # fixed deterministic arithmetic order; all arms start from same init and weights.
   flux=.015*self.flux(z)
   local=.015*torch.tanh(self.local(z))
   sk=.015*torch.einsum('ij,bjhw->bihw',skew,z)
   zbase=z+flux+local+sk
   tendency=flux+local+sk
   if arm=='stateless' or arm=='baseline' or memory_policy=='reset':slob=low(zbase)
   else:slob=slow+self.alpha*low(tendency)
   fast=zbase-slob
   if arm=='baseline':corr=torch.zeros_like(zbase)
   else:
    gate=self.head(torch.cat([slob,fast,prev],1))
    # exactly bounded action; zero gate -> zero action
    raw=torch.tanh(gate)
    corr=self.limit*rms(zbase)*raw/(rms(raw).detach()+0.1)
    cap=self.limit*rms(zbase)/(rms(corr).detach()+1.e-12)
    corr=corr*cap.clamp(max=1.0)
    if reference_action_rms is not None:
     wanted=reference_action_rms[step].to(device=corr.device,dtype=corr.dtype)
     # Deterministic post-action norm matching, without using any future target.
     corr=corr*(wanted/(rms(corr).detach()+1.e-20))
     rel=torch.abs(rms(corr)-wanted)/(wanted+1.e-12)
     if float(rel.max())>1.e-4:raise RuntimeError(f'ACTION_RMS_EXPOSURE_MATCH_FAIL_STEP{step}')
   exposure.append(rms(corr).detach().cpu())
   z=zbase+corr
   if arm in ('mhc','mhc_noaux') and memory_policy=='full':slow=slob+self.alpha*low(corr)
   else:slow=low(z)
   prev=prev+0.10*self.decoder(z)
   values.append(prev)
   lowtargets.append(low(prev));fasttargets.append(prev-low(prev))
   acts.append((rms(corr)/(rms(zbase).detach()+1.e-12)).mean().detach())
  out=torch.stack([values[3],values[4]],1)
  standard=(out,torch.stack(acts),torch.stack([lowtargets[3],lowtargets[4]],1),torch.stack([fasttargets[3],fasttargets[4]],1))
  return standard+(exposure,) if return_exposure else standard


def loss_fn(model,h,target,phase,arm,latweights):
 out,actions,lo,hi=model(h,phase,arm)
 weight=latweights.to(out.device)
 # normalized variable errors, global cosine-lat weighting
 loss=((out-target)**2*weight).mean()
 if arm!='mhc_noaux':
  lo_t=low(target.flatten(0,1)).reshape_as(target)
  hi_t=target-lo_t
  # diagnostic proxy of multiscale targets, not literal JEPA encoder
  loss=loss+0.02*((lo-lo_t)**2*weight).mean()+0.005*((hi-hi_t)**2*weight).mean()
 return loss,out,actions

def scalar_loss(model,bank,case,arm,device):
 h,y,ph=bank.case(case)
 h=h[None].to(device);y=y[None].to(device);ph=ph[None].to(device)
 w=torch.cos(torch.linspace(math.pi/2,-math.pi/2,121,device=device)).clamp_min(0).view(1,1,1,121,1)
 return loss_fn(model,h,y,ph,arm,w)

def eval_arm(model,bank,cases,arm,device):
 model.eval();records=[]
 with torch.no_grad():
  for c in cases:
   loss,out,action=scalar_loss(model,bank,c,arm,device)
   _,target,_=bank.case(c); y=target[None].to(device)
   lat=torch.cos(torch.linspace(math.pi/2,-math.pi/2,121,device=device)).view(1,1,1,121,1)
   err=((out-y)**2*lat).mean(dim=(0,2,3,4)).cpu().tolist()
   records.append({'issue':c['issue_thursday_00utc'],'W3_MSE':float(err[0]),'W4_MSE':float(err[1]),'ctrl_max_rms':float(action.max())})
 model.train()
 return records

def score(rows):
 return {key:float(np.mean([r[key] for r in rows])) for key in ('W3_MSE','W4_MSE','ctrl_max_rms')}

def paired_sampling(cases,seed):
 rng=np.random.default_rng(seed)
 return [cases[int(i)] for i in rng.integers(0,len(cases),size=STEPS)]


def save_state(model,opt,stage,seed,arm,step,extra,api=None):
 """Always write a complete local checkpoint; NEVER make a separate remote commit."""
 p=BASE/'checkpoints'/f'{stage}_seed{seed}_{arm}_step{step:04}.pt';p.parent.mkdir(parents=True,exist_ok=True)
 torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'seed':seed,'arm':arm,
  'config':{'steps':STEPS,'leads':LEADS,'alpha':model.alpha,'limit':model.limit},
  'torch_rng':torch.get_rng_state(),'metrics':extra},p)
 remote=f'{NEW_PREFIX}/{p.name}'
 return {'sha256':sha_file(p),'hf_path':remote,'bytes':p.stat().st_size,'local_path':str(p),'step':step,'seed':seed,'arm':arm}


def restore_progress_pair(api,token,report_dir):
 """Prefer new atomic progress, else merge pinned actual 429 report and original remote index."""
 from huggingface_hub import hf_hub_download
 local_pack=BASE/'evidence'/'CIDM_MHC_REAL_A1_PAIRED_PILOT_REPORTS.zip'
 if sha_file(local_pack)!=LEGACY_RESULT_SHA:raise RuntimeError('PINNED_LEGACY_429_REPORT_SHA_MISMATCH')
 with zipfile.ZipFile(local_pack) as z:
  if z.testzip() is not None:raise RuntimeError('PINNED_LEGACY_429_CRC_FAIL')
  embedded=json.loads(z.read('reports/progress.json'))
 if embedded.get('contract_source_sha')!=LEGACY_SOURCE_SHA or embedded.get('steps')!=STEPS:
  raise RuntimeError('PINNED_LEGACY_429_CONTRACT_MISMATCH')
 progress=None; origin='actual_user_429_report'
 try:
  remote=hf_hub_download(repo_id=MODEL_REPO,repo_type='model',filename=f'{NEW_PREFIX}/progress.json',token=token,force_download=True)
  recent=json.loads(Path(remote).read_text())
  if recent.get('original_training_source_sha')!=LEGACY_SOURCE_SHA or recent.get('implementation_source_sha')!=sha_file(__file__) or recent.get('steps')!=STEPS:
   raise RuntimeError('FIX1_REMOTE_PROGRESS_CONTRACT_MISMATCH')
  progress=recent;origin='atomic_fix1_progress'
 except Exception as e:
  if 'CONTRACT_MISMATCH' in str(e):raise
  log('NO FIX1 PROGRESS:',type(e).__name__,str(e)[:120])
 if progress is None:
  # The upload that failed was progress.json AFTER seed41/stateless step24 checkpoint.
  # The user's immutable progress ZIP provides that orphan's expected SHA.
  progress=dict(embedded)
  try:
   old_path=hf_hub_download(repo_id=MODEL_REPO,repo_type='model',filename='experiments/MHC_REAL_A1/progress.json',token=token,force_download=True)
   old=json.loads(Path(old_path).read_text())
   if old.get('contract_source_sha')==LEGACY_SOURCE_SHA and old.get('steps')==STEPS:
    old_snaps={x['hf_path']:x['sha256'] for x in old.get('checkpoints',[])}
    new_snaps={x['hf_path']:x['sha256'] for x in embedded.get('checkpoints',[])}
    if any(name in new_snaps and new_snaps[name]!=sha for name,sha in old_snaps.items()):raise RuntimeError('LEGACY_PROGRESS_CHECKPOINT_HASH_CONFLICT')
    # Current uploaded result is a later real run observation, not a guess.
    log('OLD HF PROGRESS VERIFIED against later immutable uploaded 429 report')
  except Exception as e:
   if 'HASH_CONFLICT' in str(e):raise
   log('OLD HF PROGRESS not usable, keep pinned actual report:',type(e).__name__,str(e)[:100])
 log('PROGRESS RESTORE',origin,len(progress.get('results',[])),'finished arms',len(progress.get('checkpoints',[])),'snapshot references')
 jdump(report_dir/'resume_origin.json',{'source':origin,'uploaded_429_sha256':LEGACY_RESULT_SHA,'old_training_sha256':LEGACY_SOURCE_SHA,
  'loaded_completed':len(progress.get('results',[])),'loaded_snapshots':len(progress.get('checkpoints',[]))})
 return progress


def verified_resume_checkpoint(api,token,snapshots,seed,arm,device):
 """Inspect NEW and LEGACY pinned checkpoint records; recover latest valid remote state.
 Missing candidate may be an orphan/local-only failure: fall back; SHA mismatch hard fails.
 """
 from huggingface_hub import hf_hub_download
 eligible=[]
 for snap in snapshots:
  if snap.get('seed') not in (None,seed) or snap.get('arm') not in (None,arm):continue
  path=snap.get('hf_path','')
  if not (path.startswith('experiments/MHC_REAL_A1/') or path.startswith(NEW_PREFIX+'/')):continue
  import re
  m=re.search(rf'pilot_seed{seed}_{arm}_step(\d+)\.pt$',path)
  if not m:continue
  step=int(m.group(1))
  if step>STEPS:raise RuntimeError('SAVED_STEP_EXCEEDS_CONTRACT')
  eligible.append((step,snap))
 for step,snap in sorted(eligible,key=lambda z:z[0],reverse=True):
  try:
   fn=hf_hub_download(repo_id=MODEL_REPO,repo_type='model',filename=snap['hf_path'],token=token)
  except Exception as e:
   # File not found may be caused by previously failed commit. Authentication/rate limits must not be suppressed.
   msg=str(e)
   if any(t in msg for t in ('404','EntryNotFound','RemoteEntryNotFound')):
    log('REMOTE CHECKPOINT ABSENT; FALL BACK',snap['hf_path']);continue
   raise
  if sha_file(fn)!=snap['sha256']:
   raise RuntimeError('RESUME_CHECKPOINT_SHA256_MISMATCH '+snap['hf_path'])
  state=torch.load(fn,map_location=device,weights_only=True)
  if int(state['step'])!=step or state['seed']!=seed or state['arm']!=arm or state['config']['steps']!=STEPS:
   raise RuntimeError('RESUME_CHECKPOINT_CONTENT_CONTRACT_FAIL')
  log('VERIFIED HF RESUME',seed,arm,'step',step,snap['hf_path'])
  return step,state,snap
 return 0,None,None


def commit_atomic_progress(api,token,checkpoint,progress_json,seed,arm,step):
 """One HF commit for checkpoint + progress, atomically. One commit per >=16 steps."""
 from huggingface_hub import CommitOperationAdd
 path=Path(checkpoint['local_path'])
 operations=[CommitOperationAdd(path_in_repo=checkpoint['hf_path'],path_or_fileobj=str(path)),
   CommitOperationAdd(path_in_repo=f'{NEW_PREFIX}/progress.json',path_or_fileobj=str(progress_json))]
 api.create_commit(repo_id=MODEL_REPO,repo_type='model',operations=operations,
                   commit_message=f'MHC A1 FIX1 atomic seed{seed} {arm} step{step}',token=token)
 # Single repository listing to confirm BOTH exist in committed tree.
 listed=set(api.list_repo_files(repo_id=MODEL_REPO,repo_type='model'))
 if checkpoint['hf_path'] not in listed or f'{NEW_PREFIX}/progress.json' not in listed:
  raise RuntimeError('ATOMIC_HF_CHECKPOINT_AND_INDEX_MISSING')
 log('HF ONE-COMMIT CHECKPOINT+INDEX',seed,arm,step,checkpoint['sha256'][:12])

def frozen_memory_eval(model,bank,cases,device):
 """Factual versus memory-reset matched action RMS, same seed17 trained MHC weights.
 No training or future target conditioning. All 2016 validation cases, 2017 remains sealed.
 """
 model.eval();rows=[]
 with torch.no_grad():
  for c in cases:
   h,t,phase=bank.case(c)
   h=h[None].to(device);t=t[None].to(device);phase=phase[None].to(device)
   factual,actions,_,_,per_step=model(h,phase,arm='mhc',return_exposure=True)
   reset,other_actions,_,_,matched=model(h,phase,arm='mhc',memory_policy='reset',
    reference_action_rms=per_step,return_exposure=True)
   weight=torch.cos(torch.linspace(math.pi/2,-math.pi/2,121,device=device)).view(1,1,121,1)
   def mse(x):
    return (((x-t)**2)*weight).mean(dim=(0,2,3,4)).cpu().tolist()
   a,b=mse(factual),mse(reset)
   diff=float(torch.max(torch.stack([torch.abs(x-y)/(x+1e-12) for x,y in zip(per_step,matched)])).item())
   if diff>1e-4:raise RuntimeError('REAL_FROZEN_ACTION_RMS_NOT_MATCHED')
   rows.append({'issue':c['issue_thursday_00utc'],'full_W3_MSE':float(a[0]),'full_W4_MSE':float(a[1]),
    'reset_equalRMS_W3_MSE':float(b[0]),'reset_equalRMS_W4_MSE':float(b[1]),
    'max_exposure_relative_error':diff,'reference_max_control_rms':float(actions.max())})
 return rows


def early_stop_frozen_audit(api,token,bank,split,vs,report,report_dir,recovered,device):
 """Verify seed17 complete heads and execute a no-update memory counterfactual.
 Gate already mathematically failed from seed17; do not waste seed41 training.
 """
 done={(r['seed'],r['arm']):r for r in recovered['results']}
 if not all((17,a) in done for a in ('stateless','mhc','mhc_noaux')):return None
 base=done[(17,'stateless')]['validation']; mhc=done[(17,'mhc')]['validation']
 gain={w:100*(base[w]-mhc[w])/max(base[w],1.e-15) for w in ('W3_MSE','W4_MSE')}
 if min(gain.values())>=0.5:return None  # No early stop if preregistered seed gate passed.
 log('KNOWN HARD GATE FAIL: seed17',gain,'=> STOP FURTHER OPTIMIZER WORK')
 # Verify original three completed remote model checkpoints, no trusting manifest alone.
 models={};verified={}
 for arm in ARMS:
  step,state,snap=verified_resume_checkpoint(api,token,recovered['checkpoints'],17,arm,device)
  if step!=STEPS or state is None:raise RuntimeError('EARLY_STOP_REQUIRES_VERIFIED_SEED17_STEP64_'+arm)
  models[arm]=MHCWeather().to(device)
  models[arm].load_state_dict(state['model']);models[arm].eval()
  verified[arm]={'step':step,'sha256':snap['sha256'],'hf_path':snap['hf_path']}
 # Original 12-item validation score replay: contract must be exactly preserved.
 for arm in ARMS:
  current=score(eval_arm(models[arm],bank,vs,arm,device))
  old=done[(17,arm)]['validation']
  for k in ('W3_MSE','W4_MSE'):
   if abs(current[k]-old[k])>5.e-5*max(abs(old[k]),1.e-10):
    raise RuntimeError('REPLAY_12_ISSUE_SCORE_MISMATCH_'+arm+'_'+k)
  log('EXACT-SET SEED17 VAL REPLAY',arm,current['W3_MSE'],current['W4_MSE'])
 # At this point, use all already designated 2016 validation cases, NEVER 2017.
 rows=frozen_memory_eval(models['mhc'],bank,split['validation'],device)
 out=report_dir/'frozen_2016_memory_ablation.csv'
 with open(out,'w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 def aggregate(k):return float(np.mean([r[k] for r in rows]))
 benefits={}
 for w in ('W3','W4'):
  full=aggregate('full_'+w+'_MSE');reset=aggregate('reset_equalRMS_'+w+'_MSE')
  benefits[w]={'full_mse':full,'reset_equal_rms_mse':reset,
   'reset_worsening_pct':100*(reset-full)/max(full,1.e-15),
   'full_better_windows':sum(r['reset_equalRMS_'+w+'_MSE']>r['full_'+w+'_MSE'] for r in rows),
   'issues':len(rows)}
 report.update(status='MHC_REAL_A1_EARLY_GATE_FAIL_FROZEN_AUDIT_COMPLETED_NO_TRAINING',
   reason='seed17 already fails preregistered >=0.5% both W3/W4 MHC-v-Stateless; seed41 training cannot restore conjunctive Gate',
   seed17_mhc_vs_stateless_pct=gain,seed17_completed_remote_verified=verified,
   seed41_future_training_skipped=True,
   all_2016_frozen_memory_reset_equal_action_rms=benefits,independent_2017_audit_opened=False,
   official_aiwq_rpss=None,is_official_submission=False)
 jdump(report_dir/'frozen_audit.json',report)
 jdump(report_dir/'status.json',report)
 try:sync=github_sync_light([Path(__file__),report_dir/'status.json',report_dir/'frozen_audit.json'])
 except Exception as e:sync={'status':'FAILED_RETRY_LATER','error_type':type(e).__name__,'message':str(e)[:250]}
 jdump(report_dir/'github_sync.json',sync)
 pack,side=write_archive(report_dir)
 # Only ONE optional HF commit; if current throttle blocks, retain the local ZIP with valid science status.
 from huggingface_hub import CommitOperationAdd
 try:
  api.create_commit(repo_id=MODEL_REPO,repo_type='model',operations=[
   CommitOperationAdd(path_in_repo=f'{NEW_PREFIX}/{pack.name}',path_or_fileobj=str(pack)),
   CommitOperationAdd(path_in_repo=f'{NEW_PREFIX}/{side.name}',path_or_fileobj=str(side))],
   commit_message='MHC A1 frozen honest early-stop audit report',token=token)
  jdump(report_dir/'hf_sync.json',{'status':'SUCCESS','path':f'{NEW_PREFIX}/{pack.name}'})
 except Exception as e:
  jdump(report_dir/'hf_sync.json',{'status':'FAILED_RETRY_LATER','error_type':type(e).__name__,'message':str(e)[:250]})
  log('HF archive could not sync; valid local report retained',type(e).__name__)
 pack,side=write_archive(report_dir)
 return report,pack


def github_sync_light(paths):
 token=get_secret('GH_TOKEN') or get_secret('GITHUB_TOKEN')
 if not token:return {'status':'SKIPPED_NO_GH_TOKEN'}
 import requests,base64
 owner,repo=GH_REPO.split('/')
 output=[]
 for path in paths:
  path=Path(path)
  if path.stat().st_size>700_000:continue
  remote='research/earth_aiwq/mhc_real_a1/'+path.name
  url=f'https://api.github.com/repos/{owner}/{repo}/contents/{remote}'
  headers={'Authorization':f'Bearer {token}','Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'}
  raw={'message':'Archive MHC Real A1 qualification light evidence','content':base64.b64encode(path.read_bytes()).decode(),'branch':'main'}
  existing=requests.get(url,headers=headers,timeout=30)
  if existing.status_code==200:raw['sha']=existing.json()['sha']
  resp=requests.put(url,headers=headers,json=raw,timeout=90)
  if resp.status_code not in (200,201):raise RuntimeError(f'GITHUB_UPLOAD_FAIL {path.name} HTTP_{resp.status_code} {resp.text[:200]}')
  output.append({'path':remote,'status':'SUCCESS','url':resp.json()['content']['html_url']})
 return {'status':'SUCCESS','files':output}

def write_archive(report_dir):
 final=BASE/'CIDM_MHC_REAL_A1_PAIRED_PILOT_REPORTS.zip'
 paths=[p for p in report_dir.rglob('*') if p.is_file()]
 # Emergency portability: if HF throttles, preserve last 2 local optimizer-complete checkpoints in ZIP.
 local_cpts=sorted((BASE/'checkpoints').glob('*.pt'),key=lambda q:q.stat().st_mtime) if (BASE/'checkpoints').exists() else []
 paths+=local_cpts[-2:]
 src=Path(__file__)
 if src.exists():paths.append(src)
 with zipfile.ZipFile(final,'w',zipfile.ZIP_DEFLATED) as z:
  for p in paths:z.write(p,'source/'+p.name if p==src else ('portable_checkpoints/'+p.name if p.suffix=='.pt' else 'reports/'+p.name))
 with zipfile.ZipFile(final) as z:
  if z.testzip() is not None:raise RuntimeError('ZIP_CRC_FAIL')
 side=Path(str(final)+'.sha256');side.write_text(sha_file(final)+'  '+final.name+'\n')
 return final,side

def run_official(a0_zip):
 device='cuda' if torch.cuda.is_available() else 'cpu'
 if device!='cuda':raise RuntimeError('GPU_REQUIRED_FOR_OFFICIAL_PILOT_USE_SMOKE_FOR_CPU')
 report_dir=BASE/'reports';report_dir.mkdir(parents=True,exist_ok=True)
 report={'experiment':'CIDM_MHC_REAL_A1','status':'STARTED','official_aiwq_rpss':None,'is_official_submission':False,
  'dataset_kind':'ERA5_historical_reanalysis_hindcast_only','operational_predictor_ready':False,'train_seeds':SEEDS,
  'conditions':ARMS,'fixed_steps':STEPS,'test_2017_untouched_until_sealed':True,'source_sha256':sha_file(__file__)}
 api,token=get_hf(True)
 ensure_private_dataset(api);ensure_model_repo(api)
 manifest,cases,prev=read_embedded_manifest(a0_zip)
 verified=[];raw=[]
 for yr in YEARS:
  for v in VARS:
   item,src=restore_or_download(yr,v,manifest,api,token)
   verified.append({k:vv for k,vv in item.items() if k not in ('data','monday_dates') } | {'source':src})
   raw.append(item)
 jdump(report_dir/'data_verified.json',verified)
 bank=FieldBank(raw)
 # Strict target-year embargo: original A0 index allowed late-year issues whose W3/W4 labels fell in the next split.
 # This A1 repairs that cross-split *target* leakage before any training/validation.
 split={name:[c for c in cases if c['split']==name and all(c[w][:4] in SPLITS[name] for w in ('W3_monday','W4_monday'))] for name in SPLITS}
 assert all(c['W3_monday'][:4] in SPLITS['train'] and c['W4_monday'][:4] in SPLITS['train'] for c in split['train'])
 assert all(c['W3_monday'][:4]=='2016' and c['W4_monday'][:4]=='2016' for c in split['validation'])
 assert all(c['W3_monday'][:4]=='2017' and c['W4_monday'][:4]=='2017' for c in split['audit_not_fresh_official'])
 if not (len(split['train'])>=140 and len(split['validation'])>=40 and len(split['audit_not_fresh_official'])>=40):raise RuntimeError('EMBARGO_TOO_FEW_SAMPLES')
 report['target_year_embargo_counts']={k:len(v) for k,v in split.items()}
 report['original_a0_counts']={k:sum(c['split']==k for c in cases) for k in SPLITS}
 log('TARGET-YEAR EMBARGO',report['original_a0_counts'],'->',report['target_year_embargo_counts'])
 # Always inspect a fixed, preregistered 12-case validation set; 2017 audit held to the end.
 vs=[split['validation'][int(i)] for i in np.unique(np.linspace(0,len(split['validation'])-1,VAL_N,dtype=int))]
 recovered=restore_progress_pair(api,token,report_dir)
 policy=os.getenv('MHC_A1_FIX1_POLICY','gate_stop')
 if policy not in ('gate_stop','resume_full'):raise RuntimeError('UNKNOWN_A1_FIX1_POLICY')
 if policy=='gate_stop':
  early=early_stop_frozen_audit(api,token,bank,split,vs,report,report_dir,recovered,device)
  if early is not None:return early
 results=recovered.get('results',[]);history=recovered.get('history',[]);snapshots=recovered.get('checkpoints',[])

 def save_progress():
  pp=report_dir/'progress.json'
  # The legacy source is pinned for scientific continuity; new implementation hash pins the resume contract.
  jdump(pp,{'contract_source_sha':LEGACY_SOURCE_SHA,'original_training_source_sha':LEGACY_SOURCE_SHA,
   'implementation_source_sha':report['source_sha256'],'steps':STEPS,
   'results':results,'history':history,'checkpoints':snapshots,'uploaded_429_report_sha256':LEGACY_RESULT_SHA})
  return pp

 for seed in SEEDS:
  torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
  init=copy.deepcopy(MHCWeather().state_dict())
  fixed_idx=paired_sampling(split['train'],seed)
  for arm in ARMS:
   model=MHCWeather().to(device);model.load_state_dict(init)
   optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
   start_step=0
   try:
    start_step,state,last_snap=verified_resume_checkpoint(api,token,snapshots,seed,arm,device)
    if state is not None:
     model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
     torch.set_rng_state(state['torch_rng'].cpu())
     # A completed final checkpoint contains validated frozen final scores.
     if start_step==STEPS and not any(x['seed']==seed and x['arm']==arm for x in results):
      results.append({'seed':seed,'arm':arm,'validation':state['metrics'],'checkpoint_step':STEPS})
      save_progress()
      log('RECOVERED COMPLETE FROM VERIFIED CHECKPOINT',seed,arm)
    elif any(x['seed']==seed and x['arm']==arm for x in results):
     raise RuntimeError('COMPLETED_RESULT_REQUIRES_VERIFIED_STEP64_CHECKPOINT')
    # A previous uncommitted local-only step can appear in uploaded failure history.
    # Prune that unsaved suffix before safely replaying from the verified HF checkpoint.
    history[:]=[x for x in history if not (x.get('seed')==seed and x.get('arm')==arm and int(x.get('step',0))>start_step)]
    snapshots[:]=[x for x in snapshots if not (x.get('hf_path','').endswith('.pt') and
      f'pilot_seed{seed}_{arm}_step' in x.get('hf_path','') and int(x.get('hf_path','').rsplit('_step',1)[1][:-3])>start_step)]
   except Exception as e:
    raise RuntimeError(f'RESUME_INTEGRITY_STOP {seed}/{arm}: {e}') from e
   if start_step==STEPS:
    log('VERIFIED FINISHED',seed,arm);continue

   if start_step==0:
    base_case=fixed_idx[0]
    before,_,_=scalar_loss(model,bank,base_case,arm,device)
    optimizer.zero_grad(set_to_none=True);before.backward()
    gnorm=torch.sqrt(sum((p.grad.detach().square().sum() for p in model.parameters() if p.grad is not None),torch.tensor(0.,device=device)))
    hgnorm=torch.linalg.vector_norm(model.head[-1].weight.grad.detach().float())
    if not torch.isfinite(gnorm) or float(gnorm)<=1e-11 or not torch.isfinite(hgnorm) or float(hgnorm)<=1e-12:
     raise RuntimeError(f'Q0_BAD_MODEL_OR_CONTROL_GRAD_{seed}_{arm}')
    # A predetermined local one-step loss-descent probe; never used as initial Q1 state.
    params=[p for p in model.parameters() if p.grad is not None]
    old=[p.detach().clone() for p in params]
    origin=float(before.detach());trial=[]
    try:
     for radius in (3e-4,1e-4,3e-5,1e-5):
      with torch.no_grad():
       for p,oldp in zip(params,old):p.copy_(oldp-radius*p.grad/gnorm.clamp_min(1e-12))
      newloss,out,action=scalar_loss(model,bank,base_case,arm,device)
      reduction=(origin-float(newloss.detach()))/max(origin,1e-12)
      trial.append({'radius':radius,'relative_decrease':reduction,'max_action':float(action.max())})
    finally:
     with torch.no_grad():
      for p,oldp in zip(params,old):p.copy_(oldp)
    if not any(t['relative_decrease']>1e-7 and t['max_action']<=model.limit+1e-6 for t in trial):
     raise RuntimeError(f'Q0_NO_MEASURABLE_BOUNDED_DESCENT_{seed}_{arm} trials={trial}')
    log('Q0 PASS',seed,arm,'grad',float(gnorm),'headgrad',float(hgnorm), 'best_descent',max(t['relative_decrease'] for t in trial))
    # Reset Q0 diagnostic gradients; exactly shared fresh weights for all arms.
    model.load_state_dict(init);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    test_before=score(eval_arm(model,bank,vs,arm,device))
    history.append({'seed':seed,'arm':arm,'step':0,'loss_train':float(before),'grad_norm':float(gnorm),'head_grad_norm':float(hgnorm),'q0_trials':trial,**test_before})
    save_state(model,optimizer,'pilot',seed,arm,0,test_before,api);save_progress()
    # Q0 is saved locally; no HF commit is required before the first >=16-step milestone.

   for step in range(start_step+1,STEPS+1):
    model.train();optimizer.zero_grad(set_to_none=True)
    train_loss,_,acts=scalar_loss(model,bank,fixed_idx[step-1],arm,device)
    if not torch.isfinite(train_loss):raise RuntimeError('NONFINITE_TRAIN_LOSS')
    train_loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('NONFINITE_GRAD')
    optimizer.step()
    if step%8==0:log(f'{seed} {arm} train {step}/{STEPS}: {float(train_loss):.5g} action_rms_max={float(acts.max()):.6f}')
    if step in EVAL_AT:
     ev=score(eval_arm(model,bank,vs,arm,device))
     history.append({'seed':seed,'arm':arm,'step':step,'loss_train':float(train_loss),**ev})
     current=save_state(model,optimizer,'pilot',seed,arm,step,ev,api)
     if step%HF_SYNC_STEPS==0 or step==STEPS:
      remote_snap={k:v for k,v in current.items() if k!='local_path'}
      snapshots.append(remote_snap)
      if step==STEPS:
       results.append({'seed':seed,'arm':arm,'validation':ev,'checkpoint_step':STEPS})
      pp=save_progress()
      commit_atomic_progress(api,token,current,pp,seed,arm,step)
     else:
      save_progress()
   if not any(x['seed']==seed and x['arm']==arm for x in results):
    raise RuntimeError('FINISH_REQUIRES_ATOMICALLY_PERSISTED_COMPLETE_RESULT')
 # Exact zero-control forecast parity is an experiment contract, not a post-hoc claim.
 for seed in SEEDS:
  zs=[r for r in history if r.get('seed')==seed and r.get('step')==0]
  if len(zs)!=3:raise RuntimeError(f'MISSING_THREE_ZERO_HEAD_BASELINES_{seed}')
  for key in ('W3_MSE','W4_MSE'):
   ref=zs[0][key]
   if any(abs(row[key]-ref)>1e-6*max(abs(ref),1e-10) for row in zs[1:]):
    raise RuntimeError(f'Q0_INITIAL_FORECAST_PARITY_FAIL_{seed}_{key}')
 # Gate never infer rank; validation MHC-v-stateless both seeds W3/W4 improvement.
 passed=True;detail={}
 for seed in SEEDS:
  v={r['arm']:r['validation'] for r in results if r['seed']==seed}
  rel={t:100*(v['stateless'][t]-v['mhc'][t])/max(v['stateless'][t],1e-15) for t in ('W3_MSE','W4_MSE')}
  detail[str(seed)]=rel
  if min(rel.values())<0.5:passed=False
 # Audit's 2017 issues remain blind if validation mechanism criterion fails.
 audit=[]
 if passed:
  for r in results:
   seed=r['seed'];arm=r['arm']
   from huggingface_hub import hf_hub_download
   snap=next((q for q in snapshots if q.get('hf_path','').endswith(f'pilot_seed{seed}_{arm}_step{STEPS:04}.pt')),None)
   if snap is None:raise RuntimeError('AUDIT_MISSING_VERIFIED_STEP64_CHECKPOINT')
   fn=snap['hf_path']
   local=hf_hub_download(repo_id=MODEL_REPO,repo_type='model',filename=fn,token=token)
   if sha_file(local)!=snap['sha256']:raise RuntimeError('AUDIT_CHECKPOINT_SHA_MISMATCH')
   state=torch.load(local,map_location=device,weights_only=True)
   model=MHCWeather().to(device);model.load_state_dict(state['model'])
   audit.append({'seed':seed,'arm':arm,'audit_2017':score(eval_arm(model,bank,split['audit_not_fresh_official'],arm,device))})
  log('2017 BLIND AUDIT OPENED AFTER PASSING VAL SIGNAL')
 else:log('2017 AUDIT NOT OPENED: validation signal Gate failed')
 report.update(status='MHC_REAL_A1_PAIRED_PILOT_COMPLETED_NO_FORMAL_TRAINING',model_steps_per_arm=STEPS,
  train_sample_count=len(split['train']),validation_count=len(split['validation']),audit_count=len(split['audit_not_fresh_official']),
  model_parameters=sum(p.numel() for p in MHCWeather().parameters()),verified_files=len(verified),
  q1_comparison_pct=detail,validation_mechanism_signal=passed,blind_2017_opened=bool(audit),
  caveats=['single 2016 validation year; 2017 audit used exactly once','not an operational forecast: ERA5 data latency','no official climatological quantile/ECMWF RPSS','2013–15 samples too few for flagship model'])
 jdump(report_dir/'metrics.json',{'history':history,'validation_final':results,'audit_2017':audit})
 jdump(report_dir/'status.json',report)
 # Save small archive to HF model, GitHub reports; never upload raw to public GitHub.
 pack,side=write_archive(report_dir)
 from huggingface_hub import CommitOperationAdd
 api.create_commit(repo_id=MODEL_REPO,repo_type='model',operations=[
  CommitOperationAdd(path_in_repo=f'{NEW_PREFIX}/{pack.name}',path_or_fileobj=str(pack)),
  CommitOperationAdd(path_in_repo=f'{NEW_PREFIX}/{side.name}',path_or_fileobj=str(side))],
  commit_message='MHC Real A1 FIX1 final reports (one atomic commit)',token=token)
 sync=github_sync_light([Path(__file__),report_dir/'status.json',report_dir/'metrics.json'])
 jdump(report_dir/'github_sync.json',sync)
 return report,pack

def smoke():
 torch.set_num_threads(2)
 rng=np.random.default_rng(7)
 for H,W in ((7,13),(121,240)):
  h=torch.from_numpy(rng.normal(size=(1,8,2,H,W)).astype('f'))
  t=torch.from_numpy(rng.normal(size=(1,2,2,H,W)).astype('f'))
  ph=torch.tensor([[0.,1.]])
  m=MHCWeather(d=8)
  a=[m(h,ph,arm)[0].detach() for arm in ARMS]
  assert all(torch.allclose(a[0],x,atol=1e-6,rtol=0) for x in a[1:]),'Q0_ZERO_HEAD_PARITY'
  with torch.no_grad():
   nn.init.normal_(m.head[-1].weight,mean=0,std=0.015)
  b,c=m(h,ph,'mhc')[0],m(h,ph,'stateless')[0]
  assert torch.isfinite(b).all() and torch.isfinite(c).all()
  assert not torch.equal(b,c),'PERSISTENT_STATE_NOT_DISTINGUISHABLE'
  loss=b.square().mean();loss.backward()
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()),'BAD_GRAD'
 log('SMOKE PASS: 7x13 and native 121x240, forward zero-head parity, persistent state, 5-step backward')
 return {'status':'SMOKE_ONLY_NO_REAL_DATA','native_grid':GRID,'zero_head_parity':'PASS','gradient':'PASS'}

def main():
 mode=os.getenv('MHC_REAL_A1_MODE','official')
 a0=BASE/'evidence'/'CIDM_MHC_REAL_A0_DATA_BRIDGE_REPORTS.zip'
 report_dir=BASE/'reports';report_dir.mkdir(parents=True,exist_ok=True)
 try:
  if mode=='smoke':report=smoke();jdump(report_dir/'status.json',report);pack,side=write_archive(report_dir)
  else:report,pack=run_official(a0)
 except Exception as e:
  log('FAIL CLOSED:',type(e).__name__,str(e))
  report={'status':('BLOCKED_HF_429_RESUMABLE' if '429' in str(e) else 'BLOCKED_NO_TRAINING'),'error_type':type(e).__name__,'error':str(e),'traceback_tail':traceback.format_exc().splitlines()[-12:]}
  jdump(report_dir/'status.json',report)
  pack,side=write_archive(report_dir)
  try:
   api,token=get_hf(False)
   if token:
    if '429' not in str(e) and 'rate limit' not in str(e).lower():
     api.upload_file(path_or_fileobj=str(pack),path_in_repo=f'{NEW_PREFIX}/{pack.name}',repo_id=MODEL_REPO,repo_type='model')
    else:log('HF RATE LIMITED: keep portable local ZIP; retry later; do not create another commit')
  except Exception as ee:log('HF FAILED REPORT SYNC:',str(ee)[:200])
 print('RUN:',report['status'],'REPORT ZIP:',pack,'SHA256:',sha_file(pack))
 return report
if __name__=='__main__':main()
