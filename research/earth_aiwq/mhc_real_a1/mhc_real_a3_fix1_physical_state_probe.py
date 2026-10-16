"""CIDM-MHC-Real A3/P0 FIX1 — frozen physical-history state sufficiency qualification.

- Reuses the official 2012–2017 ECMWF/AIWQ raw-file SHA contract and actual seed17
  64-step MHC weather checkpoint from A2. Original 2017 audit remains SEALED.
- Does not modify forecast-model weights, issue-time features, or predictor outputs.
- 2013–14 train; 2015 hyperparameter tune; 2016 previously seen validation.
- Physical features use ONLY the eight historical MSLP/TAS weekly fields at issue.
- Coarse zonal-mean residual proxy is NOT AIWQ RPSS or a proof of a Markov state.
- Private HF Dataset, one-shot HF Model archival, lightweight GitHub, NO GOOGLE DRIVE.
"""
from __future__ import annotations
import os,sys,csv,io,json,hashlib,math,zipfile,traceback,importlib.util
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(os.getenv('MHC_REAL_A3_HOME','/content/MHC_REAL_A3')).resolve()
ROOT.mkdir(parents=True,exist_ok=True)
PARENT_HOME=Path(os.getenv('MHC_REAL_A2_HOME',str(ROOT))).resolve()
A2_OFFICIAL_SHA='29f0c7c524d6af3cb2b8f22fc17b97fd924ce174f51948788465658dc23513a4'
BLOCKED_A3_SHA='98cf5e6912f1fabd72bc0a2072bf9192e9674875f3e44fae14858e5255260a05'
A2_SOURCE_SHA='4d6edafa009d09813de15241b8c485b193578bd9782117855f69547f96bf5063'
SHAM_REPEATS=int(os.getenv('MHC_A3_SHAM_REPEATS','16'))
if SHAM_REPEATS<8 or SHAM_REPEATS>64:raise ValueError('SHAM_REPEATS must be between 8 and 64')
RIDGE_GRID=(0.1,1.,10.,100.)
SEED=20261016

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for data in iter(lambda:f.read(2**20),b''):h.update(data)
 return h.hexdigest()

def dumpj(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding='utf-8')

def parent_source():
 p=PARENT_HOME/'mhc_real_a2_conditional_information.py'
 if not p.exists():p=PARENT_HOME/'evidence'/'mhc_real_a2_conditional_information.py'
 if not p.exists():raise RuntimeError('MISSING_PINNED_A2_SOURCE '+str(p))
 expected=A2_SOURCE_SHA
 if sha(p)!=expected:raise RuntimeError('A2_SOURCE_SHA_FAIL')
 spec=importlib.util.spec_from_file_location('mhc_real_a2_frozen_parent',str(p))
 m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 return m

def official_a2_result():
 p=PARENT_HOME/'evidence'/'A2_USER_REAL_CONDITIONAL_INFORMATION.zip'
 if not p.is_file() or sha(p)!=A2_OFFICIAL_SHA:raise RuntimeError('A2_OFFICIAL_EVIDENCE_HASH_FAIL')
 with zipfile.ZipFile(p) as z:
  if z.testzip() is not None:raise RuntimeError('A2_OFFICIAL_ZIP_CRC_FAIL')
  r=json.loads(z.read('reports/status.json'))
 if r.get('status')!='FROZEN_CONDITIONAL_INFORMATION_COMPLETED_NO_TRAINING':raise RuntimeError('A2_NOT_FORMALLY_COMPLETED')
 if r.get('sealed_2017_untouched') is not True:raise RuntimeError('AUDIT_OPENED')
 if r.get('gate',{}).get('strong_conditional_information_signal') is not False:raise RuntimeError('PARENT_A2_GATE_UNEXPECTED')
 return r

@torch.no_grad()
def physical_history_features(h, lat_pool):
 """[B,8,2,H,W] -> [B, 48], eight causal physical operators x two channels x three latitude belts."""
 if h.ndim!=5 or h.shape[1]!=8 or h.shape[2]!=2:raise RuntimeError('HISTORY_SHAPE_INVALID')
 last=h[:,-1]
 v=(last, h[:,-4:].mean(dim=1), h.mean(dim=1),
    last-h[:,-2], h[:,-4:].mean(dim=1)-h[:,:4].mean(dim=1),
    h.std(dim=1,unbiased=False), h[:,-4:].std(dim=1,unbiased=False),
    (last-F.avg_pool2d(last,kernel_size=5,stride=1,padding=2,count_include_pad=False)).abs())
 x=torch.cat([lat_pool(t).flatten(1) for t in v],dim=1)
 if x.shape[-1]!=48 or not torch.isfinite(x).all():raise RuntimeError('PHYSICAL_FEATURES_INVALID')
 return x.float()

def add_physical_features(parent,model,bank,cases,device):
 arr=[]
 for ix,c in enumerate(cases):
  h,_,_=bank.case(c)
  h=h[None].to(device)
  f=physical_history_features(h,parent.lat_pool)
  arr.append(f[0].cpu().numpy().astype('f8'))
 return np.stack(arr)

def fit_ridge(X,Y,lam):
 # Numerically stable Gaussian ridge with free intercept; one SVD for all lambda candidates.
 mu=X.mean(0); ymu=Y.mean(0)
 XC=X-mu;YC=Y-ymu
 U,S,VT=np.linalg.svd(XC,full_matrices=False)
 return mu,ymu,U,S,VT,U.T@YC

def pred_svd(pack,X,lam):
 mu,ymu,U,S,VT,UY=pack
 W=VT.T@((S/(S*S+lam))[:,None]*UY)
 return (X-mu)@W+ymu

def regress(fit,tune,val,lead,features,lam_grid=RIDGE_GRID):
 """2015 tune lambda, 2013–15 refit, fit-only standardization. Nothing from 2016 enters fitting."""
 bF=fit['base'][lead];bT=tune['base'][lead];bV=val['base'][lead]
 fF=features['fit'];fT=features['tune'];fV=features['validation']
 if fF is None:
  xF,xT,xV=bF,bT,bV
 else:
  xF=np.concatenate([bF,fF],axis=1)
  xT=np.concatenate([bT,fT],axis=1)
  xV=np.concatenate([bV,fV],axis=1)
 yF=fit['residual'][lead];yT=tune['residual'][lead]
 mean=xF.mean(0);std=np.maximum(xF.std(0),1e-5)
 xFs=(xF-mean)/std;xTs=(xT-mean)/std
 fac=fit_ridge(xFs,yF,1)
 vals={float(l):float(np.mean((pred_svd(fac,xTs,l)-yT)**2)) for l in lam_grid}
 chosen=sorted(vals.items(),key=lambda kv:(kv[1],kv[0]))[0][0]
 xall=np.concatenate([xF,xT],axis=0);yall=np.concatenate([yF,yT],axis=0)
 mean=xall.mean(0);std=np.maximum(xall.std(0),1e-5)
 fin=fit_ridge((xall-mean)/std,yall,1)
 resid=pred_svd(fin,(xV-mean)/std,chosen)
 err=val['forecast'][lead]+resid-val['truth'][lead]
 return np.mean(err**2,axis=1),{'chosen_lambda':chosen,'tune_mse':vals[str(chosen)] if str(chosen) in vals else vals[chosen], 'features':int(xF.shape[1])}

def quarterly_groups(dates):
 return np.asarray([int(str(d)[5:7])-1 for d in dates])//3

def permute(fit,tune,val,lead,typ,seed):
 rng=np.random.default_rng(seed)
 ret={}
 for key,part in [('fit',fit),('tune',tune),('validation',val)]:
  n=len(part['dates'])
  idx=np.arange(n)
  if typ=='global':idx=rng.permutation(n)
  elif typ=='seasonal':
   bins=quarterly_groups(part['dates'])
   for q in range(4):
    which=np.flatnonzero(bins==q)
    if len(which)>1:idx[which]=rng.permutation(which)
  else:raise ValueError(typ)
  ret[key]=idx
 return ret

def bootstrap_ci(per_issue,kind,n_boot=1500):
 """4-week circular moving-block bootstrap, descriptive only on reused 2016 VAL."""
 n=len(next(iter(per_issue.values())))
 rng=np.random.default_rng(SEED+13)
 blocks=int(math.ceil(n/4))
 diff=[]
 for _ in range(n_boot):
  starts=rng.integers(n,size=blocks)
  ix=np.concatenate([np.mod(np.arange(s,s+4),n) for s in starts])[:n]
  if kind=='physical_vs_instant':a='physical';b='instant'
  elif kind=='physical_vs_memory':a='physical';b='memory'
  else:raise ValueError(kind)
  m1=np.mean(per_issue[a][ix]);m2=np.mean(per_issue[b][ix]);diff.append(100*(m2-m1)/max(m2,1e-20))
 return [float(q) for q in np.quantile(diff,[.025,.5,.975])]

def diagnostics(fit,tune,val,phys,sham_repeats=SHAM_REPEATS):
 results={};per_rows=[];sham_rows=[]
 for lead,label in enumerate(('W3','W4')):
  y={}
  models={'instant':{'fit':None,'tune':None,'validation':None},
          'memory':{k:part['mem'][lead] for k,part in [('fit',fit),('tune',tune),('validation',val)]},
          'physical':phys}
  detail={}
  for kind,feat in models.items():
   y[kind],detail[kind]=regress(fit,tune,val,lead,feat)
  # sham controls for each history feature family, global and season-stratified
  controls={}
  for source in ('memory','physical'):
   for typ in ('global','seasonal'):
    key=source+'_'+typ
    for j in range(sham_repeats):
     idx=permute(fit,tune,val,lead,typ,SEED+lead*100000+j*13+(97 if source=='memory' else 191))
     original=models[source]
     alt={k:original[k][idx[k]] for k in ('fit','tune','validation')}
     sham,_=regress(fit,tune,val,lead,alt)
     controls.setdefault(key,[]).append(float(sham.mean()))
     sham_rows.append({'lead':label,'source':source,'permutation':typ,'replicate':j,'mean_mse':float(sham.mean()),
          'improvement_vs_instant_pct':float(100*(y['instant'].mean()-sham.mean())/max(y['instant'].mean(),1e-20))})
  stats={k:float(v.mean()) for k,v in y.items()}
  n=len(val['dates'])
  physical_over_inst=100*(stats['instant']-stats['physical'])/stats['instant']
  physical_over_memory=100*(stats['memory']-stats['physical'])/stats['memory']
  memory_over_instant=100*(stats['instant']-stats['memory'])/stats['instant']
  phy_win_inst=int((y['physical']<y['instant']).sum())
  phy_win_mem=int((y['physical']<y['memory']).sum())
  # Require physical state beats high percentile of BOTH global and season-preserving shams
  placebo95={s:float(np.quantile(controls['physical_'+s],.05)) for s in ('global','seasonal')}
  res={'mean_mse':stats,'memory_vs_instant_pct':float(memory_over_instant),
       'physical_vs_instant_pct':float(physical_over_inst),
       'physical_vs_memory_pct':float(physical_over_memory),
       'physical_better_windows_vs_instant':phy_win_inst,'physical_better_windows_vs_memory':phy_win_mem,
       'physical_vs_instant_4week_bootstrap_CI_pct':bootstrap_ci(y,'physical_vs_instant'),
       'physical_vs_memory_4week_bootstrap_CI_pct':bootstrap_ci(y,'physical_vs_memory'),
       'sham_MSE_5pct_threshold':placebo95,
       'n_2016_issues':n,'ridge_settings':detail}
  res['physical_signal_gate']=bool(physical_over_inst>=.5 and physical_over_memory>=.2
       and phy_win_inst>=32 and phy_win_mem>=32
       and stats['physical']<min(placebo95.values())
       and res['physical_vs_instant_4week_bootstrap_CI_pct'][0]>0.)
  results[label]=res
  for i,date in enumerate(val['dates']):per_rows.append({'issue':date,'lead':label,
     'instant_mse':float(y['instant'][i]),'old_memory_mse':float(y['memory'][i]),'physical_mse':float(y['physical'][i])})
 return results,per_rows,sham_rows

def make_archive():
 out=ROOT/'CIDM_MHC_REAL_A3_FIX1_PHYSICAL_STATE_REPORTS.zip'
 rd=ROOT/'reports'
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
  for file in sorted(rd.glob('*')):
   if file.is_file():z.write(file,arcname='reports/'+file.name)
  z.write(Path(__file__),arcname='source/'+Path(__file__).name)
 with zipfile.ZipFile(out) as z:
  if z.testzip() is not None:raise RuntimeError('ZIP_CRC_FAIL')
 (ROOT/(out.name+'.sha256')).write_text(sha(out)+'  '+out.name+'\n')
 return out,ROOT/(out.name+'.sha256')

def save_csv(path,rows):
 if not rows:return
 with Path(path).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def preflight_provenance(parent):
 parent_source_file=Path(parent.__file__).resolve()
 parent_source_sha=sha(parent_source_file)
 if parent_source_sha!=A2_SOURCE_SHA:raise RuntimeError('A2_MODULE_SOURCE_PIN_FAIL')
 blocked_path=PARENT_HOME/'evidence'/'A3_USER_BLOCKED_REPORT.zip'
 if not blocked_path.is_file() or sha(blocked_path)!=BLOCKED_A3_SHA:
  raise RuntimeError('A3_BLOCKED_ORIGIN_EVIDENCE_PIN_FAIL')
 json.dumps({'origin_a2_source_sha256':parent_source_sha,'origin_a3_blocked_report_sha256':sha(blocked_path)},allow_nan=False)
 return parent_source_sha

def official():
 parent=parent_source();a2=official_a2_result();paths,prior=parent.evidence()
 # FIX1: verify and serialize provenance BEFORE recovering large weather files.
 parent_source_sha=preflight_provenance(parent)
 m=parent.load_parent(paths['legacy_source'])
 device='cuda' if torch.cuda.is_available() else 'cpu'
 if device!='cuda':raise RuntimeError('OFFICIAL_GPU_REQUIRED_FOR_198_FULL_GRID_ROLLOUTS')
 api,token=m.get_hf(True)
 m.ensure_private_dataset(api);m.ensure_model_repo(api)
 manifest,cases,_=m.read_embedded_manifest(paths['a0'])
 raw=[];rawinfo=[]
 for year in m.YEARS:
  for v in m.VARS:
   data,source=m.restore_or_download(year,v,manifest,api,token)
   raw.append(data);rawinfo.append({'year':year,'var':v,'sha256':data['sha256'],'source':source})
 bank=m.FieldBank(raw);split=parent.get_case_split(m,cases)
 from huggingface_hub import hf_hub_download
 file=hf_hub_download(repo_id=m.MODEL_REPO,repo_type='model',filename=parent.MODEL_HF_PATH,token=token)
 parent.pin(file,parent.MODEL_SHA)
 state=torch.load(file,map_location=device,weights_only=True)
 if state.get('step')!=64 or state.get('seed')!=17 or state.get('arm')!='mhc':raise RuntimeError('PINNED_CHECKPOINT_META_FAIL')
 model=m.MHCWeather().to(device).eval();model.load_state_dict(state['model'])
 val12=[split['validation'][int(i)] for i in np.unique(np.linspace(0,len(split['validation'])-1,m.VAL_N,dtype=int))]
 replay=m.score(m.eval_arm(model,bank,val12,'mhc',device))
 for k in ('W3_MSE','W4_MSE'):
  wanted=a2['parent_replay_12_dates'][k]
  if abs(replay[k]-wanted)>5e-5*max(abs(wanted),1e-10):raise RuntimeError('ORIGINAL_FORECAST_REPLAY_FAIL_'+k)
 parts={p:parent.collect_cases(m,model,bank,split[p],device,p) for p in ('fit','tune','validation')}
 # Historic A2 exact comparator replay: same raw, same trained model, same future-truth targets.
 original_scores,_,_=parent.score(parts['validation'],parts['fit'],parts['tune'])
 for lead in ('W3','W4'):
  for method in ('instant','memory','sham'):
   got=original_scores[lead][method]['mse'];want=a2['metrics'][lead][method]['mse']
   if abs(got-want)>2e-6*max(abs(want),1e-10):raise RuntimeError('A2_REPLAY_FAIL_'+lead+'_'+method)
 phys={p:add_physical_features(parent,model,bank,split[p],device) for p in ('fit','tune','validation')}
 scores,rows,shams=diagnostics(parts['fit'],parts['tune'],parts['validation'],phys)
 both=all(scores[lead]['physical_signal_gate'] for lead in ('W3','W4'))
 reportdir=ROOT/'reports';reportdir.mkdir(parents=True,exist_ok=True)
 dumpj(reportdir/'status.json',{'status':'FROZEN_PHYSICAL_STATE_INFORMATION_COMPLETED_NO_TRAINING',
    'fix1_patch':'module.__file__ provenance SHA plus early no-download serialization precheck',
    'origin_a2_report_sha256':A2_OFFICIAL_SHA,'origin_a2_source_sha256':parent_source_sha,
    'origin_a3_blocked_report_sha256':BLOCKED_A3_SHA,
    'source_sha256':sha(__file__),'checkpoint_sha256':parent.MODEL_SHA,
    'data_file_hashes':rawinfo,'original_12_issue_replay':replay,
    'old_A2_three_variant_score_replay':'PASS','split_sizes':{k:len(v) for k,v in split.items()},
    'sealed_2017_untouched':True,'permutation_replicates_per_type':SHAM_REPEATS,
    'score_type':'coarse_zonal_mean_residual_MSE_NOT_AIWQ_RPSS',
    'metrics':scores,'gate':{'physical_state_information_prequalified_diagnostic_only':both,
    'full_mhc_training_authorized':False,'note':'2016 reused validation, only exploratory mechanism qualification'},
    'weather_model_optimizer_updates':0})
 save_csv(reportdir/'per_issue.csv',rows)
 save_csv(reportdir/'sham_replicates.csv',shams)
 # Archive is IMMUTABLE once uploaded: sidecar must identify the exact HF bytes.
 try:gh=m.github_sync_light([Path(__file__),reportdir/'status.json'])
 except Exception as e:gh={'status':'FAILED_RETRY_LATER','reason':str(e)[:450]}
 dumpj(reportdir/'github_sync.json',gh)
 dumpj(reportdir/'hf_upload_intent.json',{'status':'PENDING_SEPARATE_RECEIPT',
  'rule':'The immutable ZIP is packed once; actual HF upload outcome is in external upload_receipt.json'})
 archive,side=make_archive()
 receipt={'status':'NOT_ATTEMPTED','archive_sha256':sha(archive)}
 if os.getenv('MHC_REAL_A3_SYNC_HF','1')=='1':
  try:
   from huggingface_hub import CommitOperationAdd
   commit=api.create_commit(repo_id=m.MODEL_REPO,repo_type='model',token=token,operations=[
     CommitOperationAdd(path_in_repo='experiments/MHC_REAL_A3_FIX1/'+archive.name,path_or_fileobj=str(archive)),
     CommitOperationAdd(path_in_repo='experiments/MHC_REAL_A3_FIX1/'+side.name,path_or_fileobj=str(side))],
     commit_message='MHC Real A3 FIX1 repaired module provenance and completed frozen physical history probe')
   receipt={'status':'SUCCESS','archive_sha256':sha(archive),'commit_url':str(getattr(commit,'commit_url',''))}
  except Exception as e:receipt={'status':'FAILED_RETRY_LATER','archive_sha256':sha(archive),'reason':str(e)[:450]}
 else:receipt={'status':'SKIPPED_EXPLICITLY_DISABLED','archive_sha256':sha(archive)}
 dumpj(ROOT/'upload_receipt.json',receipt)
 # NEVER repack: this preserves identical bytes between user's ZIP and remote HF artifact.
 return {'status':'FROZEN_PHYSICAL_STATE_INFORMATION_COMPLETED_NO_TRAINING','scores':scores,'zip':str(archive),'sha':sha(archive),'sync':receipt}

def smoke():
 parent=parent_source();a2=official_a2_result();parent.evidence()
 assert preflight_provenance(parent)==A2_SOURCE_SHA
 m=parent.load_parent(PARENT_HOME/'evidence'/'mhc_real_a1_fix1_atomic_resume.py')
 r=np.random.default_rng(SEED)
 h=torch.from_numpy(r.normal(size=(1,8,2,13,19)).astype('f'))
 feat=physical_history_features(h,parent.lat_pool)
 assert feat.shape==(1,48) and torch.isfinite(feat).all()
 # Multiple controlled causal probes synthetic, no real-weather skill is claimed.
 def build(n,year):
  base=r.normal(size=(n,56));memory=r.normal(size=(n,48));physical=r.normal(size=(n,48))
  resp=.25*base[:,:6]+.2*physical[:,:6]+r.normal(size=(n,6))*.1
  return {'base':[base,base],'mem':[memory,memory],'residual':[resp,resp],
     'forecast':[np.zeros_like(resp)]*2,'truth':[resp,resp],
     'dates':[f'{year}-{i//4+1:02d}-{(i%4)*7+1:02d}' for i in range(n)]},physical
 fit,fphy=build(20,2013);tune,tphy=build(12,2015);val,vphy=build(16,2016)
 scores,rows,shams=diagnostics(fit,tune,val,{'fit':fphy,'tune':tphy,'validation':vphy},sham_repeats=8)
 assert len(rows)==32 and len(shams)==2*2*2*8 and all(np.isfinite(scores[k]['mean_mse']['physical']) for k in scores)
 reportdir=ROOT/'reports';reportdir.mkdir(parents=True,exist_ok=True)
 dumpj(reportdir/'status.json',{'status':'SMOKE_ONLY_NO_REAL_WEATHER','official_qualification':'NOT_RUN',
   'physical_feature_shape':list(feat.shape),'test_results':scores,'no_training':True})
 save_csv(reportdir/'per_issue.csv',rows);save_csv(reportdir/'sham_replicates.csv',shams)
 archive,side=make_archive()
 return {'status':'SMOKE_ONLY_NO_REAL_WEATHER','zip':str(archive),'sha':sha(archive),'rows':len(rows)}

def main():
 try:
  if os.getenv('MHC_REAL_A3_MODE','official')=='smoke':return smoke()
  return official()
 except Exception as e:
  rd=ROOT/'reports';rd.mkdir(parents=True,exist_ok=True)
  dumpj(rd/'status.json',{'status':'BLOCKED_NO_TRAINING','fix1_patch':'module.__file__ provenance SHA',
        'origin_a3_blocked_report_sha256':BLOCKED_A3_SHA,'error_type':type(e).__name__,
        'error':str(e),'traceback':traceback.format_exc().splitlines()[-15:]})
  archive,side=make_archive()
  return {'status':'BLOCKED_NO_TRAINING','error':str(e),'zip':str(archive),'sha':sha(archive)}

if __name__=='__main__':print('[A3]',main(),flush=True)
