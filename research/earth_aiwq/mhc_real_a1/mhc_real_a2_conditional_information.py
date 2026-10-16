"""MHC-Real A2/P0: frozen conditional-information gate for slow Markov memory.
Actual previous seed17 MHC weights, official AIWQ ERA5 2012-2017, 2016 VALIDATION only.
No new optimization of weather model; ridge regression diagnostic train only on 2013-15.
Not official AI Weather Quest RPSS or operational forecasting.
No Google Drive; HF private Dataset+Model, optional bundled result sync.
"""
from __future__ import annotations
import os,sys,json,hashlib,zipfile,math,traceback,datetime as dt,csv,importlib.util,io
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(os.getenv('MHC_REAL_A2_HOME','/content/MHC_REAL_A2')).resolve()
ROOT.mkdir(parents=True,exist_ok=True)
EVIDENCE={'a0':'c4c45e8c0e48ee5f29b78006710f4ab17de537ac0c76ef3e8bb9b332b6a37fe3',
          'a1':'989de95a8bba3fdacc8de007a68aae4945f3198e310f6018d7cba8235cf7de4c',
          'a1fix1':'5e8001317ea90df6d9b38cb6bbf88d6cf1933b5db9fdc4e71d558a3e4ffddf5c',
          'legacy_source':'9574866fc485c0868e3ac7b00c6d64db959f6353eaa3a569c60fd2b8ade84d9e'}
NP_SEED=20261016
MODEL_SHA='e55f47fb3ecd1f1518e0ec629d1f24311b2e05f6a8fd07fe04ac7de1948528d6'
MODEL_HF_PATH='experiments/MHC_REAL_A1/pilot_seed17_mhc_step0064.pt'
YEAR_SPLIT={'fit':(2013,2014),'tune':(2015,), 'validation':(2016,)}
LAMBDA_GRID=[0.1,1.,10.,100.]
LAT_BANDS=(-90.,-23.5,23.5,90.)
TARGET_GAIN_PCT=0.1  # diagnostic requirement, not proof of Markov sufficiency


def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def savej(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=str))

def pin(p,sha):
 if not Path(p).is_file() or digest(p)!=sha:raise RuntimeError('EVIDENCE_HASH_FAIL '+str(p))
 return Path(p)

def evidence():
 p=ROOT/'evidence'
 sources={'a0':p/'A0_REAL_DATA_BRIDGE.zip','a1':p/'A1_429_REPORT.zip',
  'a1fix1':p/'A1_FIX1_USER_OFFICIAL.zip','legacy_source':p/'mhc_real_a1_fix1_atomic_resume.py'}
 for key,path in sources.items():pin(path,EVIDENCE[key])
 with zipfile.ZipFile(sources['a1fix1']) as z:
  if z.testzip():raise RuntimeError('ZIP_CORRUPT_A1_FIX1')
  status=json.loads(z.read('reports/status.json'))
  if status.get('status')!='MHC_REAL_A1_EARLY_GATE_FAIL_FROZEN_AUDIT_COMPLETED_NO_TRAINING':
   raise RuntimeError('NOT_ORIGINAL_FROZEN_AUDIT')
  if status.get('independent_2017_audit_opened') is not False:raise RuntimeError('SEALED_AUDIT_LEAK')
  snap=status['seed17_completed_remote_verified']['mhc']
  if snap['sha256']!=MODEL_SHA or snap['hf_path']!=MODEL_HF_PATH or snap['step']!=64:
   raise RuntimeError('PINNED_SEED17_CHECKPOINT_DIFFERENT')
 return sources,status

def load_parent(source):
 os.environ['MHC_A1_HOME']=str(ROOT)
 spec=importlib.util.spec_from_file_location('mhc_real_a1_fix1_atomic_resume',str(source))
 m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 return m

def lat_pool(t,latbands=LAT_BANDS):
 """Area-weighted region mean; shape [1,C,H,W] -> [1,C,3]; no labels in features."""
 h=t.shape[-2]
 lat=torch.linspace(90,-90,h,device=t.device,dtype=t.dtype)
 weights=lat.mul(math.pi/180.).cos().clamp_min(0.)
 out=[]
 for a,b in zip(latbands[:-1],latbands[1:]):
  mask=((lat>=a)&(lat < b if b<90 else lat<=b)).to(t.dtype)*weights
  if float(mask.sum())<1e-8:raise RuntimeError('EMPTY_LAT_BAND')
  y=(t*mask[None,None,:,None]).sum(dim=(-2,-1))/(mask.sum()*t.shape[-1])
  out.append(y)
 return torch.stack(out,-1)

def probe_features(model,h,phase,truth,device):
 """Uses a hook on the actual head input so instrumenting leaves forward physics bit-identical."""
 captured=[]
 def hook(module,args):captured.append(args[0].detach().clone())
 handle=model.head.register_forward_pre_hook(hook)
 try:
  with torch.inference_mode(): output=model(h,phase,arm='mhc')[0]
 finally:handle.remove()
 if len(captured)!=5:raise RuntimeError('HEAD_INPUT_HOOK_FAIL')
 base=[];mem=[];residual=[];forecast=[];actual=[];exposure=[]
 for j,step in enumerate((3,4)):
  inp=captured[step]; d=model.d
  slow=inp[:,:d]; fast=inp[:,d:2*d]; prev=inp[:,2*d:]
  zbase=slow+fast
  instant=F.avg_pool2d(zbase,kernel_size=5,stride=1,padding=2,count_include_pad=False)
  innovation=slow-instant
  xf=torch.cat([lat_pool(zbase).flatten(1),lat_pool(prev).flatten(1),phase],1)
  fm=lat_pool(innovation).flatten(1)
  yt=lat_pool(truth[:,j]); yp=lat_pool(output[:,j]);r=yt-yp
  base.append(xf[0].cpu().numpy().astype('f8'))
  mem.append(fm[0].cpu().numpy().astype('f8'))
  residual.append(r[0].flatten().cpu().numpy().astype('f8'))
  forecast.append(yp[0].flatten().cpu().numpy().astype('f8'))
  actual.append(yt[0].flatten().cpu().numpy().astype('f8'))
  exposure.append({'slow_rms':float(slow.square().mean().sqrt()),'innovation_rms':float(innovation.square().mean().sqrt()),
     'normalized_innovation_rms':float(innovation.square().mean().sqrt()/(zbase.square().mean().sqrt()+1e-12))})
 return base,mem,residual,forecast,actual,exposure

def collect_cases(m,model,bank,cases,device,tag):
 base=[[],[]];mem=[[],[]];err=[[],[]];out=[[],[]];truth=[[],[]];diagnostics=[];dates=[]
 model.eval()
 for idx,case in enumerate(cases):
  h,y,ph=bank.case(case); h=h[None].to(device);y=y[None].to(device);ph=ph[None].to(device)
  x,iv,r,p,t,ex=probe_features(model,h,ph,y,device)
  dates.append(case['issue_thursday_00utc'])
  for lead in range(2):
   base[lead].append(x[lead]);mem[lead].append(iv[lead]);err[lead].append(r[lead]);out[lead].append(p[lead]);truth[lead].append(t[lead])
   diagnostics.append({'partition':tag,'issue':dates[-1],'lead':('W3','W4')[lead],**ex[lead]})
  if (idx+1)%20==0:print('[A2]',tag,idx+1,'/',len(cases),flush=True)
 return {'dates':dates,'base':[np.stack(k) for k in base],'mem':[np.stack(k) for k in mem],
         'residual':[np.stack(k) for k in err],'forecast':[np.stack(k) for k in out],
         'truth':[np.stack(k) for k in truth], 'diagnostics':diagnostics}

def normalizer(x):
 mu=x.mean(0);sd=np.maximum(x.std(0),1e-5)
 return mu,sd

def solve_ridge(x,y,lam):
 # SVD solves centered Gaussian ridge including unpenalized intercept, safe for p>n.
 xmean=x.mean(0);ymean=y.mean(0)
 xx=x-xmean;yy=y-ymean
 u,s,vt=np.linalg.svd(xx,full_matrices=False)
 weights=vt.T@((s/(s*s+lam))[:,None]*(u.T@yy))
 return (xmean,ymean,weights)

def predict_fit(fit,x):
 xm,ym,w=fit;return (x-xm)@w+ym

def train_predict(fit_part,tune_part,val_part,lead,variant,sham=False):
 # Standardize using FIT PART 2013-14 only for lambda tuning; rest fit 2013-15.
 btrain=np.concatenate([fit_part['base'][lead],tune_part['base'][lead]],0)
 mtrain=np.concatenate([fit_part['mem'][lead],tune_part['mem'][lead]],0)
 bfit=fit_part['base'][lead];mfit=fit_part['mem'][lead]
 btune=tune_part['base'][lead];mtune=tune_part['mem'][lead]
 bval=val_part['base'][lead];mval=val_part['mem'][lead]
 if variant=='instant':
  xfit=bfit;xtune=btune;xtrain=btrain;xval=bval
 elif variant=='memory':
  xfit=np.concatenate([bfit,mfit],axis=1);xtune=np.concatenate([btune,mtune],axis=1)
  xtrain=np.concatenate([btrain,mtrain],axis=1);xval=np.concatenate([bval,mval],axis=1)
 elif variant=='sham':
  # Randomly permuted innovation, equal feature width but no aligned causal information.
  # This is a non-deployable negative control for diagnostic purposes only.
  rng=np.random.default_rng(NP_SEED+lead)
  mf=mfit[rng.permutation(len(mfit))]
  mt=mtune[rng.permutation(len(mtune))]
  merged=np.concatenate([mf,mt],axis=0)
  mv=mval[rng.permutation(len(mval))]
  xfit=np.concatenate([bfit,mf],1);xtune=np.concatenate([btune,mt],1)
  xtrain=np.concatenate([btrain,merged],1);xval=np.concatenate([bval,mv],1)
 else:raise ValueError(variant)
 mu,sd=normalizer(xfit)
 yfit=fit_part['residual'][lead];ytune=tune_part['residual'][lead]
 trials=[]
 for lmb in LAMBDA_GRID:
  r=solve_ridge((xfit-mu)/sd,yfit,lmb)
  pred=predict_fit(r,(xtune-mu)/sd)
  trials.append((float(np.mean((ytune-pred)**2)),lmb))
 chosen=sorted(trials)[0][1]  # tuning uses 2015 only; no 2016 choices
 # Refit to 2013-15; recompute preprocessing without 2016.
 mu,sd=normalizer(xtrain)
 ytrain=np.concatenate([yfit,ytune],0)
 final=solve_ridge((xtrain-mu)/sd,ytrain,chosen)
 p=predict_fit(final,(xval-mu)/sd)
 return p,chosen, {'features':xtrain.shape[1], 'train_samples':len(ytrain),'lambda_tune_MSE':float(min(q[0] for q in trials))}

def score(val,fit,tune):
 scores={}; rows=[];diag={}
 for lead,key in enumerate(('W3','W4')):
  y=val['truth'][lead];basepred=val['forecast'][lead]
  baseline=float(np.mean((basepred-y)**2))
  scores[key]={'coarse_unadjusted_mse':baseline}
  arrs={}
  for variant in ('instant','memory','sham'):
   predicted,lmb,other=train_predict(fit,tune,val,lead,variant)
   new=basepred+predicted
   loss=float(np.mean((new-y)**2))
   scores[key][variant]={'mse':loss,'train_only_lambda':lmb,'diagnostic':other}
   arrs[variant]=np.mean((new-y)**2,axis=1)
  inst=scores[key]['instant']['mse'];mem=scores[key]['memory']['mse'];sham=scores[key]['sham']['mse']
  scores[key]['memory_over_instant_pct']=100*(inst-mem)/max(inst,1e-20)
  scores[key]['memory_over_sham_pct']=100*(sham-mem)/max(sham,1e-20)
  scores[key]['memory_better_windows_vs_instant']=int((arrs['memory']<arrs['instant']).sum())
  scores[key]['memory_better_windows_vs_sham']=int((arrs['memory']<arrs['sham']).sum())
  scores[key]['count']=len(val['dates'])
  for i,date in enumerate(val['dates']):
   rows.append({'issue':date,'lead':key,'instant_band_mse':float(arrs['instant'][i]),
    'memory_band_mse':float(arrs['memory'][i]),'sham_band_mse':float(arrs['sham'][i])})
 # Strict diagnostic-only. 2017 cannot be opened regardless of outcome.
 gate=all(scores[key]['memory_over_instant_pct']>=TARGET_GAIN_PCT and
          scores[key]['memory_over_sham_pct']>=TARGET_GAIN_PCT and
          scores[key]['memory_better_windows_vs_instant']>=32 and
          scores[key]['memory_better_windows_vs_sham']>=32 for key in ('W3','W4'))
 return scores,rows, {'strong_conditional_information_signal':gate,'eligibility_for_formal_training':False,
  'message':'A positive diagnostic is NOT a physical Markov state proof or permission for large training.'}

def package(report_dir):
 out=ROOT/'CIDM_MHC_REAL_A2_P0_CONDITIONAL_INFORMATION_REPORTS.zip'
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(report_dir.glob('*')):
   if p.is_file():z.write(p,'reports/'+p.name)
  z.write(Path(__file__),'source/'+Path(__file__).name)
 with zipfile.ZipFile(out) as z:
  assert z.testzip() is None
 side=Path(str(out)+'.sha256');side.write_text(digest(out)+'  '+out.name+'\n')
 return out,side

def get_case_split(m,cases):
 split={}
 for k,years in YEAR_SPLIT.items():
  split[k]=[c for c in cases if c['split']==('train' if k in ('fit','tune') else 'validation') and
    int(c['issue_thursday_00utc'][:4]) in years and
    all(int(c[w][:4]) in years for w in ('W3_monday','W4_monday'))]
 if {k:len(v) for k,v in split.items()}!={'fit':101,'tune':49,'validation':48}:
  # Fit 2013-14 and tune 2015 are themselves truth-year isolated; 3 cross-boundary train examples are excluded.
  raise RuntimeError('DATE_SPLIT_AUDIT_FAIL '+str({k:len(v) for k,v in split.items()}))
 return split

def run():
 paths,status=evidence();m=load_parent(paths['legacy_source'])
 device='cuda' if torch.cuda.is_available() else 'cpu'
 if device!='cuda':raise RuntimeError('OFFICIAL_GPU_REQUIRED_TO_PROCESS_FULL_ERA5_SET')
 api,token=m.get_hf(True)
 m.ensure_private_dataset(api);m.ensure_model_repo(api)
 # A0 source contains only 12 annual SHA hash records; files pulled exclusively from private HF or official archive.
 manifest,cases,_=m.read_embedded_manifest(paths['a0'])
 verified=[];raw=[]
 for year in m.YEARS:
  for var in m.VARS:
   field,location=m.restore_or_download(year,var,manifest,api,token)
   verified.append({'year':year,'var':var,'sha256':field['sha256'],'source':location})
   raw.append(field)
 bank=m.FieldBank(raw)
 split=get_case_split(m,cases)
 from huggingface_hub import hf_hub_download
 fn=hf_hub_download(repo_id=m.MODEL_REPO,repo_type='model',filename=MODEL_HF_PATH,token=token)
 pin(fn,MODEL_SHA)
 state=torch.load(fn,map_location=device,weights_only=True)
 if int(state['step'])!=64 or int(state['seed'])!=17 or state['arm']!='mhc':raise RuntimeError('CHECKPOINT_META_FAIL')
 model=m.MHCWeather().to(device);model.load_state_dict(state['model']);model.eval()
 report_dir=ROOT/'reports';report_dir.mkdir(parents=True,exist_ok=True)
 # Reproduce archived official same 12 date scores and 48 date 2016 frozen mean within fp tolerance.
 val12=[split['validation'][int(i)] for i in np.unique(np.linspace(0,len(split['validation'])-1,m.VAL_N,dtype=int))]
 measured=m.score(m.eval_arm(model,bank,val12,'mhc',device))
 with zipfile.ZipFile(paths['a1']) as z:old=json.loads(z.read('reports/progress.json'))
 original=next(x['validation'] for x in old['results'] if x['seed']==17 and x['arm']=='mhc')
 for key in ('W3_MSE','W4_MSE'):
  if abs(measured[key]-original[key])>5.e-5*max(1e-10,abs(original[key])):
   raise RuntimeError('ORIGINAL_12_ISSUE_REPLAY_DIFFERENCE_'+key)
 fit=collect_cases(m,model,bank,split['fit'],device,'2013-2014_fit')
 tune=collect_cases(m,model,bank,split['tune'],device,'2015_tune')
 val=collect_cases(m,model,bank,split['validation'],device,'2016_validation')
 scores,rows,gate=score(val,fit,tune)
 savej(report_dir/'status.json',{'status':'FROZEN_CONDITIONAL_INFORMATION_COMPLETED_NO_TRAINING',
  'source_sha256':digest(__file__),'model_checkpoint_sha256':MODEL_SHA,
  'parent_replay_12_dates':measured,'training_years':[2013,2014,2015], 'validation_years':[2016],
  'sealed_2017_untouched':True,'records':{k:len(v) for k,v in split.items()},
  'raw_sha_verified':verified,'score_type':'coarse_zonal_mean_residual_MSE_NOT_AIWQ_RPSS',
  'gate':gate,'metrics':scores, 'parameter_updates_on_mhc':0})
 with (report_dir/'per_issue.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 with (report_dir/'innovation_exposure.csv').open('w',newline='') as f:
  rr=fit['diagnostics']+tune['diagnostics']+val['diagnostics']
  writer=csv.DictWriter(f,fieldnames=list(rr[0]));writer.writeheader();writer.writerows(rr)
 # No commit per record. At most one optional HF commit, failed attempts explicitly recorded and locally archived.
 pack,side=package(report_dir)
 if os.getenv('MHC_REAL_A2_SYNC_HF','1')=='1':
  from huggingface_hub import CommitOperationAdd
  try:
   api.create_commit(repo_id=m.MODEL_REPO,repo_type='model',token=token,
    operations=[CommitOperationAdd(path_in_repo='experiments/MHC_REAL_A2_P0/'+pack.name,path_or_fileobj=str(pack)),
                CommitOperationAdd(path_in_repo='experiments/MHC_REAL_A2_P0/'+side.name,path_or_fileobj=str(side))],
    commit_message='MHC-Real A2 frozen conditional information audit: single atomic commit')
   savej(report_dir/'hf_sync.json',{'status':'SUCCESS'})
  except Exception as e:savej(report_dir/'hf_sync.json',{'status':'FAILED_RETRY_LATER','error':str(e)[:500]})
 else:savej(report_dir/'hf_sync.json',{'status':'SKIPPED_EXPLICITLY_DISABLED','reason':'Set MHC_REAL_A2_SYNC_HF=1 for one atomic HF commit.'})
 pack,side=package(report_dir)
 try:
  gh=m.github_sync_light([Path(__file__),report_dir/'status.json'])
 except Exception as e:gh={'status':'FAILED_RETRY_LATER','error':str(e)[:300]}
 savej(report_dir/'github_sync.json',gh)
 pack,side=package(report_dir)
 return {'status':'FROZEN_CONDITIONAL_INFORMATION_COMPLETED_NO_TRAINING','scores':scores,'zip':str(pack),'sha':digest(pack)}

def smoke():
 """Honest offline program-only test, not a weather experiment."""
 _,_=evidence();m=load_parent(ROOT/'evidence'/'mhc_real_a1_fix1_atomic_resume.py')
 rng=np.random.default_rng(NP_SEED)
 model=m.MHCWeather().eval()
 h=torch.from_numpy(rng.normal(size=(1,8,2,13,19)).astype('f'))
 ph=torch.tensor([[0.,1.]],dtype=torch.float32)
 tru=torch.from_numpy(rng.normal(size=(1,2,2,13,19)).astype('f'))
 b,memory,resid,pred,truth,exposure=probe_features(model,h,ph,tru,'cpu')
 assert len(b)==2 and len(memory)==2 and b[0].shape==(16*3+2*3+2,)
 assert all(np.isfinite(z).all() for z in b+memory+resid)
 assert all(x['innovation_rms']>1e-9 for x in exposure)
 # fixed supervised residual proxy synthetic test.
 def gen(n):
  b0=rng.normal(size=(n,len(b[0])));v0=rng.normal(size=(n,len(memory[0])))
  tgt0=0.3*b0[:,:6]+0.1*v0[:,:6]+rng.normal(size=(n,6))*.2
  return {'base':[b0,b0],'mem':[v0,v0], 'residual':[tgt0,tgt0],
   'forecast':[np.zeros_like(tgt0)]*2,'truth':[tgt0,tgt0],
   'dates':[f'2016-01-{i+1:02d}' for i in range(n)]}
 ft=gen(20);tu=gen(10);va=gen(12)
 metrics,rows,gate=score(va,ft,tu)
 assert len(rows)==24 and np.isfinite(metrics['W3']['memory']['mse'])
 rp=ROOT/'reports';rp.mkdir(exist_ok=True)
 savej(rp/'status.json',{'status':'SMOKE_ONLY_NO_REAL_DATA','mathematics':'PASS','real_inference':'NOT_EXECUTED'})
 pack,side=package(rp)
 return {'status':'SMOKE_ONLY_NO_REAL_DATA','zip':str(pack),'sha':digest(pack),'row_count':len(rows)}

def main():
 mode=os.getenv('MHC_REAL_A2_MODE','official')
 try:r=smoke() if mode=='smoke' else run()
 except Exception as e:
  rp=ROOT/'reports';rp.mkdir(parents=True,exist_ok=True)
  savej(rp/'status.json',{'status':'BLOCKED_NO_TRAINING','error_type':type(e).__name__,
   'error':str(e),'traceback':traceback.format_exc().splitlines()[-15:]})
  pack,side=package(rp)
  r={'status':'BLOCKED_NO_TRAINING','error':str(e),'zip':str(pack),'sha':digest(pack)}
 print('[A2]',r['status'],r.get('scores',''),r.get('zip',''),flush=True)
 return r
if __name__=='__main__':main()
