"""CIDM-MHC-Real A6/P0: bounded causal history adapter, development gate before fresh test.

No new weather backbone training. Uses the fixed A1 seed17-64 trained network and
2013-2015 only for diagnostic calibration; 2016 for DEVELOPMENT safety check.
Only if both controls pass, evaluate 2020-21 ONCE. 2017 audit remains sealed.
Continuous coarse lat-belt MSE IS NOT global AI Weather Quest RPSS.
No Google Drive. HF Dataset private annual data + HF Model frozen checkpoint.
"""
from __future__ import annotations
import os,sys,math,csv,json,hashlib,zipfile,traceback,importlib.util,datetime as dt
from pathlib import Path
import numpy as np
import torch

HOME=Path(os.getenv('MHC_A6_HOME','/content/MHC_REAL_A6')).resolve()
HOME.mkdir(parents=True,exist_ok=True)
A5_SRC_SHA='c8edeb5e5531dcf89110a25053e1787279e30ac2a54af235e77cd60d44aa4544'
A5_ACTUAL_SHA='0c5c74bc58b5729dd99219305b58433a819b133f080648190336d8493b5117e7'
FRESH=(2020,2021)
RISK_CAP=0.25
ALPHAS=(0.5,1.0)
BETAS=(0.0,0.05,0.10,0.25)
TUNE_MIN_GAIN_PCT=0.20
DEV_MIN_GAIN_PCT=0.20
MAX_TAIL_CVAR_RATIO=1.05


def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def savej(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n',encoding='utf8')

def preflight():
 src=HOME/'mhc_real_a5_fresh_year_season_gate.py'
 parent=HOME/'evidence'/'A5_USER_FRESH_2018_2019_REPORT.zip'
 if not src.is_file() or sha(src)!=A5_SRC_SHA:raise RuntimeError('A5_ORIGINAL_CODE_SHA_MISMATCH')
 if not parent.is_file() or sha(parent)!=A5_ACTUAL_SHA:raise RuntimeError('A5_OFFICIAL_RESULT_SHA_MISMATCH')
 with zipfile.ZipFile(parent) as z:
  if z.testzip():raise RuntimeError('A5_OFFICIAL_CRC_FAIL')
  r=json.loads(z.read('reports/status.json'))
  if r['status']!='FRESH_2018_2019_EVALUATION_COMPLETED_NO_WEATHER_TRAINING':raise RuntimeError('A5_ORIGIN_NOT_OFFICIAL')
  if r.get('2017_truth_unread') is not True:raise RuntimeError('2017_SEALED_CONTRACT_FAIL')
  if r.get('joint_diagnostic_gate') is not False:raise RuntimeError('A5_GATE_UNEXPECTED')
  for k in ('W3','W4'):
   score=r['fresh_year_results'][k]['Q1_Q3_physical_else_instant']['gain_vs_instant_pct']
   if score>=0:raise RuntimeError('A5_NEGATIVE_SCORE_UNEXPECTED_'+k)
  # Fail loudly before expensive downloading if report could not serialize.
  json.dumps({'parent_sha':sha(parent),'status':'PRE-FLIGHT-PASS'},allow_nan=False)
 return r

def load_parent():
 src=HOME/'mhc_real_a5_fresh_year_season_gate.py'
 spec=importlib.util.spec_from_file_location('a6_pinned_a5_parent',str(src))
 parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
 parent.BASE=HOME;parent.EVIDENCE=HOME/'evidence'
 parent.preflight()
 return parent

def get_feature(part,physical,lead,arm):
 if arm=='instant':return np.asarray(part['base'][lead],dtype='f8')
 if arm=='old_memory':return np.concatenate([part['base'][lead],part['mem'][lead]],axis=-1).astype('f8')
 if arm=='physical':return np.concatenate([part['base'][lead],physical],axis=-1).astype('f8')
 raise ValueError(arm)

def ridge_predict(a3,fit,tune,eval_part,fp,tp,ep,lead,arm,force_lambda=None):
 xf=get_feature(fit,fp,lead,arm);xt=get_feature(tune,tp,lead,arm);xe=get_feature(eval_part,ep,lead,arm)
 yf=fit['residual'][lead];yt=tune['residual'][lead]
 mu=xf.mean(0);sd=np.maximum(xf.std(0),1e-5)
 pack=a3.fit_ridge((xf-mu)/sd,yf,1)
 scores={float(l):float(np.mean((a3.pred_svd(pack,(xt-mu)/sd,l)-yt)**2)) for l in a3.RIDGE_GRID}
 chosen=float(sorted(scores.items(),key=lambda v:(v[1],v[0]))[0][0])
 if force_lambda is not None and abs(chosen-force_lambda)>1e-9:raise RuntimeError('2015_LAMBDA_REPLAY_CHANGED_'+arm)
 xall=np.concatenate([xf,xt]);yall=np.concatenate([yf,yt])
 mu=xall.mean(0);sd=np.maximum(xall.std(0),1e-5)
 pred=a3.pred_svd(a3.fit_ridge((xall-mu)/sd,yall,1),(xe-mu)/sd,chosen)
 out=np.asarray(eval_part['forecast'][lead],dtype='f8')+pred
 return out,{'lambda_selected_2015_only':chosen,'features':xf.shape[1],'tune_residual_mse':scores[chosen]}

def ridge_predict_fit_only(a3,fit,eval_part,fp,ep,lead,arm,lam):
 xf=get_feature(fit,fp,lead,arm);xe=get_feature(eval_part,ep,lead,arm)
 y=np.asarray(fit['residual'][lead],dtype='f8');mu=xf.mean(0);sd=np.maximum(xf.std(0),1e-5)
 return np.asarray(eval_part['forecast'][lead],dtype='f8')+a3.pred_svd(a3.fit_ridge((xf-mu)/sd,y,1),(xe-mu)/sd,lam)

def risk_g(physical):
 """Known-at-issue physical-state regime: last 4 week std / 8 week std.

The eight 6-wide physical feature groups are causal. No future truth, no
outcome-based condition, no quarter/phase label or hand-chosen Q1 selector.
"""
 p=np.asarray(physical,dtype='f8')
 if p.ndim!=2 or p.shape[1]!=48:raise RuntimeError('PHYSICAL_FEATURES_WRONG_WIDTH')
 vol8=np.maximum(np.abs(p[:,30:36]).mean(-1),1e-5)
 vol4=np.abs(p[:,36:42]).mean(-1)
 ratio=np.minimum(vol4/vol8,100)
 gate=np.where(ratio<=1.0,1.0,1.0/ratio)
 if not np.all(np.isfinite(gate)) or np.any((gate<0)|(gate>1)):raise RuntimeError('GATE_NONFINITE')
 return gate.astype('f8')

def compose(pred,phy_feat,alpha,beta,adaptive):
 i,m,p=pred['instant'],pred['old_memory'],pred['physical']
 if any(q.shape!=i.shape for q in (m,p)):raise RuntimeError('PRED_SHAPE_DIFFERENT')
 g=risk_g(phy_feat) if adaptive else np.ones(len(phy_feat))
 if not (0<=beta<=alpha<=1):raise RuntimeError('NONCONVEX_COMBINATION')
 # Three nonnegative weights, sum exactly one: 1-alpha, alpha-beta*g, beta*g.
 return (1-alpha)*i+(alpha-beta*g)[:,None]*m+(beta*g)[:,None]*p

def mse_per_issue(y,truth):
 return np.mean(np.square(np.asarray(y,dtype='f8')-np.asarray(truth,dtype='f8')),axis=1)

def cvar(x,frac=0.2):
 return float(np.sort(np.asarray(x))[-max(1,int(np.ceil(len(x)*frac))):].mean())

def dev_stats(ref,alt):
 ref=np.asarray(ref,dtype='f8');alt=np.asarray(alt,dtype='f8')
 return {'mse':float(alt.mean()),'reference_mse':float(ref.mean()),
  'gain_pct':float(100*(ref.mean()-alt.mean())/max(ref.mean(),1e-20)),
  'wins':int(np.sum(alt<ref)),'n':len(alt),
  'tail_cvar_ratio':float(cvar(alt)/max(cvar(ref),1e-20)),
  'max_worsening_abs':float(np.max(alt-ref))}

def choose_dev(train_tune,dev,train,train_phys,tune_phys,dev_phys,a3):
 """Tune weights on 2015 *only*, then require positive 2016 development safety.

 Train 2013-14 only for 2015 residual predictions, eliminating 2015 label
 fitting. Final ridge refit 2013-15 for 2016, 2020+ as in original A3.
 """
 preds_tune={}; preds_dev={}; ridge_detail={}
 for lead in (0,1):
  preds_tune[lead]={};preds_dev[lead]={};ridge_detail[lead]={}
  for arm in ('instant','old_memory','physical'):
   # fixed lambda grid learned on 2015 for subsequent model; evaluate fit-only
   pdev,info=ridge_predict(a3,train,train_tune,dev,train_phys,tune_phys,dev_phys,lead,arm)
   ptune=ridge_predict_fit_only(a3,train,train_tune,train_phys,tune_phys,lead,arm,info['lambda_selected_2015_only'])
   preds_tune[lead][arm]=ptune;preds_dev[lead][arm]=pdev;ridge_detail[lead][arm]=info
 # Score vs OLD MHC, not just Instant. No use of future data to select.
 candidates=[]
 for alpha in ALPHAS:
  for beta in BETAS:
   if beta>alpha:continue
   for adaptive in (False,True) if beta>0 else (False,):
    scores=[];safe=True
    for lead in (0,1):
     y=train_tune['truth'][lead];ref=mse_per_issue(preds_tune[lead]['old_memory'],y)
     cand=mse_per_issue(compose(preds_tune[lead],tune_phys,alpha,beta,adaptive),y)
     stat=dev_stats(ref,cand); scores.append(stat)
     if stat['tail_cvar_ratio']>MAX_TAIL_CVAR_RATIO:safe=False
    objective=sum(s['mse']+0.20*s['tail_cvar_ratio']*s['mse'] for s in scores)
    candidates.append({'alpha':alpha,'beta':beta,'adaptive':adaptive,'tune':scores,'objective':float(objective),'safe':safe})
 candidates=sorted(candidates,key=lambda x:(not x['safe'],x['objective'],x['beta'],x['alpha'],x['adaptive']))
 # Strictly compare adaptive to matched-alpha constant beta on 2015:
 adaptive=[x for x in candidates if x['adaptive'] and x['safe'] and x['beta']>0]
 if adaptive:
  chosen=adaptive[0]
 else:
  chosen=None
 dev_results={}
 if chosen is not None:
  for lead,name in enumerate(('W3','W4')):
   yr=dev['truth'][lead]
   r=mse_per_issue(preds_dev[lead]['old_memory'],yr)
   i=mse_per_issue(preds_dev[lead]['instant'],yr)
   yp=compose(preds_dev[lead],dev_phys,chosen['alpha'],chosen['beta'],True)
   op=mse_per_issue(yp,yr)
   const=compose(preds_dev[lead],dev_phys,chosen['alpha'],chosen['beta'],False)
   dev_results[name]={'vs_old_memory':dev_stats(r,op),'vs_instant':dev_stats(i,op),
     'vs_same_coeff_constant':dev_stats(mse_per_issue(const,yr),op),
     'old_memory_vs_instant':dev_stats(i,r)}
 # Train-only selection Gate (2015) AND reused development-year Gate (2016).
 qualified=bool(chosen is not None and all(s['gain_pct']>=TUNE_MIN_GAIN_PCT for s in chosen['tune']) and
  all(dev_results[k]['vs_old_memory']['gain_pct']>=DEV_MIN_GAIN_PCT and
      dev_results[k]['vs_old_memory']['tail_cvar_ratio']<=MAX_TAIL_CVAR_RATIO and
      dev_results[k]['vs_instant']['gain_pct']>0 for k in ('W3','W4')))
 return {'chosen':chosen,'diagnostic_candidates':candidates,'development':dev_results,
   'qualified_to_unseal_2020_2021_once':qualified,'ridge_details':ridge_detail}

def do_report(state,rows,mode='official'):
 rd=HOME/'reports';rd.mkdir(parents=True,exist_ok=True)
 savej(rd/'status.json',state)
 if rows:
  with (rd/'per_issue.csv').open('w',encoding='utf8',newline='') as f:
   w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 out=HOME/'CIDM_MHC_REAL_A6_P0_BOUNDED_CAUSAL_ADAPTER_REPORTS.zip'
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
  for fi in sorted(rd.glob('*')):
   if fi.is_file():z.write(fi,arcname='reports/'+fi.name)
  z.write(Path(__file__),'source/'+Path(__file__).name)
 with zipfile.ZipFile(out) as z:
  if z.testzip():raise RuntimeError('REPORT_ZIP_CRC_FAIL')
 side=Path(str(out)+'.sha256');side.write_text(sha(out)+'  '+out.name+'\n',encoding='utf8')
 return out,side

def upload_receipt(api,token,m,out,side):
 """ONE optional atomic HF commit; GitHub uploads lightweight report/source; never repack the ZIP."""
 receipt={'archive_sha256':sha(out),'hf':'SKIPPED_DEFAULT','github':'SKIPPED_NO_TOKEN'}
 if os.getenv('MHC_A6_SYNC_HF','1')=='1':
  try:
   from huggingface_hub import CommitOperationAdd
   commit=api.create_commit(repo_id=m.MODEL_REPO,repo_type='model',token=token,operations=[
    CommitOperationAdd(path_in_repo='experiments/MHC_REAL_A6_P0/'+out.name,path_or_fileobj=str(out)),
    CommitOperationAdd(path_in_repo='experiments/MHC_REAL_A6_P0/'+side.name,path_or_fileobj=str(side))],
    commit_message='MHC Real A6 immutable development gate/fresh-year diagnostic (not AIWQ submission)')
   receipt['hf']='SUCCESS';receipt['hf_commit_url']=str(getattr(commit,'commit_url',''))
  except Exception as exc:receipt['hf']='FAILED_RETRY_LATER';receipt['hf_error']=str(exc)[:350]
 if os.getenv('MHC_A6_SYNC_GITHUB','1')=='1':
  try:
   gh_token=m.get_secret('GH_TOKEN') or m.get_secret('GITHUB_TOKEN')
   if not gh_token:
    receipt['github']='SKIPPED_NO_GH_TOKEN'
   else:
    import requests,base64
    paths=(Path(__file__),HOME/'reports'/'status.json')
    published=[]
    for loc in paths:
     name='research/earth_aiwq/mhc_real_a6/'+loc.name
     url='https://api.github.com/repos/Mr-wuff/AI-Weather-Quest/contents/'+name
     headers={'Authorization':'Bearer '+gh_token,'Accept':'application/vnd.github+json'}
     old=requests.get(url,headers=headers,timeout=25)
     if old.status_code not in (200,404):raise RuntimeError('GITHUB_GET_FAILED_'+str(old.status_code))
     payload={'message':'Archive MHC Real A6 bounded adapter qualification',
       'content':base64.b64encode(loc.read_bytes()).decode(),'branch':'main'}
     if old.status_code==200:payload['sha']=old.json()['sha']
     r=requests.put(url,headers=headers,json=payload,timeout=90)
     if r.status_code not in (200,201):raise RuntimeError('GITHUB_PUT_FAILED_'+str(r.status_code)+' '+r.text[:100])
     published.append(name)
    receipt['github']={'status':'SUCCESS','paths':published}
  except Exception as exc:receipt['github']={'status':'FAILED_RETRY_LATER','error':str(exc)[:350]}
 savej(HOME/'sync_receipt.json',receipt)
 return receipt

def official():
 previous=preflight();a5=load_parent();a3,a2,m,paths=a5.parent_modules()
 if not torch.cuda.is_available():raise RuntimeError('OFFICIAL_REQUIRES_COLAB_GPU_FOR_REAL_GLOBAL_FORECAST')
 device='cuda';api,token=m.get_hf(True);m.ensure_private_dataset(api);m.ensure_model_repo(api)
 manifest,devcases,_=m.read_embedded_manifest(paths['a0'])
 raw=[];hashinfo=[]
 for year in m.YEARS:
  for var in m.VARS:
   item,source=m.restore_or_download(year,var,manifest,api,token)
   raw.append(item);hashinfo.append({'year':year,'var':var,'sha256':item['sha256'],'source':source})
 bank=m.FieldBank(raw)
 split=a2.get_case_split(m,devcases)
 from huggingface_hub import hf_hub_download
 snap=hf_hub_download(repo_id=m.MODEL_REPO,repo_type='model',filename=a2.MODEL_HF_PATH,token=token)
 a2.pin(snap,a2.MODEL_SHA)
 ckpt=torch.load(snap,map_location=device,weights_only=True)
 if (ckpt.get('arm'),ckpt.get('seed'),ckpt.get('step'))!=('mhc',17,64):raise RuntimeError('FROZEN_PARENT_META_CHANGED')
 model=m.MHCWeather().to(device).eval();model.load_state_dict(ckpt['model'])
 # Archived original 12-date replay is a strong gate.
 oldjson=previous['original_2016_replay']
 oldsel=[split['validation'][int(i)] for i in np.unique(np.linspace(0,len(split['validation'])-1,m.VAL_N,dtype=int))]
 replay=m.score(m.eval_arm(model,bank,oldsel,'mhc',device))
 for lead in ('W3_MSE','W4_MSE'):
  if abs(replay[lead]-oldjson[lead])>5e-5*max(abs(oldjson[lead]),1e-20):raise RuntimeError('ORIGINAL_MODEL_REPLAY_FAIL_'+lead)
 samples={p:a2.collect_cases(m,model,bank,split[p],device,p) for p in ('fit','tune','validation')}
 phys={p:a3.add_physical_features(a2,model,bank,split[p],device) for p in ('fit','tune','validation')}
 # Original A3 ridge-vs-manual scores must be consistent before trusting the adapter.
 for lead in (0,1):
  for arm in ('instant','old_memory','physical'):
   obs,_=ridge_predict(a3,samples['fit'],samples['tune'],samples['validation'],
      phys['fit'],phys['tune'],phys['validation'],lead,arm)
   e=mse_per_issue(obs,samples['validation']['truth'][lead])
   feat={'fit':None,'tune':None,'validation':None} if arm=='instant' else (
     {'fit':phys['fit'],'tune':phys['tune'],'validation':phys['validation']} if arm=='physical' else
     {key:samples[key]['mem'][lead] for key in ('fit','tune','validation')})
   ref,_=a3.regress(samples['fit'],samples['tune'],samples['validation'],lead,feat)
   if not np.allclose(e,ref,rtol=1e-9,atol=1e-12):raise RuntimeError('A3_RIDGE_REPLAY_FAIL_'+arm)
 dev=choose_dev(samples['tune'],samples['validation'],samples['fit'],phys['fit'],phys['tune'],phys['validation'],a3)
 status={'status':'DEVELOPMENT_ONLY_GATE_FAIL_NO_EXTERNAL_OPEN','experiment':'A6_P0_CAUSAL_BOUNDED_ADAPTER',
  'source_parent_a5_sha':A5_SRC_SHA,'original_a5_report_sha':A5_ACTUAL_SHA,
  'original_2016_replay':replay,'data_hashes_2012_2017':hashinfo,
  'true_2017_eval_used':False,'2020_2021_labels_accessed':False,'new_weather_model_updates':0,
  'score_type':'COARSE_LAT_BELT_RESIDUAL_MSE_NOT_OFFICIAL_AIWQ_RPSS',
  'development_fit_years':[2013,2014],'development_tune_year':2015,
  'development_diagnostic_year':2016,'development_gate':dev,
  'external_eval_authorized':False,'formal_cidm_training_authorized':False}
 if not dev['qualified_to_unseal_2020_2021_once']:
  out,side=do_report(status,[])
  receipt=upload_receipt(api,token,m,out,side)
  return {'status':status['status'],'file':str(out),'sha256':sha(out),'external_years_not_opened':True,'sync':receipt}
 # Only now allow the one-time genuinely fresh 2020-2021 benchmark.
 # Raw 2018-19 source files are needed ONLY as past observed 2020 causal context,
 # and are verified against immutable A5 archive. No 2018-19 forecast truths scored.
 ext=[];source_pin={(a['year'],a['var']):a['sha256'] for a in previous['new_year_source_hashes']['entries']}
 for year in (2018,2019):
  for var in m.VARS:
   record,unused=a5.pull_new_field(m,year,var,api,token)
   if unused is not None:raise RuntimeError('REQUIRE_OLD_2018_2019_PRIVATE_HF_CACHE_NOT_REDOWNLOAD')
   if record['sha256']!=source_pin[(year,var)]:raise RuntimeError('2018_2019_CAUSAL_INPUT_RAW_SHA_CHANGED')
   ext.append(record)
 # Restore/update just the needed 2020-21 source files, in one private commit.
 a5.NEW_YEARS=FRESH
 raw_fresh,manifest_new=a5.hydrate_and_persist_new(m,api,token)
 bank2=m.FieldBank(raw+ext+raw_fresh)
 dates=[date for v,date in bank2.x if v=='mslp']
 cases=a5.build_untouched_issues(FRESH,dates)
 future=a2.collect_cases(m,model,bank2,cases,device,'2020_2021_TRULY_FRESH')
 future_phys=a3.add_physical_features(a2,model,bank2,cases,device)
 metrics={};rows=[]
 for lead,name in enumerate(('W3','W4')):
  pred={}
  for arm in ('instant','old_memory','physical'):
   pred[arm],_=ridge_predict(a3,samples['fit'],samples['tune'],future,phys['fit'],phys['tune'],future_phys,
      lead,arm,force_lambda=dev['ridge_details'][lead][arm]['lambda_selected_2015_only'])
  cp=dev['chosen'];y=compose(pred,future_phys,cp['alpha'],cp['beta'],True)
  truth=future['truth'][lead]
  m0=mse_per_issue(pred['old_memory'],truth);i0=mse_per_issue(pred['instant'],truth)
  p0=mse_per_issue(pred['physical'],truth);ym=mse_per_issue(y,truth)
  mets={'adapter_vs_old_memory':dev_stats(m0,ym),'adapter_vs_instant':dev_stats(i0,ym),
    'physical_vs_instant':dev_stats(i0,p0),'memory_vs_instant':dev_stats(i0,m0),'by_year':{}}
  for year in FRESH:
   ix=np.asarray([j for j,date in enumerate(future['dates']) if date.startswith(str(year)+'-')],dtype=int)
   mets['by_year'][str(year)]={'adapter_vs_memory':dev_stats(m0[ix],ym[ix]),
                               'adapter_vs_instant':dev_stats(i0[ix],ym[ix])}
  metrics[name]=mets
  for j,date in enumerate(future['dates']):
   rows.append({'issue':date,'lead':name,'year':int(date[:4]),'instant_mse':float(i0[j]),
    'old_memory_mse':float(m0[j]),'always_physical_mse':float(p0[j]),
    'bounded_adapter_mse':float(ym[j]),'preissue_risk_gate':float(risk_g(future_phys)[j])})
 gates={lead:(r['adapter_vs_old_memory']['gain_pct']>=0.5 and r['adapter_vs_instant']['gain_pct']>=0.5 and
              r['adapter_vs_old_memory']['wins']>=int(math.ceil(0.60*r['adapter_vs_old_memory']['n'])) and
              all(r['by_year'][str(y)]['adapter_vs_memory']['gain_pct']>=0 for y in FRESH))
         for lead,r in metrics.items()}
 status.update({'status':'NEW_2020_2021_FROZEN_ADAPTER_EVALUATION_COMPLETED',
  '2020_2021_labels_accessed':True,'fresh_years':list(FRESH),'fresh_issues':len(cases),
  'new_data_manifest':manifest_new,'fresh_results':metrics,'external_gate':gates,
  'joint_external_gate':all(gates.values()),
  'external_eval_authorized':True,'formal_cidm_training_authorized':False,
  'note':'One-time offline new-year residual diagnostic, not official RPSS or operational data availability proof.'})
 out,side=do_report(status,rows)
 # Published ZIP immutable, upload receipt is a *separate* file.
 receipt=upload_receipt(api,token,m,out,side)
 return {'status':status['status'],'file':str(out),'sha256':sha(out),
         'joint_external_gate':status['joint_external_gate'],'sync':receipt}

def smoke():
 prev=preflight();a5=load_parent();a3,a2,m,paths=a5.parent_modules()
 rng=np.random.default_rng(17)
 def make(n,year):
  x=rng.normal(size=(n,56)); mem=rng.normal(size=(n,48));phys=rng.normal(size=(n,48));
  # Positive recent std, positive previous long std, realistic width.
  phys[:,30:42]=np.abs(phys[:,30:42])+0.3
  r=.08*x[:,:6]+.03*mem[:,:6]+.03*phys[:,:6]+rng.normal(0,.02,(n,6))
  return ({'base':[x,x],'mem':[mem,mem],'residual':[r,r],
   'forecast':[np.zeros_like(r)]*2,'truth':[r,r],
   'dates':[f'{year}-{(i//4)%12+1:02d}-{(i%4)*7+1:02d}' for i in range(n)]}, phys)
 f,fp=make(28,2013);t,tp=make(16,2015);v,vp=make(16,2016)
 res=choose_dev(t,v,f,fp,tp,vp,a3)
 assert 'qualified_to_unseal_2020_2021_once' in res
 assert all(x['beta']<=x['alpha'] for x in res['diagnostic_candidates'])
 assert np.min(risk_g(vp))>=0 and np.max(risk_g(vp))<=1
 # verify no target read or causal sample creation in failure branch.
 status={'status':'SMOKE_ONLY_NO_NEW_WEATHER_DATA',
  'evidence_a5_sha256':sha(HOME/'evidence'/'A5_USER_FRESH_2018_2019_REPORT.zip'),
  'outcome':res,'2020_2021_labels_accessed':False,
  'new_model_optimizer_steps':0,'original_parent_loaded':True}
 out,side=do_report(status,[])
 return {'status':status['status'],'file':str(out),'sha256':sha(out),
  'math_self_test':'PASS','deferred_external_labels':True}

def main():
 try:
  if os.getenv('MHC_A6_MODE','official')=='smoke':return smoke()
  return official()
 except Exception as ex:
  status={'status':'BLOCKED_NO_TRAINING','error_type':type(ex).__name__,
   'error':str(ex),'traceback':traceback.format_exc().splitlines()[-14:],
   'formal_cidm_training_authorized':False}
  out,side=do_report(status,[])
  return {'status':status['status'],'error':str(ex),'file':str(out),'sha256':sha(out)}

if __name__=='__main__':print('[A6]',json.dumps(main(),ensure_ascii=False,default=str),flush=True)
