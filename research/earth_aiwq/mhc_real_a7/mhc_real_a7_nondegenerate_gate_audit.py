"""CIDM-MHC Real A7/P0: nondegenerate causal gating qualification on reused development years.

Scope: frozen A1 seed17 64step backbone and A3 linear probe only. No new target years,
no official AIWQ RPSS, no 2017 truth, no 2022+ data, NO Google Drive.
A6 fresh 2020-21 already examined: its CSV is used for FORENSIC evidence only.
"""
from __future__ import annotations
import os, sys, io, csv, json, hashlib, zipfile, math, traceback, importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(os.getenv('MHC_A7_HOME','/content/MHC_REAL_A7')).resolve()
ROOT.mkdir(parents=True,exist_ok=True)
A6_ZIP_SHA='6edbe611a04e63d183d7515e9009b27d0f453b4aee3742e90821139199071706'
A6_SOURCE_SHA='99da9b0d9f1b911d0339a0310ba3ca1255a50847dd0b5d6031af4a2777c0a9d7'
BETA=.25
ALPHA=1.
MSE_GATE_MIN=.20
TAIL_MAX=1.05
STDMIN=.08
FRACTION_MIDDLE_MIN=.15
DEV_WIN_FRACTION=.55


def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()


def save_json(path,x):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n',encoding='utf8')


def load_py(path,name):
 spec=importlib.util.spec_from_file_location(name,str(path))
 m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
 return m


def preflight():
 evidence=ROOT/'evidence'/'A6_USER_OFFICIAL_REPORT.zip'
 src=ROOT/'mhc_real_a6_causal_bounded_adapter.py'
 if not evidence.is_file() or sha(evidence)!=A6_ZIP_SHA:raise RuntimeError('LOCKED_A6_ZIP_SHA_MISMATCH')
 if not src.is_file() or sha(src)!=A6_SOURCE_SHA:raise RuntimeError('LOCKED_A6_CODE_SHA_MISMATCH')
 with zipfile.ZipFile(evidence) as z:
  if z.testzip() is not None:raise RuntimeError('ZIP_CRC_FAIL')
  status=json.loads(z.read('reports/status.json'))
  rows=list(csv.DictReader(io.StringIO(z.read('reports/per_issue.csv').decode('utf8'))))
  if hashlib.sha256(z.read('source/mhc_real_a6_causal_bounded_adapter.py')).hexdigest()!=A6_SOURCE_SHA:
   raise RuntimeError('ARCHIVED_SOURCE_MISMATCH')
 if status.get('status')!='NEW_2020_2021_FROZEN_ADAPTER_EVALUATION_COMPLETED':raise RuntimeError('NOT_REAL_A6')
 if status.get('true_2017_eval_used') is not False or status.get('2020_2021_labels_accessed') is not True:raise RuntimeError('YEAR_CONTRACT_UNEXPECTED')
 if status.get('formal_cidm_training_authorized') is not False:raise RuntimeError('FORMAL_TRAIN_UNEXPECTED')
 if len(rows)!=194:raise RuntimeError('A6_CSV_LENGTH_INVALID')
 gate=np.array([float(r['preissue_risk_gate']) for r in rows]);
 if not np.isfinite(gate).all():raise RuntimeError('NONFINITE_GATE')
 # Reproduce all four A6 pooled external means and gains from the actual user CSV.
 replay={}
 for lead in ('W3','W4'):
  subset=[r for r in rows if r['lead']==lead]
  if len(subset)!=97:raise RuntimeError('ISSUE_COUNT_INVALID_'+lead)
  cols={key:np.array([float(r[key]) for r in subset]) for key in ('instant_mse','old_memory_mse','always_physical_mse','bounded_adapter_mse')}
  observed=status['fresh_results'][lead]
  for key,ref in [('old_memory_mse','adapter_vs_old_memory'),('instant_mse','adapter_vs_instant')]:
   gain=100*(cols[key].mean()-cols['bounded_adapter_mse'].mean())/cols[key].mean()
   if abs(gain-observed[ref]['gain_pct'])>1e-9:raise RuntimeError('A6_METRIC_REPLAY_FAIL_'+lead+'_'+ref)
  replay[lead]={key:float(a.mean()) for key,a in cols.items()}
 candidate=status['development_gate']['diagnostic_candidates']
 diffs=[]
 for c in candidate:
  if c['beta']<=0 or not c['adaptive']:continue
  reference=next((x for x in candidate if x['beta']==c['beta'] and x['alpha']==c['alpha'] and not x['adaptive']),None)
  if reference is not None:
   diffs.append(max(abs(x['mse']-y['mse']) for x,y in zip(c['tune'],reference['tune'])))
 audit={'official_user_a6_zip_sha256':A6_ZIP_SHA,'csv_rows':len(rows),'unique_issue_dates':len({r['issue'] for r in rows}),
  'gate_min':float(gate.min()),'gate_max':float(gate.max()),'gate_std':float(gate.std()),
  'gate_all_one':bool(np.all(gate==1)),'adaptive_vs_const_2015_max_score_difference':float(max(diffs) if diffs else float('nan')),
  'yearly_2020_21_used_before':True,'a6_recomputed_mean_scores':replay,
  'old_external_gate':status.get('external_gate'), 'old_joint_gate':status.get('joint_external_gate')}
 return status,rows,audit


def safe_ratio(physical):
 p=np.asarray(physical,dtype=np.float64)
 if p.ndim!=2 or p.shape[1]!=48 or not np.isfinite(p).all():raise RuntimeError('48_DIM_PHYSICAL_FEATURE_CONTRACT')
 # Original A3 physical_history_features: 8 operators * 6 (3 latitude bands * 2 variables).
 vol8=np.maximum(np.abs(p[:,30:36]).mean(axis=1),1e-5)
 vol4=np.abs(p[:,36:42]).mean(axis=1)
 trend=np.abs(p[:,24:30]).mean(axis=1)
 return np.minimum(vol4/vol8,100),np.minimum(trend/vol8,100)


def fit_map(physical):
 # All cutoffs are determined exclusively using 2013–2014 train-known-at-issue inputs.
 v,t=safe_ratio(physical)
 def quant(x):
  return [float(q) for q in np.quantile(x,[.1,.5,.9])]
 return {'vol4_vs_vol8':quant(v),'trend_vs_vol8':quant(t)}


def soft_rank(r,quantiles):
 lo,med,hi=quantiles
 denom=max(hi-lo,1e-5)
 return np.clip((hi-r)/denom,0.,1.)


def gate_values(physical,mapping,kind):
 vol,trend=safe_ratio(physical)
 if kind=='constant':return np.ones(len(physical),dtype='f8')
 if kind=='legacy':return np.where(vol<=1.,1.,1./np.maximum(vol,1e-9))
 if kind=='vol_quantile':return soft_rank(vol,mapping['vol4_vs_vol8'])
 if kind=='trend_quantile':return soft_rank(trend,mapping['trend_vs_vol8'])
 if kind=='hybrid':return .5*(soft_rank(vol,mapping['vol4_vs_vol8'])+soft_rank(trend,mapping['trend_vs_vol8']))
 raise ValueError(kind)


def gate_stats(x):
 x=np.asarray(x)
 return {'min':float(x.min()),'max':float(x.max()),'std':float(x.std()),
  'fraction_middle':float(np.mean((x>=.05)&(x<=.95))),
  'fraction_nonone':float(np.mean(x<.999)),
  'percentiles':[float(y) for y in np.quantile(x,[0,.1,.25,.5,.75,.9,1])],
  'nondegenerate':bool(x.std()>=STDMIN and np.mean((x>=.05)&(x<=.95))>=FRACTION_MIDDLE_MIN)}


def mix(pred,g):
 i,m,p=(np.asarray(pred[k],dtype='f8') for k in ('instant','old_memory','physical'))
 g=np.asarray(g,dtype='f8')
 if not(np.all(np.isfinite(g)) and (g>=0).all() and (g<=1).all()):raise RuntimeError('GATE_OUT_OF_RANGE')
 if m.shape!=p.shape or m.shape!=i.shape or len(g)!=len(i):raise RuntimeError('MIX_SHAPE_INCONSISTENT')
 return (1-ALPHA)*i+(ALPHA-BETA*g[:,None])*m+BETA*g[:,None]*p


def csv_report(rows,status,src):
 reports=ROOT/'reports';reports.mkdir(parents=True,exist_ok=True)
 save_json(reports/'status.json',status)
 if rows:
  with open(reports/'development_gate_rows.csv','w',newline='',encoding='utf8') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
 out=ROOT/'CIDM_MHC_REAL_A7_P0_NONDEGENERATE_GATE_REPORTS.zip'
 with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=8) as z:
  for p,arc in [(reports/'status.json','reports/status.json'),(src,'source/'+src.name)]:z.write(p,arc)
  if rows:z.write(reports/'development_gate_rows.csv','reports/development_gate_rows.csv')
 if zipfile.ZipFile(out).testzip() is not None:raise RuntimeError('RESULT_CRC_FAIL')
 side=Path(str(out)+'.sha256');side.write_text(sha(out)+'  '+out.name+'\n',encoding='utf8')
 return out,side


def diagnostic_eval(a6,a3,fit,tune,val,fp,tp,vp):
 # Uses the EXACT same fit/tune/val features and ridge method as real A6.
 fmap=fit_map(fp)
 observed={name:{key:gate_stats(gate_values(a,mapping=fmap,kind=name)) for key,a in [('fit',fp),('tune',tp),('validation',vp)]}
           for name in ('constant','legacy','vol_quantile','trend_quantile','hybrid')}
 preds_tune={};preds_val={};truths={}; details={}
 for lead,lab in enumerate(('W3','W4')):
  preds_tune[lab]={};preds_val[lab]={};details[lab]={}
  for arm in ('instant','old_memory','physical'):
   yval,info=a6.ridge_predict(a3,fit,tune,val,fp,tp,vp,lead,arm)
   ytune=a6.ridge_predict_fit_only(a3,fit,tune,fp,tp,lead,arm,info['lambda_selected_2015_only'])
   preds_tune[lab][arm]=ytune;preds_val[lab][arm]=yval;details[lab][arm]=info
  truths[lab]={'tune':np.asarray(tune['truth'][lead]),'validation':np.asarray(val['truth'][lead])}
 old=a6.choose_dev(tune,val,fit,fp,tp,vp,a3)
 # The immutable former coefficient is the one selected in A6, not retuned here.
 if os.getenv('MHC_A7_MODE','official')=='official' and (old['chosen'] is None or old['chosen']['alpha']!=1 or old['chosen']['beta']!=.25):
  raise RuntimeError('ORIGINAL_A6_DEVELOPMENT_SELECTION_CHANGED')
 # Keep all candidate methods for diagnostics, but name one hypothesis in advance, based on train-only rank.
 # The primary is hybrid: robust equal mix of relative volatility and history trend, no seasonal labels.
 hypotheses=('constant','legacy','vol_quantile','trend_quantile','hybrid')
 result={};rows=[]
 for kind in hypotheses:
  r={'feature_stats':observed[kind],'leads':{}}
  for lab in ('W3','W4'):
   measure={}
   for split,pred,phys in [('tune',preds_tune[lab],tp),('validation',preds_val[lab],vp)]:
    y=truths[lab][split]; baseline=mix(pred,np.ones(len(phys)))
    gate=gate_values(phys,fmap,kind);candidate=mix(pred,gate)
    m0=a6.mse_per_issue(baseline,y);m1=a6.mse_per_issue(candidate,y)
    st=a6.dev_stats(m0,m1)
    measure[split]=st
    for j in range(len(gate)):
     rows.append({'candidate':kind,'lead':lab,'partition':split,'index':j,
       'gate':float(gate[j]),'constant_mse':float(m0[j]),'candidate_mse':float(m1[j])})
   r['leads'][lab]=measure
  result[kind]=r
 # eligibility gate only on PRE-REGISTERED primary hybrid. 2016 is reused: exploratory safety diagnosis only.
 q=result['hybrid']
 selection={}
 for lab in ('W3','W4'):
  sel=q['leads'][lab];tune=sel['tune'];v=sel['validation']
  selection[lab]=bool(tune['gain_pct']>=MSE_GATE_MIN and v['gain_pct']>=MSE_GATE_MIN and
        v['wins']>=math.ceil(DEV_WIN_FRACTION*v['n']) and v['tail_cvar_ratio']<=TAIL_MAX and
        q['feature_stats']['tune']['nondegenerate'] and q['feature_stats']['validation']['nondegenerate'])
 return {'fit_only_mapping':fmap,'risk_ratio_distribution':{p:{'vol_ratio':_q(safe_ratio(a)[0]),'trend_ratio':_q(safe_ratio(a)[1])}
     for p,a in [('fit',fp),('tune',tp),('validation',vp)]},'candidate_results':result,
   'old_a6_development_score_parity':old['qualified_to_unseal_2020_2021_once'],
   'PRIMARY_PREDECLARED':'hybrid (same alpha=1, beta=.25, pre-issue features only)',
   'primary_dev_per_lead_gate':selection,'primary_joint_gate':all(selection.values()),
   'note':'2016 development already reused. No fresh scientific generalization or full-training permission.'},rows


def _q(x):return [float(v) for v in np.quantile(x,[0,.1,.5,.9,1])]


def official(a6,pre_audit):
 # Reuse the exact official original reader, priors and source fingerprint, without opening any further year.
 import torch
 if not torch.cuda.is_available():raise RuntimeError('OFFICIAL_NEEDS_COLAB_GPU_TO_REPRODUCE_FROZEN_WEATHER_FEATURES')
 old=a6.preflight(); a5=a6.load_parent();a3,a2,m,paths=a5.parent_modules()
 device='cuda';api,token=m.get_hf(True);m.ensure_private_dataset(api);m.ensure_model_repo(api)
 manifest,cases,_=m.read_embedded_manifest(paths['a0'])
 raw=[];hashes=[]
 for year in m.YEARS:
  for var in m.VARS:
   rec,origin=m.restore_or_download(year,var,manifest,api,token)
   raw.append(rec);hashes.append({'year':year,'var':var,'sha256':rec['sha256'],'source':origin})
 bank=m.FieldBank(raw);split=a2.get_case_split(m,cases)
 from huggingface_hub import hf_hub_download
 snap=hf_hub_download(repo_id=m.MODEL_REPO,repo_type='model',filename=a2.MODEL_HF_PATH,token=token)
 a2.pin(snap,a2.MODEL_SHA)
 ckpt=torch.load(snap,map_location=device,weights_only=True)
 if (ckpt.get('arm'),ckpt.get('seed'),ckpt.get('step'))!=('mhc',17,64):raise RuntimeError('FROZEN_PARENT_WRONG')
 model=m.MHCWeather().to(device).eval();model.load_state_dict(ckpt['model'])
 selected=[split['validation'][int(i)] for i in np.unique(np.linspace(0,len(split['validation'])-1,m.VAL_N,dtype=int))]
 replay=m.score(m.eval_arm(model,bank,selected,'mhc',device))
 for lead in ('W3_MSE','W4_MSE'):
  if abs(replay[lead]-old['original_2016_replay'][lead])>5e-5*max(abs(old['original_2016_replay'][lead]),1e-20):raise RuntimeError('MHC_REPLAY_FAIL_'+lead)
 samples={k:a2.collect_cases(m,model,bank,split[k],device,k) for k in ('fit','tune','validation')}
 phys={k:a3.add_physical_features(a2,model,bank,split[k],device) for k in ('fit','tune','validation')}
 score,rows=diagnostic_eval(a6,a3,samples['fit'],samples['tune'],samples['validation'],phys['fit'],phys['tune'],phys['validation'])
 # Hard replay: original A6 constant-mixture 2016 development MSE must match independently.
 with zipfile.ZipFile(ROOT/'evidence'/'A6_USER_OFFICIAL_REPORT.zip') as z:
  prior=json.loads(z.read('reports/status.json'))
 for k in ('W3','W4'):
  observed=score['candidate_results']['constant']['leads'][k]['validation']['mse']
  ref=prior['development_gate']['development'][k]['vs_old_memory']['mse']
  if not np.isclose(observed,ref,rtol=5e-7,atol=1e-11):
   raise RuntimeError('A6_CONSTANT_DEVELOPMENT_REPLAY_FAIL_'+k+' '+str((observed,ref)))
 out_status={'status':'A7_DEVELOPMENT_NONDEGENERACY_AUDIT_COMPLETE_NO_NEW_HOLDOUT','experiment':'MHC_REAL_A7_GATE',
  'original_a6_result_sha256':A6_ZIP_SHA,'original_a6_forensic':pre_audit,'real_frozen_replay':replay,
  'historical_source_data_hashes':hashes,'sample_counts':{k:len(samples[k]['dates']) for k in samples},
  'result':score,'true_2017_eval_used':False,'2020_2021_labels_reused_for_training':False,
  '2022plus_labels_accessed':False,'new_backbone_updates':0,
  'formal_cidm_training_authorized':False,'external_new_year_open_authorized':False,
  'score_type':'COARSE_LAT_BELT_RESIDUAL_MSE_NOT_AI_WEATHER_QUEST_RPSS'}
 return out_status,rows


def smoke(a6,audit):
 assert audit['gate_all_one'] is True and audit['adaptive_vs_const_2015_max_score_difference']<1e-13
 n=80;rng=np.random.default_rng(5)
 raw=rng.normal(size=(n,48));raw[:,30:42]=np.abs(raw[:,30:42])+.5
 raw[:40,36:42]*=0.25;raw[40:,36:42]*=2.0
 raw[:40,24:30]*=.2;raw[40:,24:30]*=3
 mapping=fit_map(raw)
 a=gate_values(raw,mapping,'hybrid');b=gate_values(raw,mapping,'constant')
 assert np.std(a)>=STDMIN and np.std(b)==0, 'MATH_GATE_DEGENERACY_TEST_FAILED'
 preds={'instant':rng.normal(size=(n,6)),'old_memory':rng.normal(size=(n,6)),'physical':rng.normal(size=(n,6))}
 assert np.array_equal(mix(preds,np.ones(n)),a6.compose(preds,raw,1,.25,False)), 'A6_CONSTANT_PARITY_FAILED'
 v=mix(preds,a);assert np.isfinite(v).all()
 return {'status':'PROGRAM_SMOKE_ONLY_NOT_REAL_WEATHER_A7','a6_evidence':audit,
  'synthetic_gate_std':float(a.std()),'synthetic_gate_nonone_fraction':float(np.mean(a<.999)),
  'constant_action_parity':'BIT_EXACT','real_data_test_executed':False,
  'formal_cidm_training_authorized':False,'2022plus_labels_accessed':False},[]


def sync_results(out,side):
 """Best-effort persist exactly once; local immutable archive is never repacked."""
 result={'archive_sha256':sha(out),'hf':'SKIPPED_DEFAULT','github':'SKIPPED_DEFAULT'}
 if os.getenv('MHC_A7_MODE','official')!='official':
  save_json(ROOT/'sync_receipt.json',result)
  return result
 def secret(k):
  v=os.getenv(k,'')
  if v:return v
  try:
   from google.colab import userdata
   return userdata.get(k) or ''
  except Exception:return ''
 if os.getenv('MHC_A7_SYNC_HF','1')=='1':
  token=secret('HF_TOKEN')
  if not token:result['hf']='SKIPPED_NO_TOKEN'
  else:
   try:
    from huggingface_hub import HfApi,CommitOperationAdd
    api=HfApi(token=token)
    ops=[CommitOperationAdd(path_in_repo='experiments/MHC_REAL_A7_P0/'+q.name,path_or_fileobj=str(q)) for q in (out,side)]
    commit=api.create_commit(repo_id='wuff-mann/AI_Weather_Quest',repo_type='model',operations=ops,
                  commit_message='MHC A7 actual preissue gate nondegeneracy development audit (NOT RPSS)')
    result['hf']='SUCCESS';result['hf_commit_url']=str(getattr(commit,'commit_url',''))
   except Exception as e:result['hf']='FAILED_RETRY_LATER';result['hf_error']=str(e)[:250]
 if os.getenv('MHC_A7_SYNC_GITHUB','1')=='1':
  token=secret('GH_TOKEN') or secret('GITHUB_TOKEN')
  if not token:result['github']='SKIPPED_NO_TOKEN'
  else:
   try:
    import requests,base64
    files=[ROOT/'mhc_real_a7_nondegenerate_gate_audit.py',ROOT/'reports'/'status.json']
    paths=[]
    for q in files:
     rel='research/earth_aiwq/mhc_real_a7/'+q.name
     url='https://api.github.com/repos/Mr-wuff/AI-Weather-Quest/contents/'+rel
     headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json'}
     old=requests.get(url,headers=headers,timeout=25)
     if old.status_code not in (200,404):raise RuntimeError('GH_GET_'+str(old.status_code))
     payload={'message':'Archive official MHC Real A7 causal gate nondegeneracy diagnostic',
       'content':base64.b64encode(q.read_bytes()).decode(),'branch':'main'}
     if old.status_code==200:payload['sha']=old.json()['sha']
     r=requests.put(url,headers=headers,json=payload,timeout=60)
     if r.status_code not in (200,201):raise RuntimeError('GH_PUT_'+str(r.status_code))
     paths.append(rel)
    result['github']={'status':'SUCCESS','paths':paths}
   except Exception as e:result['github']={'status':'FAILED_RETRY_LATER','error':str(e)[:250]}
 save_json(ROOT/'sync_receipt.json',result)
 return result


def main():
 a6_path=ROOT/'mhc_real_a6_causal_bounded_adapter.py'
 try:
  old,rows,audit=preflight();a6=load_py(a6_path,'mhc_a7_pinned_a6')
  a6.HOME=ROOT
  status,detail=(smoke(a6,audit) if os.getenv('MHC_A7_MODE','official')=='smoke' else official(a6,audit))
 except Exception as exc:
  status={'status':'BLOCKED_NO_TRAINING','exception':type(exc).__name__,'error':str(exc),
   'traceback_tail':traceback.format_exc().splitlines()[-18:],
   '2022plus_labels_accessed':False,'formal_cidm_training_authorized':False}
  detail=[]
 out,side=csv_report(detail,status,Path(__file__))
 receipt=sync_results(out,side)
 return {'status':status['status'],'zip':str(out),'sha256':sha(out),'sync':receipt,
  'joint_development_gate':status.get('result',{}).get('primary_joint_gate'),
  'score_type':'COARSE_PROBE_NOT_AI_WEATHER_QUEST_RPSS','model_training_updates':0}

if __name__=='__main__':print(json.dumps(main(),ensure_ascii=False,indent=2),flush=True)
