"""CIDM-v2 B2-R3: frozen-backbone information-sufficiency pilot.
Only frozen original models and SMALL matched trainable probes. This is NOT full model retraining.
Three science outcomes: latent-specific readout gain, observable-only calibration gain, no reliable gain.
"""
from __future__ import annotations
import os,io,math,json,hashlib,zipfile,random,copy,csv,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from benchmark import Config,Windows,fetch_official,make_cache,load_data,assess as canonical_assess,UnifiedModel,make_synthetic_cache
from b2_r1_flux_control import get_core,check_zip,assess as fixed_assess

PARENT_SHA='69552b29e169ec22ee85d9c1b1a678fdb188f6f25dc18f33b927b88f4dddffd4'
TASKS=('burgers_1d','reacdiff_2d','swe_2d')
LEADS=(1,3,6,9,12)
SEED=17

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for t in iter(lambda:f.read(1<<20),b''):h.update(t)
 return h.hexdigest()

def js(p,x):
 Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding='utf-8')

def rows_csv(p,rows):
 if not rows:return
 with open(p,'w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def normalize_sample(x):
 scale=x.detach().square().mean(dim=(1,2,3),keepdim=True).sqrt().clamp_min(.1)
 return torch.tanh(x/scale)

class ReadoutProbe(nn.Module):
 """Identical trainable parameters and dimensions for OBS and LATENT; source varies only in frozen features."""
 def __init__(self,channels,width=16,hidden=24):
  super().__init__()
  self.head=nn.Sequential(nn.Conv2d(width+3*channels+1,hidden,3,padding=1),nn.SiLU(),
                          nn.Conv2d(hidden,channels,1))
  nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
 def forward(self,common,projection):
  correction=.25*torch.tanh(self.head(torch.cat([common,projection],dim=1)))
  return correction

class FixedProjection:
 def __init__(self,d,ch,width=16,seed=20261014,device='cpu'):
  g=torch.Generator(device='cpu').manual_seed(seed)
  self.p_lat=(torch.randn(width,d,1,1,generator=g)/math.sqrt(d)).to(device)
  self.p_obs=(torch.randn(width,3*ch,1,1,generator=g)/math.sqrt(3*ch)).to(device)
 def feats(self,z,common,channels):
  raw=common[:,:3*channels]
  lat=F.conv2d(normalize_sample(z),self.p_lat)
  obs=F.conv2d(normalize_sample(raw),self.p_obs)
  return lat.detach(),obs.detach()

@torch.no_grad()
def rollout(core,h,leads=LEADS):
 """Exact original R5 dynamics, no correction feedback or future truth in prediction path."""
 mm=core.model
 raised=h.ndim==4
 hh=h.unsqueeze(-2) if raised else h
 B,L,C,H,W=hh.shape
 assert L==mm.hist==4
 coef=mm.coefficients();z=mm.encode(hh.reshape(B,L*C,H,W))
 prev=hh[:,-1];initial=prev; skew=mm.skew-mm.skew.T
 states=[]
 for t in range(1,max(leads)+1):
  z=z+coef['flux']*mm.flux(z)+coef['local']*torch.tanh(mm.local(z))+.035*torch.einsum('ij,bjhw->bihw',skew,z)
  inc=coef['out']*torch.tanh(mm.decode(z));x=prev+inc
  if t in leads:states.append((t,z.detach(),x.detach(),inc.detach(),initial.detach()))
  prev=x
 return states

def common_features(x,inc,initial,lead):
 # Values are observational, causally available under frozen baseline rollout.
 B,C,H,W=x.shape
 k=x.new_full((B,1,H,W),float(lead)/12.0)
 return torch.cat([initial.clamp(-5,5),x.clamp(-5,5),inc.clamp(-5,5),k],dim=1).detach()

def make_features(states,channels,projection):
 for lead,z,x,inc,initial in states:
  common=common_features(x,inc,initial,lead)
  latent,observable=projection.feats(z,common,channels)
  yield lead,x,common,latent,observable

@torch.no_grad()
def original_parity(core,h):
 states=rollout(core,h,tuple(range(1,13)))
 manual=torch.stack([s[2] for s in states],dim=1)
 model=core(h,12,mode='off')
 if h.ndim==4:manual=manual.squeeze(-2)
 delta=(manual-model).abs().max().item()
 if delta>2e-6:raise RuntimeError('FROZEN_FORWARD_PARITY_MISMATCH:'+str(delta))
 return delta

@torch.no_grad()
def eval_heads(core,heads,projection,ds,indices,device,batch,leads=LEADS):
 core.eval();[v.eval() for v in heads.values()]
 C=ds.h.shape[2];ssd=ds.sd.to(device).reshape(1,-1,1,1)**2
 scores={s:{t:[0.0,0] for t in leads} for s in ['baseline',*heads]}
 for i in range(0,len(indices),batch):
  samples=[ds[int(j)] for j in indices[i:i+batch]]
  h=torch.stack([v[0] for v in samples]).to(device);y=torch.stack([v[1] for v in samples]).to(device)
  if h.ndim==4:y=y.unsqueeze(-2)
  states=rollout(core,h,leads)
  for lead,x,common,lat,obs in make_features(states,C,projection):
   target=y[:,lead-1]
   preds={'baseline':x,'observable':x+heads['observable'](common,obs),
           'latent':x+heads['latent'](common,lat)}
   for name,pred in preds.items():
    diff=(pred-target).square()*ssd
    scores[name][lead][0]+=diff.sum().item();scores[name][lead][1]+=diff.numel()
 result={s:{str(t):scores[s][t][0]/max(scores[s][t][1],1) for t in leads} for s in scores}
 for name in result:
  result[name]['late_mse']=(sum(scores[name][t][0] for t in [9,12])/max(sum(scores[name][t][1] for t in [9,12]),1))
  result[name]['all_mse']=(sum(scores[name][t][0] for t in leads)/max(sum(scores[name][t][1] for t in leads),1))
 return result

def index_plan(n):
 # fixed disjoint halves: val-selection and val-audit. Never test tuning or train on validation.
 if n<12:raise RuntimeError('INSUFFICIENT_VAL_WINDOWS: require >=12 for independent audit split')
 ids=np.arange(n);cut=n//2
 return ids[:cut].tolist(),ids[cut:].tolist()

def q0_check(archive,external_diag):
 """Read-only integrity and predeclared indication from actual FIX6 results. Q0 does NOT select a fitted model."""
 if sha(archive)!=PARENT_SHA:raise RuntimeError('B0_PARENT_SHA256_MISMATCH')
 with zipfile.ZipFile(archive) as z:
  m,c=check_zip(z)
  if m['gate']['completed_runs']!=18:raise RuntimeError('INCOMPLETE_PARENT_RUNS')
 import pandas as pd
 data=pd.read_csv(external_diag)
 assert len(data)==45 and len(data[['task','seed','lead']].drop_duplicates())==45
 assert set(data.task)==set(TASKS) and set(data.seed)=={17,41,73} and set(data.lead)==set(LEADS)
 if data[['pred_increment_cos_truth','pred_increment_power_ratio','self_reencode_increment_gap_rel']].isna().any().any():
  raise RuntimeError('FIX6_DIAGNOSTIC_MISSING_OR_NONFINITE')
 l=data[data.lead==12].groupby('task').mean(numeric_only=True)
 evidence={'burgers_bad_direction':bool(l.loc['burgers_1d','pred_increment_cos_truth']<.2),
   'swe_underresponse':bool(l.loc['swe_2d','pred_increment_power_ratio']<.6),
   'rd_direction_better_than_burgers':bool(l.loc['reacdiff_2d','pred_increment_cos_truth']>l.loc['burgers_1d','pred_increment_cos_truth']+.2)}
 return {'gate':'PILOT_ALLOWED' if all(evidence.values()) else 'BLOCKED_Q0_MECHANISM_NOT_REPRODUCED',
         'evidence':evidence,'three_task_lead12':l[['pred_increment_cos_truth','pred_increment_power_ratio','self_reencode_increment_gap_rel']].round(6).to_dict(orient='index'),
         'source_sha256':sha(external_diag),'parent_sha256':PARENT_SHA}

def pilot_one(task,archive,cfg,root,output,updates=100,device='cpu',smoke=False):
 """Small matched-probe TRAINING ONLY, frozen backbone. Each has identical initialization and optimizer budget."""
 out=Path(output)/task;out.mkdir(parents=True,exist_ok=True)
 if smoke:
  cache=make_synthetic_cache(cfg)
  # Synthetic 2D or lifted 1D supported by same logic, NO official result.
  core=UnifiedModel('V2_Ported_R5',1,8,12,4).to(device).eval()
  batch=2
 else:
  with zipfile.ZipFile(archive) as zp:
   m,contract=check_zip(zp)
   path=fetch_official(Path(root)/'official_data',task)
   cache=make_cache(task,path,cfg,Path(root)/'window_cache')
   ref=next(v for v in contract['tasks'] if v['task']==task)
   assert cache['signature']['source_md5']==ref['source_md5']
   assert cache['ids']==ref['ids']
   assert all(len(cache['splits'][s]['h'])==ref['windows'][s] for s in ('train','val','test'))
   core,weight_sha=get_core(zp,cache,task,SEED,device)
   bs=cfg.batch_1d if cache['splits']['test']['h'].ndim==4 else cfg.batch_2d
   ref_score=float(json.loads(zp.read(f'{task}__V2_Ported_R5__seed{SEED}.json'))['test']['model_12'])
   replay=fixed_assess(core,'baseline',cache,'test',device,batch=bs)['rmse12']
   if abs(replay-ref_score)>max(1e-5,ref_score*.001):raise RuntimeError(f'BASELINE_REPLAY_FAIL_CLOSED {task} {replay} != {ref_score}')
   js(out/'parent_replay.json',{'task':task,'seed':SEED,'checkpoint_sha256':weight_sha,'archive_test_rmse12':ref_score,'replay_test_rmse12':replay,'pass':True})
   print('[BASELINE REPLAY PASS]',task,SEED,ref_score,flush=True)
  batch=cfg.batch_1d if cache['splits']['train']['h'].ndim==4 else cfg.batch_2d
 core.requires_grad_(False);core.eval()
 train=Windows(cache['splits']['train']['h'],cache['splits']['train']['y'],cache['mean'],cache['std'])
 val=Windows(cache['splits']['val']['h'],cache['splits']['val']['y'],cache['mean'],cache['std'])
 if smoke:sel_ids,audit_ids=[0,1],[2,3]
 else:sel_ids,audit_ids=index_plan(len(val))
 C=int(cache['splits']['train']['h'].shape[2]);d=core.model.flux.net[-1].out_channels
 proj=FixedProjection(d,C,seed=20261014,device=device)
 torch.manual_seed(20261014);head_obs=ReadoutProbe(C).to(device)
 torch.manual_seed(20261014);head_lat=ReadoutProbe(C).to(device)
 heads={'observable':head_obs,'latent':head_lat}
 assert sum(p.numel() for p in head_obs.parameters())==sum(p.numel() for p in head_lat.parameters())
 opts={s:torch.optim.AdamW(heads[s].parameters(),lr=8e-4,weight_decay=1e-4) for s in heads}
 g=torch.Generator(device='cpu').manual_seed(20261014+SEED)
 rows=[];best={'observable':(float('inf'),None,0),'latent':(float('inf'),None,0)}
 with torch.no_grad():
  smoke_x=torch.stack([train[0][0],train[1][0]]).to(device)
  parity=original_parity(core,smoke_x)
 print('[TRAIN PRECHECK]',task,'orig parity',parity,'readout params',sum(p.numel() for p in head_obs.parameters()),flush=True)
 checkpoints=[0,updates] if smoke else sorted(set([0,*range(20,updates+1,20),updates]))
 def evaluate_at(step):
  v=eval_heads(core,heads,proj,val,sel_ids,device,batch)
  for s in heads:
   mse=v[s]['late_mse']
   if mse<best[s][0]:best[s]=(mse,copy.deepcopy(heads[s].state_dict()),step)
  r={'task':task,'step':step,'baseline_late':v['baseline']['late_mse'],
     'observable_late':v['observable']['late_mse'],'latent_late':v['latent']['late_mse'],
     'latent_vs_observable_ratio':v['latent']['late_mse']/max(v['observable']['late_mse'],1e-20)}
  rows.append(r);rows_csv(out/'pilot_curve.csv',rows)
  print('[PILOT]',task,'step',step,'baseline',round(r['baseline_late'],8),'obs',round(r['observable_late'],8),'latent',round(r['latent_late'],8),flush=True)
 evaluate_at(0)
 for step in range(1,updates+1):
  ids=torch.randint(0,len(train),(batch,),generator=g).tolist()
  samples=[train[i] for i in ids]
  h=torch.stack([v[0] for v in samples]).to(device);y=torch.stack([v[1] for v in samples]).to(device)
  if h.ndim==4:y=y.unsqueeze(-2)
  with torch.no_grad():states=rollout(core,h)
  for o in opts.values():o.zero_grad(set_to_none=True)
  loss={s:0.0 for s in heads}
  for lead,x,common,lat,obs in make_features(states,C,proj):
   target=y[:,lead-1]
   for s,feature in (('observable',obs),('latent',lat)):
    estimate=x+heads[s](common,feature)
    loss[s]=loss[s]+(estimate-target).square().mean()/len(LEADS)
  for s in heads:
   if not torch.isfinite(loss[s]):raise RuntimeError('NONFINITE_TRAINING_LOSS:'+task+':'+s)
   loss[s].backward();torch.nn.utils.clip_grad_norm_(heads[s].parameters(),1.)
   opts[s].step()
  if step in checkpoints:
   evaluate_at(step)
   js(out/'progress.json',{'task':task,'updates_completed':step,'best':{k:{'selection_val_mse':v[0],'best_step':v[2]} for k,v in best.items()},'time':time.time()})
 # Independent audit split: never used for step selection or learning.
 for s in heads:
  assert best[s][1] is not None
  heads[s].load_state_dict(best[s][1]);heads[s].eval()
 audit=eval_heads(core,heads,proj,val,audit_ids,device,batch)
 # Readout trained from train only, step selected from first validation half; second half untouched.
 summary={'task':task,'seed':SEED,'updates':updates,'core_frozen':True,'matched_probe_params':sum(p.numel() for p in heads['latent'].parameters()),
   'selection_windows':len(sel_ids),'audit_windows':len(audit_ids),'best_step':{s:best[s][2] for s in best},
   'audit':audit,'late_ratios':{s:audit[s]['late_mse']/max(audit['baseline']['late_mse'],1e-20) for s in heads},
   'latent_vs_observable_late_ratio':audit['latent']['late_mse']/max(audit['observable']['late_mse'],1e-20),
   'qualifier':'SMOKE_NO_SCIENCE' if smoke else 'SINGLE_SEED_READOUT_PILOT_ONLY'}
 js(out/'pilot_summary.json',summary)
 for s in heads:
  torch.save({'head':heads[s].state_dict(),'task':task,'seed':SEED,'probe_type':s,'best_step':best[s][2],
              'original_backbone_frozen':True,'selection_mse':best[s][0],
              'fixed_projection_seed':20261014},out/f'{s}_best_probe.pt')
 return summary

def run(parent,diagnostic_csv,root,output,updates=100,smoke=False):
 out=Path(output);out.mkdir(parents=True,exist_ok=True)
 if smoke:
  cfg=Config(profile='smoke',history=4,train_horizon=6,eval_horizon=12)
  results=[pilot_one('smoke_2d',None,cfg,root,out,updates=2,smoke=True)]
  status='PROGRAM_ONLY_SMOKE_NO_SCIENTIFIC_CLAIM'
  q0={'gate':'SMOKE'}
 else:
  q0=q0_check(parent,diagnostic_csv)
  js(out/'Q0_frozen_gate.json',q0)
  if q0['gate']!='PILOT_ALLOWED':
   status='Q0_FAIL_NO_PROBE_TRAINING'; results=[]
  else:
   with zipfile.ZipFile(parent) as zp:
    manifest,_=check_zip(zp)
    cfg=Config(**manifest['config'])
   results=[]
   for task in TASKS:
    results.append(pilot_one(task,parent,cfg,root,out,updates=updates,device='cuda' if torch.cuda.is_available() else 'cpu'))
   status='SINGLE_SEED_PILOT_COMPLETED_NOT_FULL_TRAINING'
 science={'experiment':'CIDM-v2 B2-R3/P1 FROZEN BACKBONE READOUT SUFFICIENCY PILOT',
          'status':status,'parent_hash':PARENT_SHA,'trained_backbone_steps':0,
          'pilot_probe_updates_per_head_per_task':updates if not smoke and results else 0,
          'task_seed_count':len(results),'results':results,'q0':q0,
          'qualification':'NO_CIDM_V2_FULL_MODEL_QUALIFICATION',
          'evidence_limits':['Readout probes are trainable small models; results are not frozen counterfactuals.',
          'Seed17 only; no multi-seed confirmation or formal v1 comparison.',
          'Validation halves used separately for checkpoint selection and independent pilot audit.',
          'Decoder-correction probes do not feed corrections back to backbone, so no closed-loop architecture inference.',
          'Test labels are not used for training/selection; test split only strict baseline model replay.']}
 if not smoke and len(results)==3:
  rr={x['task']:x for x in results}
  def improve(task):
   x=rr[task]
   return x['late_ratios']['latent']<=.97 and x['latent_vs_observable_late_ratio']<=.98
  gate=improve('burgers_1d') and improve('swe_2d') and rr['reacdiff_2d']['late_ratios']['latent']<=1.02
  science['pilot_gate']='CANDIDATE_FOR_THREE_SEED_REPLICATION' if gate else 'PILOT_NO_UNIFIED_LATENT_SPECIFIC_GAIN'
  science['pilot_gate_policy']={'burgers_and_swe_latent_gain_vs_base_min':.03,
                               'burgers_and_swe_latent_gain_vs_matched_obs_min':.02,
                               'rd_max_regression_vs_base':.02,'pilot_is_not_architecture_gate':True}
 js(out/'report.json',science)
 return science
