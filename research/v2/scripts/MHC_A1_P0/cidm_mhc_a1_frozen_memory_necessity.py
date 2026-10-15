"""CIDM-MHC A1/P0 frozen persistent-memory necessity audit.

No weights change. Official uses exact original A0-FIX1 trained heads and frozen B0
backbone. Only current-state slow-memory representation is intervened on; no future
truth or lead labels are used to form actions. All historical inputs hash locked.
"""
from __future__ import annotations
import csv,io,json,math,hashlib,zipfile,traceback,copy,time
from pathlib import Path
import numpy as np
import torch
import cidm_mhc_a0_qualification as mhc
from b2_r6_causal_spatial_controller import PARENT_SHA,TASK_LIST,make_cache_checked
from b2_r1_flux_control import check_zip
from benchmark import Config

OFFICIAL_PRIOR_SHA='afde1a7f7face201108aaeb2b472fbc5a4f7cbd2c84945808a507ef387c1ff38'
MODES=('full','reset','blend_half')
LEADS=(1,3,6,9,12)


def verify_zip(path,expected_sha=OFFICIAL_PRIOR_SHA):
 path=Path(path)
 if mhc.sha(path)!=expected_sha:raise RuntimeError(f'ARCHIVED_A0_FIX1_SHA256_MISMATCH {path}')
 with zipfile.ZipFile(path) as z:
  if z.testzip() is not None:raise RuntimeError('PRIOR_ZIP_CRC_FAIL')
  manifest=json.loads(z.read('FILE_SHA256_MANIFEST.json'))
  for entry in manifest:
   payload=z.read(entry['path'])
   if hashlib.sha256(payload).hexdigest()!=entry['sha256'] or len(payload)!=entry['bytes']:
    raise RuntimeError(f'PRIOR_INNER_SHA_FAIL {entry["path"]}')
  status=json.loads(z.read('report.json'))
  if status.get('status')!='MHC_FIX1_Q0_OR_RESOURCE_GATE_FAIL_NO_FORMAL_TRAINING':
   raise RuntimeError(f'UNEXPECTED_PRIOR_STATUS {status}')
 return {'sha256':expected_sha,'crc':'PASS','entries':len(manifest),'status':status['status']}


def read_original_heads(prior_zip,task,channels,latent_dim,device):
 out={}
 with zipfile.ZipFile(prior_zip) as z:
  for arm in mhc.ARMS:
   n=f'{task}_{arm}_mini_pilot.pt'
   if n not in z.namelist():raise RuntimeError('MISSING_TRAINED_HEAD_'+n)
   buf=io.BytesIO(z.read(n))
   checkpoint=torch.load(buf,map_location='cpu',weights_only=True)
   if checkpoint.get('task')!=task or checkpoint.get('arm')!=arm or checkpoint.get('steps')!=24:raise RuntimeError('WRONG_HEAD_METADATA')
   h=mhc.MarkovClosureHead(channels,latent_dim).to(device).eval()
   h.load_state_dict(checkpoint['state_dict'],strict=True)
   h.requires_grad_(False)
   out[arm]=h
 return out


def _phys_scale(ds, predictions):
 return mhc.physical_scale(ds,predictions)


@torch.no_grad()
def rollout_ablate(core,head,h,memory=1.,action_match=None,reference_actions=None,return_detail=False):
 """Canonical z is authoritative; memory=1 exactly original MHC, 0 instant reset.

 action_match: None or 'per_sample_step' matching factual full-controller RMS (observed
 causally from the same issue-time full rollout, never from future truth).
 """
 if not 0. <= memory <= 1.:raise RuntimeError('INVALID_MEMORY_MIX')
 if action_match not in (None,'per_sample_step'):raise RuntimeError('INVALID_AMPLITUDE_MODE')
 one_d=(h.ndim==4)
 hh=h.unsqueeze(-2) if one_d else h
 b,his,c,H,W=hh.shape
 mm=core.model;coeff=mm.coefficients();skew=mm.skew-mm.skew.T
 z=mm.encode(hh.reshape(b,his*c,H,W));prev=hh[:,-1]
 slow=mhc.lowpass(z);fast=z-slow
 preds=[];actions=[];effects=[];memory_rms=[];cosines=[]
 for step in range(12):
  flux=coeff['flux']*mm.flux(z)
  local=coeff['local']*torch.tanh(mm.local(z))
  skewpart=.035*torch.einsum('ij,bjhw->bihw',skew,z)
  tendency=flux+local+skewpart
  base=z+flux+local+skewpart  # EXACT original arithmetic order
  persistent=slow+mhc.SLOW_RATE*mhc.lowpass(tendency)
  instant=mhc.lowpass(base)
  slow_base=memory*persistent+(1.-memory)*instant if memory not in (0.,1.) else (persistent if memory==1. else instant)
  fast_base=base-slow_base
  gate=head(slow_base,fast_base,prev)
  us=mhc.lowpass(tendency);uf=tendency-us
  norm_s=us/mhc.rms(us).clamp_min(1e-6)
  norm_f=uf/mhc.rms(uf).clamp_min(1e-6)
  slow_raw=mhc.lowpass(gate[:,0:1]*norm_s)
  fast_raw=gate[:,1:2]*norm_f-mhc.lowpass(gate[:,1:2]*norm_f)
  raw=(slow_raw+fast_raw)/math.sqrt(2)
  target=mhc.BUDGET*mhc.rms(base).clamp_min(1e-4)
  corr=target*raw/(mhc.rms(raw)+0.05)
  corr=corr*torch.clamp(target/(mhc.rms(corr)+1e-12),max=1.0)
  if action_match=='per_sample_step':
   if reference_actions is None:raise RuntimeError('MISSING_CAUSAL_REFERENCE_ACTIONS')
   ref=reference_actions[step]
   reference_norm=mhc.rms(ref)
   active=reference_norm>1e-12
   if bool((active & (mhc.rms(corr)<1e-12)).any()):raise RuntimeError('ZERO_ACTION_PREVENTS_RMS_MATCH')
   corr=corr*torch.where(active,reference_norm/(mhc.rms(corr)+1e-24),torch.zeros_like(reference_norm))
   if bool((mhc.rms(corr)>target+1e-9).any()):raise RuntimeError('ACTUAL_LATENT_BUDGET_OVERFLOW')
   if bool(active.any()):
    rel=float(((mhc.rms(corr)-reference_norm).abs()/(reference_norm+1e-20))[active].max())
    if rel>1e-4:raise RuntimeError(f'RMS_MATCH_PARITY_FAIL {rel}')
  if bool((mhc.rms(corr)>target+1e-9).any()):raise RuntimeError('LATENT_RMS_EXPOSURE_EXCEEDED')
  zn=base+corr
  slow=slow_base+memory*mhc.SLOW_RATE*mhc.lowpass(corr)+(1.-memory)*mhc.lowpass(corr) if memory not in (0.,1.) else (slow_base+mhc.SLOW_RATE*mhc.lowpass(corr) if memory==1. else mhc.lowpass(zn))
  # for half memory exactly use slow_base + .625*LP(corr).  Full path is EXACT original.
  fast=zn-slow
  z=zn
  pnew=prev+coeff['out']*torch.tanh(mm.decode(zn))
  preds.append(pnew.squeeze(-2) if one_d else pnew)
  effects.append((mhc.rms(corr)/mhc.rms(base).clamp_min(1e-4)).flatten())
  memory_rms.append((mhc.rms(persistent-instant)/mhc.rms(base).clamp_min(1e-4)).flatten())
  actions.append(corr.detach())
  prev=pnew
 return torch.stack(preds,dim=1),torch.stack(effects,dim=1),actions,torch.stack(memory_rms,dim=1)


@torch.no_grad()
def full_baseline_check(core,heads,cache,task,device,archive_zip,split='val',ids=None):
 """A0-FIX1 officially archived 16 audit window individual scores required."""
 if ids is None:ids=range(16,32)
 import pandas as pd
 ds=mhc.get_data(cache,split)
 with zipfile.ZipFile(archive_zip) as z:
  old=pd.read_csv(io.BytesIO(z.read(task+'_audit_per_window.csv')))
 old=old.set_index(['window','arm','lead'])
 max_rel=0.;max_abs=0.;row_count=0
 for ix in ids:
  x,y=mhc.get_batch(ds,[ix],device)
  for arm in ('baseline',)+mhc.ARMS:
   old_pred=mhc.run_rollout(core,None if arm=='baseline' else heads[arm],x,arm=arm,truncate=True)[0]
   # compare independently implemented full-memory new path to exact original old
   if arm in ('mhc_hierarchical','mhc_no_aux'):
    new_pred=rollout_ablate(core,heads[arm],x,memory=1.)[0]
    diff=float((new_pred-old_pred).abs().max());max_abs=max(max_abs,diff)
    if diff>2e-6:raise RuntimeError(f'FULL_MHC_REPLAY_MISMATCH {task} {arm} window{ix} {diff}')
   sc=mhc.physical_scale(ds,old_pred)
   for lead in LEADS:
    stored=float(old.loc[(ix,arm,lead),'mse_physical'])
    mse=float(((old_pred[:,lead-1]-y[:,lead-1])*sc[:,0]).square().mean())
    rel=abs(mse-stored)/max(1e-12,abs(stored))
    max_rel=max(max_rel,rel);row_count+=1
    if rel>0.005:raise RuntimeError(f'ARCHIVED_MSE_REPLAY_MISMATCH {task} {arm} w{ix} lead{lead} {rel}')
 return {'task':task,'archived_mse_max_rel_diff':max_rel,'full_controller_forecast_max_abs_diff':max_abs,'rows_checked':row_count}


@torch.no_grad()
def evaluate_task(core,heads,cache,task,device,out_dir,archive_zip,smoke=False):
 ds=mhc.get_data(cache,'val')
 ids=list(range(2,4)) if smoke else list(range(16,32))
 records=[];shadow=[]
 for i,ix in enumerate(ids):
  x,y=mhc.get_batch(ds,[ix],device)
  scale=mhc.physical_scale(ds,y)
  old_baseline=mhc.run_rollout(core,None,x,arm='baseline')[0]
  variants={'baseline':(old_baseline,torch.zeros(1,12,device=device))}
  # Four experimental conditions for each trained head: factual, reset, half-memory,
  # and equal-realized-RMS reset/half-memory to distinguish action magnitude from memory information.
  for arm in ('mhc_hierarchical','mhc_no_aux'):
   head=heads[arm]
   factual,acts,reference,_=rollout_ablate(core,head,x,1.)
   variants[arm+'__full']=(factual,acts)
   for key,m,amp in [('reset',0.,None),('blend_half',.5,None),('reset_equal_rms',0.,'per_sample_step'),('blend_equal_rms',.5,'per_sample_step')]:
    pred,actual,_,mem=rollout_ablate(core,head,x,m,amp,reference_actions=reference)
    variants[arm+'__'+key]=(pred,actual)
    for step in range(12):
     shadow.append({'task':task,'window':ix,'head':arm,'variant':key,'lead':step+1,
                    'rel_action':float(actual[:,step].mean()),'original_rel_action':float(acts[:,step].mean()),
                    'persistent_vs_instant_latent_rms':float(mem[:,step].mean())})
  no_train=heads['stateless_closure']
  p,a,_=mhc.run_rollout(core,no_train,x,arm='stateless_closure',truncate=True)
  variants['trained_stateless']=(p,a)
  for name,(pred,act) in variants.items():
   for lead in LEADS:
    mse=float(((pred[:,lead-1]-y[:,lead-1])*scale[:,0]).square().mean())
    records.append({'task':task,'window':ix,'lead':lead,'variant':name,'mse_physical':mse,
                    'rel_action':float(act[:,lead-1].mean())})
  if (i+1)%4==0:
   mhc.write_rows(out_dir/f'{task}_partial_window_metrics.csv',records)
   print('[A1]',task,'completed window',ix,'/',ids[-1],flush=True)
 mhc.write_rows(out_dir/f'{task}_window_metrics.csv',records)
 mhc.write_rows(out_dir/f'{task}_memory_action_diagnostics.csv',shadow)
 return records,shadow


def summary(rows):
 import pandas as pd
 df=pd.DataFrame(rows)
 out={}
 for task,g in df.groupby('task'):
  g=g[g.lead.isin([9,12])]
  m=g.groupby('variant').mse_physical.mean()
  original=m['baseline'];factual=m['mhc_hierarchical__full'];stateless=m['trained_stateless']
  reset=m['mhc_hierarchical__reset'];equal=m['mhc_hierarchical__reset_equal_rms']
  per=g.groupby(['window','variant']).mse_physical.mean().unstack('variant')
  out[task]={'late_mse':{k:float(v) for k,v in m.items()},'mhc_factual_gain_pct':100*(1-factual/original),
       'original_stateless_gain_pct':100*(1-stateless/original),
       'memory_removal_late_deterioration_pct':100*(reset/factual-1),
       'memory_removal_equal_rms_deterioration_pct':100*(equal/factual-1),
       'memory_reset_worse_windows':int((per['mhc_hierarchical__reset']>per['mhc_hierarchical__full']).sum()),
       'memory_reset_equal_rms_worse_windows':int((per['mhc_hierarchical__reset_equal_rms']>per['mhc_hierarchical__full']).sum()),
       'windows':len(per)}
 return out


def main(parent,work,out,prior_report,smoke=False,device=None):
 work=Path(work);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 evidence=verify_zip(prior_report)
 mhc.dump(out/'historical_evidence_pin.json',evidence)
 device='cpu' if smoke else (device or ('cuda' if torch.cuda.is_available() else 'cpu'))
 allrows=[];allshadow=[];cases=[];parities=[]
 if smoke:
  for task,core,cache in mhc.smoke_case():
   heads,_=mhc.make_arms(1,int(core.model.skew.shape[0]),'cpu')
   # Only program smoke: force nonzero output, NOT official trained control heads.
   for arm in heads:
    with torch.no_grad():heads[arm].net[-1].bias.copy_(torch.tensor([0.07,-0.03]))
   cases.append((task,core,cache,heads))
 else:
  if parent is None or mhc.sha(parent)!=PARENT_SHA:raise RuntimeError('MISSING_OR_WRONG_PARENT')
  with zipfile.ZipFile(parent) as z:
   manifest,_=check_zip(z);cfg=Config(**manifest['config'])
   for task in TASK_LIST:
    if task=='burgers_1d':continue # no valid A0-FIX1 Q1 checkpoint, no invented results
    core,cache,replay=make_cache_checked(task,z,cfg,work/'official_runtime','cpu')
    mhc.dump(out/f'{task}_parent_replay.json',replay)
    ch=int(cache['splits']['train']['h'].shape[2]);dim=int(core.model.skew.shape[0]);
    heads=read_original_heads(prior_report,task,ch,dim,device)
    cases.append((task,core,cache,heads))
 for task,core,cache,heads in cases:
  core=core.to(device).eval().requires_grad_(False)
  heads={k:v.to(device).eval().requires_grad_(False) for k,v in heads.items()}
  if not smoke:
   parity=full_baseline_check(core,heads,cache,task,device,prior_report)
   parities.append(parity);mhc.dump(out/f'{task}_strict_prior_parity.json',parity)
  else:
   x,_=mhc.get_batch(mhc.get_data(cache,'train'),[0],device)
   original=mhc.run_rollout(core,heads['mhc_hierarchical'],x,arm='mhc_hierarchical')[0]
   new=rollout_ablate(core,heads['mhc_hierarchical'],x,1.)[0]
   err=float((new-original).abs().max())
   if err>2e-6:raise RuntimeError('SMOKE_PARITY_FAIL '+str(err))
   parities.append({'task':task,'forward_diff':err})
  rows,shadow=evaluate_task(core,heads,cache,task,device,out,prior_report,smoke=smoke)
  allrows.extend(rows);allshadow.extend(shadow)
  mhc.dump(out/'progress_summary.json',{'task_completed':[p['task'] for p in parities],'task_pending':[t for t in ('reacdiff_2d','swe_2d') if t not in [p['task'] for p in parities]],'train_allowed':False})
 summ=summary(allrows)
 mhc.dump(out/'summary.json',summ)
 mhc.write_rows(out/'all_window_metrics.csv',allrows)
 mhc.write_rows(out/'all_memory_action_diagnostics.csv',allshadow)
 # Analysis gate is intentionally diagnostic-only: no selection of weights or treatment using audit windows.
 gate={'status':'PROGRAM_ONLY_SMOKE_NO_SCIENCE' if smoke else 'FROZEN_MEMORY_NECESSITY_COMPLETED_NO_TRAINING',
       'train_allowed':False,'strict_historical_replays':parities,'results':summ,'updates':0,
       'contract':'MHC trained-head fixed; intervention acts only on slow/fast readout, not original z update; equal RMS matched using factual causal action only',
       'limits':['seed17 only','all windows are reused original parent validation not unseen outer test',
                 'no independent physical Markov sufficiency proof','frozen necessity not new trainable architectural advantage',
                 'no burgers Q1 head because Burgers failed prior A0-FIX1 Q0']}
 mhc.dump(out/'memory_gate.json',gate)
 mhc.dump(out/'report.json',gate)
 return gate
