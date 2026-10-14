"""CIDM-v2 B2-R4/P0: frozen decoder tangent/latent-operator controllability audit.

NO training; one-step label-informed ORACLE interventions are RETROSPECTIVE diagnostic
reference corrections and MUST NEVER be marketed as deployable forecasts. Official test
split is used ONLY to replay unchanged historical baseline checkpoints.
"""
from __future__ import annotations
import os, math, io, json, csv, zipfile, hashlib, random, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from benchmark import Config, Windows, fetch_official, make_cache, make_synthetic_cache, UnifiedModel
from b2_r1_flux_control import check_zip, get_core, assess as baseline_assess
from b2_r3_readout_probe import original_parity

PARENT_SHA='69552b29e169ec22ee85d9c1b1a678fdb188f6f25dc18f33b927b88f4dddffd4'
TASKS=('burgers_1d','reacdiff_2d','swe_2d')
SEEDS=(17,41,73)
LEADS=(1,3,6,9,12)
WINDOWS_PER_SEED=8
BUDGETS=(.01,.05)


def file_sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for x in iter(lambda:f.read(1<<21),b''):h.update(x)
 return h.hexdigest()


def write_json(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


def write_csv(path,rows):
 if not rows:return
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 with open(p,'w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def sample_rms(x):
 return x.float().square().flatten(1).mean(1).clamp_min(1e-30).sqrt().view(-1,1,1,1)


def flatten_cos(a,b):
 af=a.flatten(1);bf=b.flatten(1)
 return (af*bf).sum(1)/(af.norm(dim=1)*bf.norm(dim=1)).clamp_min(1e-20)


def scalar_list(t):return [float(x) for x in t.detach().cpu().flatten()]


def span_project(grad,vectors):
 # Numerical orthonormalization on individual samples (nonzero vectors only).
 B=grad.shape[0];g=grad.flatten(1);result=torch.zeros_like(g);Q=[]
 for vv in vectors:
  v=vv.flatten(1).detach().float()
  for q in Q: v=v-(v*q).sum(1,keepdim=True)*q
  vnorm=v.norm(dim=1,keepdim=True)
  valid=vnorm>1e-9
  q=torch.where(valid,v/vnorm.clamp_min(1e-9),torch.zeros_like(v))
  Q.append(q)
  result=result+(g*q).sum(1,keepdim=True)*q
 frac=(result.square().sum(1)/g.square().sum(1).clamp_min(1e-30)).clamp(min=0,max=1)
 return result.reshape_as(grad),frac


@torch.no_grad()
def exact_rollout_states(core,history,leads=LEADS):
 mm=core.model
 h=history.unsqueeze(-2) if history.ndim==4 else history
 B,L,C,H,W=h.shape
 z=mm.encode(h.reshape(B,L*C,H,W));prev=h[:,-1];coeff=mm.coefficients();skew=mm.skew-mm.skew.T
 states={}
 for step in range(1,max(leads)+1):
  flux=coeff['flux']*mm.flux(z)
  local=coeff['local']*torch.tanh(mm.local(z))
  skew_part=.035*torch.einsum('ij,bjhw->bihw',skew,z)
  z=z+flux+local+skew_part
  logit=mm.decode(z)
  increment=coeff['out']*torch.tanh(logit)
  current=prev+increment
  if step in leads:
   states[step]={'z':z.detach(),'prev':prev.detach(),'pred':current.detach(),
                 'flux':flux.detach(),'local':local.detach(),'skew':skew_part.detach(),
                 'logit':logit.detach()}
  prev=current
 return states


def oracle_batch(core,states,targets,sd,lead,seed=0):
 """Future truth is ONLY used to calculate diagnostic gradient and upper bounds.
 This is NOT an issue-time deployable correction and never trains parameters."""
 mm=core.model;coeff=mm.coefficients();z=states['z'].detach().float();pred=states['pred'].detach().float();prev=states['prev'].detach().float()
 truth=targets[:,lead-1].detach().float()
 sd=sd.float().to(z.device).reshape(1,-1,1,1)
 with torch.enable_grad():
  zvar=z.clone().requires_grad_(True)
  predvar=prev+coeff['out']*torch.tanh(mm.decode(zvar))
  reconstruction=float((predvar.detach()-pred).abs().max().item())
  if reconstruction>2e-5:raise RuntimeError('EXACT_READOUT_MISMATCH:'+str(reconstruction))
  loss_per=((predvar-truth)*sd).square().flatten(1).mean(1)
  grad=torch.autograd.grad(loss_per.sum(),zvar,create_graph=False)[0].detach()
  if not torch.isfinite(grad).all():raise RuntimeError('ORACLE_NONFINITE_GRAD')
 base=loss_per.detach().float().clamp_min(1e-22)
 span,frac=span_project(grad,[states[k] for k in ('flux','local','skew')])
 gnorm=sample_rms(grad);snorm=sample_rms(span)
 normal=grad/gnorm.clamp_min(1e-18)
 span_normal=span/snorm.clamp_min(1e-18)
 # Fixed seeded reference; no information from future labels in random control.
 generator=torch.Generator(device='cpu').manual_seed(int(seed))
 noise=torch.randn(z.shape,generator=generator,dtype=z.dtype,device='cpu').to(z.device)
 noise_normal=noise/sample_rms(noise).clamp_min(1e-18)
 zscale=sample_rms(z).clamp_min(1.0)
 out={'tangent_oracle_future_truth_used':1,
      'base_mse_physical':scalar_list(base),
      'decoder_abslogit_gt2':scalar_list((states['logit'].abs()>2).float().flatten(1).mean(1)),
      'grad_rms':scalar_list(gnorm),
      'operator_grad_projection_fraction':scalar_list(frac),
      'cos_descent_flux':scalar_list(flatten_cos(-grad,states['flux'])),
      'cos_descent_local':scalar_list(flatten_cos(-grad,states['local'])),
      'cos_descent_skew':scalar_list(flatten_cos(-grad,states['skew'])),
      'latent_rms':scalar_list(sample_rms(z)),
      'max_readout_reconstruction_error':reconstruction}
 for budget in BUDGETS:
  for name,move in [('full_gradient_oracle',normal),('operator_span_oracle',span_normal),('random_control',noise_normal)]:
   shifted=z-budget*zscale*move if name!='random_control' else z+budget*zscale*move
   with torch.no_grad():
    estimate=prev+coeff['out']*torch.tanh(mm.decode(shifted))
    after=((estimate-truth)*sd).square().flatten(1).mean(1)
   out[f'{name}_mse_ratio_at_{budget:.2f}']=scalar_list(after/base)
 return out


def calc(task,seed,core,cache,device,indices,smoke=False):
 core.eval().requires_grad_(False)
 val=Windows(cache['splits']['val']['h'],cache['splits']['val']['y'],cache['mean'],cache['std'])
 sd=val.sd.to(device)
 rows=[];batch=2 if not task.endswith('_1d') else 4
 for start in range(0,len(indices),batch):
  group=indices[start:start+batch]
  samples=[val[i] for i in group]
  h=torch.stack([x[0] for x in samples],0).to(device)
  y=torch.stack([x[1] for x in samples],0).to(device)
  if y.ndim==4:y=y.unsqueeze(-2)
  states=exact_rollout_states(core,h)
  for lead,s in states.items():
   # Predictions and states are read-only; ground truth only in oracle_batch (diagnostics).
   diag=oracle_batch(core,s,y,sd,lead,seed=seed*100003+lead*1009+start)
   for ii,idx in enumerate(group):
    row={'task':task,'seed':seed,'window_index':idx,'lead':lead,
         'smoke_only':int(smoke)}
    for key,v in diag.items():
     row[key]=v[ii] if isinstance(v,list) else v
    rows.append(row)
 return rows


def aggregations(rows):
 by={}
 for r in rows:by.setdefault((r['task'],r['seed'],r['lead']),[]).append(r)
 result=[]
 for (task,seed,lead),xx in sorted(by.items()):
  rr={'task':task,'seed':seed,'lead':lead,'windows':len(xx)}
  for k in xx[0]:
   if k in rr or k in ('window_index','smoke_only','task','seed','lead','tangent_oracle_future_truth_used'):continue
   vv=np.asarray([v[k] for v in xx],dtype=float)
   if not np.isfinite(vv).all():raise RuntimeError('NONFINITE_DIAGNOSTIC:'+k)
   rr[k]=float(vv.mean())
  result.append(rr)
 return result


def interpret(rows):
 """Predeclared descriptive labels, NOT permission for formal training."""
 a=aggregations(rows);out={}
 for task in TASKS:
  for lead in (1,12):
   selected=[r for r in a if r['task']==task and r['lead']==lead]
   if not selected:continue
   avg=lambda k:sum(x[k] for x in selected)/len(selected)
   out[f'{task}_lead{lead}']={
    'grad_oracle_ratio_1pct':avg('full_gradient_oracle_mse_ratio_at_0.01'),
    'grad_oracle_ratio_5pct':avg('full_gradient_oracle_mse_ratio_at_0.05'),
    'operator_span_ratio_5pct':avg('operator_span_oracle_mse_ratio_at_0.05'),
    'random_ratio_5pct':avg('random_control_mse_ratio_at_0.05'),
    'operator_gradient_capture_fraction':avg('operator_grad_projection_fraction'),
    'decoder_saturation_fraction':avg('decoder_abslogit_gt2')}
 # Diagnostics never grants model training; mechanistic categories descriptive only.
 return out


def run(parent,root,out,smoke=False):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 if smoke:
  cfg=Config(profile='smoke',history=4,train_horizon=6,eval_horizon=12)
  cache=make_synthetic_cache(cfg)
  # 1D and 2D synthetic each, run through exact original ported backbone
  rows=[]
  for task,one_dim in [('smoke_1d',True),('smoke_2d',False)]:
   src=cache if not one_dim else {
    **cache,'splits':{k:{'h':v['h'].mean(dim=-2) if v['h'].ndim==5 else v['h'],
                         'y':v['y'].mean(dim=-2) if v['y'].ndim==5 else v['y']}
                          for k,v in cache['splits'].items()}}
   ch=int(src['splits']['train']['h'].shape[2]);core=UnifiedModel('V2_Ported_R5',ch,8,12,4).eval()
   # For a 2D grid synthetic smoke, must keep channel and width matches expected.
   r=calc(task,17,core,src,'cpu',[0,1],smoke=True)
   rows.extend(r)
  write_csv(out/'sample_oracle_diagnostics.csv',rows)
  write_csv(out/'lead_seed_summary.csv',aggregations(rows))
  result={'experiment':'B2-R4/P0','status':'PROGRAM_ONLY_SMOKE','sample_rows':len(rows),
          'warning':'Synthetic smoke is NOT official PDEBench evidence; label-conditioned oracle is NOT deployable.'}
 else:
  if file_sha(parent)!=PARENT_SHA:raise RuntimeError('PARENT_SHA_MISMATCH')
  allrows=[];replays=[];cfg=None
  with zipfile.ZipFile(parent) as zp:
   manifest,contract=check_zip(zp)
   cfg=Config(**manifest['config'])
   for task in TASKS:
    datafile=fetch_official(Path(root)/'official_data',task)
    cache=make_cache(task,datafile,cfg,Path(root)/'window_cache')
    contract_row=next(v for v in contract['tasks'] if v['task']==task)
    if cache['signature']['source_md5']!=contract_row['source_md5'] or cache['ids']!=contract_row['ids']:
     raise RuntimeError('PDEBENCH_DATA_SIGNATURE_OR_SPLIT_MISMATCH:'+task)
    if any(len(cache['splits'][s]['h'])!=contract_row['windows'][s] for s in ('train','val','test')):
     raise RuntimeError('WINDOW_COUNT_MISMATCH:'+task)
    for seed in SEEDS:
     device='cuda' if torch.cuda.is_available() else 'cpu'
     core,checkpoint_hash=get_core(zp,cache,task,seed,device)
     batch=cfg.batch_1d if task.endswith('_1d') else cfg.batch_2d
     original=float(json.loads(zp.read(f'{task}__V2_Ported_R5__seed{seed}.json'))['test']['model_12'])
     test=baseline_assess(core,'baseline',cache,'test',device,batch=batch)['rmse12']
     if abs(test-original)>max(1e-5,original*.001):
      raise RuntimeError(f'FAIL_CLOSED_BASELINE_REPLAY {task} {seed}: {test} vs {original}')
     tset=Windows(cache['splits']['val']['h'],cache['splits']['val']['y'],cache['mean'],cache['std'])
     h=torch.stack([tset[i][0] for i in range(min(2,len(tset)))],0).to(device)
     parity=original_parity(core,h)
     if parity>2e-6:raise RuntimeError('OFFICIAL_PARITY_MISMATCH')
     replays.append({'task':task,'seed':seed,'checkpoint_sha256':checkpoint_hash,
         'expected_test_rmse12':original,'replay_test_rmse12':test,'max_forward_difference':parity})
     indices=list(range(min(WINDOWS_PER_SEED,len(tset))))
     if len(indices)!=WINDOWS_PER_SEED:raise RuntimeError('VAL_WINDOWS_TOO_FEW')
     part=calc(task,seed,core,cache,device,indices,smoke=False)
     allrows.extend(part)
     write_csv(out/'sample_oracle_diagnostics.csv',allrows)
     write_csv(out/'exact_replay.csv',replays)
     print('[B2-R4]',task,'seed',seed,'rows',len(part),'test RMSE replay',test,flush=True)
  summary=aggregations(allrows)
  write_csv(out/'lead_seed_summary.csv',summary)
  missing=sorted(set((t,s,l) for t in TASKS for s in SEEDS for l in LEADS)-set((r['task'],r['seed'],r['lead']) for r in summary))
  if missing or len(replays)!=9 or len(summary)!=45 or len(allrows)!=9*5*WINDOWS_PER_SEED:
   raise RuntimeError('INCOMPLETE_MECHANISM_COVERAGE:'+repr(missing))
  result={'experiment':'CIDM-v2 B2-R4/P0 Decoder Tangent and Operator Span Audit',
          'status':'FROZEN_TANGENT_DIAGNOSTIC_COMPLETED_NO_TRAINING',
          'parent_sha256':PARENT_SHA,'official_baseline_replays':len(replays),
          'task_seed_lead_groups':len(summary),'diagnostic_sample_rows':len(allrows),
          'interventions':'ORACLE_FUTURE_TRUTH_DIAGNOSTIC_ONLY',
          'source_of_oracle_information':'Future truth on validation used ONLY to compute gradients and numerical diagnostic reference values, never as a forecast input.',
          'summary':interpret(allrows),
          'qualification':'DIAGNOSTIC_ONLY_DOES_NOT_AUTHORIZE_TRAINING',
          'limits':['Oracle label-conditioned latent shifts are not deployable and must not be compared to AI Weather Quest models.',
                    'Decoder tangent controllability is local to learned latent coordinates and fixed perturbation budgets.',
                    'Operator-span projection is only three global latent update directions per sample; low dimension limits raw captured fraction.',
                    'Previous B0 checkpoint selection used validation; this is not fresh holdout science.',
                    'No optimizers, no finetuning, no future truth entered model rollout.']}
 write_json(out/'report.json',result)
 return result
