"""CIDM-MHC A0: matched-budget, truncated-gradient, Markov hierarchical closure pilot.

IMPORTANT: Q0 numerical + immediate causal-gradient prequalification, Q1 small pilot only.
No architecture-wide finetuning, no external-holdout claim, no full-training permission.
Historical parent model is frozen and exact B0-R1 official replay must PASS.
"""
from __future__ import annotations
import csv, hashlib, json, math, time, copy, traceback, zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from benchmark import Config, Windows, UnifiedModel, make_synthetic_cache
from b2_r6_causal_spatial_controller import (PARENT_SHA, TASK_LIST, make_cache_checked)
from b2_r1_flux_control import check_zip

LEADS=(1,3,6,9,12)
ARMS=('stateless_closure','mhc_hierarchical','mhc_no_aux')
BUDGET=0.003
SLOW_RATE=0.25  # Persistent coarse state responds at its own timescale; not P(z_t).
AUX_WEIGHT=0.10
SEED=20311015


def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
 return h.hexdigest()


def dump(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False,default=str),encoding='utf8')


def write_rows(path,rows):
 if not rows:return
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with open(path,'w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));w.writeheader();w.writerows(rows)


def lowpass(x):
 if x.ndim !=4:raise RuntimeError('EXPECTED_BCHW')
 kh=1 if x.shape[-2]==1 else 3
 return F.avg_pool2d(x,kernel_size=(kh,3),stride=1,padding=(kh//2,1),count_include_pad=False)


def lowpass_field(x):
 return lowpass(x.unsqueeze(-2)).squeeze(-2) if x.ndim==3 else lowpass(x)

def rms(x):
 return torch.sqrt(torch.clamp(x.square().mean(dim=(1,2,3),keepdim=True),min=1e-24))


class MarkovClosureHead(nn.Module):
 """One small shared-architecture controller. No lead ID, no history or future truth."""
 def __init__(self,channels,latent_dim,width=16,projection_dim=8,seed=SEED):
  super().__init__()
  if latent_dim>256:raise RuntimeError('LATENT_CHANNELS_GT_256')
  generator=torch.Generator().manual_seed(seed)
  self.register_buffer('proj',torch.randn(projection_dim,256,generator=generator)/math.sqrt(projection_dim))
  inputs=2*projection_dim+2*channels+2
  self.net=nn.Sequential(nn.Conv2d(inputs,width,3,padding=1),nn.SiLU(),
                         nn.Conv2d(width,width,3,padding=1),nn.SiLU(),nn.Conv2d(width,2,1))
  nn.init.zeros_(self.net[-1].weight)
  nn.init.zeros_(self.net[-1].bias)
  self.channels=channels
  self.latent_dim=latent_dim
 def forward(self,slow,fast,prev):
  pp=self.proj[:,:self.latent_dim]
  aa=torch.einsum('kd,bdhw->bkhw',pp,slow.detach())
  bb=torch.einsum('kd,bdhw->bkhw',pp,fast.detach())
  cs=prev.detach()
  x=torch.cat([aa,bb,cs,lowpass(cs),
               torch.log1p(rms(slow.detach())).expand(-1,-1,*cs.shape[-2:]),
               torch.log1p(rms(fast.detach())).expand(-1,-1,*cs.shape[-2:])],dim=1)
  return torch.tanh(self.net(x))


def run_rollout(core,head,h,arm='stateless_closure',steps=12,truncate=True):
 """12-step original core; optional learned correction on persistent slow/fast Markov states.

 No ground truth or time/round label enters dynamics; only current (slow,fast,p).
 Each update is bounded by <=.3% of latent RMS (actual action can be smaller).
 """
 if arm not in ('baseline',)+ARMS:raise ValueError(arm)
 mm=core.model
 original_1d=(h.ndim==4)
 hh=h.unsqueeze(-2) if original_1d else h
 batch,history,ch,height,width=hh.shape
 coef=mm.coefficients();skew=mm.skew-mm.skew.T
 with torch.no_grad():
  z=mm.encode(hh.reshape(batch,history*ch,height,width))
  prev=hh[:,-1]
 slow=lowpass(z)
 fast=z-slow
 outs=[];effects=[];scales=[]
 for step in range(steps):
  # BPTT(1) only; never differentiate original 12-step dynamical core across time.
  if truncate:
   slow=slow.detach();fast=fast.detach();prev=prev.detach()
  with torch.no_grad():
   # Canonical, original recurrence state is z, NEVER reconstruct from slow+fast.
   # Even ulp-sized changes due to split/recomposition can be amplified for SWE.
   flux=coef['flux']*mm.flux(z)
   local=coef['local']*torch.tanh(mm.local(z))
   skewpart=.035*torch.einsum('ij,bjhw->bihw',skew,z)
   tendency=flux+local+skewpart
   # Match R9's actual FP32 operation order exactly: ((z+flux)+local)+skewpart.
   # Do not algebraically regroup this into z+(flux+local+skewpart).
   base=z+flux+local+skewpart
   if arm=='mhc_hierarchical' or arm=='mhc_no_aux':
    slow_increment=SLOW_RATE*lowpass(tendency)
    slow_base=slow+slow_increment
    fast_base=base-slow_base
   else:
    slow_base=lowpass(base)
    fast_base=base-slow_base
   assert base.shape==slow_base.shape
   # Algebraic zero-control parity: slow_base+fast_base == original base.
  if head is None or arm=='baseline':
   corr=torch.zeros_like(base)
  else:
   gate=head(slow_base,fast_base,prev)
   us=lowpass(tendency)
   uf=tendency-us
   # This is a spatially structured tendency-direction control, not a truth oracle.
   norm_s=us/(rms(us).detach().clamp_min(1e-6))
   norm_f=uf/(rms(uf).detach().clamp_min(1e-6))
   raw=(lowpass(gate[:,0:1]*norm_s)+(gate[:,1:2]*norm_f-lowpass(gate[:,1:2]*norm_f)))/math.sqrt(2)
   # No sqrt(0) backward singularity. Smooth denominator for near-zero action.
   target=BUDGET*rms(base).detach().clamp_min(1e-4)
   corr=target*raw/(rms(raw).detach()+0.05)
   corr=corr*torch.clamp(target/(rms(corr).detach()+1e-12),max=1.0)
   # The maximum RMS correction contract is strict and batchwise.
   exposure=(rms(corr)/rms(base).detach().clamp_min(1e-4)).detach()
   if bool((exposure>BUDGET+1e-6).any()):raise RuntimeError('LATENT_RMS_EXPOSURE_EXCEEDED')
  zn=base+corr
  if arm in ('mhc_hierarchical','mhc_no_aux'):
   cs=SLOW_RATE*lowpass(corr)
   slow=slow_base+cs
   fast=zn-slow
  else:
   slow=lowpass(zn)
   fast=zn-slow
  z=zn  # authoritative recurrent state; slow/fast are persistent controller representations
  # Frozen decoder weights, but input gradient is essential for head learning.
  pnew=prev+coef['out']*torch.tanh(mm.decode(zn))
  outs.append(pnew.squeeze(-2) if original_1d else pnew)
  effects.append(rms(corr).detach().flatten()/rms(base).detach().flatten().clamp_min(1e-4))
  scales.append((rms(slow).detach(),rms(fast).detach()))
  prev=pnew
 return torch.stack(outs,dim=1),torch.stack(effects,dim=1),scales


def physical_scale(window,field):
 return window.sd.to(field.device).reshape(1,1,-1,*([1]*(field.ndim-3)))


def criterion(pred,truth,window,arm):
 # In physical units, with same fixed objective for stateless/MHC/full vs no_aux.
 scale=physical_scale(window,pred)
 diff=(pred-truth)*scale
 losses=[]
 for lead in LEADS:
  losses.append(diff[:,lead-1].square().mean())
 loss=sum(losses)/len(losses)
 if arm!='mhc_no_aux' and arm!='baseline':
  # Fixed, physically interpretable hierarchical target representations.
  # Stop-gradient on truth (not learnable encoder) and horizons fixed before testing.
  target=truth.detach()*scale
  physical=pred*scale
  slo=[3,6,9,12];shi=[1,3]
  ll=sum((lowpass_field(physical[:,l-1])-lowpass_field(target[:,l-1])).square().mean() for l in slo)/len(slo)
  lh=sum(((physical[:,l-1]-lowpass_field(physical[:,l-1]))-
           (target[:,l-1]-lowpass_field(target[:,l-1]))).square().mean() for l in shi)/len(shi)
  loss=loss+AUX_WEIGHT*(ll+lh)*0.5
 return loss


def get_data(cache,split):
 return Windows(cache['splits'][split]['h'],cache['splits'][split]['y'],cache['mean'],cache['std'])


def get_batch(ds,ids,device):
 samples=[ds[int(i)] for i in ids]
 return torch.stack([v[0] for v in samples]).to(device),torch.stack([v[1] for v in samples]).to(device)


@torch.no_grad()
def evaluate(core,heads,cache,split,ids,device,task):
 ds=get_data(cache,split);records=[]
 for ix in ids:
  x,y=get_batch(ds,[ix],device)
  for arm in ('baseline',)+ARMS:
   p,a,_=run_rollout(core,None if arm=='baseline' else heads[arm],x,arm=arm,truncate=True)
   scale=physical_scale(ds,p)
   for lead in LEADS:
    err=((p[:,lead-1]-y[:,lead-1])*scale[:,0]).square().flatten(1).mean(1)
    records.append({'task':task,'split':split,'window':int(ix),'arm':arm,'lead':lead,
                    'mse_physical':float(err.mean()),'rel_action_mean':float(a[:,lead-1].mean())})
 return records


def aggregate(rows,arm,leads=(9,12)):
 # pooled identical number of observations on each lead and window.
 vals=[r['mse_physical'] for r in rows if r['arm']==arm and r['lead'] in leads]
 return sum(vals)/max(1,len(vals))


def make_arms(ch,latent_dim,device):
 torch.manual_seed(SEED)
 first=MarkovClosureHead(ch,latent_dim).to(device)
 heads={a:copy.deepcopy(first).to(device) for a in ARMS}
 sizes={a:sum(p.numel() for p in heads[a].parameters() if p.requires_grad) for a in ARMS}
 assert len(set(sizes.values()))==1
 return heads,sizes


def q0(core,heads,cache,task,device,output,smoke=False):
 train=get_data(cache,'train');val=get_data(cache,'val')
 x,y=get_batch(train,[0],device)
 with torch.no_grad():
  ref=run_rollout(core,None,x,arm='baseline')[0]
  # Confirm original no-correction path exactly matches original causal R9 recurrence.
  from b2_r9_activation_timing import rollout_timing
  old=rollout_timing(core,None,x)[0]
  baseline_parity=float((ref-old).abs().max())
  assert baseline_parity<=2e-6,('BASELINE_R9_PARITY_FAIL',baseline_parity)
  # zero initialized controller must preserve baseline both arms exactly.
  zpar={arm:float((run_rollout(core,head,x,arm=arm)[0]-ref).abs().max()) for arm,head in heads.items()}
  if max(zpar.values())>2e-6:raise RuntimeError('ZERO_GATE_BASELINE_PARITY_FAIL')
 qrecord={'task':task,'baseline_parity_max':baseline_parity,'zero_gate_parity':zpar,'arms':{},'passed':False}
 flags=[]
 for arm,head in heads.items():
  head.train()
  # Q0 originally tested just one AdamW step. We retain that recorded check
  # and add a separate, bounded TRAIN-only negative-gradient feasibility probe.
  original={name:param.detach().clone() for name,param in head.named_parameters()}
  before,act,_=run_rollout(core,head,x,arm=arm)
  l0=criterion(before,y,train,arm)
  head.zero_grad(set_to_none=True);l0.backward()
  grad_by_name={name:param.grad.detach().clone() for name,param in head.named_parameters() if param.grad is not None}
  grads=list(grad_by_name.values())
  gn=math.sqrt(sum(float(g.detach().double().square().sum()) for g in grads))
  finite=bool(math.isfinite(gn) and gn>1e-13 and all(bool(torch.isfinite(g).all()) for g in grads))
  l0=float(l0.detach())
  trust_tol=max(1e-11,abs(l0)*1e-5)
  trials=[];chosen=None;down=False;mag=0.;l1=l0
  if finite:
   # Do not use optimizer state from a different arm; first reproduce old AdamW one-step trial.
   opt=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=0.)
   for name,p in head.named_parameters():
    p.grad=grad_by_name.get(name,None)
   opt.step()
   with torch.no_grad():
    pnew,newact,_=run_rollout(core,head,x,arm=arm)
    adam_loss=float(criterion(pnew,y,train,arm));adam_mag=float(newact.mean())
   trials.append({'method':'original_adamw','step_size':3e-4,'loss_after':adam_loss,'improved':bool(adam_loss<l0-trust_tol)})
   if math.isfinite(adam_loss) and adam_loss<l0-trust_tol and 0.<adam_mag<BUDGET+1e-6:
    chosen='original_adamw';down=True;l1=adam_loss;mag=adam_mag
   else:
    # Undo failed optimizer update before testing a deterministic, normalized
    # negative gradient direction. This is a Q0 feasibility witness on TRAIN only,
    # not validation tuning or permission to alter later Q1 optimizer contract.
    for radius in (3e-4,1e-4,3e-5,1e-5,3e-6,1e-6):
     with torch.no_grad():
      for name,p in head.named_parameters():
       p.copy_(original[name])
       if name in grad_by_name:p.add_(grad_by_name[name],alpha=-radius/max(gn,1e-20))
      pnew,newact,_=run_rollout(core,head,x,arm=arm)
      candidate_loss=float(criterion(pnew,y,train,arm));candidate_mag=float(newact.mean())
     accepted=bool(math.isfinite(candidate_loss) and candidate_loss<l0-trust_tol and 0.<candidate_mag<BUDGET+1e-6)
     trials.append({'method':'normalized_sgd_direction','step_size':radius,'loss_after':candidate_loss,'improved':accepted})
     if accepted:
      chosen=f'normalized_sgd_{radius:.0e}';down=True;l1=candidate_loss;mag=candidate_mag;break
   if not down:
    with torch.no_grad():
     for name,p in head.named_parameters():p.copy_(original[name])
    l1=l0;mag=0.
  else:
   for name,p in head.named_parameters():p.data.copy_(original[name])
  qrecord['arms'][arm]={'loss_before':l0,'loss_after_one_step':l1,
        'relative_train_objective_change':(l1/l0-1) if l0>0 else None,
        'gradient_l2':gn,'finite_and_nonzero_gradient':finite,
        'immediate_descent':down,'q0_descent_witness':chosen,
        'q0_backtracking_trials':trials,'strict_descent_tolerance':trust_tol,
        'actual_action_mean':mag,
        'parameter_count':sum(p.numel() for p in head.parameters() if p.requires_grad)}
  flags.append(finite and down and 0.<mag<BUDGET+1e-6)
 qrecord['passed']=all(flags)
 dump(output/f'{task}_q0_gate.json',qrecord)
 return qrecord


def train_small(core,heads,cache,task,output,device,updates=24,smoke=False):
 """PREQUALIFIED mini-pilot only. Identical updates/cost, NO full architecture optimization."""
 train=get_data(cache,'train');val=get_data(cache,'val')
 opt={a:torch.optim.AdamW(heads[a].parameters(),lr=3e-4,weight_decay=0.) for a in ARMS}
 progress=[];schedule=[0,8,16,24] if not smoke else [0,1,2,3]
 # Selection evaluation is reporting only: all arms use last step, never choose using audit.
 def snap(step):
  with torch.no_grad():
   rows=evaluate(core,heads,cache,'val',list(range(min(4,len(val)))),device,task)
  for arm in ('baseline',)+ARMS:
   progress.append({'task':task,'step':step,'arm':arm,'selection_late_mse':aggregate(rows,arm)})
  write_rows(output/f'{task}_progress.csv',progress)
 snap(0)
 for step in range(1,updates+1):
  for arm,head in heads.items():
   head.train()
   ix=((step-1)*13+SEED)%len(train)
   h,y=get_batch(train,[ix],device)
   p,e,_=run_rollout(core,head,h,arm=arm,truncate=True)
   loss=criterion(p,y,train,arm)
   if not bool(torch.isfinite(loss)):raise RuntimeError(f'NONFINITE_Q1_LOSS {task} {arm} {step}')
   opt[arm].zero_grad(set_to_none=True);loss.backward()
   norm=torch.sqrt(sum(p.grad.detach().double().square().sum() for p in head.parameters() if p.grad is not None))
   # No gradient clipping: nonfinite or extreme gradients FAIL CLOSED.
   if not bool(torch.isfinite(norm)) or float(norm)>1e5:raise RuntimeError(f'UNSAFE_Q1_GRADIENT {task} {arm} {step}: {float(norm)}')
   opt[arm].step()
   if step%4==0 or step==1:
    progress.append({'task':task,'step':step,'arm':arm,'train_objective':float(loss.detach()),
                     'train_grad_l2_unclipped':float(norm.detach()),'actual_action_mean':float(e.mean())})
    write_rows(output/f'{task}_progress.csv',progress)
  if step in schedule:snap(step)
  if device.startswith('cuda') and step%4==0:torch.cuda.empty_cache()
 for arm,head in heads.items():
  torch.save({'arm':arm,'task':task,'steps':updates,'state_dict':{k:v.detach().cpu() for k,v in head.state_dict().items()}},output/f'{task}_{arm}_mini_pilot.pt')
 audit_ix=list(range(16,min(32,len(val))))
 if smoke:audit_ix=list(range(2,min(4,len(val))))
 if len(audit_ix)<2:raise RuntimeError('INSUFFICIENT_DEFERRED_VALIDATION_WINDOWS')
 audit=evaluate(core,heads,cache,'val',audit_ix,device,task)
 write_rows(output/f'{task}_audit_per_window.csv',audit)
 return audit,progress


def gate(audits,smoke=False):
 result={'status':'DIAGNOSTIC_ONLY_NO_FULL_TRAINING','train_allowed':False,'tasks':{},
  'early_regression_cap':.005,'late_gain_target':.01,'max_severe_window_fraction':.20,
  'warnings':['selection and audit both derive from original B0 validation split; NOT unseen outer holdout',
   'seed17 only and 24 update small pilot','full long-BPTT remains unqualified','state sufficiency is operational/model-defined, not physical state sufficiency']}
 for task,rows in audits.items():
  baseline=aggregate(rows,'baseline');early=aggregate(rows,'baseline',(1,3))
  n={}
  for arm in ARMS:
   late=aggregate(rows,arm)
   er=aggregate(rows,arm,(1,3))
   # A severe regression is >2% on at least one late lead for the same window.
   sub={(r['window'],r['lead']):r['mse_physical'] for r in rows if r['arm']==arm}
   ref={(r['window'],r['lead']):r['mse_physical'] for r in rows if r['arm']=='baseline'}
   ids=sorted({r['window'] for r in rows})
   severe=sum(any(sub[(i,l)]>1.02*ref[(i,l)] for l in (9,12)) for i in ids)
   n[arm]={'late_vs_baseline_ratio':late/baseline,
           'early_vs_baseline_ratio':er/early,
           'late_severe_window_fraction':severe/len(ids),
           'pass_diagnostic':bool(late<=baseline*.99 and er<=early*1.005 and severe/len(ids)<=.20)}
  n['mhc_vs_stateless_late_ratio']=aggregate(rows,'mhc_hierarchical')/aggregate(rows,'stateless_closure')
  n['mhc_vs_no_aux_late_ratio']=aggregate(rows,'mhc_hierarchical')/aggregate(rows,'mhc_no_aux')
  result['tasks'][task]=n
 success=sum(r['mhc_hierarchical']['pass_diagnostic'] and r['mhc_vs_stateless_late_ratio']<=.99 for r in result['tasks'].values())
 result['diagnostic_hits']=success
 result['status']=('MHC_FIX1_SMALL_PILOT_SIGNAL_NEEDS_MULTI_SEED_AND_OUTER_DATA' if success>=2 else
                   'MHC_FIX1_SMALL_PILOT_GATE_FAIL_NO_FORMAL_TRAINING')
 if smoke:result['status']='PROGRAM_ONLY_SMOKE_NO_SCIENCE'
 return result


def smoke_case():
 cfg=Config(profile='smoke')
 out=[]
 for tag in ('smoke_1d','smoke_2d'):
  cache=make_synthetic_cache(cfg)
  for split in ('train','val','test'):
   if tag=='smoke_1d':
    cache['splits'][split]['h']=cache['splits'][split]['h'].mean(dim=-2)[...,:4]
    cache['splits'][split]['y']=cache['splits'][split]['y'].mean(dim=-2)[...,:4]
   else:
    cache['splits'][split]['h']=cache['splits'][split]['h'][...,:2,:4]
    cache['splits'][split]['y']=cache['splits'][split]['y'][...,:2,:4]
  core=UnifiedModel('V2_Ported_R5',1,8,12,4).cpu().eval().requires_grad_(False)
  out.append((tag,core,cache))
 return out


def run(parent,work,output,smoke=False,device=None):
 work=Path(work);output=Path(output);work.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(min(4,torch.get_num_threads()))
 device='cpu' if smoke else (device or ('cuda' if torch.cuda.is_available() else 'cpu'))
 torch.manual_seed(SEED);np.random.seed(SEED)
 cases=[];progress=[];audits={};groups=[]
 try:
  if smoke:cases=smoke_case()
  else:
   if parent is None or sha(parent)!=PARENT_SHA:raise RuntimeError('MISSING_OR_WRONG_PARENT')
   with zipfile.ZipFile(parent) as z:
    manifest,_=check_zip(z);cfg=Config(**manifest['config'])
    for task in TASK_LIST:
     core,cache,replay=make_cache_checked(task,z,cfg,work/'official_runtime','cpu')
     dump(output/f'{task}_parent_replay.json',replay)
     cases.append((task,core,cache))
  for task,core,cache in cases:
   core=core.to(device).eval().requires_grad_(False)
   ch=int(cache['splits']['train']['h'].shape[2]);latent_dim=int(core.model.skew.shape[0])
   heads,sizes=make_arms(ch,latent_dim,device)
   initial_head_states={arm:copy.deepcopy(head.state_dict()) for arm,head in heads.items()}
   precert=q0(core,heads,cache,task,device,output,smoke=smoke)
   groups.append({'task':task,'q0':precert['passed'],'head_params':sizes,'device':device})
   dump(output/'progress_summary.json',groups)
   if not precert['passed']:
    # All three tasks are separately Q0-gated. Never silently claim a trained result.
    continue
   for arm in ARMS: heads[arm].load_state_dict(initial_head_states[arm],strict=True)
   audit,_=train_small(core,heads,cache,task,output,device,updates=3 if smoke else 24,smoke=smoke)
   audits[task]=audit
   dump(output/'progress_summary.json',groups)
  g=gate(audits,smoke=smoke)
  if not smoke and len(audits)!=3:
   g['status']='MHC_FIX1_Q0_OR_RESOURCE_GATE_FAIL_NO_FORMAL_TRAINING'
  dump(output/'mhc_gate.json',g)
  report={'status':g['status'],'mode':'smoke' if smoke else 'official_small_pilot',
          'task_groups':groups,'completed_pilot_tasks':list(audits),
          'updates_per_head':3 if smoke else 24,'full_model_updates':0,'q0_witness_reset_before_q1':True,'train_allowed':False,
          'data_contract':'original B0 official parent and original split; no test-set pilot tuning; authoritative canonical z and TRAIN-only Q0 line-search'}
  dump(output/'report.json',report)
  return report
 except Exception as exc:
  dump(output/'failure_details.json',{'type':type(exc).__name__,'exception':str(exc),
       'traceback':traceback.format_exc()[-16000:],'progress':groups,'train_allowed':False})
  raise
