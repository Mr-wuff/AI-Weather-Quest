"""CIDM-v2 B2-R6/P1: frozen-backbone causal local-direction controller qualification.
This is a seed17 diagnostic small-head pilot, not full CIDM-v2 training.
Inference never sees future truth; train supervision uses only official TRAIN samples.
Paired baseline vs natural local operator directions vs fixed orthogonal-rotated directions.
"""
from __future__ import annotations
import copy, csv, hashlib, io, json, math, os, time, zipfile
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from benchmark import Config, Windows, UnifiedModel, make_cache, make_synthetic_cache, TASKS
from b2_r1_flux_control import check_zip, get_core, assess as baseline_assess

PARENT_SHA='69552b29e169ec22ee85d9c1b1a678fdb188f6f25dc18f33b927b88f4dddffd4'
REPORT_SHA='0586af4b7f943b546637d828ae24cf38471b802f139138755132ce1e2784a196'
TASK_LIST=('burgers_1d','reacdiff_2d','swe_2d')
SEED=17
LEADS=(1,3,6,9,12)
RATIO_BUDGET=0.010


def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for x in iter(lambda:f.read(2**22),b''):h.update(x)
 return h.hexdigest()


def save_json(path,data):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding='utf8')


def write_csv(path,rows):
 if not rows:return
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 with open(p,'w',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def validate_preflight(parent,prior_report):
 if sha(parent)!=PARENT_SHA:raise RuntimeError('PARENT_SHA256_MISMATCH')
 with zipfile.ZipFile(parent) as z:
  m,contract=check_zip(z)
  assert m['gate']['completed_runs']==18
 if sha(prior_report)!=REPORT_SHA:raise RuntimeError('B2R5_REAL_REPORT_SHA256_MISMATCH')
 with zipfile.ZipFile(prior_report) as z:
  if z.testzip():raise RuntimeError('B2R5_REPORT_CRC_ERROR')
  r=json.loads(z.read('report.json')); import pandas as pd
  d=pd.read_csv(io.BytesIO(z.read('sample_spatial_span_oracle.csv')))
  m={x['path']:x['sha256'] for x in json.loads(z.read('FILE_SHA256_MANIFEST.json'))}
  for key,hsh in m.items():
   if hashlib.sha256(z.read(key)).hexdigest()!=hsh:raise RuntimeError('B2R5_INNER_MANIFEST_HASH_MISMATCH:'+key)
 assert r['status']=='FROZEN_SPATIAL_SPAN_AUDIT_COMPLETED_NO_TRAINING'
 assert r['baseline_replays']==9 and r['groups']==45 and r['sample_rows']==360
 assert len(d)==360 and set(d.seed)=={17,41,73} and set(d.task)==set(TASK_LIST)
 evidence={}
 for task,g in d[d.lead==12].groupby('task'):
  natural=float(g.grad_spatial3_frac.mean());rnd=float(g.grad_random_spatial3_frac.mean())
  pooled=lambda kind:float((g.base_mse_physical*g[f'{kind}_mse_ratio_0.010']).sum()/g.base_mse_physical.sum())
  evidence[task]={'natural_grad_capture':natural,'matched_random_grad_capture':rnd,
                  'natural_1pct_pooled':pooled('spatial3'),'matched_random_1pct_pooled':pooled('random_spatial3'),
                  'prequalified':bool(natural>rnd and pooled('spatial3')<pooled('random_spatial3'))}
 if set(evidence)!=set(TASK_LIST) or not all(x['prequalified'] for x in evidence.values()):
  raise RuntimeError('B2R5_EVIDENCE_NOT_SUFFICIENT_FOR_CAUSAL_PILOT')
 return {'status':'Q0_ORACLE_GEOMETRY_SUPPORTS_SMALL_PILOT_ONLY','evidence':evidence,
         'parent_sha256':sha(parent),'prior_zip_sha256':sha(prior_report),
         'truth_based_oracle_is_not_forecast_skill':True}


class CausalCoefficientHead(nn.Module):
 """Condition on predicted state only. Output 3 per-location causal gates."""
 def __init__(self,channels,width=16,proj_dim=8,seed=3030):
  super().__init__()
  self.channels=channels
  g=torch.Generator(device='cpu').manual_seed(seed)
  self.register_buffer('projection',torch.randn(proj_dim,256,generator=g)/math.sqrt(proj_dim))
  self.net=nn.Sequential(nn.Conv2d(proj_dim+2*channels+5,width,3,padding=1),nn.SiLU(),
                         nn.Conv2d(width,width,3,padding=1),nn.SiLU(),nn.Conv2d(width,3,1))
  nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def forward(self,z,prev,initial,amplitudes,lead):
  D=z.shape[1];P=self.projection[:,:D]
  if D>256: raise RuntimeError('LATENT_CHANNELS_EXCEED_256_PROJECTION_CONTRACT')
  if P.shape[1]!=D:raise RuntimeError('PROJECTION_SHAPE_ERROR')
  p=torch.einsum('kd,bdhw->bkhw',P,z.detach())
  b,_,h,w=z.shape
  t=z.new_full((b,1,h,w),lead/12)
  # baseline-only causal scalars; no target data, no sample IDs, no task label
  x=torch.cat([p,prev.detach().clamp(-6,6),initial.detach().clamp(-6,6),
     amplitudes.detach().clamp(0,20),t,z.square().mean(dim=1,keepdim=True).sqrt().detach()],1)
  return self.net(x).tanh()


def orthogonal_matrix(d,seed,device,dtype):
 g=torch.Generator(device='cpu').manual_seed(seed)
 a=torch.randn(d,d,generator=g,dtype=torch.float32)
 Q,R=torch.linalg.qr(a)
 Q=Q*torch.sign(torch.diag(R)).unsqueeze(0)
 return Q.to(device=device,dtype=dtype)


def effect_basis(Fs,Ls,Ss,rotation=None):
 basis=[Fs,Ls,Ss]
 if rotation is not None:
  basis=[torch.einsum('ij,bjhw->bihw',rotation,b) for b in basis]
 # strict same local Gram matrix/norm/rank under the orthogonal random control
 norms=torch.stack([b.square().mean(dim=1,keepdim=True).sqrt() for b in basis],dim=1)
 normed=[b / norms[:,k].clamp_min(1e-6) for k,b in enumerate(basis)]
 return torch.stack(normed,dim=1),norms.flatten(1,2)


def forecast(core,head,h,rotate=False,Q=None,steps=12,budget=RATIO_BUDGET,truncate=False):
 mm=core.model
 raised=h.ndim==4
 hh=h.unsqueeze(-2) if raised else h
 B,L,C,H,W=hh.shape
 c=mm.coefficients()
 z=mm.encode(hh.reshape(B,L*C,H,W))
 prev=hh[:,-1]
 initial=prev
 sk=mm.skew-mm.skew.T
 pred=[];diagnostics=[]
 for lead in range(1,steps+1):
  if truncate:z=z.detach();prev=prev.detach()
  with torch.no_grad():
   f=c['flux']*mm.flux(z)
   l=c['local']*torch.tanh(mm.local(z))
   s=.035*torch.einsum('ij,bjhw->bihw',sk,z)
   z_base=z+f+l+s
   basis,amps=effect_basis(f,l,s,Q if rotate else None)
  if head is None:
   z_next=z_base
   rms=torch.zeros(B,device=z.device,dtype=z.dtype)
  else:
   gates=head(z_base,prev,initial,amps,lead)
   delta=(gates.unsqueeze(2)*basis).sum(dim=1)/math.sqrt(3)
   # stable, sample-wise latent RMS budget; exact zero-gate parity with original
   zrms=z_base.square().mean(dim=(1,2,3),keepdim=True).sqrt().detach().clamp_min(1e-4)
   delta=budget*zrms*delta
   drms=delta.square().mean(dim=(1,2,3),keepdim=True).sqrt()
   cap=(budget*zrms)/(drms.detach().clamp_min(1e-8))
   delta=delta*cap.clamp(max=1.0)
   z_next=z_base+delta
   rms=(delta.square().mean(dim=(1,2,3)).sqrt()/zrms.flatten())
  x=prev+c['out']*torch.tanh(mm.decode(z_next))
  pred.append(x.squeeze(-2) if raised else x)
  if lead in LEADS:diagnostics.append({'lead':lead,'relative_delta_rms':float(rms.detach().mean())})
  z=z_next;prev=x
 return torch.stack(pred,1),diagnostics


def make_synthetic_1d(cfg):
 t=make_synthetic_cache(cfg)
 for s in ('train','val','test'):
  t['splits'][s]['h']=t['splits'][s]['h'].squeeze(-2)
  t['splits'][s]['y']=t['splits'][s]['y'].squeeze(-2)
 return t

@torch.no_grad()
def eval_case(core,heads,Q,ds,inds,device,batch=2):
 outputs={k:{n:[0.,0] for n in LEADS} for k in ('baseline','operator','rotated')}
 amp={}
 for off in range(0,len(inds),batch):
  arr=[ds[int(i)] for i in inds[off:off+batch]]
  h=torch.stack([x[0] for x in arr]).to(device);y=torch.stack([x[1] for x in arr]).to(device)
  for typ,head in (('baseline',None),('operator',heads['operator']),('rotated',heads['rotated'])):
   pred,_=forecast(core,head,h,rotate=(typ=='rotated'),Q=Q)
   scale=ds.sd.to(device)
   for lead in LEADS:
    sq=((pred[:,lead-1]-y[:,lead-1])*scale.reshape(1,-1,*([1]*(pred.ndim-3)))).square()
    outputs[typ][lead][0]+=float(sq.sum());outputs[typ][lead][1]+=sq.numel()
 return {kind:{str(lead):sq/max(n,1) for lead,(sq,n) in vals.items()} for kind,vals in outputs.items()}


def aggregate(scores,leads):
 return sum(scores[str(k)] for k in leads)/len(leads)


def make_cache_checked(task,parent_z,cfg,root,device):
 from benchmark import fetch_official
 path=fetch_official(Path(root)/'official_data',task)
 cache=make_cache(task,path,cfg,Path(root)/'window_cache')
 _,contract=check_zip(parent_z)
 ref=next(x for x in contract['tasks'] if x['task']==task)
 if cache['signature']['source_md5']!=ref['source_md5'] or cache['ids']!=ref['ids']:
  raise RuntimeError('ORIGINAL_B0_DATA_CONTRACT_MISMATCH:'+task)
 for split in ('train','val','test'):
  if len(cache['splits'][split]['h'])!=ref['windows'][split]:
   raise RuntimeError('ORIGINAL_B0_SPLIT_WINDOW_MISMATCH:'+task+split)
 core,ck_hash=get_core(parent_z,cache,task,SEED,device)
 recorded=float(json.loads(parent_z.read(f'{task}__V2_Ported_R5__seed{SEED}.json'))['test']['model_12'])
 bs=cfg.batch_1d if cache['splits']['test']['h'].ndim==4 else cfg.batch_2d
 actual=baseline_assess(core,'baseline',cache,'test',device,batch=bs)['rmse12']
 if abs(actual-recorded)>max(1e-5,recorded*0.001):raise RuntimeError('FROZEN_ORIGINAL_TEST_REPLAY_FAIL:'+task)
 return core,cache,{'task':task,'seed':SEED,'checkpoint_sha256':ck_hash,'archived_rmse12':recorded,'replay_rmse12':actual}


def pilot(task,core,cache,device,out,updates=60,smoke=False):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 core.requires_grad_(False).eval()
 ch=int(cache['splits']['train']['h'].shape[2]);d=int(core.model.skew.shape[0]);hproj=core.model
 if d>256:raise RuntimeError('LATENT_DIM_EXCEEDS_PROJECTION_256')
 torch.manual_seed(22026);natural=CausalCoefficientHead(ch,seed=707).to(device)
 torch.manual_seed(22026);rotated=CausalCoefficientHead(ch,seed=707).to(device)
 rotated.load_state_dict(copy.deepcopy(natural.state_dict()))
 heads={'operator':natural,'rotated':rotated}
 pn=sum(p.numel() for p in natural.parameters()); pr=sum(p.numel() for p in rotated.parameters())
 assert pn==pr
 Q=orthogonal_matrix(d,32207,device,torch.float32)
 train=Windows(cache['splits']['train']['h'],cache['splits']['train']['y'],cache['mean'],cache['std'])
 val=Windows(cache['splits']['val']['h'],cache['splits']['val']['y'],cache['mean'],cache['std'])
 sel=list(range(0,min(8,len(val)//2)))
 audit=list(range(len(val)//2,len(val)))
 if smoke: sel=[0,1];audit=[2,3]
 if len(sel)<2 or len(audit)<2:raise RuntimeError('INSUFFICIENT_VALIDATION_WINDOWS')
 batch=2 if smoke or cache['splits']['train']['h'].ndim==5 else 4
 with torch.no_grad():
  h=torch.stack([train[i][0] for i in range(2)]).to(device)
  p0,_=forecast(core,None,h)
  p1,_=forecast(core,natural,h)
  p2,_=forecast(core,rotated,h,Q=Q,rotate=True)
  frozen=core(h,12,mode='off')
  parity=max(float((p0-frozen).abs().max()),float((p1-frozen).abs().max()),float((p2-frozen).abs().max()))
  if parity>2e-6:raise RuntimeError('CAUSAL_HEAD_ZERO_INITIALIZATION_PARITY_FAIL')
  print('[PARITY]',task,parity,'matched params',pn,flush=True)
 opts={k:torch.optim.AdamW(m.parameters(),lr=5e-4,weight_decay=1e-5) for k,m in heads.items()}
 g=torch.Generator().manual_seed(20261014)
 history=[];best={k:{'score':float('inf'),'state':None,'step':None} for k in heads}
 resume_file=out/'paired_controller_resume.pt'
 start_step=0
 if resume_file.exists() and not smoke:
  prior=torch.load(resume_file,map_location='cpu',weights_only=True)
  if prior.get('task')!=task or prior.get('updates_contract')!=updates or prior.get('parameters_per_head')!=pn:
   raise RuntimeError('INCOMPATIBLE_RESUME_CONTRACT:'+str(resume_file))
  for k in heads:
   heads[k].load_state_dict(prior['model'][k]);opts[k].load_state_dict(prior['optimizer'][k]);best[k]=prior['best'][k]
   for state in opts[k].state.values():
    for key,v in state.items():
     if torch.is_tensor(v):state[key]=v.to(device)
  g.set_state(prior['train_rng']);history=prior['history'];start_step=prior['step']
  print('[AUTO RESUME]',task,'at update',start_step,flush=True)
 best_step=set([0,updates]+list(range(20,updates+1,20)))
 def checkpoint(step):
  v=eval_case(core,heads,Q,val,sel,device,batch=min(batch,2))
  for k in heads:
   score=aggregate(v[k],LEADS)
   if math.isfinite(score) and score<best[k]['score']:
    best[k]={'score':score,'state':copy.deepcopy(heads[k].state_dict()),'step':step}
  row={'step':step,'task':task,'baseline':aggregate(v['baseline'],LEADS),
   'operator':aggregate(v['operator'],LEADS),'rotated':aggregate(v['rotated'],LEADS),
   'operator_late':aggregate(v['operator'],(9,12)),
   'rotated_late':aggregate(v['rotated'],(9,12))}
  history.append(row);write_csv(out/'training_curve.csv',history)
  save_json(out/'checkpoint_progress.json',{'task':task,'updates_completed':step,
     'best_steps':{k:best[k]['step'] for k in heads},'parity_max':parity})
  print('[B2R6 PILOT]',task,row,flush=True)
  if not smoke:
   torch.save({'task':task,'step':step,'updates_contract':updates,'parameters_per_head':pn,
      'model':{k:heads[k].state_dict() for k in heads},
      'optimizer':{k:opts[k].state_dict() for k in heads},
      'best':best,'history':history,'train_rng':g.get_state()},resume_file)
 if start_step==0:checkpoint(0)
 for step in range(start_step+1,updates+1):
  ids=torch.randint(0,len(train),(batch,),generator=g).tolist()
  arr=[train[int(i)] for i in ids]
  h=torch.stack([x[0] for x in arr]).to(device)
  y=torch.stack([x[1] for x in arr]).to(device)
  loss_vals={}
  for typ in ('operator','rotated'):
   opts[typ].zero_grad(set_to_none=True)
   p,_=forecast(core,heads[typ],h,Q=Q,rotate=(typ=='rotated'),truncate=True)
   # Physical-unit scaled MSE, balanced selected leads; all labels from TRAIN only
   scale=train.sd.to(device).reshape(1,1,-1,*([1]*(p.ndim-3)))
   loss=sum(((p[:,k-1]-y[:,k-1])*scale.squeeze(1)).square().mean() for k in LEADS)/len(LEADS)
   if not torch.isfinite(loss):raise RuntimeError('NONFINITE_PILOT_LOSS:'+task+typ)
   loss.backward();nn.utils.clip_grad_norm_(heads[typ].parameters(),1.)
   opts[typ].step();loss_vals[typ]=float(loss.detach())
  if step in best_step:checkpoint(step)
 for k in heads:
  heads[k].load_state_dict(best[k]['state']);heads[k].eval()
 final=eval_case(core,heads,Q,val,audit,device,batch=min(2,batch))
 vratio=lambda k,leads:aggregate(final[k],leads)/max(aggregate(final['baseline'],leads),1e-20)
 opr=vratio('operator',(9,12)); rnd=vratio('rotated',(9,12))
 # A risk gate for early lead; the candidate must outperform equally parameterized random control
 approved=(opr<=.97 and opr/max(rnd,1e-20)<=.99 and
           vratio('operator',(1,3))<=1.02 and all(math.isfinite(v) for s in final.values() for v in s.values()))
 summary={'task':task,'seed':SEED,'updates':updates,'parameters_per_head':pn,
   'core_params_trainable':0,'selection_windows':sel,'audit_windows':audit,
   'best_step':{k:best[k]['step'] for k in heads},'original_forward_parity_max':parity,
   'audit':final,'operator_late_to_baseline':opr,'rotated_late_to_baseline':rnd,
   'operator_late_to_rotated':opr/max(rnd,1e-20),'operator_early_to_baseline':vratio('operator',(1,3)),
   'gate':('PILOT_LOCAL_ADVANTAGE' if approved else 'PILOT_NO_LOCAL_ADVANTAGE'),
   'limits':'This validation was used to select the pretrained B0 backbone. Not a fresh outer holdout.'}
 save_json(out/'pilot_summary.json',summary)
 for k in heads:
  torch.save({'head':heads[k].state_dict(),'task':task,'seed':SEED,'updates':updates,
     'training_model':'FROZEN_CIDM_V2','controller_type':k,'fixed_basis_rotation_seed':32207,
     'fixed_projection_seed':707,'best_step':best[k]['step']},out/f'{k}_best.pt')
 return summary


def run(parent,prior_report,root,output,updates=60,device=None,smoke=False,only_task=None):
 output=Path(output);output.mkdir(parents=True,exist_ok=True)
 device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
 if smoke:
  # 1D and 2D isolated smoke; no internet and no official train qualification.
  cfg=Config(profile='smoke')
  res=[]
  for task,cache in [('smoke_1d',make_synthetic_1d(cfg)),('smoke_2d',make_synthetic_cache(cfg))]:
   model=UnifiedModel('V2_Ported_R5',1,8,12,4).to(device)
   res.append(pilot(task,model,cache,device,output/task,updates=2,smoke=True))
  status='PROGRAM_ONLY_SMOKE_NO_SCIENCE'
  gate={'status':'SMOKE_ONLY'};replay=[]
 else:
  gate=validate_preflight(parent,prior_report)
  save_json(output/'Q0_FROZEN_GEOMETRY_GATE.json',gate)
  with zipfile.ZipFile(parent) as z:
   m,_=check_zip(z)
   cfg=Config(**m['config'])
   tasks=(only_task,) if only_task else TASK_LIST
   res=[];replay=[]
   for task in tasks:
    core,cache,record=make_cache_checked(task,z,cfg,root,device)
    replay.append(record);write_csv(output/'exact_replay_seed17.csv',replay)
    res.append(pilot(task,core,cache,device,output/task,updates=updates,smoke=False))
  groups={r['task']:r for r in res}
  if set(groups)==set(TASK_LIST):
   # RD: physical regression may not exceed 2% vs original (no need to surpass random specifically)
   rd=groups['reacdiff_2d']['operator_late_to_baseline']<=1.02
   good=(all(groups[k]['gate']=='PILOT_LOCAL_ADVANTAGE' for k in ('burgers_1d','swe_2d')) and rd)
   status='PILOT_SINGLE_SEED_PROMISING_FRESH_MULTISEED_REQUIRED' if good else 'PILOT_SINGLE_SEED_FAIL_NO_FULL_TRAINING'
  else:status='PARTIAL_PILOT_DIAGNOSTIC_ONLY'
 science={'experiment':'CIDM_v2_B2_R6_P1_Causal_LocalCoefficient_Qualification',
  'status':status,'q0':gate,'reports':res,'replays':replay,
  'frozen_parent':True,'parent_sha256':PARENT_SHA if not smoke else None,
  'no_future_truth_in_forward':True,'training_labels_only_from_official_train':True,
  'deployment_safety':'No future truth used in inference/closed loop; no truth-informed oracle policy applied.',
  'training_contract':'Seed17, budget-matched operator vs orthogonally rotated equal-rank control, frozen core, no formal training',
  'limitations':['Original B0 checkpoint was val-selected; no fresh outer holdout.',
                 'Truncated-gradient BPTT(1) gives actual closed-loop rollouts but approximate gradients.',
                 'Different basis geometry is not strict causality proof; parameter count and operator local Gram matched.',
                 'No full multi-seed retraining nor eight-PDE superiority nor AI Weather Quest RPSS qualification.']}
 save_json(output/'report.json',science)
 return science

if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--smoke',action='store_true');a.add_argument('--output',default='B2R6_SMOKE');args=a.parse_args()
 print(run(None,None,'/tmp/b2r6',args.output,smoke=args.smoke)['status'])
