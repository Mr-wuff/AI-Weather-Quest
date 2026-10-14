"""CIDM-v2 B2-R5/P0 frozen spatial operator-span geometry audit.

Future validation truth ONLY used for retrospective diagnostic gradients.
This code does not train, use test labels in diagnostics, or issue a deployable forecast.
"""
from __future__ import annotations
import hashlib,zipfile,io,json,csv,os,math,random
from pathlib import Path
import numpy as np
import torch
from benchmark import Config,Windows,fetch_official,make_cache,make_synthetic_cache,UnifiedModel
from b2_r1_flux_control import check_zip,get_core,assess as baseline_assess
from b2_r3_readout_probe import original_parity
from b2_r4_tangent_controllability import exact_rollout_states,sample_rms,span_project,flatten_cos,write_json,write_csv,scalar_list

PARENT_SHA='69552b29e169ec22ee85d9c1b1a678fdb188f6f25dc18f33b927b88f4dddffd4'
TASKS=('burgers_1d','reacdiff_2d','swe_2d'); SEEDS=(17,41,73);LEADS=(1,3,6,9,12)
VAL_START=8;WINDOWS_PER_SEED=8
BUDGETS=(.001,.003,.01,.03,.05)
METHODS=('full','global3','spatial3','random_spatial3','random_ambient')

def file_sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for x in iter(lambda:f.read(1<<22),b''):h.update(x)
 return h.hexdigest()

def per_site_projection(grad,vectors,mask_by_slot=None):
 """Orthogonally project each D-dim spatial pixel gradient on K<=3 local operator D-vectors.
     Unlike global3, weights may vary by grid point. Rank-matched random spatial
     basis is required: an expanded span necessarily captures more gradient.
 """
 assert grad.ndim==4
 qs=[];result=torch.zeros_like(grad);masks=[]
 for vv in vectors:
  v=vv.detach().to(grad.dtype)
  for q in qs:v=v-(v*q).sum(1,keepdim=True)*q
  # Each local basis must be sufficiently nonzero; avoid tiny norm / numerical noise.
  nn=v.square().sum(1,keepdim=True).sqrt()
  active=(nn>1e-7)
  if mask_by_slot is not None:active=active & mask_by_slot[len(qs)]
  q=torch.where(active, v/nn.clamp_min(1e-7), torch.zeros_like(v))
  masks.append(active)
  qs.append(q)
  result=result+(grad*q).sum(1,keepdim=True)*q
 denom=grad.square().flatten(1).sum(1).clamp_min(1e-30)
 frac=(result.square().flatten(1).sum(1)/denom).clamp(0,1)
 return result,frac,masks

def normalize_rms(x):
 rms=sample_rms(x)
 return torch.where(rms>1e-14,x/rms.clamp_min(1e-14),torch.zeros_like(x))

def comparison(core,state,truth,sd,seed):
 mm=core.model;coeff=mm.coefficients()
 z=state['z'].detach().float();prev=state['prev'].detach().float()
 truth=truth.detach().float()
 sd=sd.float().to(z.device).reshape(1,-1,1,1)
 with torch.enable_grad():
  var=z.detach().clone().requires_grad_(True)
  p=prev+coeff['out']*torch.tanh(mm.decode(var))
  assert (p.detach()-state['pred']).abs().max()<2e-5
  loss=((p-truth)*sd).square().flatten(1).mean(1)
  grad=torch.autograd.grad(loss.sum(),var)[0].detach()
 if not torch.isfinite(grad).all():raise RuntimeError('NONFINITE_GRAD')
 vec=[state[k].detach().float() for k in ('flux','local','skew')]
 glob,frac_g=span_project(grad,vec)
 site,frac_s,operator_masks=per_site_projection(grad,vec)
 gen=torch.Generator(device='cpu').manual_seed(int(seed))
 rb=[torch.randn(z.shape,generator=gen,dtype=z.dtype) for _ in range(3)]
 rb=[v.to(z.device) for v in rb]
 random_site,frac_rand,_=per_site_projection(grad,rb,mask_by_slot=operator_masks)
 random_amb=torch.randn(z.shape,generator=gen,dtype=z.dtype).to(z.device)
 # Gradient-conditioned projection into a random subspace is also an oracle.
 moves={'full':-normalize_rms(grad), 'global3':-normalize_rms(glob),
        'spatial3':-normalize_rms(site), 'random_spatial3':-normalize_rms(random_site),
        'random_ambient':normalize_rms(random_amb)}
 zscale=sample_rms(z).clamp_min(1.)
 metrics={'base_mse_physical':scalar_list(loss.detach()),
          'grad_global3_frac':scalar_list(frac_g),'grad_spatial3_frac':scalar_list(frac_s),
          'grad_random_spatial3_frac':scalar_list(frac_rand),
          'spatial_operator_mean_rank':scalar_list(torch.stack([mask.float().squeeze(1) for mask in operator_masks],0).sum(0).flatten(1).mean(1)),
          'grad_rms':scalar_list(sample_rms(grad)),
          'decoder_abslogit_gt2':scalar_list((state['logit'].abs()>2).float().flatten(1).mean(1)),
          'latent_rms':scalar_list(sample_rms(z)),
          'future_truth_gradient_oracle':1}
 for method,move in moves.items():
  for budget in BUDGETS:
   with torch.no_grad():
    shifted=z+budget*zscale*move
    predicted=prev+coeff['out']*torch.tanh(mm.decode(shifted))
    after=((predicted-truth)*sd).square().flatten(1).mean(1)
   metrics[f'{method}_mse_ratio_{budget:.3f}']=scalar_list(after/loss.detach().clamp_min(1e-22))
 # Oracle within-sample optimal move is only available with future truth; do not call this a forecast.
 for method in METHODS:
  stacked=torch.stack([torch.tensor(metrics[f'{method}_mse_ratio_{b:.3f}']) for b in BUDGETS],0)
  minimum=torch.cat([torch.ones((1,z.shape[0])),stacked],0).amin(0)
  metrics[f'{method}_best_truth_line_search_ratio']=minimum.tolist()
 return metrics

def per_task(cache,core,task,seed,device,indices,smoke=False):
 val=Windows(cache['splits']['val']['h'],cache['splits']['val']['y'],cache['mean'],cache['std'])
 sd=val.sd.to(device)
 rows=[];batch=2 if not task.endswith('_1d') else 4
 for offset in range(0,len(indices),batch):
  ids=indices[offset:offset+batch]
  h=torch.stack([val[i][0] for i in ids]).to(device)
  y=torch.stack([val[i][1] for i in ids]).to(device)
  if y.ndim==4:y=y.unsqueeze(-2)
  st=exact_rollout_states(core,h,leads=LEADS)
  for lead in LEADS:
   metrics=comparison(core,st[lead],y[:,lead-1],sd,int(seed)*1000003+int(lead)*101+offset)
   for j,idx in enumerate(ids):
    r={'task':task,'seed':seed,'lead':lead,'window_index':idx,'smoke_only':int(smoke)}
    for k,v in metrics.items():r[k]=v[j] if isinstance(v,list) else v
    rows.append(r)
 return rows

def aggregate(rows):
 grouped={}
 for r in rows: grouped.setdefault((r['task'],r['seed'],r['lead']),[]).append(r)
 result=[]
 for key,samples in sorted(grouped.items()):
  task,seed,lead=key
  row={'task':task,'seed':seed,'lead':lead,'windows':len(samples)}
  bases=np.array([r['base_mse_physical'] for r in samples],dtype=np.float64)
  if np.any(~np.isfinite(bases)) or np.any(bases<0):raise RuntimeError('BAD_BASE_LOSS')
  for k in samples[0]:
   if k in row or k in ('window_index','task','seed','lead','smoke_only','future_truth_gradient_oracle'):continue
   v=np.array([r[k] for r in samples],dtype=np.float64)
   if not np.isfinite(v).all():raise RuntimeError('NONFINITE_'+k)
   if '_ratio' in k:
    row[k+'_macro']=float(v.mean())
    row[k+'_pooled']=float(np.dot(v,bases)/bases.sum())
    row[k+'_win_fraction']=float((v<1).mean())
    row[k+'_p90']=float(np.quantile(v,.9))
   else:row[k]=float(v.mean())
  result.append(row)
 return result

def summary_report(rows,replays,smoke=False):
 agg=aggregate(rows)
 tasks=sorted(set(r['task'] for r in rows))
 report={'experiment':'CIDM-v2 B2-R5/P0 Spatial vs Global Operator Span Retrospective Audit',
         'status':'PROGRAM_ONLY_SMOKE' if smoke else 'FROZEN_SPATIAL_SPAN_AUDIT_COMPLETED_NO_TRAINING',
         'baseline_replays':len(replays),'groups':len(agg),'sample_rows':len(rows),
         'budgets':BUDGETS,'methods':METHODS,'validation_indices':[VAL_START,VAL_START+WINDOWS_PER_SEED-1],
         'future_truth_use':'ONLY retrospective gradients and oracle line search; NEVER deployment',
         'metric_warning':'Use pooled=sum(after)/sum(before) as primary error ratio. Unweighted macro(sample ratios) is susceptible to tiny baseline errors.',
         'interpretation_warning':'spatial3 has 3 local coefficients per pixel rather than 3 global coefficients; compare against dimension-matched random_spatial3 oracle. Local oracle does NOT train causal coefficients.',
         'qualification':'DIAGNOSTIC_ONLY_NO_TRAINING_PERMISSION','lead12':{}}
 for task in tasks:
  xx=[r for r in rows if r['task']==task and r['lead']==12]
  if not xx:continue
  bases=np.array([r['base_mse_physical'] for r in xx])
  info={'samples':len(xx)}
  for k in ('grad_global3_frac','grad_spatial3_frac','grad_random_spatial3_frac','spatial_operator_mean_rank','decoder_abslogit_gt2'):
   info[k]=float(np.mean([r[k] for r in xx]))
  for m in METHODS:
   k=f'{m}_mse_ratio_0.010';ratio=np.array([r[k] for r in xx]);info[m+'_0.01_pooled']=float(np.dot(ratio,bases)/bases.sum());info[m+'_0.01_macro']=float(np.mean(ratio))
  report['lead12'][task]=info
 return report,agg

def run(parent,root,out,smoke=False):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 allrows=[];replays=[]
 if smoke:
  cfg=Config(profile='smoke',history=4,train_horizon=6,eval_horizon=12)
  cache=make_synthetic_cache(cfg)
  for task,is1 in [('smoke_1d',True),('smoke_2d',False)]:
   src=cache if not is1 else {**cache,'splits':{k:{'h':v['h'].mean(dim=-2) if v['h'].ndim==5 else v['h'],
                'y':v['y'].mean(dim=-2) if v['y'].ndim==5 else v['y']} for k,v in cache['splits'].items()}}
   ch=int(src['splits']['train']['h'].shape[2]);core=UnifiedModel('V2_Ported_R5',ch,8,12,4).eval().requires_grad_(False)
   ids=[0,1]
   allrows.extend(per_task(src,core,task,17,'cpu',ids,True))
 else:
  if file_sha(parent)!=PARENT_SHA:raise RuntimeError('PARENT_SHA_MISMATCH')
  with zipfile.ZipFile(parent) as zp:
   manifest,contract=check_zip(zp);cfg=Config(**manifest['config'])
   for task in TASKS:
    data=fetch_official(Path(root)/'official_data',task)
    cache=make_cache(task,data,cfg,Path(root)/'window_cache')
    con=next(c for c in contract['tasks'] if c['task']==task)
    if cache['signature']['source_md5']!=con['source_md5'] or cache['ids']!=con['ids']:
     raise RuntimeError('PDEBENCH_MD5_OR_SPLIT_MISMATCH:'+task)
    if any(len(cache['splits'][s]['h'])!=con['windows'][s] for s in ('train','val','test')):raise RuntimeError('PDEBENCH_WINDOW_COUNT_MISMATCH:'+task)
    for seed in SEEDS:
     device='cuda' if torch.cuda.is_available() else 'cpu'
     core,sha=get_core(zp,cache,task,seed,device)
     core.eval().requires_grad_(False)
     batch=cfg.batch_1d if task.endswith('_1d') else cfg.batch_2d
     orig=float(json.loads(zp.read(f'{task}__V2_Ported_R5__seed{seed}.json'))['test']['model_12'])
     check=baseline_assess(core,'baseline',cache,'test',device,batch=batch)['rmse12']
     if abs(check-orig)>max(1e-5,orig*.001):raise RuntimeError('BASELINE_REPLAY_FAIL')
     val=Windows(cache['splits']['val']['h'],cache['splits']['val']['y'],cache['mean'],cache['std'])
     h=torch.stack([val[i][0] for i in range(2)]).to(device)
     parity=original_parity(core,h)
     if parity>2e-6:raise RuntimeError('BASELINE_PARITY_FAIL')
     replays.append({'task':task,'seed':seed,'checkpoint_sha256':sha,'expected_test_rmse12':orig,'replay_test_rmse12':check,'max_forward_difference':parity})
     if len(val)<VAL_START+WINDOWS_PER_SEED:raise RuntimeError('INSUFFICIENT_VAL_WINDOWS')
     allrows.extend(per_task(cache,core,task,seed,device,list(range(VAL_START,VAL_START+WINDOWS_PER_SEED))))
     write_csv(out/'sample_spatial_span_oracle.csv',allrows)
     write_csv(out/'exact_replay.csv',replays)
     print('[B2-R5]',task,'seed',seed,'rows',len(allrows),flush=True)
  if len(replays)!=9 or len(allrows)!=9*5*WINDOWS_PER_SEED:raise RuntimeError('INCOMPLETE_9_RUN_COVERAGE')
 report,agg=summary_report(allrows,replays,smoke)
 if not smoke and len(agg)!=45:raise RuntimeError('INCOMPLETE_45_GROUP_COVERAGE')
 write_csv(out/'sample_spatial_span_oracle.csv',allrows)
 write_csv(out/'lead_seed_spatial_summary.csv',agg)
 write_csv(out/'exact_replay.csv',replays)
 write_json(out/'report.json',report)
 return report

if __name__=='__main__':
 print(run(None,'.','./B2_R5_smoke',smoke=True))
