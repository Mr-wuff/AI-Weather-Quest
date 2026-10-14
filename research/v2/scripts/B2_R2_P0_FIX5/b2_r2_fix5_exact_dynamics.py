"""CIDM-v2 B2-R2/P0 FIX5: exact frozen observable-geometry audit.

Original B0-R1 model, original B2-R1 FIX2 wrappers, official data contract.
No trainable modules, no optimizer, no intervention to original prediction.
Observe *actual* PortedRestoredCIDM operations: skew is nn.Parameter, decode is nn.Sequential.
"""
from __future__ import annotations
import csv, hashlib, io, json, math, os, pathlib, time, zipfile
from collections import defaultdict
import numpy as np
import torch
from benchmark import Config, Windows, fetch_official, make_cache, load_data, assess as canonical_assess
from b2_r1_flux_control import (TASKLIST, SEEDS, get_core, check_zip, ControlledV2, assess as fix2_assess)

LEADS=(1,3,6,9,12)
SOURCE_HASH='8b4fd32e3f3675fd23764506267020dc61aba31c9ad54cfdf3ce6ace5d7e2c98'
PARENT_HASH='69552b29e169ec22ee85d9c1b1a678fdb188f6f25dc18f33b927b88f4dddffd4'


def rms(t):return float(t.detach().float().square().mean().sqrt().item())
def cos(a,b):
 a=a.detach().float().flatten(start_dim=1);b=b.detach().float().flatten(start_dim=1)
 an=torch.linalg.vector_norm(a,dim=1);bn=torch.linalg.vector_norm(b,dim=1)
 keep=(an>1e-10)&(bn>1e-10)&torch.isfinite(an)&torch.isfinite(bn)
 if not bool(keep.any()):return None
 return float(((a[keep]*b[keep]).sum(dim=1)/(an[keep]*bn[keep])).clamp(-1,1).mean().item())

def validate_core(core):
 assert core.kind=='V2_Ported_R5', 'Model mismatch'
 mm=core.model
 assert all(hasattr(mm,x) for x in ('encode','decode','flux','local','skew','coefficients'))
 assert isinstance(mm.skew,torch.nn.Parameter), 'skew must be trained parameter, not hookable module'
 assert callable(mm.decode) and callable(mm.flux) and callable(mm.local)
 return mm

@torch.inference_mode()
def trace_frozen(core,h,truth,task,seed,batch_index,leads=LEADS,parity_atol=2e-6):
 """Probe one baseline rollout at the exact trained operators, no future-truth inputs.

 For output diagnostic truth is used only *after* prediction to score its direction.
 Counterfactual operator deletion is local at a fixed frozen z; no retrained ablation.
 """
 mm=validate_core(core);is1d=h.ndim==4
 h5=h.unsqueeze(-2) if is1d else h
 y5=truth.unsqueeze(-2) if is1d else truth
 B,L,C,H,W=h5.shape
 assert L==mm.hist==4 and C==mm.channels and y5.shape[1]>=max(leads)
 coeff=mm.coefficients(); skew=mm.skew-mm.skew.T
 history=[x for x in h5.unbind(1)]
 z=mm.encode(h5.reshape(B,L*C,H,W))
 rows=[];forecasts=[];prev_truth=h5[:,-1]
 for t in range(1,max(leads)+1):
  z_before=z
  flux=coeff['flux']*mm.flux(z_before)
  local=coeff['local']*torch.tanh(mm.local(z_before))
  skew_delta=.035*torch.einsum('ij,bjhw->bihw',skew,z_before)
  z=z_before+flux+local+skew_delta
  dlog=mm.decode(z)
  inc=coeff['out']*torch.tanh(dlog)
  x=history[-1]+inc
  forecasts.append(x);history.append(x)
  target=y5[:,t-1]
  true_inc=target-prev_truth
  prev_truth=target
  if t not in leads:continue
  pieces={'flux':flux,'local':local,'skew':skew_delta}
  sum_delta=flux+local+skew_delta
  assert torch.all(torch.isfinite(z)) and torch.all(torch.isfinite(inc)), 'NONFINITE'
  row=dict(task=task,seed=int(seed),batch=int(batch_index),lead=t,n=B,
    flux_rms=rms(flux),local_rms=rms(local),skew_rms=rms(skew_delta),
    update_rms=rms(z-z_before),additive_residual_rel=rms((z-z_before)-sum_delta)/max(rms(z-z_before),1e-10),
    flux_local_cos=cos(flux,local),flux_skew_cos=cos(flux,skew_delta),
    flux_scale=float(coeff['flux']),local_scale=float(coeff['local']),out_scale=float(coeff['out']),
    decoder_logits_rms=rms(dlog),decoder_abs_gt2=float((dlog.abs()>2).float().mean()),
    pred_increment_rms=rms(inc),truth_increment_rms=rms(true_inc),
    pred_increment_cos_truth=cos(inc,true_inc),
    pred_increment_power_ratio=float((inc.square().mean()/(true_inc.square().mean()+1e-20)).item()),
    original_step_mse_normalized=float((x-target).square().mean()),
    latent_state_rms=rms(z),latent_step_rel=rms(sum_delta)/max(rms(z_before),1e-10))
  # The output increments have the same normalized observable units as model prediction.
  responses={}
  for key,v in pieces.items():
   inc_without=coeff['out']*torch.tanh(mm.decode(z-v))
   response=inc-inc_without
   responses[key]=response
   row[key+'_response_rms']=rms(response)
   row[key+'_response_cos_true_increment']=cos(response,true_inc)
   row[key+'_response_cos_pred_increment']=cos(response,inc)
   row[key+'_response_relative_to_inc']=rms(response)/max(rms(inc),1e-10)
   # Diagnostic-only locally removed term, fixed actual latent trajectory.
   row[key+'_local_cf_mse_change_normalized']=float(((history[-2]+inc_without-target).square().mean()-(x-target).square().mean()).item())
  all_response=sum(responses.values())
  base_increment=coeff['out']*torch.tanh(mm.decode(z_before))
  row['decoder_effect_of_total_update_rms']=rms(inc-base_increment)
  row['operator_decoder_nonlinearity_rms']=rms(all_response-(inc-base_increment))
  # The model's OWN generated history only. 4-frame history includes no future truth.
  pred_hist=torch.stack(history[-L:],dim=1)
  re_z=mm.encode(pred_hist.reshape(B,L*C,H,W))
  re_inc=coeff['out']*torch.tanh(mm.decode(re_z))
  row['self_reencode_latent_cos']=cos(z,re_z)
  row['self_reencode_increment_gap_rel']=rms(re_inc-inc)/max(rms(inc),1e-10)
  row['self_reencode_increment_cos']=cos(re_inc,inc)
  row['self_reencode_increment_rms']=rms(re_inc)
  rows.append(row)
 predicted=torch.stack(forecasts,dim=1)
 expected=ControlledV2(core,'baseline')(h,max(leads))
 if is1d:predicted=predicted.squeeze(-2)
 max_diff=float((predicted-expected).abs().max())
 if not math.isfinite(max_diff) or max_diff>parity_atol:
  raise RuntimeError(f'REAL_MODEL_FORWARD_PARITY_FAIL: {task} seed{seed} batch{batch_index}: max_diff={max_diff}')
 return rows,max_diff


def weighted_summary(rows):
 group=defaultdict(list)
 for r in rows:group[(r['task'],r['seed'],r['lead'])].append(r)
 out=[]
 for (task,seed,lead),items in sorted(group.items()):
  n=sum(x['n'] for x in items)
  v={'task':task,'seed':seed,'lead':lead,'num_windows':n,'num_batches':len(items)}
  for k in items[0]:
   if k in ('task','seed','lead','batch','n'):continue
   vals=[(x.get(k),x['n']) for x in items if x.get(k) is not None and isinstance(x.get(k),(float,int)) and math.isfinite(x[k])]
   v[k]=sum(z*w for z,w in vals)/sum(w for _,w in vals) if vals else None
  out.append(v)
 return out


def dump_csv(path,items):
 path=pathlib.Path(path)
 if not items:return
 keys=list(items[0])
 with open(path,'w',newline='',encoding='utf-8') as fp:
  wr=csv.DictWriter(fp,fieldnames=keys,extrasaction='ignore');wr.writeheader();wr.writerows(items)

def sha256(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()

def write_json(path,obj):pathlib.Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding='utf-8')

def run(parent_zip,output_dir,root_dir=None,max_val_windows=16,smoke=False):
 output_dir=pathlib.Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
 root=pathlib.Path(root_dir or output_dir.parent/'runtime');root.mkdir(parents=True,exist_ok=True)
 if smoke:return unit_test(output_dir)
 assert sha256(parent_zip)==PARENT_HASH,'PARENT_SHA256_MISMATCH'
 assert sha256(pathlib.Path(__file__).with_name('b2_r1_flux_control.py'))==SOURCE_HASH,'FIX2_DRIVER_SHA256_MISMATCH'
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(min(4,torch.get_num_threads()))
 results=[];replays=[];checks=[];status='PRECHECK'
 (output_dir/'experiment_contract.json').write_text(json.dumps(dict(
   experiment='B2_R2_P0_FIX5',official_only=True,train=False,leads=LEADS,
   val_window_limit=max_val_windows,seed_list=SEEDS,task_list=TASKLIST,
   parent_sha256=PARENT_HASH,fix2_source_sha256=SOURCE_HASH,
   selection_set='none',threshold_search='none',expected_task_seed=9),indent=2),encoding='utf-8')
 try:
  with zipfile.ZipFile(parent_zip) as zp:
   manifest,contract=check_zip(zp)
   assert manifest['gate']['completed_runs']==18
   config=Config(**manifest['config']);
   for task in TASKLIST:
    data_root=pathlib.Path(os.environ.get('CIDM_B2_DATA_ROOT',str(root/'official_data')))
    official=fetch_official(data_root,task)
    cache=make_cache(task,official,config,root/'window_cache')
    ref=next(z for z in contract['tasks'] if z['task']==task)
    assert cache['signature']['source_md5']==ref['source_md5']
    assert cache['ids']==ref['ids']
    for split in ('train','val','test'):
     assert len(cache['splits'][split]['h'])==ref['windows'][split]
    for seed in SEEDS:
     core,checkpoint_sha=get_core(zp,cache,task,seed,device)
     original_batch=config.batch_1d if cache['splits']['test']['h'].ndim==4 else config.batch_2d
     orig=float(json.loads(zp.read(f'{task}__V2_Ported_R5__seed{seed}.json'))['test']['model_12'])
     canonical=canonical_assess(core,load_data(cache,'test',original_batch,seed,False),device,config,mode='off')['model_12']
     wrapped=fix2_assess(core,'baseline',cache,'test',device,batch=original_batch)['rmse12']
     tol=max(1e-5,orig*.001)
     row=dict(task=task,seed=seed,checkpoint_sha256=checkpoint_sha,archived_rmse12=orig,
       canonical_rmse12=canonical,fix2_rmse12=wrapped,batch=original_batch,
       replay_pass=bool(abs(canonical-orig)<=tol and abs(wrapped-canonical)<=tol))
     replays.append(row);dump_csv(output_dir/'exact_replay.csv',replays)
     if not row['replay_pass']:raise RuntimeError('OFFICIAL_REPLAY_FAIL_CLOSED: '+str(row))
     print(f'[REPLAY PASS] {task} seed{seed} archived={orig:.8g} canonical={canonical:.8g} fixed2={wrapped:.8g}',flush=True)
     val_ds=Windows(cache['splits']['val']['h'],cache['splits']['val']['y'],cache['mean'],cache['std'])
     nb=min(len(val_ds),max_val_windows)
     for start in range(0,nb,original_batch):
      xx=[val_ds[j] for j in range(start,min(nb,start+original_batch))]
      h=torch.stack([x[0] for x in xx]).to(device)
      truth=torch.stack([x[1] for x in xx]).to(device)
      diag,parity=trace_frozen(core,h,truth,task,seed,start//original_batch)
      results.extend(diag);checks.append(dict(task=task,seed=seed,batch=start//original_batch,parity=max(float(parity),0.),windows=h.shape[0]))
     summary=weighted_summary(results)
     dump_csv(output_dir/'observable_samples.csv',results)
     dump_csv(output_dir/'lead_seed_summary.csv',summary)
     dump_csv(output_dir/'forward_parity.csv',checks)
     print(f'[DIAGNOSTIC] {task} seed{seed}: {len([x for x in summary if x["task"]==task and x["seed"]==seed])} leads, {nb} validation windows',flush=True)
     write_json(output_dir/'progress.json',{'completed_task_seed':[[r['task'],r['seed']] for r in replays],
       'all_replay_pass':all(r['replay_pass'] for r in replays),'valid_trace_rows':len(results),'time':time.time()})
     del core
     if device.type=='cuda':torch.cuda.empty_cache()
  summary=weighted_summary(results)
  expected={(task,int(seed),lead) for task in TASKLIST for seed in SEEDS for lead in LEADS}
  actual={(r['task'],int(r['seed']),int(r['lead'])) for r in summary}
  valid=not(expected-actual) and len(replays)==9 and len(results)>0 and all(r['replay_pass'] for r in replays)
  maxerr=max(r['additive_residual_rel'] for r in results)
  paritymax=max(r['parity'] for r in checks)
  valid=valid and maxerr<=1e-4 and paritymax<=2e-6
  status='FROZEN_DIAGNOSTIC_COMPLETED_NO_TRAINING' if valid else 'INCOMPLETE_OR_INTEGRITY_FAIL_NO_TRAINING'
  report={'experiment':'CIDM-v2 B2-R2/P0 FIX5','status':status,'official_data':'MD5_VERIFIED_BY_CANONICAL_BENCHMARK',
    'replay_models':len(replays),'covered_family_seed_leads':len(actual),'expected_family_seed_leads':len(expected),
    'raw_batch_lead_rows':len(results),'missing_coverage':sorted([list(k) for k in (expected-actual)]),
    'max_additive_relative_residual':maxerr,'max_forward_abs_difference':paritymax,
    'stage':'FROZEN_DIAGNOSTIC_ONLY','train_steps':0,'qualification':'NOT_EVALUATED',
    'notice':'Lead metrics are on validation windows; test was used only for archived baseline replay, not candidate selection.',
    'scientific_limits':['Frozen effects are local single-step counterfactuals, not a retrained or full-rollout ablation.',
     'Latent values depend on learned coordinates; no direct physical conservation conclusion.',
     'No hyperparameter search or performance-improvement claim; only mechanism diagnosis.']}
  write_json(output_dir/'report.json',report)
  print('[FIX5 FINAL]',json.dumps(report,ensure_ascii=False),flush=True)
  return report
 except BaseException as exc:
  write_json(output_dir/'failure.json',{'status':'BLOCKED_NO_TRAINING','error':repr(exc),'completed_replay':len(replays),'partial_rows':len(results)})
  raise

@torch.inference_mode()
def unit_test(output_dir):
 pathlib.Path(output_dir).mkdir(parents=True,exist_ok=True)
 from benchmark import UnifiedModel
 results=[];parities=[]
 for task,dims,channels in [('smoke_1d',(2,4,1,16),1),('smoke_2d',(2,4,2,8,16),2)]:
  model=UnifiedModel('V2_Ported_R5',channels,8,12,4).eval()
  torch.manual_seed(17)
  h=torch.randn(*dims)*.01
  y=torch.randn(2,12,*dims[2:])*.01
  r,err=trace_frozen(model,h,y,task,17,0)
  assert len(r)==5 and err<=2e-6 and max(q['additive_residual_rel'] for q in r)<1e-4
  assert all(q.get('flux_response_rms') is not None and q.get('self_reencode_increment_gap_rel') is not None for q in r)
  results.extend(r);parities.append(err)
 report={'experiment':'B2_R2_FIX5_CPU_SMOKE','status':'PROGRAM_ONLY_SMOKE_PASS','rows':len(results),
         'max_forward_abs_difference':max(parities), 'max_additive_relative_residual':max(x['additive_residual_rel'] for x in results)}
 write_json(pathlib.Path(output_dir)/'smoke_report.json',report)
 return report
