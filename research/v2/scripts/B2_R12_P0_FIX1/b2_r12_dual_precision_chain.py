"""CIDM-v2 B2-R12/P0: frozen FP32-vs-FP64 full-chain autograd integrity audit.

Uses the ORIGINAL CIDM-v2 and B2-R6 weights, 2 TRAIN windows, and B2-R10
credit objective. No optimizer or future-label predictor input.  No attempt
is made to interpret FP64 as improved deployed forecast accuracy.
"""
from __future__ import annotations
import copy, csv, json, math, time, zipfile
from pathlib import Path
import torch
import b2_r11_backward_attribution as r11
from b2_r1_flux_control import check_zip

ARCHIVE_FIX1_SHA = 'b37e0539102afb4047028db72b8c93f32305f81adbd42551e0e3fcedb0eb9a6d'

def norm(t):
    return float(torch.linalg.vector_norm(t.detach().reshape(-1).double()))

def catparam(params):
    return torch.cat([(p.grad.detach().reshape(-1) if p.grad is not None
                        else torch.zeros_like(p).reshape(-1)) for p in params]).double()

def finite(t,name):
    if not bool(torch.isfinite(t).all()): raise RuntimeError('NONFINITE_'+name)

def write_csv(dest,rows):
    if not rows: return
    dest=Path(dest)
    with dest.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader();w.writerows(rows)

def write_json(dest,data):
    Path(dest).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False,default=str),encoding='utf-8')

# B2-R12/P0 FIX1 - memory-safe exact per-example streaming and independent local VJP.
# Intentionally preserves original FP32/FP64 dynamics and two-TRAIN-window objective.
import os, gc

def _sample_precision_pass(core, head, h, y, sd, Q, dtype, task, kind, precision, sample_index):
    """Full BPTT for *one* original TRAIN window; free whole DAG before local VJPs.

    Loss is the same per-example mean reduction used by the original batch=2 run.
    Caller averages full head gradients and time components over the same 2 windows.
    """
    params=tuple(head.parameters())
    h=h.to(dtype=dtype,device=next(core.parameters()).device)
    y=y.to(dtype=dtype,device=h.device)
    sd=sd.to(dtype=dtype,device=h.device)
    q=Q.to(dtype=dtype,device=h.device) if Q is not None else None
    pred,loss,recs=r11.full_forward(core,head,h,y,sd,q)
    finite(pred,f'{task}_{kind}_{precision}_sample{sample_index}_prediction')
    finite(loss,f'{task}_{kind}_{precision}_sample{sample_index}_loss')
    extra=[]
    for rec in recs:
        extra.extend((rec['gates'],rec['base'],rec['f'],rec['l'],rec['s'],rec['z_old']))
    # ONE backward through entire trajectory, no retain_graph=True. Extract all
    # necessary adjoints in the SAME autograd invocation before graph release.
    req=params+tuple(extra)
    vals=torch.autograd.grad(loss,req,allow_unused=True,retain_graph=False)
    pvals=vals[:len(params)]; adjvals=vals[len(params):]
    full=torch.cat([torch.zeros_like(p).reshape(-1).double() if v is None else v.detach().reshape(-1).double()
                    for p,v in zip(params,pvals)]).cpu()
    finite(full,f'{task}_{kind}_{precision}_sample{sample_index}_head_grad')
    per_step=[]
    for j,rec in enumerate(recs):
        gate_adj,base_adj,f_adj,l_adj,s_adj,z_adj=adjvals[j*6:(j+1)*6]
        if any(v is None for v in (gate_adj,base_adj,f_adj,l_adj,s_adj)):
            raise RuntimeError(f'MISSING_COTANGENT {task} {kind} {precision} sample={sample_index} lead={j+1}')
        for label,v in [('gate',gate_adj),('base',base_adj),('flux',f_adj),('local',l_adj),('skew',s_adj)]:
            finite(v,f'{task}_{kind}_{precision}_{label}_sample{sample_index}_lead{j+1}')
        # local frozen HEAD replay with detached causal features. No full-trajectory graph.
        g2=head(rec['base'].detach(),rec['prev_old'].detach(),rec['initial'].detach(),rec['amps'].detach(),j+1)
        replay=float((g2.detach()-rec['gates'].detach()).abs().max())
        if replay>(2e-6 if dtype==torch.float32 else 2e-11):
            raise RuntimeError(f'HEAD_REPLAY_MISMATCH {task} {kind} {precision} sample={sample_index} lead={j+1} replay={replay}')
        pp=torch.autograd.grad(g2,params,grad_outputs=gate_adj.detach(),allow_unused=True)
        p=torch.cat([(torch.zeros_like(a).flatten() if v is None else v.detach().flatten()).double()
                     for a,v in zip(params,pp)]).cpu()
        finite(p,f'{task}_{kind}_{precision}_sample{sample_index}_part_{j+1}')
        # Independently reconstruct LOCAL VJPs with exact frozen operators and
        # original cotangents. In particular, don't hold 12-step graph for VJPs.
        z=rec['z_old'].detach().requires_grad_(True)
        mm=core.model;coeff=mm.coefficients();skew=mm.skew-mm.skew.T
        f2=coeff['flux']*mm.flux(z)
        vf=torch.autograd.grad(f2,z,grad_outputs=f_adj.detach())[0]
        l2=coeff['local']*torch.tanh(mm.local(z))
        vl=torch.autograd.grad(l2,z,grad_outputs=l_adj.detach())[0]
        s2=.035*torch.einsum('ij,bjhw->bihw',skew,z)
        vs=torch.autograd.grad(s2,z,grad_outputs=s_adj.detach())[0]
        sq=lambda v:norm(v)**2
        per_step.append({'parameter_part':p,'gate_adj_sq':sq(gate_adj),'identity_sq':sq(base_adj),
                         'flux_sq':sq(vf),'local_sq':sq(vl),'skew_sq':sq(vs),
                         'latent_sq':(sq(z_adj) if z_adj is not None else None),
                         'replay':replay})
        del g2,pp,p,z,f2,l2,s2,vf,vl,vs
    out={'loss':float(loss.detach()),'full':full,'pred':pred.detach().cpu().double(), 'steps':per_step}
    del recs, vals,adjvals,extra,loss,pred
    gc.collect()
    if h.device.type=='cuda':torch.cuda.empty_cache()
    return out


def numerical_pass(core,head,h,y,sd,Q,precision,task,kind):
    """Identical 2-window objective, streamed over original TRAIN windows 0/1.

    Shared **CPU** backend by default makes FP32-vs-FP64 comparisons unconfounded
    by different CUDA vs CPU convolution implementations. A user may opt into
    CUDA for BOTH dtypes via CIDM_B2R12_FIX1_COMMON_DEVICE=cuda, but the script
    refuses it for GPUs with <32 GiB of total memory, to avoid old 14.6GiB OOM.
    """
    common=os.environ.get('CIDM_B2R12_FIX1_COMMON_DEVICE','cpu').strip().lower()
    if common not in ('cpu','cuda'):raise RuntimeError('COMMON_BACKEND_MUST_BE_CPU_OR_CUDA')
    if common=='cuda':
        if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 32*1024**3:
            raise RuntimeError('COMMON_CUDA_BACKEND_REQUIRES_AT_LEAST_32G_GPU_USE_CPU_INSTEAD')
    dev=torch.device(common)
    dtype=torch.float32 if precision=='fp32' else torch.float64
    # Copy from CPU first: never clone an entire CUDA model before transferring.
    core=copy.deepcopy(core.to('cpu')).to(device=dev,dtype=dtype).eval().requires_grad_(False)
    head=copy.deepcopy(head.to('cpu')).to(device=dev,dtype=dtype).eval().requires_grad_(True)
    Q=Q.cpu() if Q is not None else None
    h=h.cpu();y=y.cpu();sd=sd.cpu()
    n=int(h.shape[0]);
    if n<1:raise RuntimeError('NO_TRAIN_WINDOWS')
    accumulator=None;parts=None;sum_perstep=None;loss_acc=0.;preds=[]
    for i in range(n):
        case=_sample_precision_pass(core,head,h[i:i+1],y[i:i+1],sd,Q,dtype,task,kind,precision,i)
        w=1./n
        loss_acc+=w*case['loss'];preds.append(case['pred'])
        accumulator=case['full']*w if accumulator is None else accumulator+case['full']*w
        if parts is None:
            parts=[w*x['parameter_part'] for x in case['steps']]
            sum_perstep=[{k:(x[k]*w*w if k!='replay' else x[k]) for k in ('gate_adj_sq','identity_sq','flux_sq','local_sq','skew_sq','latent_sq','replay')}
                         for x in case['steps']]
        else:
            for j,x in enumerate(case['steps']):
                parts[j]+=w*x['parameter_part']
                for k in ('gate_adj_sq','identity_sq','flux_sq','local_sq','skew_sq','latent_sq'):
                    if x[k] is not None:sum_perstep[j][k]=(sum_perstep[j][k] or 0)+x[k]*w*w
                sum_perstep[j]['replay']=max(sum_perstep[j]['replay'],x['replay'])
        del case
        gc.collect()
        if dev.type=='cuda':torch.cuda.empty_cache()
    pred=torch.cat(preds,dim=0)
    full=accumulator
    gnorm=norm(full)
    recon=torch.stack(parts).sum(dim=0)
    rel=norm(recon-full)/max(gnorm,1e-30)
    steps=[]
    for t,(p,v) in enumerate(zip(parts,sum_perstep),start=1):
        get=lambda key:math.sqrt(v[key]) if v[key] is not None else None
        a=get('identity_sq');b=get('flux_sq')
        steps.append({'task':task,'controller':kind,'precision':precision,'lead':t,
           'parameter_part_l2':norm(p),'gate_adjoint_l2':get('gate_adj_sq'),
           'identity_vjp_l2':a,'flux_vjp_l2':b,'local_vjp_l2':get('local_sq'),
           'skew_vjp_l2':get('skew_sq'),
           'flux_to_identity_vjp_ratio':b/max(a,1e-30),
           'latent_adjoint_l2':get('latent_sq'),
           'local_gate_replay_max_abs':v['replay']})
    summary={'task':task,'controller':kind,'precision':precision,
       'loss':loss_acc,'full_gradient_l2_fp64_reduce':gnorm,
       'full_gradient_max_abs':float(full.abs().max()),
       'partition_relative_error':rel,'partition_abs_residual_l2':norm(recon-full),
       'cancellation_condition':sum(norm(x) for x in parts)/max(norm(recon),1e-30),
       'first_step_flux_vjp_l2':steps[0]['flux_vjp_l2'],
       'step1_parameter_part_l2':steps[0]['parameter_part_l2'],
       'full_gradient_is_elementwise_finite':True,
       'head_partition_strict_pass':rel<=1e-4,
       'common_backend':common, 'streamed_train_windows':n,
       'graph_retention':False,
       'scoring_contract':'mean of original TRAIN window 0 and 1 per-window identical 12-step physical loss'}
    return summary,steps,pred,full


def paired_case(task,core,cache,r6zip,device,heads_used=('operator','rotated'),limit=2):
    """All cross-precision cases use SAME engine, weights, TRAIN sample ids and loss.

    The original archived seed17 checkpoint is validated before this method runs.
    """
    ch=int(cache['splits']['train']['h'].shape[2]);d=int(core.model.skew.shape[0])
    matched='swe_2d' if task=='smoke_2d' else ('burgers_1d' if task=='smoke_1d' else task)
    heads=r11.load_heads(r6zip,matched,ch,'cpu')
    Q=r11.orthogonal_matrix(d,32207,'cpu',torch.float32)
    w=r11.Windows(cache['splits']['train']['h'],cache['splits']['train']['y'],cache['mean'],cache['std'])
    ids=list(range(min(limit,len(w))));examples=[w[i] for i in ids]
    h=torch.stack([a for a,b in examples]).cpu()
    y=torch.stack([b for a,b in examples]).cpu()
    sd=w.sd.cpu()
    group=[];detail=[]
    core=core.cpu()
    for kind in heads_used:
        head=heads[kind].cpu();q=Q if kind=='rotated' else None
        a,rows_a,pa,ga=numerical_pass(core,head,h,y,sd,q,'fp32',task,kind)
        b,rows_b,pb,gb=numerical_pass(core,head,h,y,sd,q,'fp64',task,kind)
        div=float((pa-pb).abs().max())
        reldiv=norm(pa-pb)/max(norm(pb),1e-30)
        gcval=float(torch.dot(ga,gb)/(torch.linalg.vector_norm(ga)*torch.linalg.vector_norm(gb)).clamp_min(1e-300)) if norm(ga)>0 and norm(gb)>0 else 0.
        grel=norm(ga-gb)/max(norm(gb),1e-30)
        ratio=norm(ga)/max(norm(gb),1e-30)
        for row in (a,b):
            row.update({'forward_max_abs_fp32_vs_fp64':div,'forward_relative_l2_fp32_vs_fp64':reldiv,
             'gradient_cosine_fp32_vs_fp64':gcval,
             'gradient_relative_l2_fp32_vs_fp64':grel,
             'gradient_norm_ratio_fp32_over_fp64':ratio,
             'strict_partition_gate_1e_4':1e-4})
        group.extend((a,b));detail.extend(rows_a+rows_b)
        del pa,pb,ga,gb
        gc.collect()
    return group,detail


def run(parent,r6zip,r9zip,fix2zip,r11fix1zip,work,output,smoke=False,device='cpu'):
    output=Path(output);work=Path(work)
    output.mkdir(parents=True,exist_ok=True);work.mkdir(parents=True,exist_ok=True)
    if r11.sha(r11fix1zip)!=ARCHIVE_FIX1_SHA:raise RuntimeError('R11_FIX1_UPLOADED_EVIDENCE_SHA_MISMATCH')
    with zipfile.ZipFile(r11fix1zip) as z:
        if z.testzip(): raise RuntimeError('R11_FIX1_EVIDENCE_CRC_ERROR')
        rep=json.loads(z.read('report.json'));gate=json.loads(z.read('gradient_path_gate.json'))
        if rep['status']!='FROZEN_ATTRIBUTION_DATA_COLLECTED_NUMERIC_GATE_FAIL_NO_TRAINING' or rep['temporal_rows']!=72 or gate['strict_numerical_gate_pass']:
            raise RuntimeError('R11_FIX1_EVIDENCE_GATE_CONTRACT_MISMATCH')
    if r11.sha(fix2zip)!=r11.PREV_FIX2_SHA:raise RuntimeError('R10_FIX2_EVIDENCE_SHA_MISMATCH')
    r11.credit.validate_r6(r6zip);r11.credit.validate_r9(r9zip)
    sums=[];steps=[];cases=[]
    try:
        if smoke:
            cfg=r11.Config(profile='smoke')
            for task,kind in [('smoke_1d','operator'),('smoke_2d','rotated')]:
                cache=r11.make_synthetic_cache(cfg)
                for split in ('train','val','test'):
                    if task=='smoke_1d':
                        cache['splits'][split]['h']=cache['splits'][split]['h'].mean(dim=-2)[...,:4]
                        cache['splits'][split]['y']=cache['splits'][split]['y'].mean(dim=-2)[...,:4]
                    else:
                        cache['splits'][split]['h']=cache['splits'][split]['h'][...,:2,:4]
                        cache['splits'][split]['y']=cache['splits'][split]['y'][...,:2,:4]
                core=r11.UnifiedModel('V2_Ported_R5',1,8,12,4).to(device)
                s,d=paired_case(task,core,cache,r6zip,device,(kind,),1)
                sums.extend(s);steps.extend(d)
            status='PROGRAM_ONLY_SMOKE_NO_SCIENCE'
        else:
            if parent is None or r11.sha(parent)!=r11.PARENT_SHA:raise RuntimeError('B0_PARENT_ZIP_SHA_MISMATCH')
            with zipfile.ZipFile(parent) as z:
                manifest,_=check_zip(z);cfg=r11.Config(**manifest['config'])
                for task in r11.TASK_LIST:
                    core,cache,replay=r11.make_cache_checked(task,z,cfg,work,device)
                    s,d=paired_case(task,core,cache,r6zip,device)
                    sums.extend(s);steps.extend(d)
                    cases.append({'task':task,'archived_B0_checkpoint_replay':replay,'heads':2})
                    write_csv(output/'precision_case_summary.csv',sums)
                    write_csv(output/'temporal_vjp_gain.csv',steps)
                    write_json(output/'progress.json',{'tasks_completed':[v['task'] for v in cases]})
            if len(sums)!=12 or len(steps)!=144 or len(cases)!=3:raise RuntimeError('INCOMPLETE_12_CASES_OR_144_STEPS')
            status='FROZEN_DUAL_PRECISION_FIX1_COMPLETED_NO_TRAINING'
    except Exception as e:
        write_csv(output/'precision_case_summary.csv',sums);write_csv(output/'temporal_vjp_gain.csv',steps)
        write_json(output/'failure_details.json',{'exception':repr(e),'cases_completed':cases})
        raise
    write_csv(output/'precision_case_summary.csv',sums);write_csv(output/'temporal_vjp_gain.csv',steps)
    unresolved=[d for d in sums if not d['head_partition_strict_pass']]
    swe=[d for d in sums if d['task']=='swe_2d']
    gate={'experiment':'B2_R12_P0_FIX1','status':'DIAGNOSTIC_ONLY_NO_TRAINING',
          'strict_partition_pass_all':len(unresolved)==0,
          'unresolved':unresolved,'swe_precision_cases':swe,
          'train_allowed':False,
          'note':'FP64 full trajectory replay changes numerical precision; divergent FP32/FP64 forecasts cannot be compared as identical-path gradients. Branch VJP ratios are local cotangent amplification, NOT Jacobian spectral norms.'}
    write_json(output/'precision_gate.json',gate)
    rep={'experiment':'CIDM_v2_B2_R12_P0_FIX1_MemoryBoundedDualPrecision',
         'status':status,'case_rows':len(sums),'time_rows':len(steps),'cases':cases,
         'prior_R11_FIX1_SHA256':ARCHIVE_FIX1_SHA,'original_parent_frozen':True,
         'updates':0,'train_allowed':False,'numeric_gate':gate['strict_partition_pass_all'],
         'not_independent_external_holdout':True}
    write_json(output/'report.json',rep)
    return rep
