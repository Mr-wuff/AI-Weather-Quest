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

def numerical_pass(core,head,h,y,sd,Q,precision,task,kind):
    """Exact same trajectory objective, with parameters cast to selected precision.
    FP64 changes numerical evaluation, NOT training, operator coefficients or loss.
    """
    dtype=torch.float32 if precision=='fp32' else torch.float64
    core=copy.deepcopy(core).to(dtype=dtype).eval()
    head=copy.deepcopy(head).to(dtype=dtype).eval()
    core.requires_grad_(False)
    head.requires_grad_(True)
    core.zero_grad(set_to_none=True);head.zero_grad(set_to_none=True)
    q=Q.to(dtype=dtype) if Q is not None else None
    pred,loss,records=r11.full_forward(core,head,h.to(dtype),y.to(dtype),sd.to(dtype),q)
    finite(pred,task+'_'+kind+'_'+precision+'_prediction')
    finite(loss,task+'_'+kind+'_'+precision+'_loss')
    # Keep graph for local operator VJP; never update parameters.
    loss.backward(retain_graph=True)
    params=tuple(head.parameters());full=catparam(params);finite(full,'fullgradient_'+precision)
    gnorm=norm(full)
    parts=[];steps=[]
    for r in records:
        t=int(r['lead']);gg=r['gates'].grad
        if gg is None:raise RuntimeError('MISSING_GATE_COTANGENT_'+str(t))
        finite(gg,task+'_gate_cotangent_'+precision)
        rp=head(r['base'].detach(),r['prev_old'].detach(),r['initial'].detach(),r['amps'].detach(),t)
        dif=float((rp.detach()-r['gates'].detach()).abs().max())
        replaytol=2e-6 if dtype==torch.float32 else 2e-11
        if dif>replaytol:raise RuntimeError(f'HEAD_REPLAY_MISMATCH {precision} {t} {dif}')
        pp=torch.autograd.grad(rp,params,grad_outputs=gg.detach(),allow_unused=True)
        p=torch.cat([(torch.zeros_like(a) if grad is None else grad).detach().reshape(-1).double()
                     for a,grad in zip(params,pp)])
        finite(p,'step_part_'+precision+'_'+str(t));parts.append(p)
        z=r['z_old'];f=r['f'];l=r['l'];s=r['s']
        gbase=r['base'].grad
        gf=f.grad;gl=l.grad;gs=s.grad
        if any(a is None for a in [gbase,gf,gl,gs]):raise RuntimeError('MISSING_OPERATOR_COTANGENT')
        vf,=torch.autograd.grad(f,z,grad_outputs=gf,retain_graph=True)
        vl,=torch.autograd.grad(l,z,grad_outputs=gl,retain_graph=True)
        vs,=torch.autograd.grad(s,z,grad_outputs=gs,retain_graph=True)
        vals=[norm(v) for v in [gbase,vf,vl,vs]]
        steps.append({'task':task,'controller':kind,'precision':precision,'lead':t,
                      'parameter_part_l2':norm(p),'gate_adjoint_l2':norm(gg),
                      'identity_vjp_l2':vals[0],'flux_vjp_l2':vals[1],
                      'local_vjp_l2':vals[2],'skew_vjp_l2':vals[3],
                      'flux_to_identity_vjp_ratio':vals[1]/max(vals[0],1e-30),
                      'latent_adjoint_l2':norm(z.grad) if z.grad is not None else None,
                      'local_gate_replay_max_abs':dif})
    recon=torch.stack(parts).sum(dim=0)
    rel=norm(recon-full)/max(gnorm,1e-30)
    sum_norms=sum(norm(x) for x in parts)
    cancellation=sum_norms/max(norm(recon),1e-30)
    adj_first=steps[0]['flux_vjp_l2']
    summary={'task':task,'controller':kind,'precision':precision,
             'loss':float(loss.detach()),'full_gradient_l2_fp64_reduce':gnorm,
             'full_gradient_max_abs':float(full.abs().max()),
             'partition_relative_error':rel,'partition_abs_residual_l2':norm(recon-full),
             'cancellation_condition':cancellation,
             'first_step_flux_vjp_l2':adj_first,
             'step1_parameter_part_l2':steps[0]['parameter_part_l2'],
             'full_gradient_is_elementwise_finite':True,
             'head_partition_strict_pass':rel<=1e-4}
    return summary,steps,pred.detach(),full

def paired_case(task,core,cache,r6zip,device,heads_used=('operator','rotated'),limit=2):
    ch=int(cache['splits']['train']['h'].shape[2]);d=int(core.model.skew.shape[0])
    matched='swe_2d' if task=='smoke_2d' else ('burgers_1d' if task=='smoke_1d' else task)
    heads=r11.load_heads(r6zip,matched,ch,device)
    q=r11.orthogonal_matrix(d,32207,device,torch.float32)
    w=r11.Windows(cache['splits']['train']['h'],cache['splits']['train']['y'],cache['mean'],cache['std'])
    n=min(limit,len(w));samples=[w[i] for i in range(n)]
    h=torch.stack([a for a,b in samples]).to(device);y=torch.stack([b for a,b in samples]).to(device);sd=w.sd.to(device)
    group=[];detail=[]
    for kind in heads_used:
        head=heads[kind].to(device)
        Q=q if kind=='rotated' else None
        a,rows_a,pa,ga=numerical_pass(core,head,h,y,sd,Q,'fp32',task,kind)
        b,rows_b,pb,gb=numerical_pass(core,head,h,y,sd,Q,'fp64',task,kind)
        divergence=float((pa.double()-pb.double()).abs().max())
        relative_div=norm(pa.double()-pb.double())/max(norm(pb.double()),1e-30)
        gc=float(torch.dot(ga,gb)/(torch.linalg.vector_norm(ga)*torch.linalg.vector_norm(gb)).clamp_min(1e-300)) if norm(ga)>0 and norm(gb)>0 else 0.
        grel=norm(ga-gb)/max(norm(gb),1e-30)
        ratio=norm(ga)/max(norm(gb),1e-30)
        for x in (a,b):
            x.update({'forward_max_abs_fp32_vs_fp64':divergence,
                     'forward_relative_l2_fp32_vs_fp64':relative_div,
                     'gradient_cosine_fp32_vs_fp64':gc,
                     'gradient_relative_l2_fp32_vs_fp64':grel,
                     'gradient_norm_ratio_fp32_over_fp64':ratio,
                     'strict_partition_gate_1e_4':1e-4})
        group.extend([a,b]);detail.extend(rows_a+rows_b)
        del pa,pb,ga,gb
        if device=='cuda':torch.cuda.empty_cache()
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
            status='FROZEN_DUAL_PRECISION_CONDITIONING_COMPLETED_NO_TRAINING'
    except Exception as e:
        write_csv(output/'precision_case_summary.csv',sums);write_csv(output/'temporal_vjp_gain.csv',steps)
        write_json(output/'failure_details.json',{'exception':repr(e),'cases_completed':cases})
        raise
    write_csv(output/'precision_case_summary.csv',sums);write_csv(output/'temporal_vjp_gain.csv',steps)
    unresolved=[d for d in sums if not d['head_partition_strict_pass']]
    swe=[d for d in sums if d['task']=='swe_2d']
    gate={'experiment':'B2_R12_P0','status':'DIAGNOSTIC_ONLY_NO_TRAINING',
          'strict_partition_pass_all':len(unresolved)==0,
          'unresolved':unresolved,'swe_precision_cases':swe,
          'train_allowed':False,
          'note':'FP64 full trajectory replay changes numerical precision; divergent FP32/FP64 forecasts cannot be compared as identical-path gradients. Branch VJP ratios are local cotangent amplification, NOT Jacobian spectral norms.'}
    write_json(output/'precision_gate.json',gate)
    rep={'experiment':'CIDM_v2_B2_R12_P0_FullChainDualPrecision',
         'status':status,'case_rows':len(sums),'time_rows':len(steps),'cases':cases,
         'prior_R11_FIX1_SHA256':ARCHIVE_FIX1_SHA,'original_parent_frozen':True,
         'updates':0,'train_allowed':False,'numeric_gate':gate['strict_partition_pass_all'],
         'not_independent_external_holdout':True}
    write_json(output/'report.json',rep)
    return rep
