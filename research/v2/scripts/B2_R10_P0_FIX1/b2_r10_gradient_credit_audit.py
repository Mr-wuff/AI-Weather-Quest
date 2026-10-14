"""B2-R10/P0 FIX1: stable-backward frozen gradient credit-horizon qualification.

No optimization. Compare identical 12-step autonomous forecasts under exact
1/3/12-step differentiation through trained B2-R6 heads and unchanged CIDM core.
Labels used only for TRAIN loss; NEVER as rollout/controller inputs.
"""
from __future__ import annotations
import time, math, json, copy, zipfile, csv
from pathlib import Path
import torch
from benchmark import Config, Windows, UnifiedModel, make_synthetic_cache
from b2_r6_causal_spatial_controller import (
    TASK_LIST, LEADS, SEED, PARENT_SHA, make_cache_checked,
    orthogonal_matrix, sha
)
from b2_r7_frozen_controller_gain_audit import load_heads, write_csv, write_json
from b2_r9_activation_timing import validate_r8, schedule_fraction, validate_r6, validate_r7
from b2_r1_flux_control import check_zip

R9_REAL_ZIP_SHA='3b90fded1769d199da2a8ca93e1505acc451d074f4e45e8c720be3a15861389f'
BUDGET=.003
HORIZONS=(1,3,12)
EXPECTED_LEADS=(1,3,9,12)


def validate_r9(path):
    if sha(path)!=R9_REAL_ZIP_SHA:raise RuntimeError('B2R9_EXACT_REPORT_SHA256_MISMATCH')
    with zipfile.ZipFile(path) as z:
        if z.testzip():raise RuntimeError('B2R9_REPORT_CRC_FAIL')
        data=json.loads(z.read('report.json'))
        if data['status']!='FROZEN_ACTIVATION_TIMING_COMPLETED_NO_TRAINING' or data['total_samples']!=5040:
            raise RuntimeError('B2R9_P0_REPORT_STATUS_INCOMPATIBLE')
        manifest=json.loads(z.read('FILE_SHA256_MANIFEST.json'))
        import hashlib
        for entry in manifest:
            if hashlib.sha256(z.read(entry['path'])).hexdigest()!=entry['sha256']:
                raise RuntimeError('B2R9_MANIFEST_FAIL '+entry['path'])
    return data


# FIX1: Preserve the ORIGINAL R9 forward values, but remove a subtle backward
# singularity. The old sqrt(mean(x^2)).clamp_min(eps) differentiates through
# sqrt(0) and returns NaN (0 * infinity) even though the forward is zero.
# Clamp before sqrt and then restore the exact forward zero. This is NOT a
# gradient clamp/clip and does not change any nonzero forward result.

def stable_exact_rms(x, dims=(1,2,3), keepdim=True):
    q=x.square().mean(dim=dims, keepdim=keepdim)
    return q.clamp_min(1e-30).sqrt() * (q>0).to(q.dtype)

def stable_exact_effect_basis(Fs,Ls,Ss,rotation=None):
    basis=[Fs,Ls,Ss]
    if rotation is not None:
        basis=[torch.einsum('ij,bjhw->bihw',rotation,b) for b in basis]
    norms=torch.stack([stable_exact_rms(b,dims=1,keepdim=True) for b in basis],dim=1)
    normed=[b / norms[:,k].clamp_min(1e-6) for k,b in enumerate(basis)]
    return torch.stack(normed,dim=1),norms.flatten(1,2)

def zero_singular_backward_test():
    # A targeted regression test, NOT a synthetic forecasting performance claim.
    x=torch.zeros((2,4,3,3),requires_grad=True)
    y=stable_exact_rms(x).sum()
    y.backward()
    assert torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad)==0
    f=torch.zeros((2,4,3,3),requires_grad=True)
    l=torch.randn_like(f,requires_grad=True)
    sk=torch.zeros_like(f,requires_grad=True)
    v,a=stable_exact_effect_basis(f,l,sk)
    (v.square().sum()+a.square().sum()).backward()
    assert all(torch.isfinite(q.grad).all() for q in (f,l,sk))
    return {'zero_rms_backward':'PASS','zero_operator_backward':'PASS'}

def forecast_credit(core,head,h,rotation=None,truncate_horizon=1,steps=12):
    """Faithful R9 forward, but F/L/S + decoder keep differentiable computational graph.

    Core parameters remain frozen, while gradients propagate through the frozen
    operations to earlier head actions. z/prev are detached ONLY on chunk borders.
    """
    if truncate_horizon not in HORIZONS:raise ValueError('unregistered horizon')
    mm=core.model
    one_d=(h.ndim==4)
    hh=h.unsqueeze(-2) if one_d else h
    B,L,C,H,W=hh.shape
    co=mm.coefficients();z=mm.encode(hh.reshape(B,L*C,H,W));prev=hh[:,-1];initial=prev
    skew=mm.skew-mm.skew.T
    out=[]
    for lead in range(1,steps+1):
        if lead>1 and (lead-1)%truncate_horizon==0:
            z=z.detach();prev=prev.detach()
        f=co['flux']*mm.flux(z)
        l=co['local']*torch.tanh(mm.local(z))
        s=.035*torch.einsum('ij,bjhw->bihw',skew,z)
        base=z+f+l+s
        basis,amps=stable_exact_effect_basis(f,l,s,rotation)
        gates=head(base,prev,initial,amps,lead)
        raw=(gates.unsqueeze(2)*basis).sum(dim=1)/math.sqrt(3)
        rr=stable_exact_rms(raw)
        change=BUDGET*stable_exact_rms(base).clamp_min(1e-4)*raw/rr.clamp_min(1e-9)
        change=torch.where(rr>=1e-9,change,torch.zeros_like(change))
        z=base+change
        prev=prev+co['out']*torch.tanh(mm.decode(z))
        out.append(prev.squeeze(-2) if one_d else prev)
    return torch.stack(out,1)


def original_frozen_rollout(core,head,h,rotation=None):
    # Exact prior B2-R9 implementation; this is forward-parity reference.
    from b2_r9_activation_timing import rollout_timing
    return rollout_timing(core,head,h,Q=rotation,rotate=rotation is not None,
                          budget=BUDGET,onset=1,mode='fixed_step')[0]


def frozen_gradient(core,head,h,truth,scale,rotation,horizon):
    head.zero_grad(set_to_none=True)
    device=h.device
    if device.type=='cuda':
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t0=time.perf_counter()
    pred=forecast_credit(core,head,h,rotation=rotation,truncate_horizon=horizon)
    if not torch.isfinite(pred).all():raise RuntimeError('NONFINITE_FORWARD')
    forward_diff=0.
    with torch.no_grad():
        ref=original_frozen_rollout(core,head,h,rotation=rotation)
        forward_diff=float((ref-pred.detach()).abs().max())
    if forward_diff>2e-6:raise RuntimeError(f'B2R10_FORWARD_PARITY_FAILED {horizon} {forward_diff}')
    # Scale to physical units; identical objective at K1/K3/K12.
    shape=(1,1,scale.numel())+(1,)*(pred.ndim-3)
    w=scale.reshape(shape)
    losses={j:((pred[:,j-1]-truth[:,j-1])*w.squeeze(1)).square().mean() for j in EXPECTED_LEADS}
    loss=0.5*(losses[9]+losses[12])+0.125*(losses[1]+losses[3])
    if not torch.isfinite(loss):raise RuntimeError('NONFINITE_GRADIENT_LOSS')
    loss.backward()
    flat=[]
    for p in head.parameters():
        flat.append(torch.zeros_like(p).flatten() if p.grad is None else p.grad.detach().flatten())
    vec=torch.cat(flat)
    if not torch.isfinite(vec).all():
        details=[]
        for name,param in head.named_parameters():
            if param.grad is None: continue
            bad=(~torch.isfinite(param.grad)).sum().item()
            if bad: details.append({'parameter':name,'nonfinite_elements':int(bad),'total_elements':param.grad.numel()})
        raise RuntimeError('NONFINITE_HEAD_GRADIENT_DETAIL '+json.dumps({'horizon':horizon,'bad':details},ensure_ascii=False))
    if device.type=='cuda':torch.cuda.synchronize(device)
    seconds=time.perf_counter()-t0
    peak=(torch.cuda.max_memory_allocated(device)/2**20) if device.type=='cuda' else None
    return vec.detach().cpu(),{'loss':float(loss.detach()),'forward_parity_max':forward_diff,
       'gradient_l2':float(torch.linalg.vector_norm(vec)),'seconds':seconds,
       'gpu_peak_allocated_mib':peak,'objective':'0.5*(L9+L12)+0.125*(L1+L3)'}


def correlation(x,y):
    xx=torch.linalg.vector_norm(x).item(); yy=torch.linalg.vector_norm(y).item()
    if xx<1e-16 or yy<1e-16:return None
    return float(torch.dot(x.double(),y.double())/torch.linalg.vector_norm(x.double())/torch.linalg.vector_norm(y.double()))


def one_case(task,core,cache,r6zip,device,smoke=False):
    core.requires_grad_(False).eval()
    ch=int(cache['splits']['train']['h'].shape[2]);d=int(core.model.skew.shape[0])
    heads=load_heads(r6zip,'swe_2d' if task=='smoke_2d' else ('burgers_1d' if task=='smoke_1d' else task),ch,device)
    Q=orthogonal_matrix(d,32207,device,torch.float32)
    train=Windows(cache['splits']['train']['h'],cache['splits']['train']['y'],cache['mean'],cache['std'])
    inds=[0,1] if len(train)>=2 else [0]
    arr=[train[i] for i in inds]
    h=torch.stack([a[0] for a in arr]).to(device);y=torch.stack([a[1] for a in arr]).to(device)
    scale=train.sd.to(device)
    rows=[]
    for typ in ('operator','rotated'):
        head=heads[typ].train()
        for p in head.parameters():p.requires_grad_(True)
        gradients={}
        for horizon in HORIZONS:
            grad,stats=frozen_gradient(core,head,h,y,scale,(Q if typ=='rotated' else None),horizon)
            gradients[horizon]=grad
            rows.append({'task':task,'controller':typ,'horizon':horizon,**stats})
        ref=gradients[12]
        for record in rows[-3:]:
            g=gradients[record['horizon']]
            record['cosine_vs_full12']=correlation(g,ref)
            record['relative_l2_error_vs_full12']=float(torch.linalg.vector_norm((g-ref).double())/torch.linalg.vector_norm(ref.double()).clamp_min(1e-15))
        for p in head.parameters():p.requires_grad_(False)
    if any(x['forward_parity_max']>2e-6 for x in rows):raise RuntimeError('PARITY_ERROR')
    if sum(p.requires_grad for p in core.parameters())!=0:raise RuntimeError('CORE_TRAINABILITY_LEAK')
    return rows,{'task':task,'batch_indices':inds,'examples':len(inds),'core_trainable':0,
       'heads_updated':0,'reference_full_gradient_is_forecast_skill':False}


def run(parent,r6zip,r9zip,work,output,smoke=False,device='cpu'):
    work=Path(work);output=Path(output);work.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    zero_test=zero_singular_backward_test()
    write_json(output/'numerical_preflight.json',zero_test)
    r6=validate_r6(r6zip);r9=validate_r9(r9zip)
    rows=[];cases=[]
    if smoke:
        cfg=Config(profile='smoke')
        a=make_synthetic_cache(cfg);b=make_synthetic_cache(cfg)
        for split in ('train','val','test'):
            a['splits'][split]['h']=a['splits'][split]['h'].mean(dim=-2)
            a['splits'][split]['y']=a['splits'][split]['y'].mean(dim=-2)
        for task,cache in [('smoke_1d',a),('smoke_2d',b)]:
            core=UnifiedModel('V2_Ported_R5',1,8,12,4).to(device)
            r,s=one_case(task,core,cache,r6zip,device,smoke=True)
            rows+=r;cases.append(s)
        status='PROGRAM_ONLY_SMOKE_NO_SCIENCE'
    else:
        if parent is None or sha(parent)!=PARENT_SHA:raise RuntimeError('MISSING_OR_CHANGED_EXACT_B0_PARENT')
        with zipfile.ZipFile(parent) as z:
            manifest,contract=check_zip(z);cfg=Config(**manifest['config'])
            for task in TASK_LIST:
                core,cache,replay=make_cache_checked(task,z,cfg,work,device)
                try:
                    r,s=one_case(task,core,cache,r6zip,device)
                except Exception as exc:
                    write_json(output/'numeric_failure_case.json',
                       {'task':task,'exception':repr(exc),'completed':[v['task'] for v in cases],
                        'status':'BLOCKED_NUMERICAL_GRADIENT_NO_TRAINING'})
                    raise
                rows+=r;s['parent_replay']=replay;cases.append(s)
                write_csv(output/'gradient_credit_horizons.csv',rows)
                write_json(output/'progress.json',{'completed':[v['task'] for v in cases]})
        status='FROZEN_GRADIENT_CREDIT_AUDIT_COMPLETED_NO_TRAINING'
    # Evaluate structural evidence, not a skill result and not an automatic permission to train.
    paired=[]
    for task in sorted(set(r['task'] for r in rows)):
        for typ in ('operator','rotated'):
            item={r['horizon']:r for r in rows if r['task']==task and r['controller']==typ}
            cs1=item[1]['cosine_vs_full12'];cs3=item[3]['cosine_vs_full12']
            paired.append({'task':task,'controller':typ,'cos_k1_full12':cs1,'cos_k3_full12':cs3,
                 'improved_cosine_3_vs_1':None if cs1 is None or cs3 is None else cs3-cs1,
                 'full_gradient_norm':item[12]['gradient_l2'],
                 'full_gradient_gpu_mem_mib':item[12]['gpu_peak_allocated_mib'],
                 'k3_gpu_mem_mib':item[3]['gpu_peak_allocated_mib']})
    gate={'status':'DIAGNOSTIC_ONLY_NO_TRAINING','candidate_for_future_3step_pilot':False,
          'reason':'Only gradient geometry on TRAIN windows; cannot authorize training/performance claims from these diagnostics',
          'comparisons':paired,'no_full_cidm_training':True,'truth_for_actions':False}
    write_csv(output/'gradient_credit_horizons.csv',rows)
    write_csv(output/'gradient_credit_pairs.csv',paired)
    write_json(output/'gradient_qualification_gate.json',gate)
    report={'experiment':'CIDM_v2_B2_R10_P0_GradientCreditHorizon','status':status,
        'training_steps':0,'data_used_for_qualification':'TRAIN only','rows':len(rows),
        'cases':cases,'gradient_horizons':list(HORIZONS),'official_full_pde_replay':not smoke,'numerical_fix':'sqrt pre-clamp, zero-forward preserved, no gradient clipping',
        'r9_evidence_sha256':sha(r9zip),'r6_evidence_sha256':sha(r6zip),
        'original_parent_sha256':PARENT_SHA,'no_outer_holdout':True,
        'explanation':'K=12 is an exact autograd reference for the same 12-step loss; it is not a trained model improvement.'}
    write_json(output/'report.json',report)
    print('[B2-R10]',status,'cases',len(cases),'records',len(rows),flush=True)
    for p in paired:print('[GRAD CREDIT]',p['task'],p['controller'],'cos_K1',p['cos_k1_full12'],'cos_K3',p['cos_k3_full12'],flush=True)
    return report
