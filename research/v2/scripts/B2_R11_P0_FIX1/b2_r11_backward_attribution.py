"""CIDM-v2 B2-R11/P0: exact frozen 12-step backward attribution, no training.

Uses the archived B2-R10 FIX2 objective, model, control heads and normalization.
Loss uses TRAIN labels only. Attribution does not implement an inference policy.
Logs (1) each controller invocation's contribution to the full parameter
  gradient, (2) each core operator's VJP contribution to latent adjoint,
  including its normalized local-control basis path, (3) exact replay.
"""
from __future__ import annotations
import math, time, json, zipfile, csv, hashlib, copy
from pathlib import Path
import torch
from benchmark import Config,Windows,UnifiedModel,make_synthetic_cache
from b2_r6_causal_spatial_controller import TASK_LIST,SEED,PARENT_SHA,make_cache_checked,orthogonal_matrix,sha
from b2_r7_frozen_controller_gain_audit import load_heads,write_csv,write_json
from b2_r1_flux_control import check_zip
import b2_r10_gradient_credit_audit as credit

PREV_FIX2_SHA='bfc54b95d54714eee6ddd1ef3f1d158418f26f8e00a5fc42faf0b8cc5eeba6bc'
LEADS=(1,3,9,12)
BUDGET=.003

def l2(v):
    return float(torch.linalg.vector_norm(v.detach().double()))

def finite_tensor(t,name):
    if not torch.isfinite(t).all():raise RuntimeError('NONFINITE_TENSOR '+name)

def concat_grad(params):
    return torch.cat([torch.zeros_like(p).reshape(-1) if p.grad is None else p.grad.detach().reshape(-1) for p in params])

def full_forward(core,head,h,truth,scale,Q=None):
    """R9/R10 exact forward with retained local adjoints and controller outputs."""
    mm=core.model
    one_d=(h.ndim==4)
    hh=h.unsqueeze(-2) if one_d else h
    B,T,C,H,W=hh.shape
    co=mm.coefficients()
    z=mm.encode(hh.reshape(B,T*C,H,W)).detach().requires_grad_(True)
    prev=hh[:,-1].detach().requires_grad_(True)
    initial=prev
    skew=mm.skew-mm.skew.T
    records=[];pred=[]
    for lead in range(1,13):
        z_old=z
        z_old.retain_grad()
        f=co['flux']*mm.flux(z_old)
        local=co['local']*torch.tanh(mm.local(z_old))
        sk=.035*torch.einsum('ij,bjhw->bihw',skew,z_old)
        base=z_old+f+local+sk
        basis,amps=credit.stable_exact_effect_basis(f,local,sk,Q)
        gates=head(base,prev,initial,amps,lead)
        raw=(gates.unsqueeze(2)*basis).sum(dim=1)/math.sqrt(3)
        rr=credit.stable_exact_rms(raw)
        change=BUDGET*credit.stable_exact_rms(base).clamp_min(1e-4)*raw/rr.clamp_min(1e-9)
        change=torch.where(rr>=1e-9,change,torch.zeros_like(change))
        z=base+change
        prev_new=prev+co['out']*torch.tanh(mm.decode(z))
        for elem in (f,local,sk,base,gates,z,prev_new):elem.retain_grad()
        records.append({'lead':lead,'z_old':z_old,'prev_old':prev,'initial':initial,
                        'f':f,'l':local,'s':sk,'base':base,'amps':amps,
                        'gates':gates,'z_new':z,'prev_new':prev_new})
        prev=prev_new
        pred.append(prev.squeeze(-2) if one_d else prev)
    pred=torch.stack(pred,dim=1)
    shape=(1,1,scale.numel())+(1,)*(pred.ndim-3)
    w=scale.reshape(shape)
    losses={j:((pred[:,j-1]-truth[:,j-1])*w.squeeze(1)).square().mean() for j in LEADS}
    loss=.5*(losses[9]+losses[12])+.125*(losses[1]+losses[3])
    return pred,loss,records

def analyze_case(task,core,cache,r6zip,device,head_types=('operator','rotated'),max_train_examples=2):
    core.requires_grad_(False).eval()
    ch=int(cache['splits']['train']['h'].shape[2]);d=int(core.model.skew.shape[0])
    head_task= 'swe_2d' if task=='smoke_2d' else ('burgers_1d' if task=='smoke_1d' else task)
    heads=load_heads(r6zip,head_task,ch,device)
    Q=orthogonal_matrix(d,32207,device,torch.float32)
    train=Windows(cache['splits']['train']['h'],cache['splits']['train']['y'],cache['mean'],cache['std'])
    ids=list(range(min(max_train_examples,len(train))))
    arr=[train[i] for i in ids]
    h=torch.stack([t[0] for t in arr]).to(device)
    y=torch.stack([t[1] for t in arr]).to(device)
    scale=train.sd.to(device)
    result=[];steps=[];layers=[]
    for name in head_types:
        head=heads[name].to(device).train()
        for p in head.parameters():p.requires_grad_(True)
        params=tuple(head.parameters())
        head.zero_grad(set_to_none=True)
        rot=Q if name=='rotated' else None
        tic=time.perf_counter()
        out,loss,records=full_forward(core,head,h,y,scale,rot)
        finite_tensor(out,'forward '+task+' '+name)
        with torch.no_grad():
            parity=float((out.detach()-credit.original_frozen_rollout(core,head,h,rotation=rot)).abs().max())
        if parity>2e-6:raise RuntimeError(f'REPLAY_MISMATCH {task} {name} {parity}')
        finite_tensor(loss,'loss '+task+' '+name)
        loss.backward()
        full=concat_grad(params).double()
        finite_tensor(full,'full_gradient '+task+' '+name)
        fullnorm=l2(full)
        pieces=[]
        state_u=[]
        # Independent FP64 local-head VJP replay. Does not change frozen forward or
        # the FP32 12-step backward adjoints. This is a numerical *diagnostic*.
        head64=copy.deepcopy(head).to(dtype=torch.float64).eval()
        params64=tuple(head64.parameters())
        pieces64=[]
        replay_error64_max=0.0
        for r in records:
            lead=r['lead']
            gg=r['gates'].grad
            if gg is None:raise RuntimeError(f'MISSING_GATE_ADJOINT {task} {name} {lead}')
            finite_tensor(gg,f'gate_adjoint_{lead}')
            # Replay frozen input to isolate direct per-call contribution to head theta.
            g_replay=head(r['base'].detach(),r['prev_old'].detach(),
                          r['initial'].detach(),r['amps'].detach(),lead)
            if not torch.allclose(g_replay.detach(),r['gates'].detach(),atol=1e-6,rtol=1e-6):
                raise RuntimeError(f'GATE_REPLAY_MISMATCH_{lead}')
            grads=torch.autograd.grad(g_replay,params,grad_outputs=gg.detach(),allow_unused=True)
            part=torch.cat([torch.zeros_like(p).flatten() if x is None else x.detach().flatten() for p,x in zip(params,grads)]).double()
            finite_tensor(part,f'head_gradient_part_{lead}')
            pieces.append(part)
            # Recompute the head Jacobian with float64 kernel arithmetic while
            # holding the ORIGINAL FP32 gate cotangent fixed. No new labels,
            # trajectory, or optimizer steps; this is NOT a full FP64 gradient.
            g64=head64(r['base'].detach().double(),r['prev_old'].detach().double(),
                       r['initial'].detach().double(),r['amps'].detach().double(),lead)
            replay_error64_max=max(replay_error64_max,float((g64.detach()-r['gates'].detach().double()).abs().max()))
            grad64=torch.autograd.grad(g64,params64,grad_outputs=gg.detach().double(),allow_unused=True)
            part64=torch.cat([torch.zeros_like(p).flatten() if x is None else x.detach().flatten()
                              for p,x in zip(params64,grad64)])
            finite_tensor(part64,f'head_gradient_part_fp64_{lead}')
            pieces64.append(part64)
            state_u.append(l2(r['z_old'].grad))
            steps.append({'task':task,'controller':name,'lead':lead,
                'gate_adjoint_l2':l2(gg),'parameter_part_l2':l2(part),
                'parameter_part_signed_projection':float(torch.dot(part,full)/max(fullnorm,1e-30)),
                'parameter_part_cosine_full':float(torch.dot(part,full)/max(l2(part)*fullnorm,1e-30)),
                'latent_state_adjoint_l2':l2(r['z_old'].grad),
                'base_adjoint_l2':l2(r['base'].grad),
                'flux_effect_adjoint_l2':l2(r['f'].grad),
                'local_effect_adjoint_l2':l2(r['l'].grad),
                'skew_effect_adjoint_l2':l2(r['s'].grad),
                'forward_parity_max':parity})
        # Three different summation modes, with per-term FP32 and FP64 replay.
        # A very small ||sum(parts)|| compared with sum(||part||) makes the
        # reconstruction ill-conditioned; never mistake that for gate passing.
        recon=torch.stack(pieces).sum(dim=0)
        recon_fp32=torch.stack([p.float() for p in pieces]).sum(dim=0).double()
        recon_head64=torch.stack(pieces64).sum(dim=0)
        reconstruction_error=l2(recon-full)/max(fullnorm,1e-30)
        fp32_sum_rel_error=l2(recon_fp32-full)/max(fullnorm,1e-30)
        fp64_head_rel_error=l2(recon_head64-full)/max(fullnorm,1e-30)
        local_fp32_fp64_rel_error=l2(recon_head64-recon)/max(l2(recon),1e-30)
        sum_piece_norms=sum(l2(p) for p in pieces)
        cancellation_condition=sum_piece_norms/max(l2(recon),1e-30)
        absolute_residual_l2=l2(recon-full)
        head_partition_status=('PASS' if reconstruction_error<=1e-4 else 'FAIL_NUMERIC_UNRESOLVED')
        # The original 1e-4 gate stays intact. We still collect per-step VJP
        # evidence for a blocked task, but NEVER report it as qualified science.
        diagnostic={'task':task,'controller':name,'original_gate_threshold':1e-4,
            'head_partition_status':head_partition_status,
            'full_gradient_l2_fp64':fullnorm,
            'head_step_reconstruction_rel_error':reconstruction_error,
            'absolute_residual_l2_fp64':absolute_residual_l2,
            'fp32_reduced_head_sum_rel_error':fp32_sum_rel_error,
            'fp64_local_head_sum_rel_error':fp64_head_rel_error,
            'fp32_vs_fp64_local_sum_rel_error':local_fp32_fp64_rel_error,
            'per_step_norms_sum':sum_piece_norms,
            'cancellation_condition':cancellation_condition,
            'head_fp64_forward_replay_max_abs':replay_error64_max,
            'warning':'Double local-head replay does not fix an unstable upstream FP32 cotangent.'}
        for r in records:
            z=r['z_old'].detach().requires_grad_(True)
            co=core.model.coefficients();mm=core.model;skew=mm.skew-mm.skew.T
            f=co['flux']*mm.flux(z)
            local=co['local']*torch.tanh(mm.local(z))
            sk=.035*torch.einsum('ij,bjhw->bihw',skew,z)
            gf,gl,gs=(r[k].grad.detach() for k in ('f','l','s'))
            gbase=r['base'].grad.detach()
            vjp_f,=torch.autograd.grad(f,z,gf,retain_graph=True)
            vjp_l,=torch.autograd.grad(local,z,gl,retain_graph=True)
            vjp_s,=torch.autograd.grad(sk,z,gs,retain_graph=True)
            components=[gbase.double(),vjp_f.double(),vjp_l.double(),vjp_s.double()]
            recovered=sum(components)
            # Critical: z_old participates in the *previous* step's decoder branch,
            # so z_old.grad of the original full trajectory includes additional
            # graph shortcuts. It is NOT the local transition VJP reference.
            # Use an independently evaluated local VJP with the correct
            # f/l/sk direct-basis adjoints minus their already counted base path.
            other=torch.autograd.grad(
                (z+f+local+sk,f,local,sk),z,
                grad_outputs=(gbase,gf-gbase,gl-gbase,gs-gbase),allow_unused=False)[0]
            actual=other.detach().double()
            relative=l2(recovered-actual)/max(sum(l2(t) for t in components),1e-30)
            if relative>1e-3:raise RuntimeError(f'OPERATOR_VJP_PARTITION_FAILED {task} {name} lead={r["lead"]} relative={relative}')
            layers.append({'task':task,'controller':name,'lead':r['lead'],
                'direct_identity_vjp_l2':l2(gbase),
                'flux_vjp_l2':l2(vjp_f),'local_vjp_l2':l2(vjp_l),
                'skew_vjp_l2':l2(vjp_s),
                'latent_vjp_sum_l2':l2(recovered),'latent_local_reference_vjp_l2':l2(actual),'original_graph_latent_adjoint_l2':l2(r['z_old'].grad),
                'local_vjp_partition_error':relative})
        values=[l2(v) for v in pieces]
        key_idx=sorted(range(12),key=lambda i:values[i],reverse=True)
        result.append({'task':task,'controller':name,'examples':len(ids),'TRAIN_indices':ids,
            **diagnostic,
            'loss':float(loss.detach()),'forward_parity_max':parity,
            'full_gradient_l2_fp64':fullnorm,
            'head_step_reconstruction_rel_error':reconstruction_error,
            'top_step_by_norm':int(key_idx[0]+1),'top_step_gradient_l2':values[key_idx[0]],
            'early_steps_1_3_sum_norm':float(sum(values[:3])),
            'late_steps_9_12_sum_norm':float(sum(values[8:])),
            'elapsed_s':time.perf_counter()-tic})
        for p in head.parameters():p.requires_grad_(False)
    return result,steps,layers

def run(parent,r6zip,r9zip,fix2zip,work,output,smoke=False,device='cpu'):
    work=Path(work);output=Path(output);work.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    if sha(fix2zip)!=PREV_FIX2_SHA:raise RuntimeError('B2R10_FIX2_EXACT_EVIDENCE_SHA_MISMATCH')
    with zipfile.ZipFile(fix2zip) as z:
        if z.testzip():raise RuntimeError('B2R10_FIX2_EVIDENCE_CRC_ERROR')
        rep=json.loads(z.read('report.json'))
        if rep['status']!='FROZEN_GRADIENT_CREDIT_AUDIT_COMPLETED_NO_TRAINING' or rep['rows']!=18:
            raise RuntimeError('B2R10_FIX2_EVIDENCE_NOT_FULL')
        gate=json.loads(z.read('gradient_qualification_gate.json'))
        if gate['numerical_status']!='SEVERE_GRADIENT_AMPLIFICATION_NO_TRAINING':raise RuntimeError('FIX2_EVIDENCE_NOT_MATCHED')
    credit.validate_r6(r6zip);credit.validate_r9(r9zip)
    rows=[];perlead=[];ops=[];cases=[]
    if smoke:
        cfg=Config(profile='smoke')
        a=make_synthetic_cache(cfg);b=make_synthetic_cache(cfg)
        for split in ('train','val','test'):
            a['splits'][split]['h']=a['splits'][split]['h'].mean(dim=-2)
            a['splits'][split]['y']=a['splits'][split]['y'].mean(dim=-2)
            a['splits'][split]['h']=a['splits'][split]['h'][...,:4]
            a['splits'][split]['y']=a['splits'][split]['y'][...,:4]
            b['splits'][split]['h']=b['splits'][split]['h'][...,:2,:4]
            b['splits'][split]['y']=b['splits'][split]['y'][...,:2,:4]
        for task,cache in [('smoke_1d',a),('smoke_2d',b)]:
            core=UnifiedModel('V2_Ported_R5',1,8,12,4).to(device)
            res,st,op=analyze_case(task,core,cache,r6zip,device,
                        head_types=('operator',) if task=='smoke_1d' else ('rotated',),
                        max_train_examples=1)
            rows.extend(res);perlead.extend(st);ops.extend(op)
        status='PROGRAM_ONLY_SMOKE_NO_SCIENCE'
    else:
        if parent is None or sha(parent)!=PARENT_SHA:raise RuntimeError('PARENT_SHA_MISMATCH')
        with zipfile.ZipFile(parent) as z:
            manifest,contract=check_zip(z);cfg=Config(**manifest['config'])
            for task in TASK_LIST:
                core,cache,replay=make_cache_checked(task,z,cfg,work,device)
                try:
                    res,st,op=analyze_case(task,core,cache,r6zip,device)
                except Exception as e:
                    write_json(output/'failure_case.json',{'task':task,'exception':repr(e),'already_completed':cases})
                    raise
                rows.extend(res);perlead.extend(st);ops.extend(op)
                cases.append({'task':task,'parent_replay':replay,'rows':len(res),'exact_model_frozen':True})
                write_csv(output/'temporal_gradient_contributions.csv',perlead)
                write_csv(output/'operator_local_vjp.csv',ops)
                write_json(output/'progress.json',{'completed':[x['task'] for x in cases]})
        if any(r['head_partition_status']!='PASS' for r in rows):
            status='FROZEN_ATTRIBUTION_DATA_COLLECTED_NUMERIC_GATE_FAIL_NO_TRAINING'
        else:
            status='FROZEN_BACKWARD_PATH_ATTRIBUTION_COMPLETED_NO_TRAINING' 
    if not smoke and (len(rows)!=6 or len(perlead)!=72 or len(ops)!=72):
        raise RuntimeError(f'INCOMPLETE_BACKWARD_PATH_AUDIT_{len(rows)}_{len(perlead)}_{len(ops)}')
    write_csv(output/'temporal_gradient_contributions.csv',perlead)
    write_csv(output/'operator_local_vjp.csv',ops)
    write_csv(output/'case_summary.csv',rows)
    risk=[r for r in rows if r['full_gradient_l2_fp64']>1e8]
    unresolved=[r for r in rows if r['head_partition_status']!='PASS']
    gate={'status':'DIAGNOSTIC_ONLY_NO_TRAINING','completed_groups':len(rows),
          'gradient_large_cases':risk,'head_partition_unresolved':unresolved,
          'strict_numerical_gate_pass':len(unresolved)==0,'train_allowed':False,
          'note':'Relative VJP norms and parameter contribution amplitudes are diagnostic, not causal branch ablations or Jacobian spectral norms.'}
    write_json(output/'gradient_path_gate.json',gate)
    report={'experiment':'CIDM_v2_B2_R11_P0_BackwardPathAttribution',
            'status':status,'cases':cases,'number_of_controllers':len(rows),
            'temporal_rows':len(perlead),'vjp_rows':len(ops),
            'original_fix2_sha256':PREV_FIX2_SHA,
            'training_updates':0,'no_outer_holdout':True,'train_allowed':False,
            'strict_numerical_gate_pass':len(unresolved)==0,
            'attribution':'exact decomposition of the frozen head gradient and local VJPs; no weights changed'}
    write_json(output/'report.json',report)
    return report
