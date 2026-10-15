"""B2-R13/P0: frozen, SAME-STATE vs OWN-TRAJECTORY FP32/FP64 local VJPs.

Full frozen 12-step forward is used only to harvest step-local states and
loss-conditioned cotangents. All derivative probes below are independent,
zero-training, step-local Jacobian-vector products with original CIDM-v2 code.
No BPTT/training Gate, no causal operator ablation, no Jacobian spectral claim.
"""
from __future__ import annotations
import csv, copy, gc, json, math, zipfile, hashlib
from pathlib import Path
import torch
import b2_r11_backward_attribution as r11
import b2_r12_dual_precision_chain as r12
import b2_r10_gradient_credit_audit as credit
from b2_r1_flux_control import check_zip

R12_FIX1_SHA='267f8d574d98754f864227cc28e49c335035a0697ec5f0a112117218c4c5d28a'
LEADS=(1,3,6,9,12)

def vecnorm(x):
    return float(torch.linalg.vector_norm(x.detach().reshape(-1).double()))

def unit_cot(x):
    y=x.detach().double(); val=torch.linalg.vector_norm(y)
    if not torch.isfinite(val) or val<=0:raise RuntimeError('INVALID_ADJOINT_FOR_UNIT_PROBE')
    return (y/val).detach().cpu()

def finite(x,label):
    if not bool(torch.isfinite(x).all()):raise RuntimeError('NONFINITE_'+label)

def angle(a,b):
    v=(a.flatten().double()*b.flatten().double()).sum(); d=torch.linalg.vector_norm(a.double())*torch.linalg.vector_norm(b.double())
    return float(v/d.clamp_min(1e-300)) if d>0 else 0.

def relative(a,b):
    return vecnorm(a.double()-b.double())/max(vecnorm(b),1e-300)

def one_step(core,head,z,p,init,Q,lead):
    """Mathematically identical to R11.full_forward for one controlled step."""
    mm=core.model; co=mm.coefficients();skew=mm.skew-mm.skew.T
    f=co['flux']*mm.flux(z)
    l=co['local']*torch.tanh(mm.local(z))
    s=.035*torch.einsum('ij,bjhw->bihw',skew,z)
    base=z+f+l+s
    basis,amps=credit.stable_exact_effect_basis(f,l,s,Q)
    gates=head(base,p,init,amps,lead)
    raw=(gates.unsqueeze(2)*basis).sum(dim=1)/math.sqrt(3.)
    rr=credit.stable_exact_rms(raw)
    change=r11.BUDGET*credit.stable_exact_rms(base).clamp_min(1e-4)*raw/rr.clamp_min(1e-9)
    change=torch.where(rr>=1e-9,change,torch.zeros_like(change))
    zn=base+change
    pn=p+co['out']*torch.tanh(mm.decode(zn))
    return zn,pn,f,l,s

def evaluate_step(core,head,z,p,init,Q,lead,cz,cp):
    """Return both independent local VJPs wrt z_old; p held fixed."""
    dt=next(core.parameters()).dtype
    z=z.to(dtype=dt).detach().requires_grad_(True)
    p=p.to(dtype=dt).detach();init=init.to(dtype=dt).detach()
    cz=cz.to(dtype=dt).detach();cp=cp.to(dtype=dt).detach()
    q=Q.to(dtype=dt).detach() if Q is not None else None
    zn,pn,f,l,s=one_step(core,head,z,p,init,q,lead)
    for k,val in [('zn',zn),('pn',pn),('f',f),('l',l),('s',s)]:finite(val,k)
    grad_z=torch.autograd.grad(zn,z,grad_outputs=cz,retain_graph=True)[0]
    grad_p=torch.autograd.grad(pn,z,grad_outputs=cp,retain_graph=False)[0]
    for k,val in [('vjp_z',grad_z),('vjp_p',grad_p)]:finite(val,k)
    return {'latent':grad_z.detach().cpu().double(),'decoder':grad_p.detach().cpu().double(),
            'new_z':zn.detach().cpu().double(),'new_p':pn.detach().cpu().double()}

def states_and_adjoint(core,head,h,y,sd,Q,precision):
    dt=torch.float32 if precision=='fp32' else torch.float64
    h=h.to(dtype=dt);y=y.to(dtype=dt);sd=sd.to(dtype=dt)
    q=Q.to(dtype=dt) if Q is not None else None
    pred,loss,recs=r11.full_forward(core,head,h,y,sd,q)
    finite(pred,'forward_'+precision)
    if precision=='fp32':
        # same real training objective, two independent VJP probes; no parameter update.
        req=[r[k] for r in recs for k in ('z_new','prev_new')]
        g=torch.autograd.grad(loss,req,allow_unused=True,retain_graph=False)
        if any(v is None for v in g):raise RuntimeError('MISSING_FULL_CHAIN_ADJOINT')
    else:g=None
    out=[]
    for t,rec in enumerate(recs):
        d={k:rec[k].detach().cpu().clone() for k in ('z_old','prev_old','initial','z_new','prev_new')}
        if g is not None:
            d['cot_z']=unit_cot(g[t*2]);d['cot_p']=unit_cot(g[t*2+1])
        out.append(d)
    pred=pred.detach().cpu().double()
    del recs,g,loss
    gc.collect()
    return pred,out

def case(task,core,cache,r6zip,kind,number_of_windows=2,leads=LEADS):
    train=r11.Windows(cache['splits']['train']['h'],cache['splits']['train']['y'],cache['mean'],cache['std'])
    ch=int(cache['splits']['train']['h'].shape[2]);d=int(core.model.skew.shape[0])
    head_task= 'burgers_1d' if task=='smoke_1d' else ('swe_2d' if task=='smoke_2d' else task)
    head=r11.load_heads(r6zip,head_task,ch,'cpu')[kind].eval().requires_grad_(False)
    Q=r11.orthogonal_matrix(d,32207,'cpu',torch.float32) if kind=='rotated' else None
    c32=copy.deepcopy(core).cpu().float().eval().requires_grad_(False)
    c64=copy.deepcopy(core).cpu().double().eval().requires_grad_(False)
    h32=copy.deepcopy(head).cpu().float().eval().requires_grad_(True)
    h64=copy.deepcopy(head).cpu().double().eval().requires_grad_(True)
    rows=[];windows=[]
    for i in range(min(number_of_windows,len(train))):
        h,y=train[i];h=h.unsqueeze(0).cpu();y=y.unsqueeze(0).cpu();sd=train.sd.cpu()
        p32,s32=states_and_adjoint(c32,h32,h,y,sd,Q,'fp32')
        p64,s64=states_and_adjoint(c64,h64,h,y,sd,Q,'fp64')
        div=relative(p32,p64);maxdiv=float((p32-p64).abs().max())
        windows.append({'task':task,'controller':kind,'window_index':i,'fp32_fp64_forecast_rel_l2':div,'fp32_fp64_forecast_max_abs':maxdiv})
        for t in leads:
            a=s32[t-1];b=s64[t-1]
            # true FP32 full-chain cotangent: normalized and frozen before local VJP
            # random control: independent of future labels, deterministic per window and lead
            rr=torch.Generator(device='cpu').manual_seed(13013+i*101+t*19+(1 if kind=='rotated' else 0))
            rv=torch.randn(a['cot_z'].shape,generator=rr,dtype=torch.float64)
            rv/=torch.linalg.vector_norm(rv).clamp_min(1e-300)
            rp=torch.randn(a['cot_p'].shape,generator=rr,dtype=torch.float64)
            rp/=torch.linalg.vector_norm(rp).clamp_min(1e-300)
            for probe,cz,cp in (('frozen_loss_adjoint',a['cot_z'],a['cot_p']),('random_isotropic',rv,rp)):
                native32=evaluate_step(c32,h32,a['z_old'],a['prev_old'],a['initial'],Q,t,cz,cp)
                shared64=evaluate_step(c64,h64,a['z_old'],a['prev_old'],a['initial'],Q,t,cz,cp)
                own64=evaluate_step(c64,h64,b['z_old'],b['prev_old'],b['initial'],Q,t,cz,cp)
                # Strict same-dtype independent reproduction of the *actual* archived
                # 12-step rollout intermediates before interpreting any VJP.
                parity32=max(float((native32['new_z']-a['z_new'].double()).abs().max()),
                             float((native32['new_p']-a['prev_new'].double()).abs().max()))
                parity64=max(float((own64['new_z']-b['z_new'].double()).abs().max()),
                             float((own64['new_p']-b['prev_new'].double()).abs().max()))
                if parity32>2e-6 or parity64>2e-11:
                    raise RuntimeError(f'STRICT_LOCAL_FORWARD_REPLAY_FAIL {task} {kind} window={i} lead={t} fp32={parity32} fp64={parity64}')
                row={'local_forward_parity_fp32_max_abs':parity32,'local_forward_parity_fp64_max_abs':parity64,'task':task,'controller':kind,'train_window':i,'lead':t,'probe':probe,
                   'relative_state_deviation_fp32_vs_fp64':relative(a['z_old'],b['z_old']),
                   'relative_observable_deviation_fp32_vs_fp64':relative(a['prev_old'],b['prev_old']),
                   'fp32_fp64_forecast_rel_l2':div,
                   'shared_state_one_step_latent_rel_diff':relative(native32['new_z'],shared64['new_z']),
                   'shifted_state_one_step_latent_rel_diff':relative(shared64['new_z'],own64['new_z']),
                   'shared_state_one_step_observable_rel_diff':relative(native32['new_p'],shared64['new_p']),
                   'shifted_state_one_step_observable_rel_diff':relative(shared64['new_p'],own64['new_p'])}
                for branch in ('latent','decoder'):
                    x0=native32[branch];xs=shared64[branch];xo=own64[branch]
                    row[branch+'_vjp_fp32_l2']=vecnorm(x0)
                    row[branch+'_vjp_shared_fp64_l2']=vecnorm(xs)
                    row[branch+'_vjp_native_fp64_l2']=vecnorm(xo)
                    row[branch+'_same_state_precision_rel_diff']=relative(x0,xs)
                    row[branch+'_same_state_precision_cosine']=angle(x0,xs)
                    row[branch+'_trajectory_shift_rel_diff']=relative(xs,xo)
                    row[branch+'_trajectory_shift_cosine']=angle(xs,xo)
                rows.append(row)
    return rows,windows

def write_csv(path,rows):
    if not rows:return
    with open(path,'w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False,default=str),encoding='utf-8')

def run(parent,r6zip,r9zip,fix2zip,r11fix1zip,r12zip,work,output,smoke=False):
    work=Path(work);output=Path(output);work.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    if r11.sha(r12zip)!=R12_FIX1_SHA:raise RuntimeError('R12_FIXED_OFFICIAL_REPORT_SHA_MISMATCH')
    with zipfile.ZipFile(r12zip) as z:
        if z.testzip():raise RuntimeError('R12_REPORT_CRC_ERROR')
        rep=json.loads(z.read('report.json'));gate=json.loads(z.read('precision_gate.json'))
        if rep['status']!='FROZEN_DUAL_PRECISION_FIX1_COMPLETED_NO_TRAINING' or rep['case_rows']!=12 or rep['time_rows']!=144 or not gate['strict_partition_pass_all']:
            raise RuntimeError('R12_PRECISION_EVIDENCE_CONTRACT_FAILED')
    if r11.sha(r11fix1zip)!=r12.ARCHIVE_FIX1_SHA:raise RuntimeError('R11_EVIDENCE_SHA_FAILED')
    if r11.sha(fix2zip)!=r11.PREV_FIX2_SHA:raise RuntimeError('R10_EVIDENCE_SHA_FAILED')
    r11.credit.validate_r6(r6zip);r11.credit.validate_r9(r9zip)
    rows=[];wrows=[];cases=[]
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
                core=r11.UnifiedModel('V2_Ported_R5',1,8,12,4).cpu()
                a,b=case(task,core,cache,r6zip,kind,1,(1,6,12));rows.extend(a);wrows.extend(b)
                cases.append({'task':task,'controller':kind,'program_only':True})
            status='PROGRAM_ONLY_SMOKE_NO_SCIENCE'
        else:
            if parent is None or r11.sha(parent)!=r11.PARENT_SHA:raise RuntimeError('ORIGINAL_PARENT_SHA_FAILED')
            with zipfile.ZipFile(parent) as z:
                manifest,_=check_zip(z);cfg=r11.Config(**manifest['config'])
                for task in r11.TASK_LIST:
                    core,cache,replay=r11.make_cache_checked(task,z,cfg,work,'cpu')
                    for kind in ('operator','rotated'):
                        a,b=case(task,core,cache,r6zip,kind)
                        rows.extend(a);wrows.extend(b)
                        cases.append({'task':task,'controller':kind,'parent_replay':replay})
                        write_csv(output/'same_state_vjp.csv',rows)
                        write_csv(output/'window_forward_drift.csv',wrows)
                        write_json(output/'progress.json',{'completed':cases,'rows':len(rows)})
            if len(rows)!=3*2*2*len(LEADS)*2 or len(cases)!=6:
                raise RuntimeError('INCOMPLETE_OFFICIAL_R13_CASES')
            status='FROZEN_SAME_STATE_VJP_COMPLETED_NO_TRAINING'
    except Exception as e:
        write_csv(output/'same_state_vjp.csv',rows);write_csv(output/'window_forward_drift.csv',wrows)
        write_json(output/'failure_details.json',{'error':repr(e),'groups_completed':cases,'rows':len(rows)})
        raise
    write_csv(output/'same_state_vjp.csv',rows);write_csv(output/'window_forward_drift.csv',wrows)
    summary=[]
    for task in sorted({r['task'] for r in rows}):
        for kind in sorted({r['controller'] for r in rows if r['task']==task}):
            for probe in ('frozen_loss_adjoint','random_isotropic'):
                ar=[r for r in rows if r['task']==task and r['controller']==kind and r['probe']==probe]
                if not ar:continue
                for branch in ('latent','decoder'):
                    f=lambda key:sum(float(x[key]) for x in ar)/len(ar)
                    summary.append({'task':task,'controller':kind,'probe':probe,'branch':branch,
                         'mean_same_state_precision_rel_diff':f(branch+'_same_state_precision_rel_diff'),
                         'mean_trajectory_shift_rel_diff':f(branch+'_trajectory_shift_rel_diff'),
                         'mean_same_state_precision_cosine':f(branch+'_same_state_precision_cosine'),
                         'mean_trajectory_shift_cosine':f(branch+'_trajectory_shift_cosine'),
                         'row_count':len(ar)})
    write_csv(output/'vjp_summary.csv',summary)
    gate={'status':'DIAGNOSTIC_ONLY_NO_TRAINING','completed_groups':len(cases),
          'sample_probe_rows':len(rows),'train_allowed':False,
          'interpretation':'same-state VJP error isolates local precision dependence at fixed FP32 state and cotangent; shifted-state VJP difference captures sensitivity to FP32-vs-FP64 trajectory changes; not causal proof or a Jacobian spectrum bound',
          'source_report_SHA256':R12_FIX1_SHA}
    write_json(output/'same_state_gate.json',gate)
    report={'experiment':'CIDM_v2_B2_R13_P0_SameStateLocalJacobian','status':status,'cases':cases,
        'rows':len(rows),'windows':len(wrows),'updates':0,'train_allowed':False,
        'dataset_note':'official original TRAIN windows 0 and 1; previously used distribution, no fresh outer holdout'}
    write_json(output/'report.json',report)
    return report
