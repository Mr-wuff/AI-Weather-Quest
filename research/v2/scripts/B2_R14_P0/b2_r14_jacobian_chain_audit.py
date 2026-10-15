"""B2-R14/P0: Frozen 12-step adjoint transport along precision-controlled trajectories.

Three strictly comparable linearization protocols:
A: FP32 Jacobians evaluated on genuine FP32 rollout states;
B: FP64 Jacobians evaluated on CAST FP32 states (counterfactual, not a rollout);
C: FP64 Jacobians evaluated on genuine FP64 rollout states.

Same frozen terminal cotangent for all three. Per-step normalization keeps VJP
linear transport free from FP32 overflow while summing exact log10 gains.
Initial-condition input to the head is deliberately HELD FIXED: this is recurrent
state (z,p) transport, not parameter gradients or the full model Jacobian.
All future targets enter only an offline TRAIN terminal-loss probe.
"""
from __future__ import annotations
import copy, math, csv, json, zipfile, gc
from pathlib import Path
import torch
import b2_r13_same_state_vjp as r13
import b2_r11_backward_attribution as r11
from b2_r1_flux_control import check_zip

R13_OFFICIAL_SHA='a8261f9a777066795cd2004b3b0e4cd3018792082c4bec9dd3d02526cc7f5833'
R12_OFFICIAL_SHA=r13.R12_FIX1_SHA
TASKS=r11.TASK_LIST


def pyfloat(x):
    q=float(x)
    if not math.isfinite(q):raise RuntimeError('NONFINITE_DIAGNOSTIC_SCALAR')
    return q

def normpair(z,p):
    a=torch.linalg.vector_norm(z.detach().double())
    b=torch.linalg.vector_norm(p.detach().double())
    return float(torch.sqrt(a*a+b*b))

def normalize_pair(z,p):
    v=normpair(z,p)
    if not math.isfinite(v) or v<=1e-280:raise RuntimeError('INVALID_COTANGENT_MAGNITUDE')
    return z/v,p/v,v

def paircos(a,b):
    x,y=a;u,v=b
    dot=torch.sum(x.double()*u.double())+torch.sum(y.double()*v.double())
    den=normpair(x,y)*normpair(u,v)
    return pyfloat(dot/den) if den>0 else 0.0

def pairreldiff(a,b):
    x,y=a;u,v=b
    return normpair(x.double()-u.double(),y.double()-v.double())/max(normpair(u,v),1e-280)

def harvest(core,head,h,Q):
    dt=next(core.parameters()).dtype
    h=h.to(dtype=dt)
    hh=h.unsqueeze(-2) if h.ndim==4 else h
    B,T,C,H,W=hh.shape
    with torch.no_grad():
        z=core.model.encode(hh.reshape(B,T*C,H,W))
        p=hh[:,-1]
        init=p.detach().clone()
        records=[]
        for lead in range(1,13):
            a={'z_old':z.detach().cpu().clone(),'p_old':p.detach().cpu().clone()}
            z,p,*_=r13.one_step(core,head,z,p,init,(Q.to(dtype=dt) if Q is not None else None),lead)
            if not bool(torch.isfinite(z).all() and torch.isfinite(p).all()):
                raise RuntimeError('NONFINITE_ORIGINAL_FORWARD')
            a['z_new']=z.detach().cpu().clone();a['p_new']=p.detach().cpu().clone()
            records.append(a)
    return records,init.detach().cpu().clone()

def vjp_step(core,head,zold,pold,init,Q,lead,gz,gp,expected=None):
    dt=next(core.parameters()).dtype
    z=zold.to(dtype=dt).detach().requires_grad_(True)
    p=pold.to(dtype=dt).detach().requires_grad_(True)
    ini=init.to(dtype=dt).detach()
    q=Q.to(dtype=dt).detach() if Q is not None else None
    zn,pn,*_=r13.one_step(core,head,z,p,ini,q,lead)
    if expected is not None:
        ea,eb=expected
        par=max(float((zn.detach().double()-ea.double()).abs().max()),
                float((pn.detach().double()-eb.double()).abs().max()))
        tol=2e-6 if dt==torch.float32 else 2e-11
        if par>tol:raise RuntimeError(f'LOCAL_FORWARD_PARITY_FAIL t={lead} dtype={dt} error={par}')
    g=torch.autograd.grad((zn,pn),(z,p),grad_outputs=(gz.to(dtype=dt),gp.to(dtype=dt)),retain_graph=False)
    if not all(bool(torch.isfinite(v).all()) for v in g):raise RuntimeError(f'NONFINITE_LOCAL_VJP {lead} {dt}')
    return g

def terminal_probes(records,y,scale,seed):
    z=records[-1]['z_new'].double();p=records[-1]['p_new'].double()
    g=torch.Generator(device='cpu').manual_seed(seed)
    cz=torch.randn(z.shape,dtype=torch.float64,generator=g)
    cp=torch.randn(p.shape,dtype=torch.float64,generator=g)
    cz,cp,_=normalize_pair(cz,cp)
    y12=y[11].double().unsqueeze(0)
    if y12.ndim==3:y12=y12.unsqueeze(-2)
    w=scale.double().reshape(1,-1,1,1)
    resid=(p-y12)*w
    cp_loss=(2/resid.numel())*resid*w
    cz_loss=torch.zeros_like(z)
    cz_loss,cp_loss,_=normalize_pair(cz_loss,cp_loss)
    return [('random_terminal',cz,cp),('physical_lead12_terminal',cz_loss,cp_loss)]

def analyze_case(task,core,cache,r6zip,kind,windows=2,steps=12):
    train=r11.Windows(cache['splits']['train']['h'],cache['splits']['train']['y'],cache['mean'],cache['std'])
    ch=int(cache['splits']['train']['h'].shape[2]);d=int(core.model.skew.shape[0])
    head_task={'smoke_1d':'burgers_1d','smoke_2d':'swe_2d'}.get(task,task)
    head=r11.load_heads(r6zip,head_task,ch,'cpu')[kind].eval().requires_grad_(False)
    c32=copy.deepcopy(core).cpu().float().eval().requires_grad_(False)
    c64=copy.deepcopy(core).cpu().double().eval().requires_grad_(False)
    h32=copy.deepcopy(head).cpu().float().eval().requires_grad_(False)
    h64=copy.deepcopy(head).cpu().double().eval().requires_grad_(False)
    Q=r11.orthogonal_matrix(d,32207,'cpu',torch.float32) if kind=='rotated' else None
    results=[];summaries=[];drifts=[]
    for ix in range(min(windows,len(train))):
        h,y=train[ix];h=h.unsqueeze(0).cpu();y=y.cpu()
        path32,init32=harvest(c32,h32,h,Q)
        path64,init64=harvest(c64,h64,h,Q)
        end32=path32[-1];end64=path64[-1]
        forward_rel=normpair(end32['z_new'].double()-end64['z_new'].double(),end32['p_new'].double()-end64['p_new'].double()) / max(normpair(end64['z_new'],end64['p_new']),1e-280)
        drifts.append({'task':task,'controller':kind,'window':ix,'final_joint_state_rel_fp32_fp64':pyfloat(forward_rel),
                       'final_z_rel':r13.relative(end32['z_new'],end64['z_new']),
                       'final_p_rel':r13.relative(end32['p_new'],end64['p_new'])})
        for probe,cz,cp in terminal_probes(path32,y,train.sd,seed=14413+ix*109+(1 if kind=='rotated' else 0)):
            modes={
              'fp32_own':(c32,h32,path32,init32),
              'fp64_fixed_fp32_state':(c64,h64,path32,init32),
              'fp64_own':(c64,h64,path64,init64)}
            active={mode:(cz.clone(),cp.clone(),0.) for mode in modes}
            for lead in range(12,12-steps,-1):
                fresh={}
                for mode,(cc,headx,path,ini) in modes.items():
                    rec=path[lead-1]
                    g_z,g_p,old_log=active[mode]
                    expected=(rec['z_new'],rec['p_new']) if mode!='fp64_fixed_fp32_state' else None
                    a,b=vjp_step(cc,headx,rec['z_old'],rec['p_old'],ini,Q,lead,g_z,g_p,expected)
                    az,bp,step_gain=normalize_pair(a,b)
                    log_gain=math.log10(step_gain)
                    fresh[mode]=(az.detach().double(),bp.detach().double(),old_log+log_gain)
                    results.append({'task':task,'controller':kind,'window':ix,'probe':probe,'lead':lead,'mode':mode,
                                    'local_gain_log10':pyfloat(log_gain),
                                    'cumulative_gain_log10':pyfloat(old_log+log_gain),
                                    'z_adjoint_relative_share':pyfloat(r13.vecnorm(az)/max(normpair(az,bp),1e-300)),
                                    'forward_state_rel_fp32_fp64':r13.relative(path32[lead-1]['z_old'],path64[lead-1]['z_old'])})
                # record the comparison of NORMALIZED cotangent directions, separate from amplification magnitude
                x=(fresh['fp32_own'][0],fresh['fp32_own'][1])
                y_fixed=(fresh['fp64_fixed_fp32_state'][0],fresh['fp64_fixed_fp32_state'][1])
                y_native=(fresh['fp64_own'][0],fresh['fp64_own'][1])
                summaries.append({'task':task,'controller':kind,'window':ix,'probe':probe,'lead':lead,
                                  'same_state_direction_cosine':paircos(x,y_fixed),
                                  'same_state_direction_rel_diff':pairreldiff(x,y_fixed),
                                  'shifted_state_direction_cosine':paircos(y_fixed,y_native),
                                  'shifted_state_direction_rel_diff':pairreldiff(y_fixed,y_native),
                                  'same_state_log10_gain_difference':pyfloat(fresh['fp32_own'][2]-fresh['fp64_fixed_fp32_state'][2]),
                                  'shifted_state_log10_gain_difference':pyfloat(fresh['fp64_fixed_fp32_state'][2]-fresh['fp64_own'][2]),
                                  'fp32_chain_log10_gain':pyfloat(fresh['fp32_own'][2]),
                                  'fp64_fixed_state_chain_log10_gain':pyfloat(fresh['fp64_fixed_fp32_state'][2]),
                                  'fp64_own_chain_log10_gain':pyfloat(fresh['fp64_own'][2])})
                active=fresh
    return results,summaries,drifts

def independent_chain_regression(core,cache,r6zip,task,kind):
    """Independent full-graph autograd vs 12 normalized one-step VJPs, fixed init."""
    train=r11.Windows(cache['splits']['train']['h'],cache['splits']['train']['y'],cache['mean'],cache['std'])
    h,_=train[0]; h=h.unsqueeze(0)
    ch=int(cache['splits']['train']['h'].shape[2])
    htask={'smoke_1d':'burgers_1d','smoke_2d':'swe_2d'}.get(task,task)
    head=r11.load_heads(r6zip,htask,ch,'cpu')[kind].double().eval().requires_grad_(False)
    cc=copy.deepcopy(core).cpu().double().eval().requires_grad_(False)
    dim=int(cc.model.skew.shape[0]);Q=r11.orthogonal_matrix(dim,32207,'cpu',torch.float64) if kind=='rotated' else None
    states,init=harvest(cc,head,h,Q)
    g=torch.Generator(device='cpu').manual_seed(514)
    cz=torch.randn(states[-1]['z_new'].shape,dtype=torch.float64,generator=g)
    cp=torch.randn(states[-1]['p_new'].shape,dtype=torch.float64,generator=g)
    cz,cp,_=normalize_pair(cz,cp)
    z0=states[0]['z_old'].double().requires_grad_(True)
    p0=states[0]['p_old'].double().requires_grad_(True)
    z=z0;p=p0
    for t in range(1,13):z,p,*_=r13.one_step(cc,head,z,p,init.double().detach(),Q,t)
    gg=torch.autograd.grad((z,p),(z0,p0),grad_outputs=(cz,cp))
    reference=(gg[0].detach().double(),gg[1].detach().double())
    active=(cz,cp);loggain=0.
    for lead in range(12,0,-1):
        a,b=vjp_step(cc,head,states[lead-1]['z_old'],states[lead-1]['p_old'],init,Q,lead,*active)
        az,bp,stepgain=normalize_pair(a,b)
        loggain+=math.log10(stepgain)
        active=(az.detach().double(),bp.detach().double())
    fullgain=math.log10(normpair(*reference))
    err=abs(loggain-fullgain);cos=paircos(active,reference)
    if err>1e-8 or cos<1-1e-10:raise RuntimeError(f'CHAIN_JVP_VS_FULL_AUTOGRAD_MISMATCH {task} {kind} logerror={err} cos={cos}')
    return {'task':task,'controller':kind,'log10_amplitude_error':err,'direction_cosine':cos,'status':'PASS'}


def write_csv(path,items):
    if not items:return
    with open(path,'w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=list(items[0]));wr.writeheader();wr.writerows(items)

def json_write(path,x):
    Path(path).write_text(json.dumps(x,ensure_ascii=False,allow_nan=False,indent=2,default=str))

def verify_archive(zipp):
    if r11.sha(zipp)!=R13_OFFICIAL_SHA:raise RuntimeError('R13_OFFICIAL_SHA_MISMATCH')
    with zipfile.ZipFile(zipp) as z:
        if z.testzip() is not None:raise RuntimeError('R13_ZIP_CRC_FAILED')
        rep=json.loads(z.read('report.json'));gate=json.loads(z.read('same_state_gate.json'))
        assert rep['status']=='FROZEN_SAME_STATE_VJP_COMPLETED_NO_TRAINING' and rep['rows']==120 and rep['updates']==0
        assert gate['completed_groups']==6 and not gate['train_allowed']
        manifest=json.loads(z.read('FILE_SHA256_MANIFEST.json'))
        for entry in manifest:
            payload=z.read(entry['path'])
            if r11.hashlib.sha256(payload).hexdigest()!=entry['sha256'] or len(payload)!=entry['bytes']:
                raise RuntimeError('R13_INTERNAL_SHA_MISMATCH '+entry['path'])
    return True

def run(parent,r6zip,r9zip,fix2zip,r11fix1zip,r12zip,r13zip,work,output,smoke=False):
    work=Path(work);output=Path(output)
    work.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    verify_archive(r13zip)
    # all inherited evidence contracts run through exact original SHA-locked code
    if r11.sha(r12zip)!=R12_OFFICIAL_SHA:raise RuntimeError('R12_OFFICIAL_SHA_ERROR')
    if r11.sha(r11fix1zip)!=r13.r12.ARCHIVE_FIX1_SHA:raise RuntimeError('R11_SOURCE_EVIDENCE_ERROR')
    if r11.sha(fix2zip)!=r11.PREV_FIX2_SHA:raise RuntimeError('R10_SOURCE_EVIDENCE_ERROR')
    r11.credit.validate_r6(r6zip);r11.credit.validate_r9(r9zip)
    detail=[];curves=[];drifts=[];groups=[]
    try:
        if smoke:
            cfg=r11.Config(profile='smoke')
            cases=[('smoke_1d','operator'),('smoke_2d','rotated')]
            for task,kind in cases:
                cache=r11.make_synthetic_cache(cfg)
                for split in ('train','val','test'):
                    if task=='smoke_1d':
                        cache['splits'][split]['h']=cache['splits'][split]['h'].mean(dim=-2)[...,:4]
                        cache['splits'][split]['y']=cache['splits'][split]['y'].mean(dim=-2)[...,:4]
                    else:
                        cache['splits'][split]['h']=cache['splits'][split]['h'][...,:2,:4]
                        cache['splits'][split]['y']=cache['splits'][split]['y'][...,:2,:4]
                core=r11.UnifiedModel('V2_Ported_R5',1,8,12,4).cpu()
                a,b,c=analyze_case(task,core,cache,r6zip,kind,windows=1,steps=12)
                detail+=a;curves+=b;drifts+=c;independent_chain_regression(core,cache,r6zip,task,kind);groups.append({'task':task,'controller':kind,'smoke':True,'independent_full_graph_chain':'PASS'})
            status='PROGRAM_ONLY_SMOKE_NO_SCIENCE'
        else:
            if parent is None or r11.sha(parent)!=r11.PARENT_SHA:raise RuntimeError('EXACT_B0_PARENT_MISSING_OR_HASH_FAIL')
            with zipfile.ZipFile(parent) as z:
                manifest,_=r13.check_zip(z);cfg=r11.Config(**manifest['config'])
                for task in TASKS:
                    core,cache,replay=r11.make_cache_checked(task,z,cfg,work,'cpu')
                    for kind in ('operator','rotated'):
                        a,b,c=analyze_case(task,core,cache,r6zip,kind,windows=2,steps=12)
                        detail+=a;curves+=b;drifts+=c;groups.append({'task':task,'controller':kind,'parent_replay':replay})
                        write_csv(output/'vjp_chain_steps.csv',detail);write_csv(output/'vjp_chain_comparison.csv',curves)
                        json_write(output/'progress.json',{'groups':groups,'curves':len(curves)})
            if len(groups)!=6 or len(curves)!=6*2*2*12 or len(detail)!=6*2*2*12*3:
                raise RuntimeError(f'INCOMPLETE_OFFICIAL_R14 {len(groups)} {len(curves)} {len(detail)}')
            status='FROZEN_JACOBIAN_CHAIN_COMPLETED_NO_TRAINING'
    except Exception as exc:
        write_csv(output/'vjp_chain_steps.csv',detail);write_csv(output/'vjp_chain_comparison.csv',curves)
        json_write(output/'failure_details.json',{'error':repr(exc),'groups':groups,'steps':len(detail),'comparisons':len(curves)})
        raise
    write_csv(output/'vjp_chain_steps.csv',detail);write_csv(output/'vjp_chain_comparison.csv',curves)
    write_csv(output/'forward_trajectory_drift.csv',drifts)
    gate={'status':'DIAGNOSTIC_ONLY_NO_TRAINING','train_allowed':False,'metric':'unit-normalized recurrent (z,p) VJP direction and exact log-amplification, fixed initial-conditioning input','warnings':['not full model parameter gradient','one seed + two historical TRAIN windows per PDE','not full Jacobian spectral radius','not an intervention or training-performance claim']}
    json_write(output/'jacobian_chain_gate.json',gate)
    report={'status':status,'experiment':'CIDM_v2_B2_R14_P0_JacobianChainTransport','groups':len(groups),'steps':len(detail),'comparisons':len(curves),'train_windows':len(drifts),'updates':0,'train_allowed':False,'cases':groups}
    json_write(output/'report.json',report)
    return report
