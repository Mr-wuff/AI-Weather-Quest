"""CIDM-v2 B2-R8/P0: frozen, issue-time, equal-realized-RMS perturbation audit.

NO TRAINING. Truth is used only AFTER producing autoregressive forecasts to score
physical-space MSE. Same per-sample/per-step RMS exposure for Natural and Rotated.
This diagnostic on B0-preselected validation windows is NOT an independent skill claim.
"""
from __future__ import annotations
import csv,hashlib,io,json,math,zipfile
from pathlib import Path
import torch
from benchmark import Config, Windows, UnifiedModel, make_synthetic_cache
from b2_r1_flux_control import check_zip
from b2_r6_causal_spatial_controller import (
    sha, make_cache_checked, CausalCoefficientHead, orthogonal_matrix, effect_basis,
    TASK_LIST, SEED, LEADS, RATIO_BUDGET,
)
from b2_r7_frozen_controller_gain_audit import load_heads, forecast_scaled, validate_r6, write_json, write_csv

# Hash pinned to the user-provided actual real run, not the original Notebook.
R7_REAL_ZIP_SHA='9c7891b0784ce63c95dad6eb52c073ea69495f2e99adf38a4f187255397f987a'
R6_REAL_ZIP_SHA='d923eb6af43c294f28ec4f236c707feb0c4bcf1e6e5fed519bf697599730ef89'
PARENT_SHA='69552b29e169ec22ee85d9c1b1a678fdb188f6f25dc18f33b927b88f4dddffd4'
# Magnitudes are fractions of the current autoregressive model's own latent RMS.
BUDGETS=(0.,0.0005,0.001,0.003,0.005,0.01)
SIGNS=(1,-1)
PRE_REGISTERED_BUDGET=0.003


def validate_r7(r7zip):
    if sha(r7zip)!=R7_REAL_ZIP_SHA:raise RuntimeError('B2_R7_REAL_REPORT_SHA_MISMATCH')
    with zipfile.ZipFile(r7zip) as z:
        if z.testzip():raise RuntimeError('B2_R7_CRC_FAILED')
        r=json.loads(z.read('report.json'))
        if r['status']!='FROZEN_CONTROLLER_SENSITIVITY_COMPLETED_NO_TRAINING':raise RuntimeError('B2_R7_SCIENTIFIC_STATE_INVALID')
        if {v['task'] for v in r['tasks']}!=set(TASK_LIST):raise RuntimeError('B2_R7_TASK_COVERAGE_MISSING')
        for item in json.loads(z.read('FILE_SHA256_MANIFEST.json')):
            if hashlib.sha256(z.read(item['path'])).hexdigest()!=item['sha256']:
                raise RuntimeError('B2_R7_INNER_MANIFEST_MISMATCH:'+item['path'])
        parsed=list(csv.DictReader(io.StringIO(z.read('gain_lead_curves.csv').decode('utf-8'))))
        assert len(parsed)==285
        ref={(v['task'],v['method'],int(v['lead'])):float(v['mse_physical']) for v in parsed}
        return r,ref


def _norm(t):
    return t.square().mean(dim=(1,2,3),keepdim=True).sqrt()


@torch.no_grad()
def rollout_budget(core, head, h, *, rotate=False, Q=None, budget=0., sign=1, steps=12):
    mm=core.model; one_d=(h.ndim==4); hh=h.unsqueeze(-2) if one_d else h
    B,L,C,H,W=hh.shape; coeff=mm.coefficients()
    z=mm.encode(hh.reshape(B,L*C,H,W)); prev=hh[:,-1]; initial=prev
    skew=mm.skew-mm.skew.T
    preds=[]; diag={}; min_raw=float('inf')
    for lead in range(1,steps+1):
        f=coeff['flux']*mm.flux(z)
        l=coeff['local']*torch.tanh(mm.local(z))
        s=.035*torch.einsum('ij,bjhw->bihw',skew,z)
        base=z+f+l+s
        if head is None or float(budget)==0.:
            change=torch.zeros_like(base); realized=base.new_zeros((B,))
            raw_rms=base.new_zeros((B,)); gates=base.new_zeros((B,)); inactive=base.new_zeros((B,))
        else:
            basis,amps=effect_basis(f,l,s,Q if rotate else None)
            coeff_local=head(base,prev,initial,amps,lead)
            raw=(coeff_local.unsqueeze(2)*basis).sum(dim=1)/math.sqrt(3)
            raw_rms=_norm(raw)
            if not torch.isfinite(raw_rms).all():raise RuntimeError('NONFINITE_HEAD_VECTOR')
            # Near-zero controller vectors cannot be safely normalized.
            active=(raw_rms>=1e-9)
            change=sign*float(budget)*_norm(base).clamp_min(1e-4)*raw/raw_rms.clamp_min(1e-9)
            change=torch.where(active,change,torch.zeros_like(change))
            realized=(_norm(change)/_norm(base).clamp_min(1e-4)).flatten()
            raw_rms=raw_rms.flatten()
            gates=coeff_local.abs().mean(dim=(1,2,3))
            inactive=(~active).flatten().float()
            # Assert exact equal exposure before any ground-truth loss is observed.
            if active.any() and (realized[active.flatten()]-float(budget)).abs().max()>max(1e-6,float(budget)*2e-3):
                raise RuntimeError('EXPOSURE_MATCH_CONTRACT_FAIL')
        z=base+change
        out=prev+coeff['out']*torch.tanh(mm.decode(z));prev=out
        preds.append(out.squeeze(-2) if one_d else out)
        if lead in LEADS:
            diag[lead]={'used_rel':realized.cpu(), 'raw_norm':raw_rms.cpu(),
                        'gate_abs':gates.cpu(),'inactive':inactive.cpu()}
    return torch.stack(preds,1),diag


def _name(head,budget,sign):
    if head is None:return 'baseline'
    return f'{head}__b{budget*100:.3f}pct__s{sign:+d}'


@torch.no_grad()
def evaluate(core,heads,Q,val,indices,device,*,smoke=False):
    # Fully preregistered grid. All methods issue-time; never use truth for tuning.
    budgets=(0.,.001,.003) if smoke else BUDGETS
    specs=[(None,None,0.,1,False)]
    for method in ('operator','rotated'):
        for b in budgets:
            if b==0: continue  # baseline is shared and has exact zero correction
            for sign in SIGNS:
                specs.append((method,heads[method],b,sign,method=='rotated'))
    records=[]; sums={_name(n,b,sign):{int(t):[0.,0] for t in LEADS} for n,_,b,sign,_ in specs}
    stats={_name(n,b,sign):{int(t):[0.,0.,0.,0] for t in LEADS} for n,_,b,sign,_ in specs}
    parity=0.
    for pos in range(0,len(indices),2):
        ids=indices[pos:pos+2]
        h=torch.stack([val[int(i)][0] for i in ids]).to(device)
        truth=torch.stack([val[int(i)][1] for i in ids]).to(device)
        scale=val.sd.to(device).reshape(1,-1,*([1]*(truth.ndim-3)))
        # Verify B2-R6 1x head and B2-R7 1x replay on the same issue-time windows.
        prior={}
        for kind in ('operator','rotated'):
            prior[kind],_=forecast_scaled(core,heads[kind],h,rotate=(kind=='rotated'),Q=Q,steps=12,gain=1.,sign=1)
        for typ,head,b,sign,rot in specs:
            name=_name(typ,b,sign)
            pred,d=rollout_budget(core,head,h,rotate=rot,Q=Q,budget=b,sign=sign)
            if typ is None:
                parent_base,_=forecast_scaled(core,None,h,steps=12,gain=0.,sign=1)
                parity=max(parity,float((pred-parent_base).abs().max()))
            for lead in LEADS:
                err=((pred[:,lead-1]-truth[:,lead-1])*scale).square().flatten(1).mean(1)
                sums[name][lead][0]+=float(err.sum());sums[name][lead][1]+=err.numel()
                for idx,loss in zip(ids,err.cpu().tolist()):
                    records.append({'window_index':int(idx),'method':name,'lead':int(lead),'mse_physical':float(loss)})
                if head is not None:
                    di=d[int(lead)];stat=stats[name][int(lead)]
                    stat[0]+=float(di['used_rel'].sum());stat[1]+=float(di['inactive'].sum());
                    stat[2]+=float(di['gate_abs'].sum());stat[3]+=len(ids)
        # No time leakage: prior forecasts only scored retrospectively here.
        for kind in ('operator','rotated'):
            for lead in LEADS:
                err=((prior[kind][:,lead-1]-truth[:,lead-1])*scale).square().flatten(1).mean(1)
                key=f'original__{kind}__g1'
                if key not in sums:sums[key]={int(t):[0.,0] for t in LEADS}
                sums[key][int(lead)][0]+=float(err.sum());sums[key][int(lead)][1]+=err.numel()
    if parity>2e-6:raise RuntimeError('ORIGINAL_OUTPUT_PARITY_FAIL')
    curves=[]
    for name,lead_metrics in sums.items():
        for lead,(total,n) in lead_metrics.items():
            ss,ns=sums['baseline'][lead]
            s=stats.get(name,{}).get(lead,[0,0,0,0])
            curves.append({'method':name,'lead':lead,'mse_physical':total/max(n,1),
             'ratio_to_baseline':(total/max(n,1))/(ss/max(ns,1)),
             'realized_relative_rms':s[0]/max(s[3],1),
             'inactive_fraction':s[1]/max(s[3],1),
             'gate_abs_mean':s[2]/max(s[3],1)})
    return curves,records,parity


def check_replay(curves,ref,task):
    for method in ('baseline','operator__g1__s+1','rotated__g1__s+1'):
        current_name='baseline' if method=='baseline' else 'original__'+method.split('__')[0]+'__g1'
        for lead in LEADS:
            current=next(row['mse_physical'] for row in curves if row['method']==current_name and row['lead']==lead)
            old=ref[(task,method,int(lead))]
            if abs(current-old)>max(1e-8,abs(old)*0.003):
                raise RuntimeError(f'B2_R7_REPLAY_MISMATCH {task} {method} {lead}: {current} vs {old}')


def posthoc_gate(curves,task):
    # A diagnosis gate only, NOT authorization for full training or deployment.
    get=lambda m,lead:next(r for r in curves if r['method']==m and r['lead']==lead)
    bud=PRE_REGISTERED_BUDGET;natural=_name('operator',bud,1);rot=_name('rotated',bud,1)
    rev=_name('operator',bud,-1)
    late=[9,12]
    late_nat=sum(get(natural,l)['mse_physical'] for l in late)
    late_rot=sum(get(rot,l)['mse_physical'] for l in late)
    late_base=sum(get('baseline',l)['mse_physical'] for l in late)
    early=max(get(natural,l)['ratio_to_baseline'] for l in (1,3))
    return {'task':task,'diagnostic_budget_percent':bud*100,
      'late_vs_baseline_gain_pct':100*(1-late_nat/late_base),
      'late_vs_rotated_gain_pct':100*(1-late_nat/late_rot),
      'early_max_ratio_to_baseline':early,
      'directional_sign_consistency':all(get(natural,l)['mse_physical']<get(rev,l)['mse_physical'] for l in late),
      'exposure_nat_minus_rot_at_12':get(natural,12)['realized_relative_rms']-get(rot,12)['realized_relative_rms'],
      'local_diagnostic_signal':bool((late_nat/late_rot)<=0.99 and early<=1.02)}


def sample_risk(samples, budget=PRE_REGISTERED_BUDGET):
    """Posthoc paired risk; zero-error windows have an absolute+relative floor."""
    out={}
    for lead in (1,3,9,12):
        table={}
        for row in samples:
            if row['lead']==lead:
                table.setdefault(row['method'],{})[row['window_index']]=float(row['mse_physical'])
        baseline=table['baseline']
        for typ in ('operator','rotated'):
            key=_name(typ,budget,1)
            active=table[key]
            assert set(active)==set(baseline),'UNPAIRED_SAMPLE_IDS'
            ids=sorted(baseline)
            improv=[active[i]<baseline[i] for i in ids]
            small_floor=max(1e-12,sum(baseline.values())/len(ids)*1e-4)
            ratios=sorted(active[i]/max(baseline[i],small_floor) for i in ids)
            out[f'{typ}_lead{lead}']={
                'windows':len(ids), 'window_win_fraction':sum(improv)/len(ids),
                'sample_ratio_p90':ratios[min(len(ids)-1,math.ceil(.9*len(ids))-1)],
                'regression_over_2pct_fraction':sum(active[i]>baseline[i]*1.02+small_floor for i in ids)/len(ids),
                'denominator_floor':small_floor,
            }
    return out


def run(parent,r6zip,r7zip,work,output,*,smoke=False,device='cpu'):
    work=Path(work);output=Path(output)
    work.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    rr,refs=validate_r7(r7zip)
    if sha(r6zip)!=R6_REAL_ZIP_SHA:raise RuntimeError('R6_WEIGHTS_ARCHIVE_SHA_MISMATCH')
    validate_r6(r6zip)
    all_curves=[];all_samples=[];cases=[]; gates=[]
    if smoke:
        cfg=Config(profile='smoke');cache1=make_synthetic_cache(cfg)
        for split in ('train','val','test'):
            cache1['splits'][split]['h']=cache1['splits'][split]['h'].mean(dim=-2)
            cache1['splits'][split]['y']=cache1['splits'][split]['y'].mean(dim=-2)
        inputs=[('smoke_1d',cache1),('smoke_2d',make_synthetic_cache(cfg))]
        for task,cache in inputs:
            core=UnifiedModel('V2_Ported_R5',1,8,12,4).to(device).eval().requires_grad_(False)
            hch=1
            heads=load_heads(r6zip,'burgers_1d' if task=='smoke_1d' else 'swe_2d',hch,device)
            Q=orthogonal_matrix(int(core.model.skew.shape[0]),32207,device,torch.float32)
            val=Windows(cache['splits']['val']['h'],cache['splits']['val']['y'],cache['mean'],cache['std'])
            curves,samples,parity=evaluate(core,heads,Q,val,[0,1],device,smoke=True)
            for item in curves:item['task']=task
            for item in samples:item['task']=task
            all_curves.extend(curves);all_samples.extend(samples)
            cases.append({'task':task,'forward_parity_max':parity})
        status='PROGRAM_ONLY_SMOKE_NO_SCIENCE'
    else:
        if parent is None or sha(parent)!=PARENT_SHA:raise RuntimeError('PINNED_18_MODEL_PARENT_REQUIRED')
        with zipfile.ZipFile(parent) as z:
            m,contract=check_zip(z);cfg=Config(**m['config'])
            for task in TASK_LIST:
                core,cache,replay=make_cache_checked(task,z,cfg,work,device)
                old=next(x for x in rr['tasks'] if x['task']==task)
                if replay['checkpoint_sha256']!=old['replay']['checkpoint_sha256']:
                    raise RuntimeError('CHECKPOINT_SHA_CHANGED:'+task)
                hch=int(cache['splits']['val']['h'].shape[2]);heads=load_heads(r6zip,task,hch,device)
                Q=orthogonal_matrix(int(core.model.skew.shape[0]),32207,device,torch.float32)
                val=Windows(cache['splits']['val']['h'],cache['splits']['val']['y'],cache['mean'],cache['std'])
                ix=old['readout_windows']
                if ix!=list(range(16,32)):raise RuntimeError('AUDIT_WINDOW_CONTRACT_CHANGED')
                curves,samples,parity=evaluate(core,heads,Q,val,ix,device,smoke=False)
                check_replay(curves,refs,task)
                if parity>2e-6:raise RuntimeError('PARENT_FORWARDS_CHANGED')
                diagnostic_gate=posthoc_gate(curves,task)
                diagnostic_gate['paired_window_risk']=sample_risk(samples)
                gates.append(diagnostic_gate)
                cases.append({'task':task,'replay':replay,'baseline_parity_max':parity,
                              'audit_windows':ix,'frozen_controller_only':True})
                for item in curves:item['task']=task
                for item in samples:item['task']=task
                all_curves.extend(curves);all_samples.extend(samples)
                write_csv(output/'equal_rms_lead_curves.csv',all_curves)
                write_csv(output/'sample_mse.csv',all_samples)
                write_json(output/'live_progress.json',{'completed_tasks':[v['task'] for v in cases]})
        status='FROZEN_EQUAL_RMS_AUDIT_COMPLETED_NO_TRAINING'
    write_csv(output/'equal_rms_lead_curves.csv',all_curves)
    write_csv(output/'sample_mse.csv',all_samples)
    write_json(output/'matched_exposure_gate.json',{'status':'DIAGNOSTIC_ONLY','per_task':gates,
           'does_not_authorize': ['full-training','hyperparameter deployment','outer-holdout claims'],
           'registered_budget_fraction':PRE_REGISTERED_BUDGET})
    report={'experiment':'CIDM_v2_B2_R8_P0_FrozenEqualRMS_CausalControl_Diagnostic',
            'status':status,'cases':cases,'training_steps':0,
            'r6_real_archive_sha256':R6_REAL_ZIP_SHA,'r7_real_archive_sha256':R7_REAL_ZIP_SHA,
            'expected_parent_sha256':PARENT_SHA,
            'budgets':list(BUDGETS),'signs':list(SIGNS),'num_curves':len(all_curves),
            'num_samples':len(all_samples),'registered_decision_budget':PRE_REGISTERED_BUDGET,
            'no_future_truth_used_in_control':True,'labels_scoring_only':True,
            'outer_holdout':False,
            'notes':['Unit-norm controller action is evaluated at equal per-step/per-window latent-RMS exposure.',
                    'Natural vs Rotated comparison is exposure-matched, not trained-policy matched.',
                    'No validation budget maximization or deployment tuning; B0 checkpoint was val-selected.',
                    'Oracle gradient geometry from B2-R5 is not used in causal corrections.']}
    write_json(output/'report.json',report)
    return report
