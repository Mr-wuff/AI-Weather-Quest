"""CIDM-v2 B2-R9/P0: causal activation timing under equal per-step or total RMS-energy.

Frozen original backbone and pretrained B2-R6 heads; ground truth enters ONLY scoring.
The fixed lead-dependent schedules are retrospective ablations, NOT a learned controller
or a validated deployment policy. All B0 checkpoint selection predates this study.
"""
from __future__ import annotations
import csv,hashlib,io,json,math,zipfile
from pathlib import Path
import torch
from benchmark import Config,Windows,UnifiedModel,make_synthetic_cache
from b2_r1_flux_control import check_zip
from b2_r6_causal_spatial_controller import sha,make_cache_checked,orthogonal_matrix,effect_basis,TASK_LIST,LEADS
from b2_r7_frozen_controller_gain_audit import load_heads,write_csv,write_json
from b2_r8_equal_exposure_audit import (validate_r7,validate_r6,rollout_budget,
  R6_REAL_ZIP_SHA,R7_REAL_ZIP_SHA,PARENT_SHA)

R8_REAL_ZIP_SHA='ad8334d8ac0632ccd29965305bc613d6628fdcc89b34906ba30ca64604947159'
PRIMARY_BUDGET=.003
SECONDARY_BUDGET=.001
ONSETS=(1,4,7)
STEPS=12
VAL_INDICES=list(range(16,32))


def validate_r8(archive):
    if sha(archive)!=R8_REAL_ZIP_SHA:raise RuntimeError('B2_R8_REAL_ZIP_SHA_MISMATCH')
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise RuntimeError('B2_R8_ZIP_CRC_FAIL')
        rep=json.loads(z.read('report.json'))
        if rep.get('status')!='FROZEN_EQUAL_RMS_AUDIT_COMPLETED_NO_TRAINING':
            raise RuntimeError('B2_R8_NOT_FROZEN_AUDIT_COMPLETE')
        if set(x['task'] for x in rep.get('cases',[]))!=set(TASK_LIST):
            raise RuntimeError('B2_R8_NOT_THREE_TASKS')
        manifest=json.loads(z.read('FILE_SHA256_MANIFEST.json'))
        for ent in manifest:
            if hashlib.sha256(z.read(ent['path'])).hexdigest()!=ent['sha256']:
                raise RuntimeError('B2_R8_INNER_SHA_MISMATCH '+ent['path'])
        rows=list(csv.DictReader(io.StringIO(z.read('sample_mse.csv').decode())))
        if len(rows)!=5040:raise RuntimeError('B2_R8_EXPECTED_5040_SAMPLES')
        per={(row['task'],int(row['window_index']),row['method'],int(row['lead'])):float(row['mse_physical']) for row in rows}
        return rep,per


def rms(x):
    return x.square().mean(dim=(1,2,3),keepdim=True).sqrt()


def spec_name(kind,budget,onset,mode):
    if kind=='baseline':return 'baseline'
    return f'{kind}__b{budget*100:.3f}pct__onset{onset}__{mode}'


def schedule_fraction(budget,onset,mode):
    if mode=='fixed_step' or onset==1:return float(budget)
    if mode=='matched_energy':return float(budget)*math.sqrt(STEPS/(STEPS-onset+1))
    raise ValueError(mode)


@torch.no_grad()
def rollout_timing(core,head,h,*,Q=None,rotate=False,budget=.003,onset=1,mode='fixed_step'):
    mm=core.model;one_d=(h.ndim==4)
    hh=h.unsqueeze(-2) if one_d else h
    B,L,C,H,W=hh.shape;coeff=mm.coefficients()
    z=mm.encode(hh.reshape(B,L*C,H,W));prev=hh[:,-1];initial=prev
    skew=mm.skew-mm.skew.T
    preds=[]; per_lead={}; exposure=hh.new_zeros((B,));min_raw=float('inf')
    step_fraction=schedule_fraction(budget,onset,mode)
    for lead in range(1,STEPS+1):
        f=coeff['flux']*mm.flux(z)
        l=coeff['local']*torch.tanh(mm.local(z))
        s=.035*torch.einsum('ij,bjhw->bihw',skew,z)
        base=z+f+l+s
        if head is None or lead<onset:
            change=torch.zeros_like(base);rel=base.new_zeros((B,))
            coeff_abs=base.new_zeros((B,));inactive=base.new_zeros((B,))
        else:
            basis,amps=effect_basis(f,l,s,Q if rotate else None)
            gates=head(base,prev,initial,amps,lead)
            raw=(gates.unsqueeze(2)*basis).sum(dim=1)/math.sqrt(3)
            rr=rms(raw)
            active=(rr>=1e-9)
            change=step_fraction*rms(base).clamp_min(1e-4)*raw/rr.clamp_min(1e-9)
            change=torch.where(active,change,torch.zeros_like(change))
            rel=(rms(change)/rms(base).clamp_min(1e-4)).flatten()
            coeff_abs=gates.abs().mean(dim=(1,2,3))
            inactive=(~active).flatten().float()
            if active.any():
                mismatch=(rel[active.flatten()]-step_fraction).abs().max().item()
                if mismatch>max(1e-6,step_fraction*2e-3):
                    raise RuntimeError('R9_PER_SAMPLE_EXPOSURE_CONTRACT_FAIL '+str(mismatch))
        exposure=exposure+rel.square()
        z=base+change
        out=prev+coeff['out']*torch.tanh(mm.decode(z));prev=out
        preds.append(out.squeeze(-2) if one_d else out)
        if lead in LEADS:
            per_lead[lead]={'rel':rel.detach().cpu(),'coef':coeff_abs.detach().cpu(),
                            'inactive':inactive.detach().cpu(), 'cum_energy':exposure.detach().cpu()}
    return torch.stack(preds,1),per_lead


def specs_for(smoke):
    budgets=(PRIMARY_BUDGET,) if smoke else (SECONDARY_BUDGET,PRIMARY_BUDGET)
    specs=[('baseline',None,0.,1,'fixed_step')]
    for kind in ('operator','rotated'):
        for budget in budgets:
            for onset in ONSETS:
                modes=('fixed_step',) if onset==1 else ('fixed_step','matched_energy')
                for mode in modes:specs.append((spec_name(kind,budget,onset,mode),kind,budget,onset,mode))
    return specs


@torch.no_grad()
def evaluate(core,heads,Q,val,indices,device,*,smoke=False):
    specs=specs_for(smoke);rows=[];curves=[];totals={name:{lead:[0.,0] for lead in LEADS} for name,*_ in specs}
    diag={name:{lead:[0.,0.,0.,0.,0.] for lead in LEADS} for name,*_ in specs}
    assert set(indices)==(set([0,1]) if smoke else set(VAL_INDICES))
    parity_max=0.;prefix_parity_max=0.;exposure_maxdiff=0.
    for ix in range(0,len(indices),2):
        ids=indices[ix:ix+2];arr=[val[int(i)] for i in ids]
        h=torch.stack([a[0] for a in arr]).to(device)
        truth=torch.stack([a[1] for a in arr]).to(device)
        scale=val.sd.to(device).reshape(1,-1,*([1]*(truth.ndim-3)))
        pred_baseline,base_diag=rollout_timing(core,None,h)
        all_preds={'baseline':(pred_baseline,base_diag)}
        for name,kind,budget,onset,mode in specs[1:]:
            p,d=rollout_timing(core,heads[kind],h,Q=Q,rotate=(kind=='rotated'),budget=budget,onset=onset,mode=mode)
            all_preds[name]=(p,d)
            if onset>1:
                prefix_parity_max=max(prefix_parity_max,float((p[:,:onset-1]-pred_baseline[:,:onset-1]).abs().max()))
        for name,kind,budget,onset,mode in specs:
            pred,dd=all_preds[name]
            for lead in LEADS:
                loss=((pred[:,lead-1]-truth[:,lead-1])*scale).square().flatten(1).mean(1)
                totals[name][lead][0]+=float(loss.sum());totals[name][lead][1]+=len(ids)
                for window,loss_value in zip(ids,loss.cpu().tolist()):
                    rows.append({'task':'','window_index':int(window),'method':name,
                                 'lead':int(lead),'mse_physical':float(loss_value)})
                if kind is not None:
                    v=dd[lead];acc=diag[name][lead]
                    acc[0]+=float(v['rel'].sum());acc[1]+=float(v['coef'].sum())
                    acc[2]+=float(v['inactive'].sum());acc[3]+=float(v['cum_energy'].sum());acc[4]+=len(ids)
        # Compare Natural v Rotated realized step budget and integral exposure.
        for budget in ((PRIMARY_BUDGET,) if smoke else (SECONDARY_BUDGET,PRIMARY_BUDGET)):
            for onset in ONSETS:
                for mode in (('fixed_step',) if onset==1 else ('fixed_step','matched_energy')):
                    n=all_preds[spec_name('operator',budget,onset,mode)][1]
                    r=all_preds[spec_name('rotated',budget,onset,mode)][1]
                    for lead in LEADS:
                        exposure_maxdiff=max(exposure_maxdiff,float((n[lead]['rel']-r[lead]['rel']).abs().max()))
    if prefix_parity_max>1e-6:raise RuntimeError('B2_R9_ZERO_PREFIX_NOT_IDENTICAL '+str(prefix_parity_max))
    if exposure_maxdiff>2e-6:raise RuntimeError('B2_R9_NAT_ROT_NOT_EXPOSURE_MATCHED '+str(exposure_maxdiff))
    for name,kind,b,onset,mode in specs:
        for lead in LEADS:
            sm,n=totals[name][lead];bm,bn=totals['baseline'][lead]
            a=diag[name][lead]
            curves.append({'task':'','method':name,'lead':lead,'mse_physical':sm/n,
                'ratio_to_baseline':(sm/n)/(bm/bn),
                'realized_rel_rms':a[0]/max(a[4],1),
                'cumulative_rms_energy':a[3]/max(a[4],1),
                'gate_abs_mean':a[1]/max(a[4],1),
                'inactive_fraction':a[2]/max(a[4],1),
                'onset':onset,'mode':mode,'registered_budget':b})
    return curves,rows,{'zero_prefix_max':prefix_parity_max,'exposure_difference_max':exposure_maxdiff,'num_windows':len(indices)}


def compare_r8_replay(samples,archive_ref,task,smoke=False):
    if smoke:return 0.
    got={(int(r['window_index']),r['method'],int(r['lead'])):float(r['mse_physical']) for r in samples}
    checks=[]
    for window in VAL_INDICES:
        for lead in LEADS:
            names={'baseline':'baseline'}
            for b in (SECONDARY_BUDGET,PRIMARY_BUDGET):
                for k in ('operator','rotated'):
                    names[spec_name(k,b,1,'fixed_step')]=f'{k}__b{b*100:.3f}pct__s+1'
            for current,archived in names.items():
                v=got[(window,current,lead)];ref=archive_ref[(task,window,archived,lead)]
                diff=abs(v-ref)
                if diff>max(1e-8,abs(ref)*0.003):
                    raise RuntimeError(f'B2_R8_WINDOW_REPLAY_MISMATCH {task}/{window}/{current}/{lead} {v} {ref}')
                checks.append(diff)
    return max(checks)


def summarize_risk(samples,task,budget=PRIMARY_BUDGET):
    lookup={(int(r['window_index']),r['method'],int(r['lead'])):float(r['mse_physical']) for r in samples}
    output=[]
    for onset in ONSETS:
        for mode in (('fixed_step',) if onset==1 else ('fixed_step','matched_energy')):
            natural=spec_name('operator',budget,onset,mode);rot=spec_name('rotated',budget,onset,mode)
            for lead in (1,3,9,12):
                ids=VAL_INDICES
                base=[lookup[i,'baseline',lead] for i in ids]
                n=[lookup[i,natural,lead] for i in ids];r=[lookup[i,rot,lead] for i in ids]
                floor=max(1e-12,sum(base)/len(base)*1e-4)
                sample_ratios=sorted(a/max(b,floor) for a,b in zip(n,base))
                output.append({'task':task,'onset':onset,'mode':mode,'lead':lead,
                 'natural_vs_baseline_ratio':sum(n)/sum(base),
                 'rotated_vs_baseline_ratio':sum(r)/sum(base),
                 'natural_win_rate':sum(a<b for a,b in zip(n,base))/len(base),
                 'natural_beats_rotated_rate':sum(a<b for a,b in zip(n,r))/len(base),
                 'natural_regression_over_2pct':sum(a>b*1.02+floor for a,b in zip(n,base))/len(base),
                 'sample_ratio_p90':sample_ratios[math.ceil(.9*len(sample_ratios))-1],
                 'floor':floor})
    return output


def gate_task(curves,risk,task):
    arr={r['method']:{'first':r} for r in []}  # placeholder-free: explicit indexed curve lookup below
    refs={(r['method'],int(r['lead'])):r for r in curves}
    signals=[]
    for onset in ONSETS:
        for mode in (('fixed_step',) if onset==1 else ('fixed_step','matched_energy')):
            n=spec_name('operator',PRIMARY_BUDGET,onset,mode)
            r=spec_name('rotated',PRIMARY_BUDGET,onset,mode)
            late_n=sum(refs[n,ld]['mse_physical'] for ld in (9,12));late_r=sum(refs[r,ld]['mse_physical'] for ld in (9,12))
            late_base=sum(refs['baseline',ld]['mse_physical'] for ld in (9,12))
            early=max(refs[n,ld]['ratio_to_baseline'] for ld in (1,3))
            tail=max(v['natural_regression_over_2pct'] for v in risk if v['onset']==onset and v['mode']==mode and v['lead'] in (9,12))
            gain_base=100*(1-late_n/late_base);gain_rot=100*(1-late_n/late_r)
            # Conditional diagnostic gate only. Even success does NOT authorize training without outer holdout.
            pass_signal=(gain_base>=1.5 and gain_rot>=1.0 and early<=1.005 and tail<=.2)
            signals.append({'onset':onset,'mode':mode,'late_gain_over_base_pct':gain_base,
              'late_gain_over_rotated_pct':gain_rot,'early_max_ratio':early,
              'worst_late_window_regression_fraction':tail,'candidate_diagnostic_pass':bool(pass_signal)})
    return {'task':task,'decisions':signals,'any_diagnostic_candidate':any(v['candidate_diagnostic_pass'] for v in signals)}


def run(parent,r6zip,r7zip,r8zip,work,output,*,smoke=False,device='cpu'):
    work=Path(work);output=Path(output);work.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    validate_r6(r6zip);r7_report,_=validate_r7(r7zip);_,r8_ref=validate_r8(r8zip)
    all_curves=[];all_samples=[];all_risks=[];cases=[];gate=[]
    if smoke:
        cfg=Config(profile='smoke');s1=make_synthetic_cache(cfg)
        for sp in ('train','val','test'):
            s1['splits'][sp]['h']=s1['splits'][sp]['h'].mean(dim=-2)
            s1['splits'][sp]['y']=s1['splits'][sp]['y'].mean(dim=-2)
        inputs=[('smoke_1d',s1),('smoke_2d',make_synthetic_cache(cfg))]
        for name,cache in inputs:
            core=UnifiedModel('V2_Ported_R5',1,8,12,4).to(device).eval().requires_grad_(False)
            hch=1;heads=load_heads(r6zip,'burgers_1d' if name=='smoke_1d' else 'swe_2d',hch,device)
            Q=orthogonal_matrix(int(core.model.skew.shape[0]),32207,device,torch.float32)
            val=Windows(cache['splits']['val']['h'],cache['splits']['val']['y'],cache['mean'],cache['std'])
            cur,smp,checks=evaluate(core,heads,Q,val,[0,1],device,smoke=True)
            for row in cur:row['task']=name
            for row in smp:row['task']=name
            all_curves+=cur;all_samples+=smp;cases.append({'task':name,**checks})
        status='PROGRAM_ONLY_SMOKE_NO_SCIENCE'
    else:
        if parent is None or sha(parent)!=PARENT_SHA:raise RuntimeError('MISSING_EXACT_B0_18_RUN_PARENT')
        with zipfile.ZipFile(parent) as z:
            m,contract=check_zip(z);cfg=Config(**m['config'])
            for task in TASK_LIST:
                core,cache,replay=make_cache_checked(task,z,cfg,work,device)
                original=next(v for v in r7_report['tasks'] if v['task']==task)
                if replay['checkpoint_sha256']!=original['replay']['checkpoint_sha256']:
                    raise RuntimeError('R9_PARENT_CHECKPOINT_CHANGED '+task)
                val=Windows(cache['splits']['val']['h'],cache['splits']['val']['y'],cache['mean'],cache['std'])
                heads=load_heads(r6zip,task,int(cache['splits']['val']['h'].shape[2]),device)
                Q=orthogonal_matrix(int(core.model.skew.shape[0]),32207,device,torch.float32)
                cur,smp,checks=evaluate(core,heads,Q,val,VAL_INDICES,device)
                maxdelta=compare_r8_replay(smp,r8_ref,task)
                for row in cur:row['task']=task
                for row in smp:row['task']=task
                risks=summarize_risk(smp,task)
                local_gate=gate_task(cur,risks,task)
                all_curves+=cur;all_samples+=smp;all_risks+=risks;gate.append(local_gate)
                cases.append({'task':task,'replay':replay,'parity_vs_b2r8_max_mse_diff':maxdelta,**checks})
                write_csv(output/'activation_lead_curves.csv',all_curves)
                write_csv(output/'paired_window_mse.csv',all_samples)
                write_csv(output/'paired_risk.csv',all_risks)
                write_json(output/'progress.json',{'completed_tasks':[v['task'] for v in cases]})
        if len(cases)!=3 or len(all_curves)!=(3*len(specs_for(False))*len(LEADS)):
            raise RuntimeError('B2_R9_INCOMPLETE_OFFICIAL_COVERAGE')
        status='FROZEN_ACTIVATION_TIMING_COMPLETED_NO_TRAINING'
    write_csv(output/'activation_lead_curves.csv',all_curves)
    write_csv(output/'paired_window_mse.csv',all_samples)
    write_csv(output/'paired_risk.csv',all_risks)
    write_json(output/'timing_gate.json',{'status':'DIAGNOSTIC_ONLY','registered_primary_budget':PRIMARY_BUDGET,
       'gates':gate,'no_training_authorized':True,'outer_holdout':False})
    result={'experiment':'CIDM_v2_B2_R9_P0_CausalActivationTiming_EnergyMatched',
      'status':status,'original_training_steps':0,'new_controller_training_steps':0,
      'cases':cases,'total_curves':len(all_curves),'total_samples':len(all_samples),
      'total_risks':len(all_risks),'onsets':list(ONSETS),
      'budgets':[SECONDARY_BUDGET,PRIMARY_BUDGET],'primary_budget':PRIMARY_BUDGET,
      'original_parent_sha256':PARENT_SHA,'r6_zip_sha256':R6_REAL_ZIP_SHA,
      'r7_zip_sha256':R7_REAL_ZIP_SHA,'r8_zip_sha256':R8_REAL_ZIP_SHA,
      'truth_used_for_actions':False,'used_val_selected_parent':True,
      'outer_holdout':False,'schedules_are_deployment_eligible':False,
      'notes':['Predetermined lead-based schedules are causal but not a trained issue-time gate.',
      'Matched-energy onsets equalize sum of per-step squared relative perturbation budgets, not model response.',
      'All losses are scored after 12-step autonomous rollout; no future labels are input to any action.']}
    write_json(output/'report.json',result)
    return result
