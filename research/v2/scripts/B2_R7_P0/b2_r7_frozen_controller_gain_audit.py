"""CIDM-v2 B2-R7/P0. Frozen B2-R6 trained-head response audit, no optimization.

Official test data only used to exact-replay original archived checkpoint.  B2-R6
pilot's validation AUDIT windows are used for posthoc, pre-registered gain/sign
sensitivity. No validation tuning, no claim of new holdout or deployable gains.
"""
from __future__ import annotations
import csv, io, json, math, zipfile, hashlib
from pathlib import Path
import torch
from benchmark import Config, Windows, UnifiedModel, make_synthetic_cache
from b2_r1_flux_control import check_zip
from b2_r6_causal_spatial_controller import (
    sha, validate_preflight, make_cache_checked, make_synthetic_1d,
    CausalCoefficientHead, orthogonal_matrix, effect_basis,
    forecast, TASK_LIST, SEED, LEADS, RATIO_BUDGET,
)

B2R6_SHA='d923eb6af43c294f28ec4f236c707feb0c4bcf1e6e5fed519bf697599730ef89'
GAINS=(0.0, 0.5, 1.0, 2.0, 4.0)
SIGNS=(1, -1)

def write_json(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding='utf-8')

def write_csv(p,rows):
    if not rows:return
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def validate_r6(reportzip):
    if sha(reportzip)!=B2R6_SHA:raise RuntimeError('B2_R6_PILOT_ZIP_SHA256_MISMATCH')
    with zipfile.ZipFile(reportzip) as z:
        if z.testzip():raise RuntimeError('B2_R6_PILOT_CRC_FAIL')
        report=json.loads(z.read('report.json'))
        if report.get('status')!='PILOT_SINGLE_SEED_FAIL_NO_FULL_TRAINING':
            raise RuntimeError('B2_R6_UNEXPECTED_OR_MISSING_ORIGINAL_SCIENCE_STATUS')
        if set(x['task'] for x in report['reports'])!=set(TASK_LIST):raise RuntimeError('B2_R6_TASK_COVERAGE_INVALID')
        for name in z.namelist():
            if name.endswith('.pt'):continue
        manifest=json.loads(z.read('FILE_SHA256_MANIFEST.json'))
        entries=manifest if isinstance(manifest,list) else manifest.get('files',[])
        for ent in entries:
            name=ent['path'];h=ent['sha256']
            if hashlib.sha256(z.read(name)).hexdigest()!=h:raise RuntimeError('REPORT_INNER_SHA_MISMATCH:'+name)
        for task in TASK_LIST:
            for typ in ('operator','rotated'):
                fn=f'{task}/{typ}_best.pt'
                if fn not in z.namelist():raise RuntimeError('MISSING_TRAINED_HEAD:'+fn)
    return report

def load_heads(report_zip,task,ch,device):
    heads={}
    with zipfile.ZipFile(report_zip) as z:
        for typ in ('operator','rotated'):
            obj=torch.load(io.BytesIO(z.read(f'{task}/{typ}_best.pt')),
                           map_location='cpu',weights_only=True)
            assert obj['task']==task and obj['controller_type']==typ and obj['seed']==SEED
            assert obj['fixed_projection_seed']==707 and obj['fixed_basis_rotation_seed']==32207
            head=CausalCoefficientHead(ch,seed=707).to(device)
            head.load_state_dict(obj['head'],strict=True)
            head.requires_grad_(False).eval()
            heads[typ]=head
    return heads

@torch.no_grad()
def forecast_scaled(core,head,h,*,rotate=False,Q=None,steps=12,
                    gain=1.0,sign=1):
    """Identical B2-R6 dynamics when gain=1 and sign=+1; changed head impulse only."""
    mm=core.model
    one_d=(h.ndim==4)
    hh=h.unsqueeze(-2) if one_d else h
    B,L,C,H,W=hh.shape
    coeff=mm.coefficients()
    z=mm.encode(hh.reshape(B,L*C,H,W))
    prev=hh[:,-1];initial=prev
    skew=mm.skew-mm.skew.T
    predictions=[]; per_lead=[]
    for lead in range(1,steps+1):
        f=coeff['flux']*mm.flux(z)
        l=coeff['local']*torch.tanh(mm.local(z))
        s=.035*torch.einsum('ij,bjhw->bihw',skew,z)
        base=z+f+l+s
        if head is not None:
            basis,amps=effect_basis(f,l,s,Q if rotate else None)
            gates=head(base,prev,initial,amps,lead)
            raw=(gates.unsqueeze(2)*basis).sum(dim=1)/math.sqrt(3)
            zrms=base.square().mean(dim=(1,2,3),keepdim=True).sqrt().clamp_min(1e-4)
            raw_delta=RATIO_BUDGET*zrms*raw
            change=(float(sign)*float(gain))*raw_delta
            r=change.square().mean(dim=(1,2,3),keepdim=True).sqrt()
            cap=(RATIO_BUDGET*float(gain)*zrms)/(r.clamp_min(1e-8))
            change=change*cap.clamp(max=1.)
            rel=change.square().mean(dim=(1,2,3)).sqrt()/zrms.flatten()
            raw_rel=raw_delta.square().mean(dim=(1,2,3)).sqrt()/zrms.flatten()
            gmean=gates.abs().mean(dim=(1,2,3))
            clipped=(cap<1.).flatten().float()
        else:
            change=torch.zeros_like(z)
            rel=z.new_zeros(B);raw_rel=z.new_zeros(B)
            gmean=z.new_zeros(B);clipped=z.new_zeros(B)
        z_next=base+change
        out=prev+coeff['out']*torch.tanh(mm.decode(z_next))
        predictions.append(out.squeeze(-2) if one_d else out)
        if lead in LEADS:
            per_lead.append({'lead':lead,'relative_delta_rms':rel.cpu(),
                             'raw_relative_delta_rms':raw_rel.cpu(),
                             'gate_abs_mean':gmean.cpu(),
                             'clipped_fraction':clipped.cpu()})
        z=z_next;prev=out
    return torch.stack(predictions,dim=1),per_lead

@torch.no_grad()
def evaluated(core,heads,Q,val,indices,device,batch=2,smoke=False):
    specs=[('baseline',None,0.,1,False)]
    # Original baseline + 5 fixed magnitudes (including zero) and sign reversals.
    gains=(0.,.5,1.5) if smoke else GAINS
    for typ in ('operator','rotated'):
        for gain in gains:
            if gain==0:specs.append((typ,typ,gain,1,typ=='rotated'))
            else:
                for sign in SIGNS:specs.append((typ,typ,gain,sign,typ=='rotated'))
    total={typ:{lead:[0.,0] for lead in LEADS} for typ,*_ in specs}
    # Name stable keys for all experiment cells.
    def key(base,gain,sign):return 'baseline' if base=='baseline' else f'{base}__g{gain:g}__s{sign:+d}'
    total={key(typ,g,sgn):{lead:[0.,0] for lead in LEADS} for typ,_,g,sgn,_ in specs}
    effects={key(typ,g,sgn):{lead:{'rel':0.,'raw_rel':0.,'gate':0.,'clip':0.,'n':0} for lead in LEADS}
             for typ,_,g,sgn,_ in specs if typ!='baseline'}
    sample=[];parity_max=0.
    for off in range(0,len(indices),batch):
        ids=indices[off:off+batch]
        arr=[val[int(i)] for i in ids]
        h=torch.stack([x[0] for x in arr]).to(device)
        truth=torch.stack([x[1] for x in arr]).to(device)
        orig,orig_d=forecast(core,heads['operator'],h,steps=12)
        direct, _=forecast_scaled(core,heads['operator'],h,steps=12,gain=1.,sign=1)
        parity_max=max(parity_max,float((orig-direct).abs().max()))
        origrot,_=forecast(core,heads['rotated'],h,rotate=True,Q=Q,steps=12)
        directrot,_=forecast_scaled(core,heads['rotated'],h,rotate=True,Q=Q,steps=12,gain=1.,sign=1)
        parity_max=max(parity_max,float((origrot-directrot).abs().max()))
        scale=val.sd.to(device).reshape(1,-1,*([1]*(truth.ndim-3)))
        for typ,headtype,gain,sign,rotated in specs:
            name=key(typ,gain,sign)
            head=None if headtype is None else heads[headtype]
            pred,diag=forecast_scaled(core,head,h,rotate=rotated,Q=Q,gain=gain,sign=sign)
            for lead in LEADS:
                err=((pred[:,lead-1]-truth[:,lead-1])*scale).square().flatten(1).mean(1)
                total[name][lead][0]+=float(err.sum()); total[name][lead][1]+=err.numel()
                for k,e in zip(ids,err.cpu().tolist()):
                    sample.append({'window_index':int(k),'method':name,'lead':lead,'mse_physical':e})
            if head is not None:
                for item in diag:
                    e=effects[name][item['lead']]
                    for field,source in [('rel','relative_delta_rms'),('raw_rel','raw_relative_delta_rms'),('gate','gate_abs_mean'),('clip','clipped_fraction')]:
                        e[field]+=float(item[source].sum())
                    e['n']+=len(ids)
    curves=[]
    for name,measures in total.items():
        for lead,(sse,n) in measures.items():
            e=effects.get(name,{}).get(lead,{})
            base_sse,base_n=total['baseline'][lead]
            curves.append({'method':name,'lead':lead,'mse_physical':sse/max(n,1),
              'ratio_to_baseline':(sse/max(n,1))/max(base_sse/max(base_n,1),1e-25),
              'used_latent_rms':e.get('rel',0)/max(e.get('n',0),1),
              'unscaled_head_rms':e.get('raw_rel',0)/max(e.get('n',0),1),
              'gate_abs_mean':e.get('gate',0)/max(e.get('n',0),1),
              'cap_applied_fraction':e.get('clip',0)/max(e.get('n',0),1),
            })
    if parity_max>2e-6:raise RuntimeError(f'B2R6_EXACT_FORECAST_PARITY_FAIL {parity_max}')
    return curves,sample,parity_max

def run(parent,reportzip,work,output,*,smoke=False,device='cpu'):
    work=Path(work);output=Path(output)
    work.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    if smoke:
        cfg=Config(profile='smoke')
        s1=make_synthetic_cache(cfg)
        # Force a genuinely rank-1 spatial grid into the 4D history interface.
        for split in ('train','val','test'):
            s1['splits'][split]['h']=s1['splits'][split]['h'].mean(dim=-2)
            s1['splits'][split]['y']=s1['splits'][split]['y'].mean(dim=-2)
        tasks=[('smoke_1d',s1),('smoke_2d',make_synthetic_cache(cfg))]
        if reportzip is not None:validate_r6(reportzip)
        stats=[];samples=[]
        for task,cache in tasks:
            core=UnifiedModel('V2_Ported_R5',1,8,12,4).to(device).eval().requires_grad_(False)
            ch=1
            if reportzip is not None:
                heads=load_heads(reportzip,'burgers_1d' if task=='smoke_1d' else 'swe_2d',ch,device)
            else:
                heads={name:CausalCoefficientHead(ch,seed=707).to(device) for name in ('operator','rotated')}
                for head in heads.values():
                    with torch.no_grad():head.net[-1].bias.copy_(torch.tensor([.01,-.03,.025],device=device))
                    head.eval().requires_grad_(False)
            val=Windows(cache['splits']['val']['h'],cache['splits']['val']['y'],cache['mean'],cache['std'])
            Q=orthogonal_matrix(int(core.model.skew.shape[0]),32207,device,torch.float32)
            c,s,p=evaluated(core,heads,Q,val,[0,1],device,batch=2,smoke=True)
            for row in c:row['task']=task
            for row in s:row['task']=task
            stats.extend(c);samples.extend(s)
        write_csv(output/'gain_lead_curves.csv',stats);write_csv(output/'sample_mse.csv',samples)
        r={'status':'PROGRAM_ONLY_SMOKE_NO_SCIENCE','groups':len(stats),'records':len(samples),'new_training':False}
        write_json(output/'report.json',r);return r
    if not (parent and reportzip):raise RuntimeError('PARENT_AND_PILOT_REPORT_REQUIRED')
    prior=validate_r6(reportzip)
    if sha(parent)!=prior['parent_sha256']:raise RuntimeError('PARENT_MISMATCH_PRIOR_REPORT')
    if any(len(z['audit_windows'])!=16 for z in prior['reports']):raise RuntimeError('R6_AUDIT_CONTRACT_INVALID')
    results=[];all_curves=[];all_samples=[]
    with zipfile.ZipFile(parent) as z:
        m,contract=check_zip(z);cfg=Config(**m['config'])
        for task in TASK_LIST:
            core,cache,replay=make_cache_checked(task,z,cfg,work,device)
            ch=int(cache['splits']['val']['h'].shape[2]);heads=load_heads(reportzip,task,ch,device)
            Q=orthogonal_matrix(int(core.model.skew.shape[0]),32207,device,torch.float32)
            val=Windows(cache['splits']['val']['h'],cache['splits']['val']['y'],cache['mean'],cache['std'])
            old=next(x for x in prior['reports'] if x['task']==task)
            indices=old['audit_windows']
            curves,samples,parity=evaluated(core,heads,Q,val,indices,device,batch=2)
            for row in curves:row['task']=task
            for row in samples:row['task']=task
            # Exact pointwise replay of already-archived B2-R6 paired validation scores.
            for typ in ('operator','rotated','baseline'):
                key='baseline' if typ=='baseline' else f'{typ}__g1__s+1'
                for lead in LEADS:
                    actual=next(row['mse_physical'] for row in curves if row['method']==key and row['lead']==lead)
                    archived=old['audit'][typ][str(lead)]
                    if abs(actual-archived)>max(1e-8,abs(archived)*0.003):
                        raise RuntimeError(f'B2R6_VAL_METRIC_REPLAY_FAILED {task}/{key}/{lead}: {actual} vs {archived}')
            b=next(x for x in curves if x['method']=='baseline' and x['lead']==12)
            g=next(x for x in curves if x['method']=='operator__g1__s+1' and x['lead']==12)
            rr={'task':task,'pilot_seed':SEED,'replay':replay,'original_forecast_parity_max':parity,
                'gain1_relative_delta_rms_lead12':g['used_latent_rms'],
                'gain1_head_gate_mean_lead12':g['gate_abs_mean'],
                'gain1_ratio_to_baseline_lead12':g['ratio_to_baseline'],
                'baseline_lead12_mse':b['mse_physical'],
                'readout_windows':indices,'outer_holdout':False}
            print('[B2R7 AUDIT]',task,rr,flush=True)
            results.append(rr);all_curves.extend(curves);all_samples.extend(samples)
            write_csv(output/'gain_lead_curves.csv',all_curves)
            write_csv(output/'sample_mse.csv',all_samples)
            write_json(output/'live_progress.json',{'completed_tasks':[r['task'] for r in results]})
    report={'experiment':'CIDM_v2_B2_R7_P0_FrozenCausalController_GainAndSignSensitivity',
      'status':'FROZEN_CONTROLLER_SENSITIVITY_COMPLETED_NO_TRAINING',
      'tasks':results,'original_B2R6_pilot_status':prior['status'],
      'b2r6_zip_sha256':sha(reportzip),'parent_sha256':sha(parent),
      'gain_list':GAINS,'signs':SIGNS,'training_steps':0,
      'no_future_truth_in_forecast':True,
      'labels_only_posthoc_audit':True,
      'no_selection_on_validation':True,
      'qualification':'DIAGNOSTIC_ONLY_NOT_FULL_TRAINING',
      'limitations':['Only archived seed17 heads; no fresh outer holdout.',
       'Amplifying trained frozen heads is not retraining or evidence of future issue-time calibration.',
       'Gradient truncated in original B2-R6 training; this frozen audit cannot prove BPTT is causal bottleneck.']}
    write_json(output/'report.json',report)
    return report
