"""CIDM-v2 B2/R1: model-free flux stability qualification, then matched tiny training.
No new parameters; no test-driven threshold tuning. Source remains B0-R1 archived v1 + PDE-ported v2.
"""
from __future__ import annotations
import os, io, json, math, zipfile, hashlib, random, time, copy, csv
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from benchmark import Config, TASKS, TASKS as ALL_TASKS, fetch_official, make_cache, Windows, UnifiedModel, ARCHIVE_SHA, sha, seed_all, load_data, assess as canonical_assess

TASKLIST=('burgers_1d','reacdiff_2d','swe_2d')
SEEDS=(17,41,73)
POLICIES=('baseline','flux_cap2','flux_cap4','flux_cap8','flux_scale075','flux_scale050')

def flux_rms(t):
    return t.square().mean(dim=(1,2,3),keepdim=True).clamp_min(1e-20).sqrt()

class ControlledV2(nn.Module):
    """Same pretrained/newly trained v2 parameters, only altered latent Flux time stepping."""
    def __init__(self, core, policy='baseline'):
        super().__init__()
        if policy not in POLICIES: raise ValueError(policy)
        self.core=core
        self.policy=policy
    def forward(self, h, steps=12, return_trace=False):
        is1d=h.ndim==4
        h5=h.unsqueeze(-2) if is1d else h
        mm=self.core.model
        B,L,C,H,W=h5.shape
        buf=list(h5.unbind(1))
        z=mm.encode(h5.reshape(B,L*C,H,W))
        coeff=mm.coefficients()
        skew=mm.skew-mm.skew.T
        f0=coeff['flux']*mm.flux(z)
        initnorm=flux_rms(f0).detach().clamp_min(1e-10)
        ys=[];tr=[]
        for k in range(steps):
            raw=coeff['flux']*mm.flux(z)
            if self.policy.startswith('flux_cap'):
                cap=float(self.policy.replace('flux_cap',''))
                factor=(cap*initnorm/(flux_rms(raw)+1e-12)).clamp(max=1.)
                f=raw*factor
            elif self.policy.startswith('flux_scale'):
                factor=float(self.policy.replace('flux_scale',''))/100.
                f=raw*factor
            else:
                f=raw
            local=coeff['local']*torch.tanh(mm.local(z))
            skew_update=.035*torch.einsum('ij,bjhw->bihw',skew,z)
            z=z+f+local+skew_update
            logits=mm.decode(z)
            inc=coeff['out']*torch.tanh(logits)
            x=buf[-1]+inc
            ys.append(x);buf.append(x)
            if return_trace:
                with torch.no_grad():
                    tr.append(dict(lead=k+1,raw_flux_rms=float(flux_rms(raw).mean()),
                                   used_flux_rms=float(flux_rms(f).mean()),
                                   local_rms=float(flux_rms(local).mean()),
                                   decoder_logit_abs_gt2=float((logits.abs()>2).float().mean()),
                                   output_increment_rms=float(inc.square().mean().sqrt())))
        y=torch.stack(ys,dim=1)
        return ((y.squeeze(-2) if is1d else y),tr) if return_trace else (y.squeeze(-2) if is1d else y)


def rms_rmse(yhat,y,sd):
    return (((yhat-y)*sd).square().sum().item(),yhat.numel())

@torch.no_grad()
def assess(core,policy,cache,split,device,steps=12,batch=None,max_windows=None,details=False):
    if batch is None: batch = 8 if cache['splits'][split]['h'].ndim==4 else 2  # MUST match archived B0 batch for 1D vs 2D
    w=Windows(cache['splits'][split]['h'],cache['splits'][split]['y'],cache['mean'],cache['std'])
    net=ControlledV2(core,policy).to(device).eval()
    sq=0.;n=0; trace=[]; seq_err=None; seq_n=0; dirs={1:[],6:[],12:[]};powers={1:[],6:[],12:[]}
    max_windows=min(len(w),max_windows) if max_windows else len(w)
    for i in range(0,max_windows,batch):
        xx=[w[j] for j in range(i,min(max_windows,i+batch))]
        h=torch.stack([v[0] for v in xx]).to(device);y=torch.stack([v[1] for v in xx]).to(device)
        result=net(h,steps,return_trace=details)
        p,tt=result if details else (result,[])
        s= w.sd.to(device)
        a,b=rms_rmse(p,y[:,:steps],s);sq+=a;n+=b
        if details:
            for t in tt:trace.append(t)
            hh=h.unsqueeze(-2) if h.ndim==4 else h
            yy=y.unsqueeze(-2) if y.ndim==4 else y
            pp=p.unsqueeze(-2) if p.ndim==4 else p
            for k in dirs:
                trueinc=yy[:,k-1]-(hh[:,-1] if k==1 else yy[:,k-2]);predinc=pp[:,k-1]-(hh[:,-1] if k==1 else pp[:,k-2]);
                cos=F.cosine_similarity(predinc.flatten(1),trueinc.flatten(1),dim=1,eps=1e-12)
                dirs[k].extend([float(x) for x in cos])
                powers[k].append(float(predinc.square().mean().clamp_min(1e-20)/trueinc.square().mean().clamp_min(1e-20)))
    out={'rmse12':math.sqrt(sq/max(n,1)),'mse12':sq/max(n,1),'num_windows':max_windows}
    if details:
        out.update({f'cos_lead{k}':float(np.mean(v)) for k,v in dirs.items()})
        out.update({f'power_lead{k}':float(np.mean(v)) for k,v in powers.items()})
        out['trace_lead12']={j:float(np.mean([t[j] for t in trace if t['lead']==12])) for j in ('raw_flux_rms','used_flux_rms','local_rms','decoder_logit_abs_gt2','output_increment_rms')}
    return out


def get_core(z,cache,task,seed,device):
    name=f'best_models/{task}__V2_Ported_R5__seed{seed}.pt'
    payload=z.read(name)
    ck=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
    assert ck['source_sha']==ARCHIVE_SHA and ck['seed']==seed and ck['model_kind']=='V2_Ported_R5'
    cfg=ck['config'];channel=cache['splits']['train']['h'].shape[2]
    core=UnifiedModel('V2_Ported_R5',channel,cfg['d'],cfg['hidden'],4).to(device)
    core.load_state_dict(ck['model'],strict=True)
    core.eval()
    return core,hashlib.sha256(payload).hexdigest()

def check_zip(z):
    assert z.testzip() is None
    r=json.loads(z.read('B0_REPORT.json'));gate=r['gate'];c=r['config']
    assert c['profile']=='pilot_r1' and c['eval_horizon']==12 and c['train_horizon']==6
    assert gate['completed_runs']==18 and gate['official_data'] and gate['all_complete']
    for task in TASKLIST:
        for seed in SEEDS:
            assert f'best_models/{task}__V2_Ported_R5__seed{seed}.pt' in z.namelist()
    return r,json.loads(z.read('data_and_architecture_contract.json'))

def output_csv(path,rows):
    if not rows:return
    with open(path,'w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)

def gate_validate(rows):
    """One policy globally selected across all three PDEs; no family-dependent runtime routing."""
    cand=[]
    baseline={(r['task'],r['seed']):r['mse12'] for r in rows if r['policy']=='baseline'}
    for policy in POLICIES[1:]:
        x=[r for r in rows if r['policy']==policy]
        ratios={(r['task'],r['seed']):r['mse12']/baseline[(r['task'],r['seed'])] for r in x}
        family={task:float(np.mean([ratios[(task,s)] for s in SEEDS])) for task in TASKLIST}
        seed_wins={task:sum(ratios[(task,s)]<1 for s in SEEDS) for task in TASKLIST}
        gm=float(math.exp(np.mean([math.log(v) for v in family.values()])))
        eligible=(gm<=.97 and all(v<=1.02 for v in family.values()) and all(v>=2 for v in seed_wins.values()))
        cand.append(dict(policy=policy,geometric_mean_mse_ratio=gm,worst_family_ratio=max(family.values()),
                         family_ratios=family,seed_wins=seed_wins,validation_qualified=bool(eligible)))
    eligible=sorted((t for t in cand if t['validation_qualified']),key=lambda r:r['geometric_mean_mse_ratio'])
    return {'chosen_policy':eligible[0]['policy'] if eligible else None,
            'state':'FROZEN_VAL_QUALIFIED_FOR_TRAINING_PREFLIGHT' if eligible else 'FROZEN_VAL_GATE_FAIL_NO_TRAINING',
            'requirements':{'global_geometric_mse_gain_min':.03,'max_each_family_mse_regression':.02,
                            'min_paired_seed_wins_per_family':2},'candidates':cand}

def _seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def train_pair(task,cache,seed,policy,outdir,device,updates=240,smoke=False):
    """Same initial parameters, same window indices, same training budget. Qualification only."""
    from benchmark import choose_match
    cfg=Config(profile='pilot_r1')
    channel=cache['splits']['train']['h'].shape[2]
    # Exact prior B0 parameter specs from checkpoint are ideal but same parameter count within this pair suffices.
    conf=choose_match('V2_Ported_R5',channel,cfg) if not smoke else {'d':8,'hidden':12}
    _seed(seed)
    init=UnifiedModel('V2_Ported_R5',channel,conf['d'],conf['hidden'],4)
    sd=copy.deepcopy(init.state_dict()); del init
    dataset=Windows(cache['splits']['train']['h'],cache['splits']['train']['y'],cache['mean'],cache['std'])
    logs=[];results=[]
    for route in ('baseline',policy):
        core=UnifiedModel('V2_Ported_R5',channel,conf['d'],conf['hidden'],4).to(device)
        core.load_state_dict(sd,strict=True)
        net=ControlledV2(core,route).to(device)
        optimizer=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
        best=float('inf');best_state=None;hist=[]
        snapshot=Path(outdir)/f'Q1_LATEST_{task}__{route}__seed{seed}.pt'
        start_step=0
        if snapshot.exists():
            saved=torch.load(snapshot,map_location=device,weights_only=True)
            if (saved.get('seed')==seed and saved.get('route')==route and
                saved.get('candidate_policy')==policy and saved.get('updates')==updates and
                saved.get('match_spec')==conf):
                net.load_state_dict(saved['model'],strict=True)
                optimizer.load_state_dict(saved['optimizer'])
                start_step=int(saved['step']);best_state=saved['best_model']
                hist=saved['history'];best=float(saved['best_rmse12'])
                print('[Q1 RESUME]',task,route,seed,'step',start_step,flush=True)
        for step in range(start_step,updates):
            net.train()
            inds=torch.randint(0,len(dataset),(2 if len(cache['splits']['train']['h'].shape)>4 else 8,),
                               generator=torch.Generator().manual_seed(seed*1000003+step)).tolist()
            samples=[dataset[i] for i in inds]
            h=torch.stack([x[0] for x in samples]).to(device)
            y=torch.stack([x[1] for x in samples]).to(device)
            optimizer.zero_grad(set_to_none=True)
            p=net(h,6);loss=F.mse_loss(p,y[:,:6]);assert torch.isfinite(loss)
            loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
            if (step+1)%max(2,updates//4)==0 or step+1==updates:
                ev=assess(net.core,route,cache,'val',device,batch=2,max_windows=2 if smoke else None)
                if ev['rmse12']<best:
                    best=ev['rmse12'];best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
                rr=dict(task=task,seed=seed,route=route,step=step+1,train_mse=float(loss.detach()),
                        val_rmse12=ev['rmse12'],best_val_rmse12=best)
                hist.append(rr);print('[Q1 train]',rr,flush=True)
                # Persist last and best weights/optimizer for diagnosis even on interrupted runs.
                snapshot_tmp=snapshot.with_suffix('.pt.tmp')
                torch.save({'model':net.state_dict(),'best_model':best_state,'optimizer':optimizer.state_dict(),
                            'step':step+1,'history':hist,'seed':seed,'route':route,
                            'candidate_policy':policy,'qualified_only':True,
                            'best_rmse12':best,'updates':updates,'match_spec':conf},snapshot_tmp)
                os.replace(snapshot_tmp,snapshot)
                # Optional HF checkpoint + GitHub small progress report; no failure may stop scientific training.
                if os.getenv('CIDM_LIVE_SYNC','1')=='1':
                    try:
                        from artifact_store import sync_training_milestone
                        ret=sync_training_milestone(snapshot,rr,
                            repo_id=os.getenv('CIDM_HF_RESULTS_REPO',''),
                            github_repo=os.getenv('CIDM_GITHUB_REPO','Mr-wuff/AI-Weather-Quest'))
                        print('[Q1 ARCHIVE]',ret,flush=True)
                    except Exception as archive_exc:
                        print('[Q1 ARCHIVE NONBLOCKING ERROR]',str(archive_exc)[:140],flush=True)

        net.load_state_dict(best_state)
        outfile=Path(outdir)/f'Q1_{task}__{route}__seed{seed}.pt'
        torch.save({'model_state':best_state,'route':route,'task':task,'seed':seed,'updates':updates,
                    'match_spec':conf,'qualification_only':True},outfile)
        results.append(dict(task=task,seed=seed,route=route,updates=updates,best_val_rmse12=best,
                            train_history=hist,checkpoint=outfile.name))
        logs+=hist
    return results,logs

def run(b0zip,outdir,root='/content/CIDM_v2_B0_FairPDE',smoke=False):
    torch.set_num_threads(min(4,torch.get_num_threads()))
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    assert sha(Path(__file__).with_name('cidm_complete_frozen.py'))==ARCHIVE_SHA
    z=zipfile.ZipFile(b0zip);r0,manifest=check_zip(z)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows=[];replay=[];caches={};diagnostic_rows=[]
    tasks=('swe_2d',) if smoke else TASKLIST
    seeds=(17,) if smoke else SEEDS
    for task in tasks:
        if smoke:
            g=torch.Generator().manual_seed(2033)
            a=torch.randn(4,16,1,8,16,generator=g)*.03
            for t in range(1,16):a[:,t]=.98*a[:,t-1]+.02*a[:,t]
            caches[task]={'splits':{k:{'h':a[:,:4], 'y':a[:,4:]} for k in ('train','val','test')},
                          'mean':torch.zeros(1),'std':torch.ones(1)}
        else:
            cfg=Config(**r0['config']);data_dir=Path(os.getenv('CIDM_B2_DATA_ROOT',str(Path(root)/'official_data')))
            official=fetch_official(data_dir,task)
            caches[task]=make_cache(task,official,cfg,Path(root)/'window_cache')
            ref=next(t for t in manifest['tasks'] if t['task']==task)
            assert caches[task]['signature']['source_md5']==ref['source_md5']
            # Check trajectory IDs (not just window counts / source MD5). A stale cache can otherwise silently pass.
            assert caches[task]['ids']==ref['ids'], f'{task}: OFFICIAL_TRAJECTORY_ID_MISMATCH'
            for split in ('train','val','test'):
                assert len(caches[task]['splits'][split]['h'])==ref['windows'][split]
        for seed in seeds:
            core,hsh=get_core(z,caches[task],task,seed,device)
            base=assess(core,'baseline',caches[task],'val',device,batch=2,max_windows=2 if smoke else None)
            if not smoke:
                # REPLAY FIX2: archival benchmark loader, exact batch size, original forward FIRST.
                # A different batch size may cause divergent GPU convolution rounding during unstable rollout.
                cfg=Config(**r0['config'])
                original_batch=cfg.batch_1d if caches[task]['splits']['test']['h'].ndim==4 else cfg.batch_2d
                orig=float(json.loads(z.read(f'{task}__V2_Ported_R5__seed{seed}.json'))['test']['model_12'])
                ref_loader=load_data(caches[task],'test',original_batch,seed,False)
                canonical=canonical_assess(core,ref_loader,device,cfg,mode='off')['model_12']
                # Run the wrapper with the SAME batch schedule and normalization.
                actual=assess(core,'baseline',caches[task],'test',device,batch=original_batch)
                # Check element-wise equality, in the original B0 test batch shape.
                hw,yy=next(iter(ref_loader));hw=hw.to(device)
                with torch.no_grad():
                    raw_out=core(hw,12,'off')
                    wrapped_out=ControlledV2(core,'baseline')(hw,12)
                    parity=float((raw_out-wrapped_out).abs().max())
                tol=max(1e-5,orig*.001)
                row=dict(task=task,seed=seed,checkpoint_sha256=hsh,archival_batch=original_batch,
                         original_rmse12=orig,canonical_rmse12=canonical,wrapped_rmse12=actual['rmse12'],
                         max_forward_parity_diff=parity,
                         original_difference=canonical-orig,wrapper_difference=actual['rmse12']-canonical,
                         pass_archive=bool(abs(canonical-orig)<=tol),
                         pass_wrapper=bool(abs(actual['rmse12']-canonical)<=tol and parity<2e-6))
                replay.append(row)
                if not row['pass_archive'] or not row['pass_wrapper']:
                    # Preserve exact evidence before fail-closed. Never enter Q0 on a mismatched baseline.
                    (out/'REPLAY_FAILURE_DIAGNOSTIC.json').write_text(json.dumps({'reason':'CANONICAL_SOURCE_OR_WRAPPER_REPLAY_MISMATCH',
                      'failed_run':row,'prior_runs':replay,'data_signature':caches[task]['signature'],
                      'historical_report_config':r0['config']},indent=2,ensure_ascii=False))
                    raise RuntimeError('B0 replay FAIL-CLOSED: '+json.dumps(row,ensure_ascii=False)+
                        '. See REPLAY_FAILURE_DIAGNOSTIC.json. Confirm checkpoint/data/cache/batch. DO NOT relax tolerance.')
                print('[B0 archived exact replay PASS]',task,seed,'batch',original_batch,
                       'archived',round(orig,8),'reference',round(canonical,8),
                       'wrapper',round(actual['rmse12'],8),'max_diff',parity,flush=True)
            # Verify unchanged baseline replay exactly against original architecture.
            w=Windows(caches[task]['splits']['val']['h'],caches[task]['splits']['val']['y'],
                      caches[task]['mean'],caches[task]['std'])
            h=torch.stack([w[0][0]]).to(device)
            with torch.no_grad():
                diff=float((ControlledV2(core,'baseline')(h,12)-core(h,12,'off')).abs().max())
            assert diff<2e-6,(task,seed,diff)
            print('[B2 R1 Q0]',task,'seed',seed,'baseline val12',base['rmse12'])
            # Causally available validation histories only; measure saturation and direction
            # on a small subset. These observations do not affect policy selection.
            dp=assess(core,'baseline',caches[task],'val',device,batch=2,max_windows=2 if smoke else 8,details=True)
            diagnostic_rows.append(dict(task=task,seed=seed,policy='baseline',
              **{k:v for k,v in dp.items() if k.startswith(('cos_lead','power_lead'))},
              **dp['trace_lead12']))
            for policy in POLICIES:
                x=base if policy=='baseline' else assess(core,policy,caches[task],'val',device,batch=2,max_windows=2 if smoke else None)
                rows.append(dict(task=task,seed=seed,policy=policy,mse12=x['mse12'],rmse12=x['rmse12'],
                                 gain_vs_baseline_pct=100*(1-x['mse12']/base['mse12']),val_windows=x['num_windows']))
            del core
            if device.type=='cuda':torch.cuda.empty_cache()
    output_csv(out/'frozen_validation_selection.csv',rows)
    output_csv(out/'baseline_validation_operator_diagnostics.csv',diagnostic_rows)
    stage={'state':'PROGRAM_ONLY_SMOKE' if smoke else 'UNKNOWN','chosen_policy':None,'candidates':[]}
    if not smoke:stage=gate_validate(rows)
    q1results=[];history=[]
    if stage['chosen_policy'] is not None and not smoke:
        print('[B2 R1] Frozen validated candidate:',stage['chosen_policy'], 'entering TRAINED Q1 seed17 only')
        for task in TASKLIST:
            ans,logs=train_pair(task,caches[task],17,stage['chosen_policy'],out,device,updates=240)
            q1results+=ans;history+=logs
        ref={r['task']:r for r in q1results if r['route']=='baseline'}
        gains={r['task']:1-r['best_val_rmse12']/ref[r['task']]['best_val_rmse12']
               for r in q1results if r['route']!= 'baseline'}
        # First-stage tests only, not automatic multi-seed formal training.
        q1pass=all(gains[t]>=.02 for t in ('burgers_1d','swe_2d')) and gains['reacdiff_2d']>=-.02
        stage['q1_trainable_prequalification']={'improvement_by_task':gains,'pass':q1pass,
          'state':'Q1_SINGLE_SEED_PREQUALIFIED_FURTHER_SEEDS_NEEDED' if q1pass else 'Q1_STOP_BEFORE_FORMAL_TRAINING'}
        stage['state']='Q1_SINGLE_SEED_PREQUALIFIED_FURTHER_SEEDS_NEEDED' if q1pass else 'Q1_STOP_BEFORE_FORMAL_TRAINING'
    report={'status':stage['state'],'data':'SYNTHETIC_SMOKE' if smoke else 'OFFICIAL_MD5_CHECKED',
            'original_B0_completed_runs':r0['gate']['completed_runs'],
            'original_B0_source':ARCHIVE_SHA,'frozen_gate':stage,
            'test_replay_only_not_used_for_selection':replay,'trainable_q1_results':q1results,
            'limitations':['Q0 hyperparameter selection uses validation only; official test used for original replay check only.',
                           'Q1 is 240-update seed17 trained qualification, NOT cross-seed scientific superiority.',
                           'Policy changes learned R5 transition and may distort a frozen checkpoint; no inference policy is adopted without retrained comparison.',
                           'Flux latent telescoping is not physical energy conservation; perturbation growth is not a Lyapunov exponent.',
                           'Original three PDE families have repeatedly informed development; not truly blind transfer.']}
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    output_csv(out/'q1_training_history.csv',history)
    resultzip=out/'CIDM_v2_B2_R1_FLUX_CONTROL_QUALIFICATION_REPORTS.zip'
    with zipfile.ZipFile(resultzip,'w',compression=zipfile.ZIP_DEFLATED) as zout:
        for file in sorted(out.glob('*')):
            if file!=resultzip and file.is_file():zout.write(file,arcname=file.name)
    assert zipfile.ZipFile(resultzip).testzip() is None
    z.close();return report,str(resultzip)
