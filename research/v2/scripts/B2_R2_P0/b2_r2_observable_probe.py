"""CIDM-v2 B2-R2/P0: frozen observable-effect and self-reencoding audit.

This is a diagnostic observer, NOT a training procedure or a proposed new architecture.
It attaches to the existing B2-R1 FIX2 ControlledV2.forward inside the canonical
`assess` evaluator. It does not change model weights, callbacks or predictions.
"""
from __future__ import annotations
import collections, contextlib, copy, hashlib, inspect, json, math, os, pathlib, time
import numpy as np
import torch

DEFAULT_STEPS = (1,3,6,9,12)


def _tensor(out):
    if torch.is_tensor(out): return out
    if isinstance(out, (list, tuple)) and out and torch.is_tensor(out[0]): return out[0]
    if isinstance(out, dict):
        for k in ('pred','prediction','output','logits'):
            if k in out and torch.is_tensor(out[k]): return out[k]
    return None


def _rms(t): return float(t.detach().float().square().mean().sqrt().item())


def _cos(a,b):
    a=a.detach().float().reshape(-1); b=b.detach().float().reshape(-1)
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()): return None
    na=torch.linalg.vector_norm(a); nb=torch.linalg.vector_norm(b)
    if na.item()<1e-12 or nb.item()<1e-12: return None
    return float((a@b/(na*nb)).clamp(-1,1).item())


def find_observer_modules(model):
    """Match actual train-time modules by names/classes, refusing ambiguous duplicates."""
    groups={k:[] for k in ('flux','local','skew','decoder')}
    for name,mod in model.named_modules():
        if not name: continue
        cls=mod.__class__.__name__.lower(); leaf=name.split('.')[-1].lower()
        for k in groups:
            if leaf==k or (k!='local' and k in cls and len(list(mod.children()))==0):
                groups[k].append((name,mod));break
    selected={}
    for k,found in groups.items():
        # Prefer explicit leaf module (not constituent convolutions).
        explicit=[x for x in found if x[0].split('.')[-1].lower()==k]
        options=explicit or found
        if len(options)==1: selected[k]=options[0]
    return selected, {k:[x[0] for x in v] for k,v in groups.items()}


class FrozenObservableProbe:
    """Observational hooks; no intervention on forward outputs or any parameters."""
    def __init__(self, max_batches=2, max_steps=12, diagnostics_steps=DEFAULT_STEPS):
        self.max_batches=int(max_batches);self.max_steps=int(max_steps)
        self.steps=set(diagnostics_steps)
        self.rows=[];self.audit=[];self._seen=collections.Counter()
        self.active=False;self.label=None;self._handles=[];self._buffers=None;self._last_latents=[];self._pred_response={}

    def begin_eval(self, task, seed, split):
        self.active=split=='val';self.label=(str(task),str(seed),str(split))
        self._seen[self.label]=0

    def end_eval(self): self.active=False; self.label=None

    @contextlib.contextmanager
    def install(self, net):
        if not self.active or self._seen[self.label]>=self.max_batches:
            yield;return
        selected, candidates=find_observer_modules(net)
        if not all(k in selected for k in ('flux','local','skew','decoder')):
            self.audit.append({'task':self.label[0],'seed':self.label[1],
                               'status':'MISSING_UNAMBIGUOUS_MODULES', 'candidates':candidates})
            yield;return
        buffers={k:[] for k in selected};self._buffers=buffers
        handles=[]
        for kind,(name,module) in selected.items():
            def hook(mod,args,out,kind=kind):
                if len(buffers[kind])>=self.max_steps:return
                tin=args[0] if args and torch.is_tensor(args[0]) else None
                tout=_tensor(out)
                if tin is None or tout is None: return
                buffers[kind].append((tin.detach().clone(),tout.detach().clone()))
            handles.append(module.register_forward_hook(hook))
        try:
            yield
        finally:
            for h in handles:h.remove()
            self._buffers=None
            self._last_latents=[q[0].detach().clone() for q in buffers['decoder']]
            self._finalize(net,buffers,selected)
            self._seen[self.label]+=1

    @torch.no_grad()
    def _finalize(self, net, buf, selected):
        lengths={k:len(v) for k,v in buf.items()}
        required=['flux','local','skew','decoder']
        n=min(lengths.values()) if lengths else 0
        if not n:
            self.audit.append({'task':self.label[0],'seed':self.label[1], 'status':'NO_HOOK_EVENTS','lengths':lengths});return
        dec=selected['decoder'][1]
        # Guard that decoder forward remains independently callable and shape-preserving.
        for t in range(min(n,self.max_steps)):
            if (t+1) not in self.steps:continue
            f_in,f=buf['flux'][t]; l_in,l=buf['local'][t]; s_in,s=buf['skew'][t]; z,d=buf['decoder'][t]
            if not (f.shape==l.shape==s.shape==z.shape==f_in.shape):
                self.audit.append({'task':self.label[0],'seed':self.label[1],'step':t+1,
                                   'status':'SHAPE_MISMATCH','shapes':{k:list(q.shape) for k,q in [('F',f),('L',l),('S',s),('z',z),('f_in',f_in)]}});continue
            if any(not torch.isfinite(q).all() for q in (f,l,s,z,d,f_in)):
                self.audit.append({'task':self.label[0],'seed':self.label[1], 'step':t+1,'status':'NONFINITE'});continue
            dz=z-f_in
            # Fit effective coefficients to EXACT forward update, do not assume
            # the archived training code uses the same coefficients as a paper sketch.
            candidates=[f,torch.tanh(l),s]
            # float64 small Gram system avoids giant matrix materialization.
            gram=torch.stack([torch.stack([torch.sum(a.double()*b.double()) for b in candidates]) for a in candidates])
            cross=torch.stack([torch.sum(a.double()*dz.double()) for a in candidates])
            try:
                coeff=(torch.linalg.pinv(gram,rcond=1e-10)@cross).to(dz.dtype)
                reconstructed=sum(coeff[i]*candidates[i] for i in range(3))
                rel_residual=_rms(dz-reconstructed)/max(_rms(dz),1e-12)
                conditioning=float(torch.linalg.cond(gram.double()).item())
            except Exception as exc:
                self.audit.append({'task':self.label[0],'seed':self.label[1], 'step':t+1,'status':'FIT_FAILED','message':repr(exc)});continue
            row={'task':self.label[0],'seed':self.label[1], 'split':self.label[2],
                 'batch':self._seen[self.label], 'step':t+1,
                 'flux_rms':_rms(f),'local_rms':_rms(l),'skew_rms':_rms(s),
                 'latent_update_rms':_rms(dz),'decoder_output_rms':_rms(d),
                 'decoder_abs_gt2':float((d.abs()>2).float().mean().item()),
                 'flux_local_cos':_cos(f,l),'flux_skew_cos':_cos(f,s),
                 'fitted_flux':float(coeff[0]),'fitted_local':float(coeff[1]),
                 'fitted_skew':float(coeff[2]),'fit_rel_residual':rel_residual,
                 'fit_gram_cond':conditioning,
                 'valid_additive_attribution':bool(rel_residual<=0.10 and math.isfinite(conditioning) and conditioning<1e10)}
            # Counterfactual deletion is only identifiable for an additive-step
            # decomposition that faithfully reconstructs the original z update.
            if row['valid_additive_attribution']:
                for i,key in enumerate(('flux','local','skew')):
                    without=z-coeff[i]*candidates[i]
                    try:
                        cf=_tensor(dec(without))
                        if cf is not None and cf.shape==d.shape:
                            # The physical readout may include additional scaling and
                            # residual. We explicitly report decoder-space responses.
                            delta=d-cf
                            self._pred_response[(self.label,self._seen[self.label],t+1,key)]=(torch.tanh(d)-torch.tanh(cf)).detach()
                            row[f'{key}_decoder_response_rms']=_rms(delta)
                            row[f'{key}_decoder_response_cos']=_cos(delta,d)
                            row[f'{key}_counterfactual_delta_z_rms']=_rms(coeff[i]*candidates[i])
                        else:
                            row[f'{key}_counterfactual_error']='DECODER_SHAPE'
                    except Exception as e:
                        row[f'{key}_counterfactual_error']=repr(e)[:200]
            self.rows.append(row)



    @torch.no_grad()
    def model_increment_alignment(self,history,raw_output):
        """Additional direction against SELF prediction increments, never future truth."""
        pred=_tensor(raw_output)
        if not self.active or pred is None or pred.ndim!=5 or history.ndim!=5:return
        if history.shape[0]!=pred.shape[0] or history.shape[2:]!=pred.shape[2:]:return
        batch=self._seen[self.label]-1
        for t in range(pred.shape[1]):
            if t+1 not in self.steps:continue
            before=history[:,-1] if t==0 else pred[:,t-1]
            delta_pred=pred[:,t]-before
            for key in ('flux','local','skew'):
                resp=self._pred_response.pop((self.label,batch,t+1,key),None)
                if resp is None or resp.shape!=delta_pred.shape:continue
                for row in self.rows:
                    if row.get('task')==self.label[0] and row.get('seed')==self.label[1] and row.get('batch')==batch and row.get('step')==t+1 and row.get('diagnostic')!='SELF_PREDICTED_REENCODE':
                        row[key+'_response_cos_pred_increment']=_cos(resp,delta_pred)
                        break

    @torch.no_grad()
    def self_reencode(self,net,history,raw_output):
        """Compare own rollout to re-encoding ONLY model-generated history (no oracle)."""
        if not self.active or self._seen[self.label]>self.max_batches: return
        pred=_tensor(raw_output)
        if pred is None or not torch.is_tensor(history) or pred.ndim!=5 or history.ndim!=5:return
        if history.shape[0]!=pred.shape[0] or history.shape[2:]!=pred.shape[2:]:return
        candidates=[net,getattr(net,'core',None),getattr(net,'model',None),getattr(net,'net',None)]
        encoder=next((q.encode for q in candidates if q is not None and callable(getattr(q,'encode',None))),None)
        decoder=next((q.decoder for q in candidates if q is not None and hasattr(q,'decoder')),None)
        if encoder is None or decoder is None:
            self.audit.append({'task':self.label[0],'seed':self.label[1],
                               'status':'SELF_REENCODE_API_UNAVAILABLE'});return
        hist=history.detach().clone()
        for t in range(min(pred.shape[1],len(self._last_latents))):
            hist=torch.cat([hist[:,1:],pred[:,t:t+1]],dim=1)
            if t+1 not in self.steps:continue
            z=self._last_latents[t]
            try:
                rz=_tensor(encoder(hist))
                if rz is None or rz.shape!=z.shape:continue
                orig_d=_tensor(decoder(z)); re_d=_tensor(decoder(rz))
                self.rows.append({'task':self.label[0],'seed':self.label[1],
                                  'split':self.label[2],'batch':self._seen[self.label]-1,
                                  'step':t+1,'diagnostic':'SELF_PREDICTED_REENCODE',
                                  'reencode_latent_cos':_cos(z,rz),
                                  'reencode_decoder_rel_gap':_rms(re_d-orig_d)/max(_rms(orig_d),1e-12),
                                  'reencode_decoder_cos':_cos(re_d,orig_d)})
            except Exception as exc:
                self.audit.append({'task':self.label[0],'seed':self.label[1],
                                   'step':t+1,'status':'REENCODE_ERROR','error':repr(exc)[:200]})


def attach_to_fix2(module, probe):
    """Hook the already-validated B2-R1 evaluation without rewriting its sampler.

    Calls the ORIGINAL assess for exactly the same official val windows, then
    instruments ControlledV2 forward. No train or test-set strategy selection.
    """
    if not hasattr(module,'assess') or not hasattr(module,'ControlledV2'):
        raise RuntimeError('Missing verified FIX2 API: assess / ControlledV2; refuse generic substitute.')
    original_assess=module.assess
    controlled=module.ControlledV2
    old_forward=controlled.forward
    def observed_forward(self,*args,**kwargs):
        with probe.install(self):
            output=old_forward(self,*args,**kwargs)
        if args and torch.is_tensor(args[0]):
            probe.model_increment_alignment(args[0],output)
            probe.self_reencode(self,args[0],output)
        return output
    def observed_assess(core,policy,cache,split,device,*args,**kwargs):
        # Obtain the actual task and seed from FIX2 run() lexical frame.
        caller=inspect.currentframe().f_back
        vals=caller.f_locals if caller else {}
        task=vals.get('task','unknown');seed=vals.get('seed','unknown')
        if policy=='baseline' and str(split).lower() in ('val','validation'):
            probe.begin_eval(task,seed,'val')
        else:probe.active=False
        try:
            return original_assess(core,policy,cache,split,device,*args,**kwargs)
        finally:probe.end_eval()
    module.assess=observed_assess;controlled.forward=observed_forward
    def detach():
        module.assess=original_assess;controlled.forward=old_forward
    return detach


def summarize(rows,audit,out_dir):
    import csv
    p=pathlib.Path(out_dir);p.mkdir(parents=True,exist_ok=True)
    fields=sorted(set().union(*(x.keys() for x in rows))) if rows else []
    if rows:
        with (p/'operator_observable_samples.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    (p/'observer_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    by={}
    for row in rows:
        key=(row['task'],row['seed'])
        by.setdefault(key,[]).append(row)
    stats=[]
    for (task,seed),r in by.items():
        vals=lambda k:[x[k] for x in r if isinstance(x.get(k), (int,float)) and math.isfinite(x[k])]
        stats.append(dict(task=task,seed=seed,num_observations=len(r),valid_attribution=sum(bool(x.get('valid_additive_attribution',False)) for x in r),
                          mean_fit_rel_residual=float(np.mean(vals('fit_rel_residual'))) if vals('fit_rel_residual') else None,
                          mean_decoder_abs_gt2=float(np.mean(vals('decoder_abs_gt2'))) if vals('decoder_abs_gt2') else None,
                          mean_flux_decoder_response_rms=float(np.mean(vals('flux_decoder_response_rms'))) if vals('flux_decoder_response_rms') else None,
                          mean_local_decoder_response_rms=float(np.mean(vals('local_decoder_response_rms'))) if vals('local_decoder_response_rms') else None))
    report={'stage':'B2-R2/P0','status':'FROZEN_DIAGNOSTIC_ONLY_NO_TRAINING',
            'qualification':'NOT_EVALUATED','number_of_rows':len(rows),
            'coverage':stats,'limitations':['Observable responses are decoder-space only, not physical target increments.',
              'No claims about true PDE physical conservation or Lyapunov exponents.',
              'Frozen ablation is NOT a capacity-matched training comparison.',
              'Truth-free diagnostic: cannot certify trajectory RMSE improvement.',
              'No direct state synchronization intervention and no future-observation inputs.']}
    (p/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def cpu_unit_test():
    class TestNet(torch.nn.Module):
        def __init__(self):
            super().__init__();self.flux=torch.nn.Conv2d(2,2,1,bias=False)
            self.local=torch.nn.Conv2d(2,2,1,bias=False)
            self.skew=torch.nn.Conv2d(2,2,1,bias=False)
            self.decoder=torch.nn.Conv2d(2,1,1,bias=False)
        def forward(self,x,steps=3,return_trace=False):
            z=x[:, -1];pred=[]
            for _ in range(steps):
                f=self.flux(z);l=self.local(z);s=self.skew(z)
                z=z+.15*f+.04*torch.tanh(l)+.035*s
                pred.append(self.decoder(z))
            p=torch.stack(pred,1)
            return (p,[]) if return_trace else p
    torch.manual_seed(2);m=TestNet();x=torch.randn(2,3,2,4,4)
    p=FrozenObservableProbe(max_batches=1,max_steps=3,diagnostics_steps=(1,3))
    p.begin_eval('unit','17','val')
    with p.install(m): y=m(x)
    p.end_eval()
    assert y.shape==(2,3,1,4,4)
    assert len(p.rows)==2,(p.rows,p.audit)
    assert max(r['fit_rel_residual'] for r in p.rows)<1e-5
    assert all(r['valid_additive_attribution'] for r in p.rows)
    assert all('flux_decoder_response_rms' in r for r in p.rows)
    return {'status':'PROGRAM_ONLY_SMOKE_PASS','rows':len(p.rows),'error':max(r['fit_rel_residual'] for r in p.rows)}

if __name__=='__main__': print(cpu_unit_test())
