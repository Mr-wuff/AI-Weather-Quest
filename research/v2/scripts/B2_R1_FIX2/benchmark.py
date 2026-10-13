"""CIDM-v2 B0-R1 PDEBench three-seed convergence-aware v1 Complete vs v2 restored Flux port.
Fairness: identical official files, trajectory ids, histories, normalization, budgets,
validation rules, and paired random seeds. V2 is channel-generalized adapter,
NOT bitwise original trained R5 checkpoint. Synthetic mode tests PROGRAM only.
"""
from __future__ import annotations
import os,sys,time,json,hashlib,random,zipfile,math,traceback,copy
from pathlib import Path
from dataclasses import dataclass,asdict
from collections import defaultdict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from cidm_complete_frozen import CIDMConfig,CIDM_BER

ARCHIVE_SHA='f9b1cf4ff7630f5540bb3adde20e4c4b81fff3218d9325ea3105bf28b1d97665'
TASKS={
 'advection_1d':('1D_Advection_Sols_beta0.4.hdf5','255674','d595bbfd2c659df995a93cd40d6ea568',4),
 'burgers_1d':('1D_Burgers_Sols_Nu0.01.hdf5','281363','e6d9a4f62baf9a29121a816b919e2770',4),
 'diff_sorp_1d':('1D_diff-sorp_NA_NA.h5','133020','9d466d1213065619d087319e16d9a938',4),
 'reacdiff_1d':('ReacDiff_Nu1.0_Rho2.0.hdf5','133183','112c01a76447162bd67c8c1073f58ca2',4),
 'cfd_1d':('1D_CFD_Rand_Eta0.01_Zeta0.01_periodic_Train.hdf5','164672','2fdd00138f1d7abc794fb953021a9f43',4),
 'reacdiff_2d':('2D_diff-react_NA_NA.h5','133017','b8d0b86064193195ddc30c33be5dc949',1),
 'swe_2d':('2D_rdb_NA_NA.h5','133021','75d838c47aa410694bdc912ea7f22282',1),
 'cfd_2d':('2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5','164687','5b21dcccaef4d2145ca579a71153c580',2),
}
PILOT=('burgers_1d','reacdiff_2d','swe_2d')
CANONICAL_TASKS=tuple(TASKS)

@dataclass
class Config:
 root:str='/content/CIDM_v2_B0_FairPDE'
 profile:str='pilot_r1' # smoke | pilot_r1 | formal8 | canonical8
 data_root:str=''
 seed:int=20261012
 match_target:int=500_000
 match_tolerance:float=.05
 history:int=4
 train_horizon:int=6
 eval_horizon:int=12
 pilot_updates:int=600
 formal_updates:int=600
 canonical_updates:int=1000
 train_traj:int=64
 val_traj:int=16
 test_traj:int=16
 windows_per_traj:int=2
 eval_every:int=60
 batch_1d:int=8
 batch_2d:int=2
 lr:float=.001
 grad_clip:float=1.0
 resume:bool=True
 num_workers:int=0
 def tasks(self):return ('smoke_2d',) if self.profile=='smoke' else PILOT if self.profile in ('pilot','pilot_r1') else CANONICAL_TASKS
 def seeds(self):return (17,) if self.profile in ('smoke','pilot') else (17,41,73) if self.profile in ('pilot_r1','formal8') else (17,41,73,101,137)
 def updates(self):return 2 if self.profile=='smoke' else self.pilot_updates if self.profile in ('pilot','pilot_r1') else self.formal_updates if self.profile=='formal8' else self.canonical_updates

def jsonsave(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str),encoding='utf8');tmp.replace(p)

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for a in iter(lambda:f.read(1<<22),b''):h.update(a)
 return h.hexdigest()

def md5(p):
 h=hashlib.md5()
 with open(p,'rb') as f:
  for a in iter(lambda:f.read(1<<22),b''):h.update(a)
 return h.hexdigest()

def seed_all(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(s)

def params(model):return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Copy of R5-R1 RawEdgeFlux, with architecture frozen; no simulator coefficients or task IDs.
class RawEdgeFlux(nn.Module):
 def __init__(self,d,h):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(2*d,h,1),nn.SiLU(),nn.Conv2d(h,d,1))
 def forward(self,z):
  out=torch.zeros_like(z)
  for dim in (-1,-2):
   q=z.roll(-1,dims=dim)
   edge=self.net(torch.cat((z,q),1))-self.net(torch.cat((q,z),1))
   out=out+edge-edge.roll(1,dims=dim)
  return out

class PortedRestoredCIDM(nn.Module):
 """Exact structural R5-R1 port to arbitrary input channel count and lifted 1D.
 Differences vs the original two-channel R5-R1: encoder in channels L*C,
 decoder out channels C, no unused train-only scale head, support singleton H=1.
 Training here is from scratch; original R5 weights are NOT load-compatible.
 """
 def __init__(self,channels=1,d=24,hidden=40,hist=4,sync_interval=6):
  super().__init__();assert d%4==0
  self.channels=channels;self.hist=hist;self.sync_interval=sync_interval
  self.encode=nn.Sequential(nn.Conv2d(hist*channels,hidden,3,padding=1),nn.SiLU(),nn.Conv2d(hidden,d,3,padding=1),nn.GroupNorm(4,d))
  self.flux=RawEdgeFlux(d,hidden)
  self.local=nn.Sequential(nn.Conv2d(d,hidden,3,padding=1),nn.SiLU(),nn.Conv2d(hidden,d,1))
  self.skew=nn.Parameter(torch.randn(d,d)*.01)
  self.decode=nn.Sequential(nn.Conv2d(d,hidden,3,padding=1),nn.SiLU(),nn.Conv2d(hidden,channels,1))
  # R5-R1 scale_readout was auxiliary train-only; B0 has no paired-scale labels.
  # Omit it instead of counting untrained, disconnected parameters in a 500K match.
  self.flux_logit=nn.Parameter(torch.tensor(.0))
  self.local_logit=nn.Parameter(torch.tensor(-.5))
  self.out_logit=nn.Parameter(torch.tensor(-.4))
  self.sync_logit=nn.Parameter(torch.tensor(-.6))
 def coefficients(self):
  return {'flux':.30*self.flux_logit.sigmoid(),'local':.15*self.local_logit.sigmoid(),
          'out':.22*self.out_logit.sigmoid(),'sync':.8*self.sync_logit.sigmoid()}
 def forward(self,h,steps=6,mode='off',use_flux=True,freeze_latent=False):
  assert mode in ('off','soft','hard')
  B,L,C,H,W=h.shape
  assert L==self.hist and C==self.channels and (H==1 or H%4==0) and W%4==0
  coeff=self.coefficients();buf=list(h.unbind(1))
  z=self.encode(h.reshape(B,L*C,H,W));sk=self.skew-self.skew.T;ps=[]
  for k in range(steps):
   if not freeze_latent:
    f=self.flux(z) if use_flux else torch.zeros_like(z)
    z=z+coeff['flux']*f+coeff['local']*torch.tanh(self.local(z))+.035*torch.einsum('ij,bjhw->bihw',sk,z)
   xn=buf[-1]+coeff['out']*torch.tanh(self.decode(z))
   ps.append(xn);buf.append(xn)
   if mode!='off' and (k+1)%self.sync_interval==0 and k+1<steps:
    ref=self.encode(torch.stack(buf[-L:],1).reshape(B,L*C,H,W))
    z=ref if mode=='hard' else z+coeff['sync']*(ref-z)
  return torch.stack(ps,1)

class V1Archived(nn.Module):
 def __init__(self,channels,d,hidden,hist=4):
  super().__init__()
  conf=CIDMConfig(channels=channels,hist_len=hist,horizon=6,latent_dim=d,hidden=hidden,
     soc_modes1=8,soc_modes2=8,port_channels=1,ber_enabled=True,soc_enabled=True)
  self.model=CIDM_BER(conf)
 def forward(self,h,steps=6):return self.model(h,horizon=steps,port=None)['pred']

class UnifiedModel(nn.Module):
 def __init__(self,kind,channels,d,hidden,hist=4):
  super().__init__();self.kind=kind
  self.model=V1Archived(channels,d,hidden,hist) if kind=='V1_CIDM_Complete' else PortedRestoredCIDM(channels,d,hidden,hist)
 def forward(self,h,steps,mode='off'):
  dim1=h.ndim==4
  hh=h.unsqueeze(-2) if dim1 else h
  y=self.model(hh,steps) if self.kind=='V1_CIDM_Complete' else self.model(hh,steps,mode=mode)
  return y.squeeze(-2) if dim1 else y

def choose_match(kind,channels,cfg,dev='cpu'):
 if cfg.profile=='smoke':return {'d':8,'hidden':12,'params':params(UnifiedModel(kind,channels,8,12,cfg.history)),'relative_error':None}
 ds=(8,12,16,20,24,28,32,40,48,56,64,72,80,96,112,128)
 hs=(12,16,20,24,32,40,48,56,64,72,80,96,112,128,144,160,192,224)
 candidates=[]
 for d in ds:
  for hidden in hs:
   model=UnifiedModel(kind,channels,d,hidden,cfg.history)
   p=params(model)
   candidates.append((abs(p-cfg.match_target),d,hidden,p))
   del model
 score,d,hidden,p=min(candidates)
 if score/cfg.match_target>cfg.match_tolerance:
  raise RuntimeError(f'NO PARAMETER MATCH {kind} channels={channels} best d={d},h={hidden},params={p},target={cfg.match_target}')
 return {'d':d,'hidden':hidden,'params':p,'relative_error':score/cfg.match_target}

class Windows(Dataset):
 def __init__(self,h,y,mean,std):
  self.h=h.float();self.y=y.float();shape=[1,1,-1]+[1]*(h.ndim-3)
  self.mn=mean.view(*shape);self.sd=std.view(*shape)
 def __len__(self):return len(self.h)
 def __getitem__(self,i):return ((self.h[i]-self.mn[0])/self.sd[0],(self.y[i]-self.mn[0])/self.sd[0])

def fetch_official(root,task):
 import requests
 name,id,expected,stride=TASKS[task]
 root=Path(root);root.mkdir(parents=True,exist_ok=True)
 dest=root/name
 if dest.exists() and md5(dest)==expected:return dest
 if dest.exists():raise RuntimeError('Existing file MD5 mismatch: '+str(dest)+'; delete or replace manually')
 part=Path(str(dest)+'.part'); url=f'https://darus.uni-stuttgart.de/api/access/datafile/{id}'
 for retry in range(3):
  pos=part.stat().st_size if part.exists() else 0
  headers={'Range':f'bytes={pos}-'} if pos else {}
  try:
   with requests.get(url,stream=True,timeout=(20,120),headers=headers) as r:
    r.raise_for_status()
    if pos and r.status_code!=206:part.unlink(missing_ok=True);pos=0
    with open(part,'ab' if pos else 'wb') as f:
     for ch in r.iter_content(chunk_size=1<<22):
      if ch:f.write(ch)
   break
  except (requests.RequestException,OSError) as exc:
   if retry==2:raise RuntimeError(f'{task}: download failed. Place verified official file manually into {root}. {exc}') from exc
   time.sleep(2*(retry+1))
 if md5(part)!=expected:raise RuntimeError(f'MD5 mismatch for {task}: {md5(part)}; expected {expected}. Do not train.')
 part.rename(dest)
 return dest

def schema(path):
 import h5py
 with h5py.File(path,'r') as f:
  if 'tensor' in f:return ('tensor',int(f['tensor'].shape[0]))
  if all(k in f for k in ('density','pressure','Vx')):return ('cfd',int(f['density'].shape[0]))
  keys=[k for k in sorted(f) if isinstance(f[k],h5py.Group) and 'data' in f[k]]
  if keys:return ('grouped',len(keys))
  raise RuntimeError(f'Unsupported data schema {list(f.keys())[:12]}')

def load_traj(task,path,idx):
 import h5py
 with h5py.File(path,'r') as f:
  if 'tensor' in f:
   a=np.asarray(f['tensor'][idx],dtype=np.float32)
   if a.ndim==2:a=a[:,None,:]
   elif a.ndim==3:
    if task.endswith('_2d'):a=a[:,None,:,:]
    elif a.shape[-1]<=8:a=np.moveaxis(a,-1,1)
    else:raise RuntimeError(f'ambiguous tensor {task} {a.shape}')
   elif a.ndim==4:
    if a.shape[-1]<=8:a=np.moveaxis(a,-1,1)
    elif a.shape[1]<=8:pass
    else:raise RuntimeError((task,a.shape))
   else:raise RuntimeError((task,a.shape))
  elif all(k in f for k in ('density','pressure','Vx')):
   arr=[np.asarray(f[k][idx],dtype=np.float32) for k in ('density','pressure','Vx')]
   if 'Vy' in f:arr.append(np.asarray(f['Vy'][idx],dtype=np.float32))
   a=np.stack(arr,axis=1)
  else:
   keys=[k for k in sorted(f) if isinstance(f[k],h5py.Group) and 'data' in f[k]]
   a=np.asarray(f[keys[idx]]['data'],dtype=np.float32)
   if a.ndim==4:a=np.moveaxis(a,-1,1)
   elif a.ndim==3:a=a[:,None,:,:]
   elif a.ndim==2:a=a[:,None,:]
   else:raise RuntimeError((task,a.shape))
 stride=TASKS[task][3]
 if stride>1:a=a[...,::stride] if a.ndim==3 else a[...,::stride,::stride]
 if not np.isfinite(a).all():raise RuntimeError(f'{task}: nonfinite trajectory {idx}')
 if a.ndim not in (3,4):raise RuntimeError(a.shape)
 # Fail closed rather than fake-convert impossible sizes.
 if (a.shape[-1]%4)!=0 or (a.ndim==4 and a.shape[-2]%4)!=0:
  raise RuntimeError(f'{task}: grid {a.shape[2:]} must be multiple of 4 for ported v2; no silent resize')
 return torch.from_numpy(np.ascontiguousarray(a))

def splits(N,cfg):
 ntest=max(1,int(.1*N));nval=max(1,int(.1*N))
 pools={'test':np.arange(0,ntest),'val':np.arange(N-nval,N),'train':np.arange(ntest,N-nval)}
 cap={'train':cfg.train_traj,'val':cfg.val_traj,'test':cfg.test_traj}
 return {k:np.asarray(v,dtype=int)[np.unique(np.linspace(0,len(v)-1,min(cap[k],len(v)),dtype=int))].tolist() for k,v in pools.items()}

def make_cache(task,path,cfg,out):
 assert cfg.history==4 and cfg.train_horizon==6 and cfg.eval_horizon>=6
 signature={'task':task,'source_md5':TASKS[task][2],'history':cfg.history,'eval_horizon':cfg.eval_horizon,
 'counts':[cfg.train_traj,cfg.val_traj,cfg.test_traj,cfg.windows_per_traj],'stride':TASKS[task][3]}
 token=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:12]
 fp=Path(out)/f'{task}_{token}.pt'
 if fp.exists():
  obj=torch.load(fp,map_location='cpu',weights_only=True)
  if obj['signature']==signature:return obj
 _,N=schema(path);ids=splits(N,cfg)
 all_splits={}; meta={}
 for sp,inds in ids.items():
  hs=[];ys=[];windows=[]
  for idx in inds:
   a=load_traj(task,path,int(idx))
   mx=a.shape[0]-cfg.history-cfg.eval_horizon
   if mx<0:raise RuntimeError(f'{task} traj {idx} only has {a.shape[0]} time steps, need {cfg.history+cfg.eval_horizon}. Try a smaller eval_horizon.')
   starts=np.unique(np.linspace(0,mx,cfg.windows_per_traj,dtype=int))
   for start in starts:
    hs.append(a[start:start+cfg.history]);ys.append(a[start+cfg.history:start+cfg.history+cfg.eval_horizon]);windows.append([int(idx),int(start)])
  all_splits[sp]={'h':torch.stack(hs),'y':torch.stack(ys)};meta[sp]=windows
 # Training-only moments, no val/test leakage. Same for both models.
 tensors=[all_splits['train']['h'].double(),all_splits['train']['y'].double()]
 cc=tensors[0].shape[2];ss=torch.zeros(cc,dtype=torch.float64);sq=torch.zeros_like(ss);count=0
 for a in tensors:
  rd=(0,1)+tuple(range(3,a.ndim));ss+=a.sum(rd);sq+=(a*a).sum(rd)
  count+=math.prod(a.shape[x] for x in rd)
 mu=ss/count;std=(sq/count-mu.square()).clamp_min(1e-12).sqrt()
 obj={'signature':signature,'splits':all_splits,'ids':ids,'windows':meta,'mean':mu.float(),'std':std.float()}
 fp.parent.mkdir(parents=True,exist_ok=True);torch.save(obj,fp)
 return obj

def make_synthetic_cache(cfg):
 # Synthetic preflight is NEVER an official PDEBench performance result.
 gen=torch.Generator().manual_seed(1000)
 spl={}
 for sp,n in [('train',8),('val',4),('test',4)]:
  a=torch.randn(n,cfg.history+cfg.eval_horizon,1,8,16,generator=gen)*.15
  for t in range(1,a.shape[1]):a[:,t]=.95*a[:,t-1]+.05*a[:,t]
  spl[sp]={'h':a[:,:cfg.history],'y':a[:,cfg.history:]}
 return {'signature':{'task':'smoke_2d','synthetic':True},'splits':spl,'ids':{},'windows':{},'mean':torch.zeros(1),'std':torch.ones(1)}

def load_data(cache,split,batch,seed,shuffle):
 n=Windows(cache['splits'][split]['h'],cache['splits'][split]['y'],cache['mean'],cache['std'])
 return DataLoader(n,batch_size=batch,shuffle=shuffle,drop_last=shuffle and len(n)>=batch,
                   generator=torch.Generator().manual_seed(seed),num_workers=0,pin_memory=torch.cuda.is_available())

@torch.no_grad()
def assess(model,loader,dev,cfg,mode='off'):
 model.eval();tot={}; cnt={}; baselines={}
 for h,y in loader:
  h=h.to(dev);y=y.to(dev);K=min(cfg.eval_horizon,y.shape[1]);p=model(h,K,mode)
  baseline=h[:,-1:].expand_as(y[:,:K])
  kvec=torch.arange(1,K+1,device=dev,dtype=y.dtype).view(1,K,*([1]*(h.ndim-2)))
  lin=h[:,-1:]+kvec*(h[:,-1:]-h[:,-2:-1])
  for k in sorted(set((cfg.train_horizon,K))):
   if k>K:continue
   for name,q in [('model',p),('persistence',baseline),('trend',lin)]:
    err=(q[:,:k]-y[:,:k]).square()
    # Per element, denormalized physical-unit RMSE; normalize separately per channel only at dataset stage
    std=loader.dataset.std if hasattr(loader.dataset,'std') else loader.dataset.sd
    # std shape [1,1,C,...]; broadcast to tensor.
    ss=(err*std.to(dev).square()).sum().item()
    key=(name,k);tot[key]=tot.get(key,0.)+ss;cnt[key]=cnt.get(key,0)+err.numel()
 return {name+'_'+str(k):math.sqrt(tot[(name,k)]/cnt[(name,k)]) for name,k in sorted(tot)}

@torch.no_grad()
def per_lead_assess(model,loader,dev,cfg):
 """Per-target-step physically denormalized RMSE; strictly test-only after checkpoint freeze."""
 model.eval();K=cfg.eval_horizon; ss=np.zeros(K,dtype=np.float64); nncounts=np.zeros(K,dtype=np.float64)
 for h,y in loader:
  h=h.to(dev); y=y.to(dev);p=model(h,K,'off')
  sd=loader.dataset.sd.to(dev)
  for k in range(K):
   e=((p[:,k:k+1]-y[:,k:k+1])*sd).square()
   ss[k]+=e.sum().item();nncounts[k]+=e.numel()
 return [float(np.sqrt(ss[k]/max(nncounts[k],1))) for k in range(K)]

@torch.no_grad()
def measure_latency(model,loader,dev,K):
    model.eval(); x=next(iter(loader))[0].to(dev)
    if x.shape[0]>1:x=x[:1]
    for _ in range(2):model(x,K,'off')
    if dev.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    t0=time.perf_counter()
    for _ in range(4):model(x,K,'off')
    if dev.type=='cuda':torch.cuda.synchronize()
    elapsed=(time.perf_counter()-t0)/4
    peak=int(torch.cuda.max_memory_allocated()) if dev.type=='cuda' else None
    return {'latency_ms_per_trajectory':1000*elapsed,'peak_allocated_gpu_bytes':peak,
       'note':'batch1; GPU warmup2, timed4, use as hardware-bound descriptive evidence. It is not equal-FLOP comparison.'}

@torch.no_grad()
def rollout_diagnostics(model,loader,dev,K):
    model.eval(); drifts=[]; flux_rms=[]; output_grows=[]
    for bi,(h,y) in enumerate(loader):
        if bi>=3:break
        h=h.to(dev);p=model(h,K,'off');
        output_grows.append(float((p[:,-1]-p[:,0]).square().mean().sqrt()))
        if model.kind=='V2_Ported_R5':
            mm=model.model
            hh=h.unsqueeze(-2) if h.ndim==4 else h
            b,l,c,hhgt,w=hh.shape; z=mm.encode(hh.reshape(b,l*c,hhgt,w)); f=mm.flux(z)
            drifts.append(float((z.square().mean()).sqrt()))
            flux_rms.append(float((f.square().mean()).sqrt()))
    return {'mean_rollout_field_change_rms':float(np.mean(output_grows)),
       'encoded_latent_rms':float(np.mean(drifts)) if drifts else None,
       'raw_flux_rms':float(np.mean(flux_rms)) if flux_rms else None}

def smoke_grad(model,dev,cfg,dim):
 model.train();shape=(2,4,model.model.model.config.channels,8,16) if model.kind=='V1_CIDM_Complete' else (2,4,model.model.channels,8,16)
 if dim==1:shape=shape[:3]+(16,)
 h=torch.randn(*shape,device=dev);p=model(h,3);assert p.shape==(shape[0],3,*shape[2:]) and bool(torch.isfinite(p).all())
 loss=p.square().mean();loss.backward()
 grads={n:float(v.grad.norm()) for n,v in model.named_parameters() if v.grad is not None}
 if model.kind=='V2_Ported_R5':
  flux=next(v for n,v in grads.items() if '.flux.net.' in n)
  assert flux>1e-12,'Flux path not differentiable'
 assert grads and max(grads.values())>0
 model.zero_grad(set_to_none=True);return {'shape':list(p.shape),'grad_nonzero':sum(x>0 for x in grads.values()),'total_param':params(model)}

def save_ckpt(path,model,opt,iteration,best,best_state,rng):
 torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'iteration':iteration,'best':best,
   'best_model':best_state,'cpu_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),
   'python_rng':random.getstate(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},path)

def fit(model,train,val,cfg,dev,runid,root,seed=17):
 # Single fixed LR across models, no differential tuning budget.
 target=cfg.updates();opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=1e-4)
 checkpoint=Path(root)/'checkpoints'/f'{runid}.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True)
 hist=[];best=float('inf');best_state=None;start=0
 if cfg.resume and checkpoint.exists():
  chk=torch.load(checkpoint,map_location='cpu',weights_only=False)
  model.load_state_dict(chk['model'],strict=True);opt.load_state_dict(chk['optimizer']);start=int(chk['iteration'])
  best=chk['best'];best_state=chk['best_model'];torch.set_rng_state(chk['cpu_rng'])
  np.random.set_state(chk['numpy_rng']);random.setstate(chk['python_rng'])
  if torch.cuda.is_available() and chk['cuda_rng'] is not None:torch.cuda.set_rng_state_all(chk['cuda_rng'])
  print('[resumed]',runid,'from',start)
 losses=[];t0=time.time()
 for step in range(start,target):
  model.train()
  # SAME per-seed per-step indices for BOTH architectures. Exact on resume.
  g=torch.Generator().manual_seed(int(seed)*1000003+step)
  ids=torch.randint(0,len(train.dataset),(train.batch_size,),generator=g).tolist()
  xs=[train.dataset[i] for i in ids]
  h=torch.stack([x[0] for x in xs]).to(dev)
  y=torch.stack([x[1] for x in xs]).to(dev)
  opt.zero_grad(set_to_none=True)
  p=model(h,cfg.train_horizon,mode='off')
  loss=F.mse_loss(p,y[:,:cfg.train_horizon]);
  if not torch.isfinite(loss):raise FloatingPointError('Non-finite train loss')
  loss.backward();nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip);opt.step()
  losses.append(float(loss.detach()))
  if (step+1)%cfg.eval_every==0 or step+1==target:
   eva=assess(model,val,dev,cfg,'off')
   score=eva['model_'+str(cfg.eval_horizon)] # pre-registered 12-step VAL selection for BOTH models
   if score<best:best=score;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
   row={'iter':step+1,'train_mse':float(np.mean(losses[-cfg.eval_every:])),
       'val_6_rmse':eva['model_'+str(cfg.train_horizon)],'val_12_rmse':eva['model_'+str(cfg.eval_horizon)],'val_rmse':score,'best_val_rmse':best,'wall_seconds':time.time()-t0}
   hist.append(row);print('[train]',runid,row,flush=True)
   save_ckpt(checkpoint,model,opt,step+1,best,best_state,None)
 if best_state is not None:model.load_state_dict(best_state,strict=True)
 return {'best_val_rmse':best,'validation_selection_horizon':cfg.eval_horizon,'history':hist,'elapsed_seconds':time.time()-t0, 'best_at_last_validation':bool(hist and hist[-1]['val_rmse'] <= best+1e-12)}

def run(cfg=None):
 cfg=cfg or Config();out=Path(cfg.root)/cfg.profile;out.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(min(4,max(1,torch.get_num_threads())))
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 frozen=Path(__file__).resolve().parent/'cidm_complete_frozen.py'
 if sha(frozen)!=ARCHIVE_SHA:raise RuntimeError('CIDM-v1 source SHA256 mismatch; STOP')
 if cfg.profile!='smoke' and dev.type!='cuda':raise RuntimeError('Real-data paired training requires GPU; use smoke for program validation.')
 print('PROFILE',cfg.profile,'device',dev,'tasks',cfg.tasks(),'seeds',cfg.seeds())
 if cfg.profile=='smoke':caches={'smoke_2d':make_synthetic_cache(cfg)}
 else:
  data_dir=Path(cfg.data_root) if cfg.data_root else Path(cfg.root)/'official_data'
  caches={}
  for task in cfg.tasks():
   path=fetch_official(data_dir,task)
   caches[task]=make_cache(task,path,cfg,Path(cfg.root)/'window_cache')
 manifests=[];results=[];failures=[]
 for task in cfg.tasks():
  cache=caches[task];sp=cache['splits']['train']['h'].shape
  channel=sp[2];dim=len(sp)-3
  spec={'task':task,'channels':channel,'spatial_dim':dim,'spatial_shape':list(sp[3:]),'windows':{k:len(v['h']) for k,v in cache['splits'].items()},'ids':cache.get('ids',{}),'source_md5':TASKS[task][2] if task!='smoke_2d' else 'synthetic'}
  match={}
  for kind in ('V1_CIDM_Complete','V2_Ported_R5'):
   match[kind]=choose_match(kind,channel,cfg)
   model=UnifiedModel(kind,channel,match[kind]['d'],match[kind]['hidden'],cfg.history).to(dev)
   g=smoke_grad(model,dev,cfg,dim);match[kind]['preflight']=g
   del model
  spec['matched']=match;manifests.append(spec)
  print('[preflight]',task,match,flush=True)
 jsonsave(out/'data_and_architecture_contract.json',{'config':asdict(cfg),'source_sha':ARCHIVE_SHA,'tasks':manifests,
    'warning':'v2 is a documented channel-port from two-channel synthetic architecture, not same checkpoint as R5-R1',
    'data_policy':'official DaRUS MD5 checked, shared windows, train-only norm, no future inference controls'})
 for seed in cfg.seeds():
  for task in cfg.tasks():
   cache=caches[task];sp=cache['splits']['train']['h'].shape;dim=len(sp)-3;channel=sp[2]
   batch=cfg.batch_1d if dim==1 else cfg.batch_2d
   tr=load_data(cache,'train',batch,seed,True)
   va=load_data(cache,'val',batch,seed,False)
   te=load_data(cache,'test',batch,seed,False)
   spec=next(x for x in manifests if x['task']==task)
   for kind in ('V1_CIDM_Complete','V2_Ported_R5'):
    runid=f'{task}__{kind}__seed{seed}';fp=out/(runid+'.json')
    if fp.exists():
     try:
      prior=json.loads(fp.read_text())
      if prior.get('status')=='COMPLETED' and prior.get('total_steps')==cfg.updates():
       results.append(prior);print('[skip completed]',runid);continue
     except:pass
    seed_all(seed)
    conf=spec['matched'][kind]
    model=UnifiedModel(kind,channel,conf['d'],conf['hidden'],cfg.history).to(dev)
    try:
     trained=fit(model,tr,va,cfg,dev,runid,out,seed=seed)
     metrics=assess(model,te,dev,cfg,'off')
     lead_rmse=per_lead_assess(model,te,dev,cfg)
     timing=measure_latency(model,te,dev,cfg.eval_horizon)
     mechanism=rollout_diagnostics(model,te,dev,cfg.eval_horizon)
     best_model_path=out/'best_models'/f'{runid}.pt';best_model_path.parent.mkdir(parents=True,exist_ok=True)
     torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
       'model_kind':kind,'config':conf,'data_signature':cache['signature'],'seed':seed,'source_sha':ARCHIVE_SHA},best_model_path)
     noflux=None
     if kind=='V2_Ported_R5':
      coefficients={k:float(v.detach()) for k,v in model.model.coefficients().items()}
      # Mechanism dependence on same learned model (not trained ablation).
      model.model.flux.forward_saved=model.model.flux.forward
      model.model.flux.forward=lambda z:torch.zeros_like(z)
      nf=assess(model,te,dev,cfg,'off');noflux=nf['model_'+str(cfg.eval_horizon)]
      del model.model.flux.forward;coefficients['no_flux_test_rmse']=noflux
     else:coefficients={}
     row={'status':'COMPLETED','seed':seed,'task':task,'model':kind,'parameter_count':params(model),
       'model_config':conf,'total_steps':cfg.updates(),'train':trained,'test':metrics,'test_per_lead_rmse':lead_rmse,
       'latency':timing,'mechanism_diagnostic':mechanism,'best_model_path':str(best_model_path),
       'coefficients':coefficients}
     jsonsave(fp,row);results.append(row)
    except Exception as e:
     row={'status':'FAILED','seed':seed,'task':task,'model':kind,'error':repr(e),'traceback':traceback.format_exc()[-2400:]}
     jsonsave(fp,row);failures.append(row);print('[FAILED]',runid,row['error'],flush=True)
    finally:
     del model
     if torch.cuda.is_available():torch.cuda.empty_cache()
 summary=[]
 for task in cfg.tasks():
  for lead in (cfg.train_horizon,cfg.eval_horizon):
   paired=[]
   for seed in cfg.seeds():
    v1=next((r for r in results if r['task']==task and r['seed']==seed and r['model']=='V1_CIDM_Complete' and r['status']=='COMPLETED'),None)
    v2=next((r for r in results if r['task']==task and r['seed']==seed and r['model']=='V2_Ported_R5' and r['status']=='COMPLETED'),None)
    if v1 is not None and v2 is not None:
     a=v1['test']['model_'+str(lead)];b=v2['test']['model_'+str(lead)];naive=min(v2['test']['persistence_'+str(lead)],v2['test']['trend_'+str(lead)])
     paired.append({'seed':seed,'v1_rmse':a,'v2_rmse':b,'v2_improvement_pct':100*(1-b/a),'best_naive_rmse':naive})
   if paired:
    summary.append({'task':task,'horizon':lead,'n_paired_seeds':len(paired),
       'v1_mean_rmse':float(np.mean([x['v1_rmse'] for x in paired])),
       'v2_mean_rmse':float(np.mean([x['v2_rmse'] for x in paired])),
       'paired_improvement_pct_mean':float(np.mean([x['v2_improvement_pct'] for x in paired])),
       'paired_results':paired,
       'all_v2_beat_v1':all(x['v2_rmse']<x['v1_rmse'] for x in paired),
       'all_v2_beat_strong_naive':all(x['v2_rmse']<x['best_naive_rmse'] for x in paired)})
 # Gate only evaluates PROGRAM/SIGNAL quality; pilot cannot qualify architecture maturity.
 expected=len(cfg.tasks())*len(cfg.seeds())*2
 complete=len([r for r in results if r.get('status')=='COMPLETED'])
 gate={'source_sha_pass':True,'completed_runs':complete,'expected_runs':expected,'all_complete':complete==expected,
       'all_config_matched':all((cfg.profile=='smoke' or s['matched'][k]['relative_error']<=cfg.match_tolerance) for s in manifests for k in ('V1_CIDM_Complete','V2_Ported_R5')),
       'all_v2_better_v1':bool(summary) and all(x['all_v2_beat_v1'] for x in summary),
       'all_v2_better_strong_naive':bool(summary) and all(x['all_v2_beat_strong_naive'] for x in summary),
       'enough_independent_seeds':len(cfg.seeds())>=3,
       'all_eight_pde_families':len(cfg.tasks())==8,
       'official_data':cfg.profile!='smoke',
       'state':'DIAGNOSTIC_ONLY' if cfg.profile in ('smoke','pilot','pilot_r1') else 'FORMAL_COMPARISON_UNQUALIFIED',
       'convergence_proven':False,'no_claim_all_pde_superiority':True}
 if cfg.profile in ('formal8','canonical8') and all([gate['all_complete'],gate['all_config_matched'],gate['all_v2_better_v1'],gate['all_v2_better_strong_naive'],gate['enough_independent_seeds'],gate['all_eight_pde_families']]):
  gate['state']='EIGHT_PDE_V1_SUPERIORITY_PRELIMINARY_PASS_EXTERNAL_MODELS_PENDING'
 jsonsave(out/'B0_REPORT.json',{'gate':gate,'config':asdict(cfg),'summary':summary,'failures':failures,'manifest':manifests})
 # Descriptive multi-seed confidence summary and frozen mechanism diagnostics.
 paired_task_summary=[]
 for row in summary:
  diffs=[x['v2_improvement_pct'] for x in row['paired_results']]
  paired_task_summary.append({'task':row['task'],'lead':row['horizon'],
     'n_seeds':len(diffs),'mean_improvement_pct':float(np.mean(diffs)),
     'median_improvement_pct':float(np.median(diffs)),
     'min_improvement_pct':float(np.min(diffs)),
     'max_improvement_pct':float(np.max(diffs)),
     'n_v2_wins':sum(x>0 for x in diffs),
     'remark':'three seeds are not sufficient to assert statistical superiority'})
 jsonsave(out/'B0_R1_PAIRED_DIAGNOSTICS.json',{'paired_task_summary':paired_task_summary,
    'warning':'12-step validation selection used for BOTH models, not comparable by itself to old 6-step selected B0 pilot; insufficient converged for v2 final claim',
    'sync_mode':'OFF for all v2 runs; not a state synchronization evaluation',
    'extra':'No SOC in v2 but v1 SOC and BER structure retained as archival baseline; interpretation is architecture-level comparison'})
 # Plot - failure plot should still be generated.
 try:
  import matplotlib.pyplot as plt
  xx=[r for r in summary if r['horizon']==cfg.eval_horizon]
  if xx:
   fig,ax=plt.subplots(figsize=(max(7,1.3*len(xx)),4));ax.bar([r['task'] for r in xx],[r['paired_improvement_pct_mean'] for r in xx]);ax.axhline(0,color='gray',lw=.8)
   ax.set_ylabel('v2 versus exact v1: paired RMSE improvement (%)');ax.tick_params(axis='x',labelrotation=25);fig.tight_layout();fig.savefig(out/'paired_comparison.png',dpi=180);plt.close(fig)
 except Exception as e:print('[plot skipped]',repr(e))
 archive=out/'CIDM_v2_B0_PDE_FAIR_REPORTS.zip'
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for p in sorted(out.glob('*.json')):z.write(p,p.name)
  if (out/'paired_comparison.png').exists():z.write(out/'paired_comparison.png','paired_comparison.png')
  if cfg.profile in ('smoke','pilot','pilot_r1') or os.getenv('CIDM_B0_ZIP_WEIGHTS','0')=='1':
   for p in sorted((out/'best_models').glob('*.pt')):z.write(p,'best_models/'+p.name)
 print('[FINAL]',gate, '\nReport ZIP:',archive,flush=True)
 return {'gate':gate,'summary':summary,'archive':str(archive),'out':str(out)}

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--profile',default=os.getenv('CIDM_B0_PROFILE','smoke'),choices=['smoke','pilot','pilot_r1','formal8','canonical8'])
 parser.add_argument('--root',default=os.getenv('CIDM_B0_ROOT','/mnt/data/CIDM_v2_B0_R1_demo'))
 parser.add_argument('--data-root',default=os.getenv('CIDM_B0_DATA_ROOT',''))
 args=parser.parse_args();run(Config(profile=args.profile,root=args.root,data_root=args.data_root))
