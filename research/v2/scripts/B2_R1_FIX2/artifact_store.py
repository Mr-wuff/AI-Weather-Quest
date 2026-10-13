"""CIDM experiment artifact resolver and publisher.
No secrets in source code. Repo discovery does not imply archives actually exist.
Uses public repository API and optional authenticated tokens; fallbacks to Google Drive/local upload.
"""
from pathlib import Path
import os,json,hashlib,zipfile,shutil,time,base64,requests,datetime
from urllib.parse import quote

BASENAMES=('CIDM_v2_B0_PDE_FAIR_REPORTS(1).zip','CIDM_v2_B0_PDE_FAIR_REPORTS.zip')

def sha256(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()

def verify_b0(p):
 try:
  with zipfile.ZipFile(p) as z:
   if z.testzip() is not None:return False,'CORRUPT_ZIP'
   names=set(z.namelist())
   r=json.loads(z.read('B0_REPORT.json'))
   g=r['gate'];cfg=r['config']
   if cfg['profile']!='pilot_r1' or cfg['eval_horizon']!=12 or cfg['train_horizon']!=6:
    return False,'WRONG_PROFILE_OR_HORIZON'
   if int(g['completed_runs'])!=18 or not g['official_data'] or not g['all_complete']:
    return False,'NOT_B0_R1_18_OFFICIAL_RUNS'
   expected={f'best_models/{t}__{k}__seed{s}.pt' for t in ('burgers_1d','reacdiff_2d','swe_2d')
             for k in ('V1_CIDM_Complete','V2_Ported_R5') for s in (17,41,73)}
   if not expected.issubset(names):return False,'MISSING_CHECKPOINTS'
   return True,'B0_R1_OFFICIAL_18_PASS'
 except Exception as e:return False,'UNREADABLE:'+type(e).__name__+':'+str(e)[:100]

def candidate_files(roots):
 seen=set()
 for root in roots:
  p=Path(root).expanduser()
  if p.is_file():c=[p]
  elif p.exists():
   c=[]
   for pat in ('CIDM_v2_B0_PDE_FAIR_REPORTS*.zip','*B0*PDE*FAIR*REPORTS*.zip'):
    c.extend(p.glob(pat))
    c.extend((p/'CIDM_v2_B0_FairPDE').glob(pat))
  else:c=[]
  for x in c:
   if str(x.resolve()) not in seen:
    seen.add(str(x.resolve()));yield x

def get_repo_secret(label):
 val=os.getenv(label,'')
 if val:return val
 try:
  from google.colab import userdata
  return userdata.get(label) or ''
 except Exception:return ''

def _req(url,token='',**kwargs):
 headers=dict(kwargs.pop('headers',{}))
 if token:headers['Authorization']='Bearer '+token
 headers.setdefault('User-Agent','CIDM-v2-B2-R1-Artifact-Resolver')
 return requests.get(url,headers=headers,timeout=25,**kwargs)

def fetch_file_to(url,dest,token='',expected_sha=None):
 dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
 tmp=dest.with_suffix(dest.suffix+'.part')
 with _req(url,token=token,stream=True) as res:
  res.raise_for_status()
  if int(res.headers.get('Content-Length','0'))>200_000_000:
   raise RuntimeError('Unexpected huge artifact (>200MB): '+url)
  with tmp.open('wb') as f:
   for data in res.iter_content(1<<20):
    if data:f.write(data)
  if tmp.stat().st_size>200_000_000:raise RuntimeError('Unexpected huge artifact (>200MB)')
 if expected_sha and sha256(tmp)!=expected_sha:
  tmp.unlink(missing_ok=True);raise RuntimeError('REMOTE_SHA256_MISMATCH')
 ok,reason=verify_b0(tmp)
 if not ok:tmp.unlink(missing_ok=True);raise RuntimeError('REMOTE_NOT_VALID_B0_R1: '+reason)
 tmp.replace(dest)
 return dest

def discover_github_release_assets(repo='Mr-wuff/AI-Weather-Quest',token=''):
 try:
  r=_req(f'https://api.github.com/repos/{repo}/releases?per_page=30',token=token)
  if not r.ok:return []
  assets=[]
  for release in r.json():
   for a in release.get('assets',[]):
    if 'B0_PDE_FAIR_REPORTS' in a.get('name',''):
     assets.append((a['browser_download_url'],f"GitHub release {release.get('tag_name','')}/{a['name']}"))
  return assets
 except Exception:return []

def discover_hf_files(repo_hint='',owner='wuff-mann',token='',max_repos=30):
 """Only actual listed HF model/dataset repos. Never manufacture repo IDs."""
 try:
  from huggingface_hub import HfApi
 except ImportError:return [],'huggingface_hub NOT INSTALLED'
 api=HfApi(token=token or None)
 repo_infos=[]
 try:
  if repo_hint:
   repo_type=os.getenv('CIDM_HF_REPO_TYPE','dataset')
   repo_infos=[(repo_hint,repo_type)]
  else:
   for info in list(api.list_datasets(author=owner,limit=max_repos)):
    repo_infos.append((info.id,'dataset'))
   for info in list(api.list_models(author=owner,limit=max_repos)):
    repo_infos.append((info.id,'model'))
  out=[]
  for repo_id,repo_type in repo_infos[:max_repos]:
   try:
    names=api.list_repo_files(repo_id,repo_type=repo_type)
    for name in names:
     if 'B0_PDE_FAIR_REPORTS' in name and name.endswith('.zip'):
      out.append((repo_id,repo_type,name))
   except Exception:continue
  return out,'SCAN_FINISHED'
 except Exception as exc:return [],'HF_SCAN_FAILED:'+str(exc)[:180]

def resolve_b0(roots,cache_dir,github_repo='Mr-wuff/AI-Weather-Quest',hf_repo='',hf_user='wuff-mann',auto_search=True):
 log=[]
 for f in candidate_files(roots):
  ok,reason=verify_b0(f);log.append({'origin':str(f),'check':reason})
  if ok:return str(f),log
 if not auto_search:raise RuntimeError('No valid local B0-R1 archive: '+str(log))
 cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
 gh=get_repo_secret('GH_TOKEN');hf=get_repo_secret('HF_TOKEN')
 for url,kind in discover_github_release_assets(github_repo,gh):
  try:
   dest=fetch_file_to(url,cache_dir/'CIDM_v2_B0_R1_FROM_GITHUB.zip',gh)
   log.append({'origin':kind,'check':'VERIFIED'});return str(dest),log
  except Exception as exc:log.append({'origin':kind,'check':'FAILED:'+str(exc)[:180]})
 hfitems,scan=discover_hf_files(hf_repo,hf_user,hf);log.append({'origin':'HF_SCAN','check':scan,'candidates':len(hfitems)})
 for rid,typ,name in hfitems:
  try:
   from huggingface_hub import hf_hub_download
   path=hf_hub_download(repo_id=rid,filename=name,repo_type=typ,token=hf or None,
                        local_dir=str(cache_dir/'hf'))
   ok,why=verify_b0(path);log.append({'origin':rid+'/'+name,'check':why})
   if ok:return path,log
  except Exception as exc:log.append({'origin':rid+'/'+name,'check':'FAILED:'+str(exc)[:160]})
 raise FileNotFoundError('Could not locate verified B0-R1 18-run archive on disk, GitHub releases, or HF. '
                         'Set CIDM_HF_REPO, or upload/mount the correct B0-R1 archive. Log: '+json.dumps(log,ensure_ascii=False))

def upload_hf(result_zip,repo_id,token,remote_path):
 if not repo_id:return 'SKIPPED_HF_REPO_NOT_CONFIGURED'
 if not token:return 'SKIPPED_HF_WRITE_TOKEN_NOT_AVAILABLE'
 from huggingface_hub import HfApi
 HfApi(token=token).upload_file(path_or_fileobj=str(result_zip),path_in_repo=remote_path,
                                repo_id=repo_id,repo_type='dataset',commit_message='Archive CIDM-v2 B2-R1 reproducible result')
 return f'UPLOADED_TO_HF_DATASET: {repo_id}/{remote_path}'

def upload_github_small(path,repo,token,remote_path):
 if not token:return 'SKIPPED_GITHUB_WRITE_TOKEN_NOT_AVAILABLE'
 p=Path(path)
 if p.stat().st_size>950_000:return 'SKIPPED_GITHUB_CONTENTS_LIMIT'
 url=f'https://api.github.com/repos/{repo}/contents/{quote(remote_path,safe="/")}'
 headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','User-Agent':'CIDM-ArtifactSync'}
 payload={'message':'Archive CIDM-v2 B2-R1 experiment report','content':base64.b64encode(p.read_bytes()).decode()}
 existing=requests.get(url,headers=headers,timeout=25)
 if existing.status_code==200:payload['sha']=existing.json()['sha']
 r=requests.put(url,headers=headers,json=payload,timeout=55)
 if r.status_code not in (200,201):raise RuntimeError('GitHub report upload '+str(r.status_code)+' '+r.text[:140])
 return f'GITHUB_OK: {repo}/{remote_path}'

def upload_github_release(path,repo,token,release_tag):
 if not token:return 'SKIPPED_GITHUB_WRITE_TOKEN_NOT_AVAILABLE'
 if not release_tag:return 'SKIPPED_GITHUB_RELEASE_TAG_NOT_CONFIGURED'
 headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','User-Agent':'CIDM-ArtifactSync'}
 url=f'https://api.github.com/repos/{repo}/releases/tags/{quote(release_tag)}'
 r=requests.get(url,headers=headers,timeout=30)
 if r.status_code==404:
  r=requests.post(f'https://api.github.com/repos/{repo}/releases',headers=headers,
                  json={'tag_name':release_tag,'name':release_tag,'draft':False,'prerelease':True},timeout=30)
 r.raise_for_status();release=r.json();name=Path(path).name
 # Versioned assets by digest; never replace silently.
 name=Path(path).stem+'_'+sha256(path)[:12]+Path(path).suffix
 url=f"https://uploads.github.com/repos/{repo}/releases/{release['id']}/assets?name={quote(name)}"
 with open(path,'rb') as f:
  res=requests.post(url,headers={**headers,'Content-Type':'application/zip'},data=f,timeout=180)
 if res.status_code==422:return 'GITHUB_RELEASE_IDENTICAL_NAME_PRESENT'
 res.raise_for_status();return 'GITHUB_RELEASE_OK: '+res.json()['browser_download_url']

def publish_results(result_zip,output_dir,project='CIDM_v2_B2_R1',hf_repo='',github_repo='Mr-wuff/AI-Weather-Quest',enable_network=True):
 """Always retains result ZIP locally; optionally publishes to HF, GitHub and mounted Drive."""
 p=Path(result_zip);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
 if not p.exists():raise FileNotFoundError(p)
 report={'result_zip':str(p),'zip_sha256':sha256(p),'size_bytes':p.stat().st_size,'operations':{},
         'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 drive=Path('/content/drive/MyDrive/CIDM_v2/experiments')/project
 if Path('/content/drive/MyDrive').exists():
  try:
   drive.mkdir(parents=True,exist_ok=True);shutil.copy2(p,drive/p.name)
   report['operations']['google_drive']='COPIED: '+str(drive/p.name)
  except Exception as exc:report['operations']['google_drive']='FAILED:'+str(exc)[:180]
 else:report['operations']['google_drive']='SKIPPED_DRIVE_NOT_MOUNTED'
 if enable_network:
  hf=get_repo_secret('HF_TOKEN');gh=get_repo_secret('GH_TOKEN')
  remote=f'research/v2/{project}/{p.name}'
  for label,fun in [('huggingface',lambda:upload_hf(p,hf_repo,hf,remote)),
                    ('github_report',lambda:upload_github_small(out/'report.json',github_repo,gh,
                        f'research/v2/experiment_runs/{project}/report.json')),
                    ('github_release',lambda:upload_github_release(p,github_repo,gh,f'{project}-archive'))]:
   try:report['operations'][label]=fun()
   except Exception as e:report['operations'][label]='FAILED: '+type(e).__name__+': '+str(e)[:300]
 else:report['operations']['network']='SKIPPED_SMOKE_OR_OFFLINE'
 target=out/'artifact_publish_manifest.json';target.write_text(json.dumps(report,ensure_ascii=False,indent=2))
 return report

def sync_training_milestone(checkpoint_path, progress_record, repo_id='',github_repo='Mr-wuff/AI-Weather-Quest'):
 """Best-effort remote checkpoint+progress sync. Never makes training dependent on network health."""
 if os.getenv('CIDM_LIVE_SYNC','1')!='1':return {'status':'DISABLED'}
 result={}
 hf=get_repo_secret('HF_TOKEN');gh=get_repo_secret('GH_TOKEN')
 label=Path(checkpoint_path).stem
 if hf and repo_id:
  try:
   result['hf']=upload_hf(checkpoint_path,repo_id,hf,f'research/v2/B2_R1_FIX2/checkpoints/{label}.pt')
  except Exception as e:result['hf']='FAILED:'+str(e)[:150]
 else:result['hf']='NO_HF_WRITE_ACCESS_OR_REPO'
 if gh:
  try:
   temp=Path(checkpoint_path).with_name(label+'_progress.json')
   temp.write_text(json.dumps(progress_record,indent=2,ensure_ascii=False))
   result['github']=upload_github_small(temp,github_repo,gh,
       f'research/v2/experiment_runs/B2_R1_FIX2/progress/{label}.json')
  except Exception as e:result['github']='FAILED:'+str(e)[:150]
 else:result['github']='NO_GITHUB_WRITE_TOKEN'
 return result
