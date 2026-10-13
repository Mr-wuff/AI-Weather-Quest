"""Remote-first experiment asset resolver and conservative evidence publisher.
Only operates on HF model repo and GitHub Releases/Contents. NO GOOGLE DRIVE.
"""
import ast,contextlib,hashlib,io,json,os,pathlib,re,shutil,sys,time,urllib.request,urllib.parse,zipfile

HF_ID='wuff-mann/AI_Weather_Quest'
GH_ID='Mr-wuff/AI-Weather-Quest'
EXPECTED_B0_SHA='69552b29e169ec22ee85d9c1b1a678fdb188f6f25dc18f33b927b88f4dddffd4'
B0_NAME='CIDM_v2_B0_PDE_FAIR_REPORTS(1).zip'
FIX2_NAME='CIDM_v2_B2_R1_FIX2_ExactReplay_AutoRepoSync_Colab_STANDALONE.ipynb'


def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()


def _hf():
    from huggingface_hub import HfApi
    return HfApi(token=os.getenv('HF_TOKEN') or None)


def discover_hf():
    try:return _hf().list_repo_files(HF_ID,repo_type='model')
    except Exception as e:
        print('[HF LIST WARN]',repr(e));return []


def github_releases():
    try:
        headers={'User-Agent':'cidm-b2r2-v1','Accept':'application/vnd.github+json'}
        key=os.getenv('GH_TOKEN') or os.getenv('GITHUB_TOKEN')
        if key:headers['Authorization']='Bearer '+key
        req=urllib.request.Request(f'https://api.github.com/repos/{GH_ID}/releases?per_page=100',headers=headers)
        with urllib.request.urlopen(req,timeout=30) as resp:return json.load(resp)
    except Exception as e:
        print('[GITHUB RELEASES WARN]',repr(e));return []


def _download(url,out):
    req=urllib.request.Request(url,headers={'User-Agent':'cidm-b2r2-v1'})
    with urllib.request.urlopen(req,timeout=100) as r,open(out,'wb') as f:
        shutil.copyfileobj(r,f)
    return out


def discover_assets(work):
    """Search remote actual filenames, not report mentions/assumed paths."""
    work=pathlib.Path(work);work.mkdir(parents=True,exist_ok=True)
    files=discover_hf(); releases=github_releases(); result=[]
    for rel in files:
        base=rel.rsplit('/',1)[-1]
        kind=('parent' if 'B0' in base and ('FAIR' in base or 'R1' in base) and base.endswith('.zip')
              else 'fix2' if ('B2_R1_FIX2' in base or base=='b2_r1_flux_control.py') and base.endswith(('.py','.ipynb','.zip'))
              else None)
        if kind:result.append({'kind':kind,'source':'HF-model','path':rel,'filename':base})
    for release in releases:
        for a in release.get('assets',[]):
            name=a.get('name','')
            kind=('parent' if 'B0' in name and name.endswith('.zip') else
                  'fix2' if 'B2_R1_FIX2' in name else None)
            if kind:result.append({'kind':kind,'source':'GitHub-Release','url':a.get('browser_download_url'),'filename':name})
    return result


def fetch_asset(item,folder):
    folder=pathlib.Path(folder);folder.mkdir(parents=True,exist_ok=True)
    out=folder/item['filename']
    if out.is_file():return out
    if item['source']=='HF-model':
        from huggingface_hub import hf_hub_download
        fp=hf_hub_download(HF_ID,filename=item['path'],repo_type='model',token=os.getenv('HF_TOKEN') or None)
        shutil.copy2(fp,out)
    elif item['source']=='GitHub-Release': _download(item['url'],out)
    else:raise ValueError(item)
    return out


def verify_parent(path):
    """Reject archives with wrong SHA or without enough real .pt checkpoint files."""
    path=pathlib.Path(path)
    if not zipfile.is_zipfile(path):raise RuntimeError('Parent is not ZIP: '+str(path))
    actual=sha256(path)
    if actual!=EXPECTED_B0_SHA:
        raise RuntimeError('Parent SHA256 mismatch! Expected '+EXPECTED_B0_SHA+' observed '+actual+'; no audit permitted.')
    with zipfile.ZipFile(path) as z:
        bad=z.testzip();names=z.namelist()
        if bad:raise RuntimeError('Parent ZIP CRC failure: '+bad)
        pt=[n for n in names if n.lower().endswith(('.pt','.pth'))]
        if len(pt)<18:raise RuntimeError(f'Parent must have >=18 weight files; only {len(pt)} detected; NO AUDIT')
    return {'status':'PASS','file':str(path),'sha256':actual,'count_weight_files':len(pt)}


def _safe_extract_zip(zpath,folder):
    """Safe, constrained, flat source extraction. Report-only archives contain no usable driver."""
    folder=pathlib.Path(folder);folder.mkdir(parents=True,exist_ok=True)
    found=[]
    with zipfile.ZipFile(zpath) as z:
        if z.testzip(): raise RuntimeError('SOURCE_ZIP_CRC_FAILED')
        for info in z.infolist():
            member=pathlib.PurePosixPath(info.filename.replace('\\','/'))
            if '..' in member.parts or member.is_absolute() or not member.name: continue
            if member.suffix.lower() not in ('.py','.ipynb'): continue
            if info.file_size>12_000_000 or info.compress_size>12_000_000:continue
            out=folder/member.name
            out.write_bytes(z.read(info));found.append(out)
    return found


def static_extract_py_from_ipynb(ipath,folder):
    """Statically unpack literal %%writefile / Path.write_text source; never execute notebook cells."""
    folder=pathlib.Path(folder);folder.mkdir(parents=True,exist_ok=True)
    obj=json.loads(pathlib.Path(ipath).read_text(encoding='utf-8'))
    extracted=[]
    for cell in obj.get('cells',[]):
        if cell.get('cell_type')!='code':continue
        s=''.join(cell.get('source',[]))
        m=re.match(r'^\s*%%writefile\s+([^\n]+\.py)\s*\n([\s\S]*)$',s)
        if m:
            name=pathlib.PurePosixPath(m.group(1).strip().strip('"\\\'')).name
            out=folder/name;out.write_text(m.group(2),encoding='utf-8');extracted.append(out);continue
        try: tree=ast.parse(s)
        except SyntaxError:continue
        env={}
        for node in tree.body:
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                try: env[node.targets[0].id]=ast.literal_eval(node.value)
                except Exception: pass
            for candidate in ast.walk(node):
                if not (isinstance(candidate,ast.Call) and isinstance(candidate.func,ast.Attribute) and candidate.func.attr=='write_text'):continue
                try:
                    if not candidate.args:continue
                    v=candidate.args[0]
                    content=env.get(v.id) if isinstance(v,ast.Name) else ast.literal_eval(v)
                    path_literals=[x.value for x in ast.walk(candidate.func.value)
                                   if isinstance(x,ast.Constant) and isinstance(x.value,str)
                                   and x.value.endswith('.py')]
                    if not isinstance(content,str) or not path_literals:continue
                    name=pathlib.PurePosixPath(path_literals[0]).name
                    out=folder/name;out.write_text(content,encoding='utf-8');extracted.append(out)
                except Exception: continue
    return extracted


def inspect_driver(path):
    """Static interface sanity check, followed by exact B0 replay at runtime before evidence may be accepted."""
    p=pathlib.Path(path)
    if not p.is_file(): return {'valid':False,'reason':'NOT_PRESENT'}
    try:
        raw=p.read_bytes()
        if len(raw)<500 or len(raw)>15_000_000: return {'valid':False,'reason':'SOURCE_SIZE_INVALID'}
        code=raw.decode('utf-8-sig')
        tree=ast.parse(code,filename=str(p))
        methods=set();classes=set()
        for item in tree.body:
            if isinstance(item,(ast.FunctionDef,ast.AsyncFunctionDef)): methods.add(item.name)
            if isinstance(item,ast.ClassDef): classes.add(item.name)
        # Allow some modules to forward-export the controlled class, but require defined public API.
        missing=[x for x in ('assess','run') if x not in methods]
        missing += [x for x in ('ControlledV2',) if x not in classes and x not in code]
        return {'valid':not missing,'missing_api':missing,'size':len(raw),'sha256':sha256(p),'reason':'OK' if not missing else 'MISSING_REQUIRED_API'}
    except Exception as e: return {'valid':False,'reason':'PARSE_ERROR: '+repr(e)[:240]}


def prepare_sources(assets,work):
    work=pathlib.Path(work);source=work/'source';source.mkdir(exist_ok=True,parents=True)
    allpaths=[];rejected=[]
    for raw in assets:
        p=pathlib.Path(raw)
        if not p.is_file():continue
        try:
            if p.suffix=='.py':
                dst=source/p.name;shutil.copy2(p,dst);allpaths.append(dst)
            elif p.suffix=='.ipynb': allpaths+=static_extract_py_from_ipynb(p,source)
            elif p.suffix=='.zip': allpaths+=_safe_extract_zip(p,source)
        except Exception as e:rejected.append({'file':p.name,'error':repr(e)[:300]})
    for p in list(allpaths):
        if p.suffix=='.ipynb':
            try:allpaths+=static_extract_py_from_ipynb(p,source)
            except Exception as e:rejected.append({'file':p.name,'error':repr(e)[:300]})
    driver=source/'b2_r1_flux_control.py'
    checked=inspect_driver(driver)
    return {'ready':checked['valid'],'source_dir':str(source),'files':sorted(set(p.name for p in allpaths)),
            'driver_validation':checked,'rejected':rejected,
            'missing':([] if checked['valid'] else ['an authentic FIX2 b2_r1_flux_control.py with ControlledV2 / assess / run API'])}


def classify_uploaded_file(path):
    """Detect report ZIP and prevent publishing it as FIX2 source."""
    p=pathlib.Path(path);name=p.name.lower()
    if p.suffix.lower()=='.zip' and 'b0' in name and 'fair' in name:
        try:return ('parent',verify_parent(p))
        except Exception as e:return ('rejected',{'reason':'INVALID_PARENT: '+str(e)[:260]})
    if p.suffix.lower() not in ('.py','.ipynb','.zip'):
        return 'rejected',{'reason':'UNSUPPORTED_EXTENSION'}
    if p.suffix.lower()=='.zip':
        try:
            with zipfile.ZipFile(p) as z:
                names=[n.lower() for n in z.namelist()]
            if not any(n.endswith('.py') or n.endswith('.ipynb') for n in names):
                return 'rejected',{'reason':'REPORT_ONLY_ZIP: no Python/Notebook source. Do not upload B2_R1_FLUX_CONTROL_QUALIFICATION_REPORTS.zip'}
        except Exception as e:return 'rejected',{'reason':'NOT_VALID_ZIP: '+repr(e)[:150]}
    return 'source_candidate',{'reason':'AWAITING_STATIC_SOURCE_VALIDATION'}

def publish_hf(path,relative):
    token=os.getenv('HF_TOKEN')
    if not token:return {'status':'SKIPPED_NO_HF_TOKEN'}
    try:
        info=_hf().upload_file(path_or_fileobj=str(path),path_in_repo=relative,repo_id=HF_ID,repo_type='model',
                               commit_message='Archive CIDM-v2 reproducible '+pathlib.Path(path).name)
        return {'status':'SUCCESS','url':str(info)}
    except Exception as e:return {'status':'FAILED','error':str(e)[:500]}


def publish_gh_release(path,tag):
    token=os.getenv('GH_TOKEN') or os.getenv('GITHUB_TOKEN')
    if not token:return {'status':'SKIPPED_NO_GH_TOKEN'}
    try:
        from urllib.error import HTTPError
        headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
                 'X-GitHub-Api-Version':'2022-11-28','User-Agent':'cidm-b2r2-v1'}
        url=f'https://api.github.com/repos/{GH_ID}/releases/tags/{tag}'
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=40) as r:release=json.load(r)
        except HTTPError as e:
            if e.code!=404:raise
            data=json.dumps({'tag_name':tag,'name':tag,'body':'Verified checkpoint/evidence archive; see manifest for hashes.'}).encode()
            req=urllib.request.Request(f'https://api.github.com/repos/{GH_ID}/releases',data=data,
                                       headers={**headers,'Content-Type':'application/json'},method='POST')
            with urllib.request.urlopen(req,timeout=40) as r:release=json.load(r)
        asset_name=pathlib.Path(path).stem+'_'+sha256(path)[:12]+pathlib.Path(path).suffix
        for prior in release.get('assets',[]):
            if prior.get('name')==asset_name:
                return {'status':'SUCCESS_ALREADY_PRESENT_SAME_HASH','url':prior.get('browser_download_url')}
        upload_url=release['upload_url'].split('{')[0]+'?name='+urllib.parse.quote(asset_name)
        import requests
        with open(path,'rb') as stream:
            resp=requests.post(upload_url,data=stream,headers={**headers,'Content-Type':'application/zip'},timeout=600)
        resp.raise_for_status();asset=resp.json()
        return {'status':'SUCCESS','url':asset.get('browser_download_url')}
    except Exception as e:return {'status':'FAILED','error':repr(e)[:500]}


def publish_github_small(path,remote):
    token=os.getenv('GH_TOKEN') or os.getenv('GITHUB_TOKEN')
    if not token:return {'status':'SKIPPED_NO_GH_TOKEN'}
    import base64,urllib.error
    headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
             'X-GitHub-Api-Version':'2022-11-28','User-Agent':'cidm-b2r2-v1'}
    url=f'https://api.github.com/repos/{GH_ID}/contents/{remote}'
    try:
        sha=None
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=20) as r:sha=json.load(r).get('sha')
        except urllib.error.HTTPError as e:
            if e.code!=404:raise
        body={'message':'Archive CIDM-v2 B2-R2 diagnostic evidence','content':base64.b64encode(pathlib.Path(path).read_bytes()).decode(),
              'branch':'main'}
        if sha:body['sha']=sha
        req=urllib.request.Request(url,data=json.dumps(body).encode(),headers={**headers,'Content-Type':'application/json'},method='PUT')
        with urllib.request.urlopen(req,timeout=90) as r:result=json.load(r)
        return {'status':'SUCCESS','commit_sha':result['commit']['sha']}
    except Exception as e:return {'status':'FAILED','error':repr(e)[:500]}


def package_results(folder,root):
    folder=pathlib.Path(folder);root=pathlib.Path(root);root.mkdir(parents=True,exist_ok=True)
    out=root/'CIDM_v2_B2_R2_P0_OBSERVABLE_ALIGNMENT_REPORTS.zip'
    manifest=[]
    for f in sorted(folder.rglob('*')):
        if f.is_file():manifest.append({'path':str(f.relative_to(folder)),'bytes':f.stat().st_size,'sha256':sha256(f)})
    (folder/'SHA256_MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in folder.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(folder))
    pathlib.Path(str(out)+'.sha256').write_text(sha256(out)+'  '+out.name+'\n')
    return out

if __name__=='__main__':
    print('B2 R2 repository helper ready; HF=',HF_ID,'GitHub=',GH_ID)
