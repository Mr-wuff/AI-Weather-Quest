# -*- coding: utf-8 -*-
"""Canonical A15 (P1–P7) HF archive loader. Refuse data drift or missing index."""
from pathlib import Path
import hashlib,json,os
from huggingface_hub import hf_hub_download

REPO="wuff-mann/AI_Weather_Quest"
PREFIX="research_artifacts/a15/v20261019"
SHA={
 "P1":"a8025c19bc9109a639891c8526bbce1a7bff18a0c5451e6e3b4d15b486b1fa99",
 "P2":"dd380fb310546b080314ecb8ad78a2edbecad68413ac21f7696ea6a9feda486c",
 "P3":"0a60d66483bdf81d9e7bdeacf72c40d7eb1e4cabd8f7d6cb6691cc61c903ebd9",
 "P4":"10e471b02b94583c0b6a59857d1f9f6339c3ab2a498845e429a621bb34a25871",
 "P5":"611dce6f6de444c62edbe0339c639d818edb209b2d41065793c74e85e12e4c51",
 "P6":"0d861561e9c9ca24fc7fc5d564283449e19452bf3addb77b0532c6df89f50a75",
 "P7":"79995d1b53c19956ffcd9b5c9794d1eef891b029bd5c4b582bbe73f3f8e866e6",
}
def file_sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def fetch(stage, token=None, repo=REPO, cache_dir=None, revision="main"):
 stage=stage.upper().replace("A15-","")
 if stage not in SHA:raise KeyError("Unknown A15 artifact: "+stage)
 token=token or os.environ.get("HF_TOKEN")
 kw=dict(repo_id=repo,repo_type="model",revision=revision,token=token)
 # The manifest is uploaded LAST only when all seven files were read-back verified.
 manifest_path=hf_hub_download(filename=PREFIX+"/manifest.json",**kw)
 manifest=json.loads(Path(manifest_path).read_text(encoding="utf8"))
 expected={x["name"]:x["sha256"] for x in manifest["artifacts"]}
 matching=[name for name,digest in expected.items() if digest==SHA[stage]]
 if len(matching)!=1:raise RuntimeError("REMOTE_MANIFEST_MISSING_OR_DRIFT: "+stage)
 name=matching[0]
 path=Path(hf_hub_download(filename=PREFIX+"/"+name,**kw))
 if file_sha(path)!=SHA[stage]:raise RuntimeError("REMOTE_SHA_MISMATCH: "+stage)
 if cache_dir:
  import shutil
  out=Path(cache_dir)/name;out.parent.mkdir(parents=True,exist_ok=True)
  if not out.exists() or file_sha(out)!=SHA[stage]:shutil.copy2(path,out)
  return str(out)
 return str(path)
# Examples:
# from a15_hf_results_loader import fetch
# result_p3=fetch("P3",token=userdata.get("HF_TOKEN"))
