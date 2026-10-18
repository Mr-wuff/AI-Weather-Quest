#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CIDM-Earth A15-P4: frozen train-only quintile probabilities on fresh 2019 dates.

GEFSv12 archive is RETROSPECTIVE, not contemporaneous operational forecasts.
This does NOT produce official ECMWF RPSS or authorize CIDM training.
"""
import io, os, json, zipfile, traceback, argparse
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import a15_p1_gefs_weekly_ensemble as gefs
import a15_p2_gefs_era5_paired as paired
import a15_p3_quantile_baseline as p3

P3_SHA="0a60d66483bdf81d9e7bdeacf72c40d7eb1e4cabd8f7d6cb6691cc61c903ebd9"
DATES=("2019010200","2019070300")
TARGETS=("mslp_W3","mslp_W4","tas_W3","tas_W4")
ARCHIVE="CIDM_EARTH_A15_P4_DELIVERY_REPORTS.zip"

def thresholds_from_p3(path):
    if paired.hash_file(path)!=P3_SHA:
        raise RuntimeError("P3_PARENT_SHA256_DRIFT: exact uploaded P3 ZIP required")
    with zipfile.ZipFile(path) as z:
        if z.testzip():raise RuntimeError("P3_PARENT_ZIP_CRC_FAILURE")
        state=json.loads(z.read("reports/STATUS.json"))
        if state.get("status")!="REAL_TRAIN_ONLY_ERA5_QUINTILE_PROXY_BASELINE_FOUR_CASES_PASS_RESEARCH_ONLY":
            raise RuntimeError("P3_PARENT_NOT_REAL")
        if state.get("climate_train_years")!=list(range(2008,2017)) or state.get("historical_samples_per_season")!=27:
            raise RuntimeError("P3_CLIMATOLOGY_CONTRACT_CHANGED")
        with np.load(io.BytesIO(z.read("reports/TRAIN_ONLY_PROXY_QUINTILES.npz")),allow_pickle=False) as a:
            if not np.array_equal(a["train_years"],np.arange(2008,2017)) or not np.array_equal(a["week_offsets"],[-1,0,1]):
                raise RuntimeError("P3_TRAINING_YEARS_CHANGED")
            q={"jan":a["january"].copy(),"jul":a["july"].copy()}
    for season,values in q.items():
        if values.shape!=(4,2,2,121,240) or not np.isfinite(values).all() or (np.diff(values,axis=0)<-1e-4).any():
            raise RuntimeError("P3_THRESHOLD_INVALID_"+season)
    return q

def smoke(path):
    q=thresholds_from_p3(path)
    p3.selftest()
    for date in DATES:
        ini,issue=gefs.init_issue(date)
        if issue.weekday()!=3:raise RuntimeError("2019 issue day incorrect")
        for w in (3,4):
            if len(gefs.required_steps(date,w))!=28 or len(paired.valid_hours(date,w))!=28:
                raise RuntimeError("Lead contract error")
    fake=np.zeros((5,2,2,121,240),np.float32)
    fake[:,0]=1000
    fake[:,1]=10
    for smooth in (0,.5):
        out=p3.score_prob_case(fake,fake[0],q["jan"],smooth)
        if set(out)!=set(TARGETS) or not all(np.isfinite(v["research_proxy_rpss"]) for v in out.values()):
            raise RuntimeError("RPS mathematical smoke failure")
    print("[A15-P4] Offline source/time/threshold/RPS smoke PASS",flush=True)

def finish(root,details,status,error=None):
    root=Path(root);reports=root/"reports";reports.mkdir(exist_ok=True,parents=True)
    state={**details,"experiment":"CIDM_EARTH_A15_P4","status":status,
           "utc":datetime.now(timezone.utc).isoformat(),"p3_parent_sha256":P3_SHA,
           "climatology_train_years":list(range(2008,2017)),"operational_input_verified":False,
           "source_is_retrospective_reforecast":True,"official_rpss":None,
           "cidm_training_steps":0,"formal_training_authorized":False,
           "unseen_2019_dates_in_prior_P2_P3":True,
           "two_dates_do_not_establish_generalization":True,
           "regrid_difference":"GEFS node-sampled 0.5->1.5 versus WB2 ERA5 conservative"}
    paired.json_write(reports/"STATUS.json",state)
    if error:(reports/"ERROR_TRACEBACK.txt").write_text(error[-30000:],encoding="utf8")
    target=root/ARCHIVE
    with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for f in sorted(reports.rglob("*")):
            if f.is_file():z.write(f,f.relative_to(root).as_posix())
    with zipfile.ZipFile(target) as z:
        if z.testzip():raise RuntimeError("OUTPUT_CRC_FAILURE")
    sha=paired.hash_file(target)
    side=Path(str(target)+".sha256")
    side.write_text(sha+"  "+ARCHIVE+"\n",encoding="ascii")
    assert paired.hash_file(target)==side.read_text().split()[0]
    print("[A15-P4]",status,"ZIP",target,"SHA256",sha,flush=True)
    return target

def run(root,p3_zip,offline=False):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    reports=root/"reports";reports.mkdir(exist_ok=True)
    # Never merge stale previous-run reports into a new attempt; preserve checkpoints.
    for f in reports.iterdir():
        if f.is_file():f.unlink()
    state={"requested_dates":list(DATES),"successful_paired_cases":0,
           "fixed_probability_smoothing":[0,.5],
           "source_s3":gefs.BUCKET,"source_verification":paired.SOURCE}
    try:
        smoke(p3_zip)
        q=thresholds_from_p3(p3_zip)
        if offline:return finish(root,state,"OFFLINE_QA_ONLY_NO_REAL_2019_DATA")
        source_details={}
        for date in DATES:
            print("[PRE-FLIGHT]",date,"checking ten S3 member-variable IDX inventories",flush=True)
            c=gefs.anonymous_client()
            req={w:gefs.required_steps(date,w) for w in (3,4)}
            found=gefs.maybe_preflight(c,date,gefs.MEMBERS,req)
            source_details[date]={m+"_"+v:a["idx_sha"] for (m,v),a in found.items()}
        paired.json_write(reports/"S3_PREFLIGHT_IDX_SHA256.json",source_details)
        scored={};metric={};case_hashes={}
        ds=None
        for date in DATES:
            print("[REAL]",date,"getting five-member full-week GEFS",flush=True)
            pred,meta=paired.new_s3_case(date,root,save_every=4)
            if ds is None:ds=paired.open_wb2()
            truth=paired.truth_cache(ds,date,root)
            season="jan" if date[4:6]=="01" else "jul"
            scored[date]={"raw_ensemble":p3.score_prob_case(pred,truth,q[season],0),
                          "fixed_half_count_smoothing":p3.score_prob_case(pred,truth,q[season],.5)}
            metric[date]=paired.score_case(pred,truth)
            out=reports/("2019_paired_"+date+".npz")
            np.savez_compressed(out,forecast=pred,truth=truth,date=date,
                                source="GEFSv12 retrospective / WB2 ERA5 labels",
                                p3_threshold_parent_sha=P3_SHA)
            case_hashes[date]=paired.hash_file(out)
            state["successful_paired_cases"]+=1
            paired.json_write(reports/"PER_CASE_PROBABILISTIC.json",scored)
            paired.json_write(reports/"PER_CASE_RMSE_CRPS.json",metric)
            paired.json_write(reports/"PROGRESS.json",state)
            print("[PAIRED]",date,"done",state["successful_paired_cases"],"/2",flush=True)
        if state["successful_paired_cases"]!=2:raise RuntimeError("Incomplete 2019 audit")
        aggregate={}
        for mode in ("raw_ensemble","fixed_half_count_smoothing"):
            aggregate[mode]={}
            for target in TARGETS:
                r=np.mean([scored[d][mode][target]["research_proxy_rps_climatology"] for d in DATES])
                f=np.mean([scored[d][mode][target]["research_proxy_rps_gefs"] for d in DATES])
                vals=[scored[d][mode][target]["research_proxy_rpss"] for d in DATES]
                aggregate[mode][target]={"proxy_rpss_pooled":float(1-f/r),
                                         "per_case_rpss":vals,
                                         "positive_both_cases":all(x>0 for x in vals)}
        state["case_npz_sha256"]=case_hashes
        state["two_case_all_four_channels_positive"]=all(
             aggregate["fixed_half_count_smoothing"][t]["positive_both_cases"] for t in TARGETS)
        state["two_case_gate_only_not_training_authorization"]=True
        paired.json_write(reports/"FROZEN_2019_PROXY_SUMMARY.json",aggregate)
        return finish(root,state,"REAL_2019_TWO_CASE_FROZEN_PROXY_EVALUATION_COMPLETE_RESEARCH_ONLY")
    except Exception:
        return finish(root,state,"BLOCKED_NOT_A_PREDICTIVE_RESULT",traceback.format_exc())

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--p3-results",required=True)
    parser.add_argument("--out",default="/content/CIDM_EARTH_A15_P4_LITE")
    parser.add_argument("--offline-smoke",action="store_true")
    args=parser.parse_args()
    run(args.out,args.p3_results,args.offline_smoke)
