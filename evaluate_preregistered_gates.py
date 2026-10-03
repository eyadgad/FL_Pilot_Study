#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np

SHIFTED=['core_mnist_rotation','core_mnist_patch','core_cifar10_rotation','core_cifar10_color']
UNEQUAL=['stress_mnist_rotation_unequal_sizes','stress_cifar10_rotation_unequal_sizes']

def load_rows(path):
    with open(path,newline='',encoding='utf-8') as f:
        out=[]
        for r in csv.DictReader(f):
            for k,v in list(r.items()):
                try:r[k]=float(v) if k not in ('experiment','scenario','method','run_dir','config_sha256') else v
                except Exception:pass
            out.append(r)
        return out

def index(rows): return {(r['experiment'],int(r['seed']),r['scenario'],r['method']):r for r in rows}

def paired(idx,exp,a,b,metric):
    vals=[]
    seeds=sorted({k[1] for k in idx if k[0]==exp and k[2]=='clean'})
    for s in seeds:
        ra=idx.get((exp,s,'clean',a));rb=idx.get((exp,s,'clean',b))
        if ra and rb:
            x=float(ra.get(metric,np.nan));y=float(rb.get(metric,np.nan))
            if np.isfinite(x) and np.isfinite(y):vals.append((s,x,y,x-y))
    return vals

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',default='aggregate_all/runs.csv');p.add_argument('--out',default='aggregate_all/preregistered_gate_report.json');a=p.parse_args()
    rows=load_rows(a.runs);idx=index(rows);report={'status':'PENDING','gates':{},'notes':[]}
    # Gate 1
    successful=[];details={}
    for exp in SHIFTED:
        v=paired(idx,exp,'ucpa','xfedalign_median','artifact_fidelity_jsd');wins=sum(x[3]<0 for x in v)
        details[exp]={'n':len(v),'wins':wins,'mean_delta':float(np.mean([x[3] for x in v])) if v else None};
        if len(v)>=5 and wins>=4: successful.append(exp)
    report['gates']['G1_local_fidelity']={'pass':len(successful)>=3,'complete':all(details[e]['n']>=5 for e in SHIFTED),'successful_families':successful,'details':details}
    # Gate 2
    successful=[];details={}
    for exp in SHIFTED:
        v=paired(idx,exp,'ucpa','local','pairwise_edi');wins=sum(x[3]<0 for x in v)
        details[exp]={'n':len(v),'wins':wins,'mean_delta':float(np.mean([x[3] for x in v])) if v else None}
        if len(v)>=5 and float(np.mean([x[3] for x in v]))<0:successful.append(exp)
    report['gates']['G2_consistency_vs_local']={'pass':len(successful)>=3,'complete':all(details[e]['n']>=5 for e in SHIFTED),'successful_families':successful,'details':details}
    # Gate 3 pooled functional non-inferiority
    ddel=[];dins=[]
    for exp in SHIFTED:
        ddel += [x[3] for x in paired(idx,exp,'ucpa','xfedalign_median','deletion_auc')]
        dins += [x[3] for x in paired(idx,exp,'ucpa','xfedalign_median','insertion_auc')]
    g3=bool(ddel and dins and np.mean(ddel)<=.01 and np.mean(dins)>=-.01)
    report['gates']['G3_functional_fidelity']={'pass':g3,'complete':len(ddel)>=20 and len(dins)>=20,'mean_deletion_delta':float(np.mean(ddel)) if ddel else None,'mean_insertion_delta':float(np.mean(dins)) if dins else None,'n':min(len(ddel),len(dins))}
    # Gate 4 mechanism, needs B unequal runs too
    pool=SHIFTED+UNEQUAL; full_wh=[];full_co=[]
    for exp in pool:
        full_wh += [x[3] for x in paired(idx,exp,'ucpa','ucpa_whole_only','artifact_fidelity_jsd')]
        full_co += [x[3] for x in paired(idx,exp,'ucpa','ucpa_coord_only','artifact_fidelity_jsd')]
    enough=len(full_wh)>=len(pool)*5 and len(full_co)>=len(pool)*5
    report['gates']['G4_two_scale_mechanism']={'pass':bool(enough and np.mean(full_wh)<0 and np.mean(full_co)<0),'complete':enough,'mean_delta_vs_whole_only':float(np.mean(full_wh)) if full_wh else None,'mean_delta_vs_coord_only':float(np.mean(full_co)) if full_co else None,'n':len(full_wh)}
    # Gate 5 CIFAR task validity
    acc=[]
    for exp in ['core_cifar10_rotation','core_cifar10_color']:
        v=[]
        for k,r in idx.items():
            if k[0]==exp and k[2]=='clean' and k[3]=='ucpa':v.append(float(r['client_task_accuracy_mean']))
        acc.append((exp,float(np.mean(v)) if v else None,len(v)))
    g5complete=all(n>=5 and m is not None for _,m,n in acc)
    report['gates']['G5_cifar_task_validity']={'pass':bool(g5complete and all(m>=.35 for _,m,n in acc)),'complete':g5complete,'details':acc}
    # Gate 6 privacy warning, not a kill gate but constrains claims
    pm=[];pv=[]
    for r in rows:
        if r['experiment'] in SHIFTED and r['scenario']=='clean' and r['method']=='ucpa':
            x=float(r.get('artifact_mia_mean_only_auc',np.nan));y=float(r.get('artifact_mia_mean_variance_auc',np.nan))
            if np.isfinite(x) and np.isfinite(y):pm.append(x);pv.append(y)
    privacy_risk=bool(pv and (np.mean(pv)-np.mean(pm)>.05 or np.mean(pv)>.65))
    report['gates']['G6_privacy_claim_constraint']={'privacy_risk_flag':privacy_risk,'mean_only_auc':float(np.mean(pm)) if pm else None,'mean_variance_auc':float(np.mean(pv)) if pv else None,'n':len(pv)}
    required=['G1_local_fidelity','G2_consistency_vs_local','G3_functional_fidelity','G4_two_scale_mechanism','G5_cifar_task_validity']
    complete=all(report['gates'][g].get('complete',True) is not False for g in required)
    if complete:
        report['status']='CONTINUE' if all(report['gates'][g]['pass'] for g in required) else 'REVISE_OR_KILL'
    Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
