#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
CORE=['core_mnist_rotation','core_mnist_patch','core_cifar10_rotation','core_cifar10_color']
ATTACKS=['robust_mnist_rotation_artifact_shift','robust_mnist_rotation_random_support']

def load(path):
 out=[]
 with open(path,newline='',encoding='utf-8') as f:
  for r in csv.DictReader(f):
   for k,v in list(r.items()):
    try:r[k]=float(v) if k not in ('experiment','scenario','method','run_dir','config_sha256') else v
    except:pass
   out.append(r)
 return out

def idx(rows):return {(r['experiment'],int(r['seed']),r['scenario'],r['method']):r for r in rows}
def paired(I,exp,a,b,m,scenario='clean'):
 vals=[];seeds=sorted({k[1] for k in I if k[0]==exp and k[2]==scenario})
 for s in seeds:
  x=I.get((exp,s,scenario,a));y=I.get((exp,s,scenario,b))
  if x and y:
   xv=float(x.get(m,np.nan));yv=float(y.get(m,np.nan))
   if np.isfinite(xv) and np.isfinite(yv):vals.append((s,xv,yv,xv-yv))
 return vals

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--runs',default='aggregate_all/runs.csv');ap.add_argument('--out',default='aggregate_all/ncsacpa_gate_report.json');a=ap.parse_args()
 rows=load(a.runs);I=idx(rows);R={'status':'PENDING','gates':{},'notes':[]}
 # G1: primary artifact fidelity vs xFedAlign across core shifted families.
 fam={};succ=0
 for e in CORE:
  v=paired(I,e,'nc_sacpa','xfedalign_median','artifact_fidelity_jsd');wins=sum(d<0 for _,_,_,d in v);md=float(np.mean([d for *_,d in v])) if v else None
  fam[e]={'n':len(v),'wins':wins,'mean_delta':md};succ += int(len(v)>=5 and wins>=4 and md<0)
 complete=all(fam[e]['n']>=5 for e in CORE);R['gates']['G1_primary_fidelity']={'pass':bool(complete and succ>=3),'complete':complete,'successful_families':succ,'details':fam}
 # G2: consistency should improve over Local in >=3/4 core families, but not at expense of G1.
 det={};succ=0
 for e in CORE:
  v=paired(I,e,'nc_sacpa','local','pairwise_edi');md=float(np.mean([d for *_,d in v])) if v else None
  det[e]={'n':len(v),'mean_delta':md};succ+=int(len(v)>=5 and md<0)
 comp=all(det[e]['n']>=5 for e in CORE);R['gates']['G2_coordination_vs_local']={'pass':bool(comp and succ>=3),'complete':comp,'successful_families':succ,'details':det}
 # G3 functional non-inferiority pooled vs xFedAlign.
 dd=[];di=[];ov=[]
 for e in CORE:
  dd += [d for *_,d in paired(I,e,'nc_sacpa','xfedalign_median','deletion_auc')]
  di += [d for *_,d in paired(I,e,'nc_sacpa','xfedalign_median','insertion_auc')]
  ov += [d for *_,d in paired(I,e,'nc_sacpa','xfedalign_median','topk_oracle_overlap')]
 comp=len(dd)>=20 and len(di)>=20 and len(ov)>=20
 pas=bool(comp and np.mean(dd)<=.01 and np.mean(di)>=-.01 and np.mean(ov)>=-.05)
 R['gates']['G3_functional_noninferiority']={'pass':pas,'complete':comp,'mean_deletion_delta':float(np.mean(dd)) if dd else None,'mean_insertion_delta':float(np.mean(di)) if di else None,'mean_overlap_delta':float(np.mean(ov)) if ov else None}
 # G4 revision mechanism: patch should improve over Phase-VII predecessor; other core families not collapse.
 vp=paired(I,'core_mnist_patch','nc_sacpa','sacpa_global_count','artifact_fidelity_jsd');patch_impr=-float(np.mean([d for *_,d in vp])) if vp else None
 other=[]
 for e in [x for x in CORE if x!='core_mnist_patch']:other += [d for *_,d in paired(I,e,'nc_sacpa','sacpa_global_count','artifact_fidelity_jsd')]
 comp=len(vp)>=5 and len(other)>=15;pas=bool(comp and patch_impr>0 and np.mean(other)<=.02)
 R['gates']['G4_local_corroboration_mechanism']={'pass':pas,'complete':comp,'patch_mean_improvement':patch_impr,'other_family_mean_delta':float(np.mean(other)) if other else None}
 # G5 task validity
 acc=[]
 for e in ['core_cifar10_rotation','core_cifar10_color']:
  vals=[float(r['client_task_accuracy_mean']) for r in rows if r['experiment']==e and r['scenario']=='clean' and r['method']=='nc_sacpa']
  acc.append((e,float(np.mean(vals)) if vals else None,len(vals)))
 comp=all(n>=5 and m is not None for _,m,n in acc);R['gates']['G5_cifar_task_validity']={'pass':bool(comp and all(m>=.35 for _,m,n in acc)),'complete':comp,'details':acc}
 # G6 communication: NC-SACPA must not exceed xFedAlign mean-artifact bytes.
 deltas=[]
 for e in CORE:deltas += [d for *_,d in paired(I,e,'nc_sacpa','xfedalign_median','communication_bytes_per_client_artifact')]
 comp=len(deltas)>=20;R['gates']['G6_communication']={'pass':bool(comp and np.max(deltas)<=1e-9),'complete':comp,'max_delta_bytes':float(np.max(deltas)) if deltas else None}
 # G7 robustness under attacks: attacked fidelity no worse than xFedAlign on pooled attacks by >.01.
 atk=[]
 for e in ATTACKS:atk += [d for *_,d in paired(I,e,'nc_sacpa','xfedalign_median','artifact_fidelity_jsd',scenario='attacked')]
 comp=len(atk)>=10;R['gates']['G7_attack_robustness']={'pass':bool(comp and np.mean(atk)<=.01),'complete':comp,'mean_attacked_fidelity_delta':float(np.mean(atk)) if atk else None,'n':len(atk)}
 req=list(R['gates']);complete=all(R['gates'][g].get('complete',True) for g in req)
 if complete:R['status']='CONTINUE' if all(R['gates'][g]['pass'] for g in req) else 'REVISE_OR_KILL'
 Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(R,indent=2),encoding='utf-8');print(json.dumps(R,indent=2))
if __name__=='__main__':main()
