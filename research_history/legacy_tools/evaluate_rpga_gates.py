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
 ap=argparse.ArgumentParser();ap.add_argument('--runs',default='aggregate_all/runs.csv');ap.add_argument('--out',default='aggregate_all/rpga_gate_report.json');a=ap.parse_args()
 rows=load(a.runs);I=idx(rows);R={'status':'PENDING','gates':{},'notes':[]}
 # G1: exact ranking-based functional invariance vs Local.
 det={};complete=True;passed=True
 for e in CORE:
  dd=paired(I,e,'rpga','local','deletion_auc');di=paired(I,e,'rpga','local','insertion_auc');to=paired(I,e,'rpga','local','topk_oracle_overlap')
  comp=len(dd)>=5 and len(di)>=5 and len(to)>=5;complete &= comp
  maxd=max([abs(x[-1]) for x in dd],default=np.inf);maxi=max([abs(x[-1]) for x in di],default=np.inf);maxt=max([abs(x[-1]) for x in to],default=np.inf)
  ok=bool(comp and maxd<=1e-4 and maxi<=1e-4 and maxt<=1e-12);passed &= ok
  det[e]={'complete':comp,'max_abs_deletion_delta':maxd if np.isfinite(maxd) else None,'max_abs_insertion_delta':maxi if np.isfinite(maxi) else None,'max_abs_topk_overlap_delta':maxt if np.isfinite(maxt) else None,'pass':ok}
 R['gates']['G1_rank_functional_invariance']={'pass':bool(complete and passed),'complete':bool(complete),'details':det}
 # G2: coordination improvement vs local in all four shifted families; at least 10% mean reduction and >=4/5 wins.
 det={};complete=True;passed=True
 for e in CORE:
  v=paired(I,e,'rpga','local','pairwise_edi');comp=len(v)>=5;complete &= comp
  rp=np.mean([x[1] for x in v]) if v else np.nan;lo=np.mean([x[2] for x in v]) if v else np.nan;wins=sum(x[3]<0 for x in v)
  ratio=float(rp/lo) if v and lo>0 else None;ok=bool(comp and ratio is not None and ratio<=.90 and wins>=4);passed &= ok
  det[e]={'n':len(v),'rpga_mean':float(rp) if np.isfinite(rp) else None,'local_mean':float(lo) if np.isfinite(lo) else None,'ratio':ratio,'wins':int(wins),'pass':ok}
 R['gates']['G2_coordination_gain']={'pass':bool(complete and passed),'complete':bool(complete),'details':det}
 # G3: sample oracle JSD noninferior to Local and better than xFedAlign pooled.
 dl=[];dx=[]
 for e in CORE:
  dl += [x[3] for x in paired(I,e,'rpga','local','sample_fidelity_jsd')]
  dx += [x[3] for x in paired(I,e,'rpga','xfedalign_median','sample_fidelity_jsd')]
 comp=len(dl)>=20 and len(dx)>=20;mdl=float(np.mean(dl)) if dl else None;mdx=float(np.mean(dx)) if dx else None
 ok=bool(comp and mdl<=.01 and mdx<0)
 R['gates']['G3_oracle_fidelity']={'pass':ok,'complete':comp,'mean_rpga_minus_local':mdl,'mean_rpga_minus_xfedalign':mdx,'n_local':len(dl),'n_xfedalign':len(dx)}
 # G4: artifact-summary fidelity may move, but must not collapse relative to xFedAlign.
 da=[]
 for e in CORE: da += [x[3] for x in paired(I,e,'rpga','xfedalign_median','artifact_fidelity_jsd')]
 comp=len(da)>=20;m=float(np.mean(da)) if da else None;ok=bool(comp and m<=.02)
 R['gates']['G4_artifact_fidelity']={'pass':ok,'complete':comp,'mean_rpga_minus_xfedalign':m,'n':len(da)}
 # G5: communication equal to xFedAlign.
 dc=[]
 for e in CORE: dc += [x[3] for x in paired(I,e,'rpga','xfedalign_median','communication_bytes_per_client_artifact')]
 comp=len(dc)>=20;mx=max(dc) if dc else None;ok=bool(comp and abs(mx)<=1e-9 and abs(min(dc))<=1e-9)
 R['gates']['G5_communication']={'pass':ok,'complete':comp,'max_delta_bytes':mx,'min_delta_bytes':min(dc) if dc else None}
 # G6: CIFAR task validity (method-independent task model; read RPGA rows).
 acc=[]
 for e in ['core_cifar10_rotation','core_cifar10_color']:
  vals=[float(r['client_task_accuracy_mean']) for r in rows if r['experiment']==e and r['scenario']=='clean' and r['method']=='rpga']
  acc.append((e,float(np.mean(vals)) if vals else None,len(vals)))
 comp=all(n>=5 and m is not None for _,m,n in acc);ok=bool(comp and all(m>=.35 for _,m,n in acc))
 R['gates']['G6_cifar_task_validity']={'pass':ok,'complete':comp,'details':acc}
 # G7: poisoning may alter magnitudes, but cannot alter RPGA ranking-based functional metrics.
 details={};complete=True;passed=True
 for e in ATTACKS:
  vals=[]
  for metric,tol in [('deletion_auc',1e-4),('insertion_auc',1e-4),('topk_oracle_overlap',1e-12)]:
   # attacked - clean for same method/seed
   seeds=sorted({k[1] for k in I if k[0]==e})
   ds=[]
   for s in seeds:
    arow=I.get((e,s,'attacked','rpga'));crow=I.get((e,s,'clean','rpga'))
    if arow and crow and np.isfinite(float(arow.get(metric,np.nan))) and np.isfinite(float(crow.get(metric,np.nan))):ds.append(float(arow[metric])-float(crow[metric]))
   vals.append((metric,ds,tol))
  comp=all(len(ds)>=5 for _,ds,_ in vals);complete &= comp
  md={metric:max([abs(x) for x in ds],default=np.inf) for metric,ds,_ in vals}
  ok=bool(comp and all(md[metric]<=tol for metric,ds,tol in vals));passed &= ok
  details[e]={'complete':comp,'max_abs_deltas':md,'pass':ok}
 R['gates']['G7_attack_rank_invariance']={'pass':bool(complete and passed),'complete':bool(complete),'details':details}
 req=list(R['gates']);allcomplete=all(R['gates'][g].get('complete',True) for g in req)
 if allcomplete:R['status']='CONTINUE' if all(R['gates'][g]['pass'] for g in req) else 'REVISE_OR_KILL'
 Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(R,indent=2),encoding='utf-8');print(json.dumps(R,indent=2))
if __name__=='__main__':main()
