from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); sys.path.insert(0,str(HERE/'cfba_core'))
import multi_proposal_smoke as mp, global_subspace as gs
from ucpa_fl.datasets import load_data,make_loader
from ucpa_fl.federated import train_fedavg,evaluate
from ucpa_fl.repro import set_seed,resolve_device
from ucpa_fl.explain import build_local_explanation
from ucpa_fl.artifacts import sanitize_artifact
from ucpa_fl.alignment import xfedalign_prior
from ucpa_fl.metrics import prepare_client_evaluation,normalize_np
RHO=.30; LAM=np.array([0,.25,.5,.75,1.0])

def weighted(rec,prior,A,lam):
 c=int(rec['pred']);l=normalize_np(np.asarray(rec['local_map'])[None])[0];p=normalize_np(np.asarray(prior[c])[None])[0];a=np.clip(float(lam)*A[c],0,1);return normalize_np(((1-a)*l+a*p)[None])[0]

def run(seed,shift,outroot):
 cfg=mp.make_cfg(shift,seed,outroot);set_seed(seed,True);device=resolve_device('auto');bundle=load_data(cfg,seed)
 class DL:
  def log(self,*a,**k):pass
  def metric(self,*a,**k):pass
  def save_json(self,*a,**k):pass
 model,_=train_fedavg(cfg,bundle,seed,device,DL());acc=[evaluate(model,make_loader(ds,128,False,seed+6000+i,0),device)['accuracy'] for i,ds in enumerate(bundle.eval_clients)]
 bundle.set_round(cfg.federation.rounds-1);loc=[];art=[];rng=np.random.default_rng(seed+70707)
 for i in range(4):
  le=build_local_explanation(model,bundle.surrogate_clients[i],bundle.artifact_clients[i],bundle.input_shape,bundle.n_classes,cfg.surrogate,seed+i*97,device,0);loc.append(le);art.append(sanitize_artifact(le.mean,le.var_mean,le.counts,cfg.artifact,rng))
 means=np.stack([a.mean for a in art]);pri=xfedalign_prior(means);sels=[];cals=[];tests=[]
 for i in range(4):
  cache=prepare_client_evaluation(model,bundle.eval_clients[i],loc[i].surrogate_state,cfg.surrogate.source,bundle.input_shape,bundle.n_classes,device,cfg.surrogate.ig_steps,seed+i*211,0,cfg.evaluation.max_eval_samples)
  se,ca,te=mp.split3(cache['records'],seed+i*101);sels.append(se);cals.append(ca);tests.append(te)
 gains=np.stack([gs.client_gain(sels[i],pri[i],bundle.n_classes,means.shape[-1]) for i in range(4)]);g=np.nanmean(gains,0);g=np.nan_to_num(g,nan=-np.inf)
 S=np.zeros_like(g)
 for c in range(bundle.n_classes):
  pos=g[c]>0
  if np.any(pos):
   v=g[c,pos]; lo=np.min(v);hi=np.max(v);S[c,pos]=.25+.75*(v-lo)/(hi-lo+1e-12)
 A=RHO+(1-RHO)*S
 policies={'local':[],'xfedalign_0p2':[],'risk_weighted_0p3':[]};diag=[]
 for i in range(4):
  policies['local'].append(lambda r:normalize_np(np.asarray(r['local_map'])[None])[0]);policies['xfedalign_0p2'].append((lambda i=i:lambda r:mp.basic_maps(r,pri[i,int(r['pred'])],.2))())
  mappers=[(lambda lam,i=i,A=A:lambda r:weighted(r,pri[i],A,lam))(lam) for lam in LAM];L=mp.losses_for_policy(model,cals[i],mappers,device,cfg.evaluation.deletion_steps,cfg.evaluation.topk_overlap_k);lam,idx,up,emp,cert=mp.choose_crc(L,LAM)
  policies['risk_weighted_0p3'].append((lambda i=i,A=A,lam=lam:lambda r:weighted(r,pri[i],A,lam))());diag.append({'lambda':lam,'certified':cert,'upper':up,'mean_weight':float(A.mean())})
 metrics={}
 for name,fns in policies.items():
  per=[]
  for i in range(4):
   maps=[fns[i](r) for r in tests[i]]
   for r in tests[i]:r['prior']=pri[i,int(r['pred'])]
   per.append(mp.metric_for_maps(model,tests[i],maps,device,cfg.evaluation.deletion_steps,cfg.evaluation.topk_overlap_k))
  avg=lambda k:float(np.mean([x[k] for x in per]));metrics[name]={k:avg(k) for k in ['sample_fidelity_jsd','deletion_auc','insertion_auc','topk_oracle_overlap','risk','risk_p90']};metrics[name]['pairwise_edi']=mp.summarize_edi(tests,pri,fns)
 return {'seed':seed,'shift':shift,'task_acc_mean':float(np.mean(acc)),'task_acc_min':float(np.min(acc)),'metrics':metrics,'diagnostics':diag}
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',default='1861,1862,1863');ap.add_argument('--shifts',default='rotation,patch');ap.add_argument('--out',default='/mnt/data/multiproposal_smoke/risk_weighted_confirm');a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);R=[]
 for sh in a.shifts.split(','):
  for s in map(int,a.seeds.split(',')):
   print('RUN',sh,s,flush=True);r=run(s,sh,out);R.append(r);(out/f'{sh}_{s}.json').write_text(json.dumps(r,indent=2));print('DONE',sh,s,r['task_acc_mean'],flush=True)
 # merge if existing
 allp=out/'all.json'; old=[]
 if allp.exists(): old=json.load(open(allp))
 key={(x['shift'],x['seed']) for x in old};old += [x for x in R if (x['shift'],x['seed']) not in key];allp.write_text(json.dumps(old,indent=2))
