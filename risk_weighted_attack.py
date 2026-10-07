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
from ucpa_fl.attacks import apply_artifact_attack
from ucpa_fl.config import AttackConfig
RHO=.30; LAM=np.array([0,.25,.5,.75,1.0])

def weighted(rec,prior,A,lam):
 c=int(rec['pred']);l=normalize_np(np.asarray(rec['local_map'])[None])[0];p=normalize_np(np.asarray(prior[c])[None])[0];a=np.clip(float(lam)*A[c],0,1);return normalize_np(((1-a)*l+a*p)[None])[0]

def policy(model,sels,cals,tests,arts,bundle,cfg,device):
 means=np.stack([a.mean for a in arts]);pri=xfedalign_prior(means)
 gains=np.stack([gs.client_gain(sels[i],pri[i],bundle.n_classes,means.shape[-1]) for i in range(4)]);g=np.nanmean(gains,0);g=np.nan_to_num(g,nan=-np.inf)
 S=np.zeros_like(g)
 for c in range(bundle.n_classes):
  pos=g[c]>0
  if np.any(pos):
   v=g[c,pos];lo=v.min();hi=v.max();S[c,pos]=.25+.75*(v-lo)/(hi-lo+1e-12)
 A=RHO+(1-RHO)*S
 pol={'xfedalign':[],'risk_weighted':[]};diag=[]
 for i in range(4):
  pol['xfedalign'].append((lambda i=i:lambda r:mp.basic_maps(r,pri[i,int(r['pred'])],.2))())
  mappers=[(lambda lam,i=i,A=A:lambda r:weighted(r,pri[i],A,lam))(lam) for lam in LAM];L=mp.losses_for_policy(model,cals[i],mappers,device,cfg.evaluation.deletion_steps,cfg.evaluation.topk_overlap_k);lam,idx,up,emp,cert=mp.choose_crc(L,LAM);pol['risk_weighted'].append((lambda i=i,A=A,lam=lam:lambda r:weighted(r,pri[i],A,lam))());diag.append({'lambda':lam,'certified':cert})
 met={}
 for name,fns in pol.items():
  per=[]
  for i in range(4):
   maps=[fns[i](r) for r in tests[i]]
   for r in tests[i]:r['prior']=pri[i,int(r['pred'])]
   per.append(mp.metric_for_maps(model,tests[i],maps,device,cfg.evaluation.deletion_steps,cfg.evaluation.topk_overlap_k))
  avg=lambda k:float(np.mean([x[k] for x in per]));met[name]={k:avg(k) for k in ['sample_fidelity_jsd','deletion_auc','insertion_auc','topk_oracle_overlap','risk']};met[name]['pairwise_edi']=mp.summarize_edi(tests,pri,fns)
 return met,diag

def run(seed):
 shift='rotation';cfg=mp.make_cfg(shift,seed,'/tmp/no');set_seed(seed,True);device=resolve_device('auto');bundle=load_data(cfg,seed)
 class DL:
  def log(self,*a,**k):pass
  def metric(self,*a,**k):pass
  def save_json(self,*a,**k):pass
 model,_=train_fedavg(cfg,bundle,seed,device,DL());acc=[evaluate(model,make_loader(ds,128,False,seed+6000+i,0),device)['accuracy'] for i,ds in enumerate(bundle.eval_clients)]
 bundle.set_round(cfg.federation.rounds-1);loc=[];arts=[];rng=np.random.default_rng(seed+70707)
 for i in range(4):
  le=build_local_explanation(model,bundle.surrogate_clients[i],bundle.artifact_clients[i],bundle.input_shape,bundle.n_classes,cfg.surrogate,seed+i*97,device,0);loc.append(le);arts.append(sanitize_artifact(le.mean,le.var_mean,le.counts,cfg.artifact,rng))
 sels=[];cals=[];tests=[]
 for i in range(4):
  cache=prepare_client_evaluation(model,bundle.eval_clients[i],loc[i].surrogate_state,cfg.surrogate.source,bundle.input_shape,bundle.n_classes,device,cfg.surrogate.ig_steps,seed+i*211,0,cfg.evaluation.max_eval_samples);se,ca,te=mp.split3(cache['records'],seed+i*101);sels.append(se);cals.append(ca);tests.append(te)
 out={'seed':seed,'task_acc':float(np.mean(acc)),'conditions':{}}
 out['conditions']['clean']={'metrics':policy(model,sels,cals,tests,arts,bundle,cfg,device)[0]}
 for kind in ['artifact_shift','random_support']:
  ac=AttackConfig(enabled=True,kind=kind,fraction=.25,strength=.25);aa,bad=apply_artifact_attack(arts,ac,seed+999);met,diag=policy(model,sels,cals,tests,aa,bundle,cfg,device);out['conditions'][kind]={'bad':bad,'metrics':met,'diag':diag}
 return out
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',default='1871,1872');ap.add_argument('--out',default='/mnt/data/multiproposal_smoke/risk_weighted_attack');a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);R=[]
 for s in map(int,a.seeds.split(',')):
  print('RUN',s,flush=True);r=run(s);R.append(r);(out/f'{s}.json').write_text(json.dumps(r,indent=2));print('DONE',s,r['task_acc'],flush=True)
 (out/'all.json').write_text(json.dumps(R,indent=2))
