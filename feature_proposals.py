from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np, torch
BASE='/mnt/data/multiproposal_smoke/multi_proposal_smoke.py'
sys.path.insert(0,'/mnt/data/multiproposal_smoke'); sys.path.insert(0,'/mnt/data/cfba_src/CFBA_full_pipeline')
import multi_proposal_smoke as mp
from ucpa_fl.datasets import load_data, make_loader
from ucpa_fl.federated import train_fedavg,evaluate
from ucpa_fl.repro import set_seed,resolve_device
from ucpa_fl.explain import build_local_explanation
from ucpa_fl.artifacts import sanitize_artifact
from ucpa_fl.alignment import xfedalign_prior
from ucpa_fl.metrics import prepare_client_evaluation,normalize_np

LAM=np.array([0,.25,.5,.75,1.0])

def learn_masks(records,prior,C,D,mode='positive'):
    W=np.zeros((C,D),float); stats=[]
    side=int(round(D**0.5))
    for c in range(C):
        rr=[r for r in records if int(r['pred'])==c]
        if len(rr)<3: stats.append({'n':len(rr),'selected':0});continue
        L=np.stack([normalize_np(np.asarray(r['local_map'])[None])[0] for r in rr]);O=np.stack([normalize_np(np.asarray(r['oracle_map'])[None])[0] for r in rr]);P=np.asarray(prior[c])[None]
        diff=np.abs(L-O)-np.abs(P-O) # positive = prior closer to oracle
        gain=diff.mean(0); se=diff.std(0,ddof=1)/np.sqrt(max(1,len(rr)))
        if mode=='positive': w=(gain>0).astype(float)
        elif mode=='significant': w=(gain>1.0*se).astype(float)
        elif mode=='soft':
            z=gain/(se+1e-6); w=1/(1+np.exp(-z)); w[gain<=0]=0
        elif mode=='block':
            if side*side!=D:
                w=(gain>0).astype(float)
            else:
                w=np.zeros(D,float); b=4
                for r0 in range(0,side,b):
                    for c0 in range(0,side,b):
                        ids=[]
                        for rr0 in range(r0,min(side,r0+b)):
                            for cc0 in range(c0,min(side,c0+b)):ids.append(rr0*side+cc0)
                        if gain[ids].mean()>0:w[ids]=1
        else:raise ValueError(mode)
        W[c]=w;stats.append({'n':len(rr),'selected':float(np.mean(w>0)),'weight_mean':float(np.mean(w))})
    return W,stats

def masked_map(rec,prior,W,lam):
    c=int(rec['pred']);l=normalize_np(np.asarray(rec['local_map'])[None])[0];p=normalize_np(np.asarray(prior[c])[None])[0];a=np.clip(float(lam)*W[c],0,1)
    return normalize_np(((1-a)*l+a*p)[None])[0]

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
    policies={'local':[],'xfedalign_0p2':[]};diag={}
    for i in range(4):
        policies['local'].append(lambda r: normalize_np(np.asarray(r['local_map'])[None])[0])
        policies['xfedalign_0p2'].append((lambda i=i: lambda r: mp.basic_maps(r,pri[i,int(r['pred'])],.2))())
    for mode in ['positive','significant','soft','block']:
        name='riskmask_'+mode;policies[name]=[];diag[name]=[]
        for i in range(4):
            W,st=learn_masks(sels[i],pri[i],bundle.n_classes,means.shape[-1],mode)
            mappers=[(lambda lam,i=i,W=W: lambda r: masked_map(r,pri[i],W,lam))(lam) for lam in LAM]
            L=mp.losses_for_policy(model,cals[i],mappers,device,cfg.evaluation.deletion_steps,cfg.evaluation.topk_overlap_k)
            lam,idx,up,emp,cert=mp.choose_crc(L,LAM)
            policies[name].append((lambda i=i,W=W,lam=lam: lambda r: masked_map(r,pri[i],W,lam))())
            diag[name].append({'lambda':lam,'mask_stats':st,'upper':up,'certified':cert})
    metrics={}
    for name,fns in policies.items():
        per=[]
        for i in range(4):
            maps=[fns[i](r) for r in tests[i]]
            for r in tests[i]:r['prior']=pri[i,int(r['pred'])]
            per.append(mp.metric_for_maps(model,tests[i],maps,device,cfg.evaluation.deletion_steps,cfg.evaluation.topk_overlap_k))
        avg=lambda k:float(np.mean([x[k] for x in per]));metrics[name]={k:avg(k) for k in ['sample_fidelity_jsd','deletion_auc','insertion_auc','topk_oracle_overlap','risk','risk_p90','movement']};metrics[name]['pairwise_edi']=mp.summarize_edi(tests,pri,fns)
    return {'seed':seed,'shift':shift,'task_acc_mean':float(np.mean(acc)),'metrics':metrics,'diagnostics':diag}

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',default='1811,1812');ap.add_argument('--shifts',default='rotation,patch');ap.add_argument('--out',default='/mnt/data/multiproposal_smoke/feature_results');a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);R=[]
 for sh in a.shifts.split(','):
  for s in map(int,a.seeds.split(',')):
   print('RUN',sh,s,flush=True);r=run(s,sh,out);R.append(r);(out/f'{sh}_{s}.json').write_text(json.dumps(r,indent=2));print('DONE',sh,s,r['task_acc_mean'],flush=True)
 (out/'all.json').write_text(json.dumps(R,indent=2))
