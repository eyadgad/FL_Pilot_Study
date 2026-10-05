#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import phase4_repa as p4
import phase4c_rotation_confirm as rot
from phase5_peer_smoothing_dev import simple_metrics
from phase7_adaptive_dev import sanitize, adaptive_peer_prior, global_median, cluster_best, norm

ROOT=Path(__file__).resolve().parent;RES=ROOT/'results';RES.mkdir(exist_ok=True)
SEEDS={701,702,703,704,705}; BETA=.6; POWER=2.; MS=2
PATCH={'strong':(.9,.1),'moderate':(.75,.25)}

def runprep(prep,domain,scenario,strength,seed,rows):
    metric=(lambda A:p4.metrics(A,prep.oracle,scenario)) if domain=='patch' else (lambda A:simple_metrics(A,prep.oracle))
    for n in p4.N_EXPLAIN:
        E,V=p4.sample_E_V(prep,n);E,V,M=sanitize(E,V,np.random.default_rng(seed*100000+n*17+3))
        base=dict(seed=seed,domain=domain,scenario=scenario,strength=strength,n_explain=n,train_acc=prep.train_acc,test_acc=prep.test_acc)
        def add(method,A,param='',diag=None):rows.append({**base,'method':method,'param':param,**(diag or {}),**metric(A)})
        add('Local',E)
        gm=[]
        for b in p4.GLOBAL_BETAS:
            A=global_median(E,b);gm.append((metric(A)['oracle_jsd'],A,b))
        _,A,b=min(gm,key=lambda x:x[0]);add('GlobalMedianOracle',A,f'b={b}')
        _,A,p=cluster_best(E,prep.oracle,seed+n,metric);add('ClusterOracle',A,p)
        # no-corroboration ablation
        P1,d1=adaptive_peer_prior(E,M,power=POWER,min_missing_support=1)
        A1=norm((1-BETA)*E+BETA*P1);add('SACPA_no_corroboration',A1,'p=2,b=.6,ms=1',d1)
        # frozen SACPA
        P,d=adaptive_peer_prior(E,M,power=POWER,min_missing_support=MS)
        A=norm((1-BETA)*E+BETA*P);add('SACPA',A,'p=2,b=.6,ms=2',d)

def main():
    if len(sys.argv)!=2:raise SystemExit('usage: phase7_sacpa_confirm.py SEED')
    seed=int(sys.argv[1]);
    if seed not in SEEDS:raise SystemExit('not a registered seed')
    Xtr,ytr,Xte,yte=p4.load_binary_mnist();rows=[]
    for name,st in PATCH.items():
        p4.PATCH_P_POS,p4.PATCH_P_NEG=st
        for sc in p4.SCENARIOS:
            print('patch',seed,name,sc,flush=True);runprep(p4.prepare_seed_scenario(Xtr,ytr,Xte,yte,seed,sc),'patch',sc,name,seed,rows)
    for sc in p4.SCENARIOS:
        print('rotation',seed,sc,flush=True);runprep(rot.prep_rotation(Xtr,ytr,Xte,yte,seed,sc),'rotation',sc,'rotation',seed,rows)
    out=RES/f'P7_sacpa_confirm_{seed}.jsonl';out.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    print('saved',len(rows),out)
if __name__=='__main__':main()
