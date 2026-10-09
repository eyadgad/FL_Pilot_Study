#!/usr/bin/env python3
"""Check full ORIGINAL offline dataset, frozen hashes, all split/teacher feasibility."""
import argparse,sys,json,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.launch_expanded import plan,sha
from study_v7.data import verify_data
from ucpa_fl.config import load_config
from ucpa_fl.datasets import load_data

def audit(dataset,root,stage='all'):
    checked=[]
    for d in (['mnist','cifar10'] if dataset=='both' else [dataset]):
        prov=verify_data(d,root)
        for cfgpath,meta,seed in plan(d,stage):
            cfg=load_config(cfgpath)
            cfg.dataset.root=str(root)
            if cfg.dataset.download or cfg.dataset.train_limit or cfg.dataset.test_limit:
                raise ValueError('Only full offline originals permitted')
            bundle=load_data(cfg,seed)
            groups=[bundle.task_clients,bundle.surrogate_clients,bundle.artifact_clients,bundle.calibration_clients,bundle.eval_clients]
            ii=np.concatenate([z.indices for grp in groups for z in grp]);u=np.unique(ii)
            if len(ii)!=len(u):raise ValueError(f'Overlapping source image indices: {cfgpath.name} seed {seed}')
            if min(len(z) for z in bundle.calibration_clients)<48 or min(len(z) for z in bundle.eval_clients)<64:
                raise ValueError(f'Insufficient genuine teacher/eval records: {cfgpath.name} seed {seed}')
            if len(bundle.test)!=10000:raise ValueError('Not the entire original test split')
            checked.append({'config':cfgpath.name,'seed':seed,'clients':len(bundle.task_clients),'dataset':d,
               'min_teacher':min(len(z) for z in bundle.calibration_clients),
               'min_eval':min(len(z) for z in bundle.eval_clients),'full_train':prov['train_count']})
            print('PREFLIGHT_PASS',cfgpath.stem,seed,flush=True)
    return checked

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',choices=['mnist','cifar10','both'],default='both')
    ap.add_argument('--data-root',required=True)
    ap.add_argument('--stage',choices=['all','primary','clients','rounds','epochs','optimizer','smoke'],default='all')
    ap.add_argument('--out',default='expanded_preflight.json')
    a=ap.parse_args();rows=audit(a.dataset,Path(a.data_root).resolve(),a.stage)
    Path(a.out).write_text(json.dumps({'status':'PASS','checks':rows,'count':len(rows)},indent=2)+'\n')
    print('PREFLIGHT_ALL_PASS',len(rows))
