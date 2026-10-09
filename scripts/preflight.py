#!/usr/bin/env python3
"""Real-data-only training/teacher/eval feasibility check for every frozen seed."""
import argparse,json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from study_v7.data import verify_data,sha
from ucpa_fl.config import load_config
from ucpa_fl.datasets import load_data

def run(dataset,root):
    proto=json.loads((ROOT/'configs/PROTOCOL.json').read_text())
    frozen=json.loads((ROOT/'configs/FROZEN_CONFIG_HASHES.json').read_text())
    datasets=['mnist','cifar10'] if dataset=='both' else [dataset]
    checks=[]
    for ds in datasets:
        prov=verify_data(ds,root)
        for scenario in proto['scenarios']:
            name=f'{ds}_{scenario}';file=ROOT/'configs'/f'{name}.yaml'
            if sha(file)!=frozen[file.name]:raise ValueError('Frozen config integrity mismatch')
            cfg=load_config(file);cfg.dataset.root=str(root)
            for seed in proto['seed_matrix'][name]:
                b=load_data(cfg,int(seed))
                groups=[b.task_clients,b.surrogate_clients,b.artifact_clients,b.calibration_clients,b.eval_clients]
                arrays=[a.indices for grp in groups for a in grp]
                import numpy as np
                allidx=np.concatenate(arrays)
                if len(allidx)!=len(np.unique(allidx)):raise ValueError('Overlapping partitions')
                nc=cfg.federation.n_clients
                if any(len(b.calibration_clients[i])<proto['teachers_per_client'] or len(b.eval_clients[i])<proto['evaluation_per_client'] for i in range(nc)):
                    raise ValueError(f'Insufficient real records for {name} seed {seed}')
                checks.append({'dataset':ds,'scenario':name,'seed':seed,'n_train':prov['train_count'],
                    'n_test':len(b.test),'min_teacher':min(map(len,b.calibration_clients)),
                    'min_eval':min(map(len,b.eval_clients))})
    return checks
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['mnist','cifar10','both'],default='both')
    ap.add_argument('--data-root',required=True);ap.add_argument('--out',default='preflight_results.json')
    args=ap.parse_args();checks=run(args.dataset,args.data_root)
    Path(args.out).write_text(json.dumps({'status':'PASS','checks':checks},indent=2)+'\n')
    print('PASS REAL DATA PARTITION PREFLIGHT',len(checks),'seed/scenarios')
