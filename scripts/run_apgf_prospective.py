#!/usr/bin/env python3
"""Prospective real-MNIST APGF train -> freeze -> accuracy gate -> attribution study.

This is a PROPOSED protocol, not a completed result. Uses original MNIST images
with no synthetic or download path. All method comparisons use one frozen CNN.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8') # before importing torch
import sys,argparse,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from ucpa_fl.config import load_config
from ucpa_fl.repro import set_seed, resolve_device
from ucpa_fl.datasets import load_data
from ucpa_fl.federated import train_fedavg
from scripts.verify_real_mnist import verify
from scripts.smoke_real_mnist import run as eval_apgf


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class Logger:
    def __init__(self,out):
        self.run_dir=Path(out);(self.run_dir/'checkpoints').mkdir(parents=True,exist_ok=True)
        self.f=(self.run_dir/'rounds.jsonl').open('w')
    def metric(self,name,value,**kwargs):
        self.f.write(json.dumps({'kind':'metric','name':name,'value':float(value),**kwargs})+'\n');self.f.flush()
    def event(self,name,**kwargs):
        self.f.write(json.dumps({'kind':'event','name':name,**kwargs})+'\n');self.f.flush()
    def close(self):self.f.close()


def frozen_configs():
    return json.loads((ROOT/'configs'/'APGF_FROZEN_YAML_HASHES.json').read_text())


def train_then_evaluate(cfgpath,seed,outroot,device='auto',reuse=False):
    cfgpath=Path(cfgpath).resolve();rule=frozen_configs()
    if cfgpath.parent!=(ROOT/'configs').resolve() or cfgpath.name not in rule:raise ValueError('Not a frozen APGF config')
    if sha(cfgpath)!=rule[cfgpath.name]:raise ValueError('Frozen config modified after pre-registration')
    cfg=load_config(cfgpath)
    if cfg.dataset.name!='mnist' or cfg.dataset.download or cfg.dataset.train_limit or cfg.dataset.test_limit:
        raise ValueError('Must use complete original MNIST; no downloads or downsampled sources')
    if seed not in cfg.seeds:raise ValueError('Seed not frozen in scenario YAML')
    if cfg.federation.rounds!=25 or cfg.federation.local_epochs!=2 or cfg.federation.n_clients!=5:
        raise ValueError('Prospective train protocol modified')
    dev=resolve_device(device)
    data=verify(ROOT/'data')
    assert data['status']=='PASS' and data['train']['rows']==60000 and data['test']['rows']==10000
    cfg.dataset.root=str(ROOT/'data')
    set_seed(seed,True)
    bundle=load_data(cfg,seed)
    from scripts.run_fagc_full_suite import _partition_hash
    p_hash=_partition_hash(bundle)
    out=Path(outroot).resolve()/cfg.experiment_name/f'seed_{seed}'
    out.mkdir(parents=True,exist_ok=True)
    ck=out/'checkpoints'/'global_task_model.pt'
    mt=out/'train_manifest.json'
    if reuse:
        if not ck.exists() or not mt.exists():raise FileNotFoundError('Missing frozen checkpoint / manifest')
        old=json.loads(mt.read_text())
        if old['seed']!=seed or old['partition_sha256']!=p_hash or old['config_sha256']!=sha(cfgpath) or old['checkpoint_sha256']!=sha(ck):
            raise ValueError('Checkpoint reuse hash mismatch')
    else:
        if ck.exists() or mt.exists():raise FileExistsError('Run already exists; only verified --reuse is allowed')
        log=Logger(out)
        try:
            model,metrics=train_fedavg(cfg,bundle,seed,dev,log)
        finally:
            log.close()
        if not ck.is_file():raise RuntimeError('CNN checkpoint was not saved')
        status='PASS' if metrics['accuracy']>=.95 and metrics['n']==10000 else 'FAIL_TASK_GATE'
        manifest={'seed':seed,'kind':cfg.shift.kind,'scenario':cfg.experiment_name,'n_original_train':60000,'n_original_test':10000,
             'train_limit':0,'full_test_n':metrics['n'],'full_test_accuracy':metrics['accuracy'],
             'rounds':cfg.federation.rounds,'local_epochs':cfg.federation.local_epochs,'n_clients':cfg.federation.n_clients,
             'checkpoint_sha256':sha(ck),'partition_sha256':p_hash,'config_sha256':sha(cfgpath),
             'status':status,'passes_gate':status=='PASS','model_type':'MNISTCNN',
             'dataset_hash':data['source_archive_sha256'],'device':str(dev)}
        mt.write_text(json.dumps(manifest,indent=2)+'\n')
        # Retain exact train/surrogate/artifact/teacher/eval index lists for audit (not pixel data).
        split={}
        for i in range(cfg.federation.n_clients):
            for label,collection in [('task',bundle.task_clients),('surrogate',bundle.surrogate_clients),
                       ('artifact',bundle.artifact_clients),('teacher',bundle.calibration_clients),('evaluation',bundle.eval_clients)]:
                split[f'client{i}_{label}']=collection[i].indices
        np.savez_compressed(out/'partition_indices.npz',**split)
    manifest=json.loads(mt.read_text())
    if not manifest['passes_gate']:raise ValueError('CNN did not pass full 10k accuracy gate, explanations stopped')
    smoke=json.loads((ROOT/'configs'/'smoke_predeclared.json').read_text())
    smoke['final_heldout_eval_per_client']=64
    report=eval_apgf(ROOT,cfg.shift.kind,seed,smoke,out/'apgf_evaluation',cfgpath,ck,mt,device=dev)
    (out/'apgf_complete_manifest.json').write_text(json.dumps({'train_manifest':manifest,'explanation_report':report},indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True,help='configs/apgf_full_<scenario>.yaml, sha256 frozen')
    p.add_argument('--seed',required=True,type=int)
    p.add_argument('--output-root',default='apgf_prospective_outputs')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--threads',type=int,default=3)
    p.add_argument('--reuse',action='store_true',help='requires verified checkpoint metadata')
    a=p.parse_args();torch.set_num_threads(a.threads)
    train_then_evaluate(a.config,a.seed,a.output_root,a.device,a.reuse)
