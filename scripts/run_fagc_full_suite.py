#!/usr/bin/env python3
"""Full end-to-end real-MNIST study for user execution: train, freeze, qualify, explain, verify.

NO synthetic data and no dataset download. No test-set tuning or automatic re-seeding
if a model fails. Every independent seed has a saved partition hash and checkpoint hash.
"""
from __future__ import annotations
import sys,json,hashlib,argparse,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ucpa_fl.config import load_config
from ucpa_fl.repro import set_seed, resolve_device
from ucpa_fl.datasets import load_data, make_loader
from ucpa_fl.federated import train_fedavg, evaluate
from ucpa_fl.models import MNISTCNN
from scripts.verify_real_mnist import verify
from scripts.run_fagc_generalized import run as run_explanations

class AuditLogger:
    def __init__(self,directory):
        self.run_dir=Path(directory);(self.run_dir/'checkpoints').mkdir(parents=True,exist_ok=True)
        self.fh=open(self.run_dir/'federated_training_metrics.jsonl','a')
    def metric(self,name,value,**kwargs):
        self.fh.write(json.dumps({'type':'metric','name':name,'value':float(value),**kwargs})+'\n');self.fh.flush()
    def event(self,name,**kwargs):
        self.fh.write(json.dumps({'type':'event','name':name,**kwargs})+'\n');self.fh.flush()
    def close(self):self.fh.close()

def _partition_hash(b):
  splits={str(i):{'task':b.task_clients[i].indices.tolist(),'surrogate':b.surrogate_clients[i].indices.tolist(),
     'artifact':b.artifact_clients[i].indices.tolist(),'teacher':b.calibration_clients[i].indices.tolist(),
     'evaluation':b.eval_clients[i].indices.tolist()} for i in range(len(b.task_clients))}
  return hashlib.sha256(json.dumps(splits,sort_keys=True).encode()).hexdigest()

def run_one(config_path,seed,output_root,teacher_n,eval_n,require_existing_checkpoint=False):
  path=Path(config_path).resolve(); cfg=load_config(path)
  if cfg.dataset.name!='mnist' or cfg.dataset.download:raise ValueError('real MNIST offline only')
  if int(seed) not in list(map(int,cfg.seeds)):raise ValueError('seed not frozen in config; do not select after looking at results')
  cfg.dataset.root=str(ROOT/'data');cfg.dataset.test_limit=0
  if verify(ROOT/'data')['status']!='PASS':raise ValueError('MNIST provenance validation failed')
  set_seed(int(seed),True);bundle=load_data(cfg,int(seed))
  if len(bundle.test)!=10000:raise ValueError('full 10k original MNIST test required')
  dest=Path(output_root)/cfg.experiment_name/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
  parhash=_partition_hash(bundle);cfgsha=hashlib.sha256(path.read_bytes()).hexdigest()
  log=AuditLogger(dest);dev=resolve_device('auto')
  ckpt=dest/'checkpoints'/'global_task_model.pt'
  if require_existing_checkpoint:
    if not ckpt.exists():raise FileNotFoundError('No reusable checkpoint')
    meta=json.loads((dest/'train_manifest.json').read_text())
    if meta['config_sha256']!=cfgsha or meta['partition_sha256']!=parhash or meta['seed']!=seed:raise ValueError('frozen training metadata mismatch')
    if meta['checkpoint_sha256']!=hashlib.sha256(ckpt.read_bytes()).hexdigest():raise ValueError('tampered checkpoint')
    model=MNISTCNN();model.load_state_dict(torch.load(ckpt,weights_only=True,map_location='cpu'));model.to(dev).eval()
    task=evaluate(model,make_loader(bundle.test,256,False,seed,0),dev)
  else:
    if ckpt.exists():raise FileExistsError('Fail closed: checkpoint exists. Use --reuse-checkpoint, do not overwrite')
    model,task=train_fedavg(cfg,bundle,int(seed),dev,log)
    if not ckpt.exists():raise RuntimeError('task checkpoint was not saved')
  log.close()
  cksha=hashlib.sha256(ckpt.read_bytes()).hexdigest()
  manifest={'seed':int(seed),'experiment':cfg.experiment_name,'dataset':'MNIST/CSV-original','n_source_train':60000,'n_source_test':10000,
    'n_task_training_limit':cfg.dataset.train_limit,'n_clients':cfg.federation.n_clients,'rounds':cfg.federation.rounds,
    'local_epochs':cfg.federation.local_epochs,'lr':cfg.federation.lr,'batch':cfg.federation.batch_size,
    'config_sha256':cfgsha,'partition_sha256':parhash,'checkpoint_sha256':cksha,'full_test_accuracy':task['accuracy'],
    'full_test_n':10000,'accuracy_gate_min':.95,'status':'PASS' if task['accuracy']>=.95 else 'FAIL_TASK_GATE'}
  (dest/'train_manifest.json').write_text(json.dumps(manifest,indent=2))
  print('Task accuracy (10000 genuine MNIST test)',cfg.experiment_name,seed,task['accuracy'],flush=True)
  if task['accuracy']<.95:
    # No explanation queries are made on an unqualified CNN.
    return {'status':'FAIL_TASK_GATE','manifest':manifest}
  explanation_root=dest/'fagc_explanations'
  result=run_explanations(cfg.shift.kind,int(seed),0,eval_n,explanation_root,path,ckpt,cfg.federation.rounds,teacher_n,device=dev)
  if result['checkpoint_sha256']!=cksha or result['partition_sha256']!=parhash:raise RuntimeError('task vs explanation split/checkpoint mismatch')
  (dest/'complete_manifest.json').write_text(json.dumps({'training':manifest,'evaluation':result},indent=2))
  print('COMPLETED qualified MNIST full pipeline',cfg.experiment_name,seed,flush=True)
  return {'status':'COMPLETE','manifest':manifest,'evaluation':result}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--seed',type=int,required=True)
 p.add_argument('--output-root',default='fagc_full_study_outputs');p.add_argument('--teacher',type=int,default=48)
 p.add_argument('--eval',type=int,default=64);p.add_argument('--threads',type=int,default=3)
 p.add_argument('--reuse-checkpoint',action='store_true');a=p.parse_args()
 torch.set_num_threads(a.threads)
 if a.teacher<1 or a.eval<1:raise SystemExit('Teacher and evaluation budgets must be positive')
 run_one(a.config,a.seed,ROOT/a.output_root,a.teacher,a.eval,a.reuse_checkpoint)
