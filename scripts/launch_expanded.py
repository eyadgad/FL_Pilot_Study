#!/usr/bin/env python3
"""Frozen multi-run launcher. Parallelism on multiple GPUs; one task per GPU by default.
CPU parallelism available via --workers, with bounded CPU threads per worker.
"""
from __future__ import annotations
import argparse,concurrent.futures,json,os,subprocess,sys,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()

def plan(dataset='both',stage='all'):
    m=json.loads((ROOT/'configs/EXPANDED_STUDY_PLAN.json').read_text())['matrix']
    hashes=json.loads((ROOT/'configs/EXPANDED_CONFIG_HASHES.json').read_text())
    for name,meta in m.items():
        if dataset!='both' and meta['dataset']!=dataset:continue
        if stage!='all' and meta['kind']!=stage:continue
        f=ROOT/'configs/expanded'/name
        if sha(f)!=hashes[name]:raise ValueError(f'Frozen config modified: {f}')
        for seed in meta['seeds']:yield f,meta,seed

def run_one(entry,device,root,output,threads,dry_run=False):
    cfg,meta,seed=entry
    dest=output/cfg.stem/f'seed_{seed}'
    if (dest/'complete_manifest.json').exists():
        report=json.loads((dest/'complete_manifest.json').read_text())
        if report.get('status')=='COMPLETE' and report.get('config_sha256')==sha(cfg):
            return {'config':cfg.stem,'seed':seed,'status':'SKIPPED_COMPLETE'}
        raise RuntimeError(f'Unverified complete output: {dest}')
    if dest.exists():
        if (dest/'train_manifest.json').exists():
            rr=json.loads((dest/'train_manifest.json').read_text())
            if rr.get('status')=='FAIL_TASK_GATE':return {'config':cfg.stem,'seed':seed,'status':'SKIPPED_FAILED_ACCURACY_GATE'}
        raise FileExistsError(f'Existing incomplete run (manual audit required): {dest}')
    cmd=[sys.executable,str(ROOT/'scripts/run_one_expanded.py'),'--config',str(cfg), '--seed',str(seed),
         '--data-root',str(root),'--output-root',str(output),'--device',device,'--threads',str(threads)]
    if dry_run:return {'config':cfg.stem,'seed':seed,'status':'PLANNED','device':device}
    logs=output/'execution_logs';logs.mkdir(parents=True,exist_ok=True)
    logfile=logs/f'{cfg.stem}_seed_{seed}.log'
    start=time.perf_counter()
    with logfile.open('w') as f:
        rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':str(threads),'MKL_NUM_THREADS':str(threads)}).returncode
    if rc!=0:raise RuntimeError(f'Run failed rc={rc} file={logfile}')
    return {'config':cfg.stem,'seed':seed,'status':'EXECUTED','seconds':round(time.perf_counter()-start,1),'log':str(logfile)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['both','mnist','cifar10'],default='both')
    p.add_argument('--stage',choices=['all','primary','clients','rounds','epochs','optimizer','smoke'],default='primary')
    p.add_argument('--data-root',default='data')
    p.add_argument('--output-root',default='expanded_results')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--allow-gpu-sharing',action='store_true',help='Opt-in: allow multiple CUDA processes per GPU; monitor VRAM')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    import torch
    if a.workers<1 or a.threads<1:raise ValueError('workers/threads must be positive')
    available=torch.cuda.device_count()
    mode='cuda' if a.device=='cuda' or (a.device=='auto' and available>0) else 'cpu'
    if mode=='cuda' and available==0:raise RuntimeError('CUDA explicitly requested but no CUDA devices found')
    if mode=='cuda' and a.workers>available and not a.allow_gpu_sharing:
        raise ValueError(f'{a.workers} workers but only {available} GPUs. Use --allow-gpu-sharing only with sufficient VRAM.')
    out=Path(a.output_root).resolve();root=Path(a.data_root).resolve()
    targets=list(plan(a.dataset,a.stage))
    if a.dry_run:
        print(json.dumps({'count':len(targets),'num_workers':a.workers,'dataset':a.dataset,'stage':a.stage,
          'devices':[f'cuda:{i%available}' if mode=='cuda' else 'cpu' for i in range(a.workers)],
          'configs':sorted(set(z[0].stem for z in targets))},indent=2));return
    # Parallel per-GPU subprocesses, not Python threads sharing a CUDA context.
    # No hidden uploads, no automatic data downloads.
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[]
        for i,e in enumerate(targets):
            assigned=f'cuda:{i%available}' if mode=='cuda' else 'cpu'
            futures.append(pool.submit(run_one,e,assigned,root,out,a.threads))
        for f in concurrent.futures.as_completed(futures):print(json.dumps(f.result()),flush=True)

if __name__=='__main__':main()
